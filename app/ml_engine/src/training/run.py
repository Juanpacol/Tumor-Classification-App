"""One training run: folds, features, selection, ablation, calibration, test and artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import TumorLabel
from src.deep_learning.bundle import sha256_of
from src.deep_learning.extractor import DeepFeatureExtractor
from src.seed import set_seed
from src.training.ablation import ablate_blocks
from src.training.artifact import write_artifact
from src.training.augmentation import AugmentationConfig, keeps_augmentation
from src.training.calibration import (
    CALIBRATION_ENSEMBLE,
    CALIBRATION_METHOD,
    calibrate,
)
from src.training.config import TrainingConfig, resolved_config
from src.training.dataset import Split, load_samples, split_samples
from src.training.evaluation import evaluate
from src.training.features import (
    CopyCache,
    FeatureCache,
    FeatureTable,
    config_fingerprint,
    copy_rows,
    make_cnn_input,
    make_copy_featurizer,
    make_featurizer,
)
from src.training.folds import (
    Splits,
    assign_folds,
    fold_record,
    fold_splits,
    used_per_fold_and_class,
)
from src.training.guarantees import leakage_guarantees
from src.training.model import (
    CandidateScore,
    build_model,
    candidate_classifiers,
    compare_classifiers,
    cross_validate_predictions,
)
from src.training.out_of_fold import out_of_fold_metrics
from src.training.report import build_report, write_report
from src.training.shortcut import metadata_baseline, recall_by_source_size
from src.training.stats import feature_stats

LABELS = [label.value for label in TumorLabel]
NO_TUMOR = TumorLabel.NO_TUMOR.value


@dataclass(frozen=True)
class TrainingResult:
    artifact_dir: Path
    report_dir: Path
    report: dict[str, Any]


def train(config: TrainingConfig, now: datetime | None = None) -> TrainingResult:
    """Run the whole pipeline and write the artifact and the report.

    The official Test split is scored once, at the end, with the calibrated
    model. Raises ``FileExistsError`` before any work if the artifact exists.
    """
    now = now or datetime.now(UTC)
    artifact_dir = config.out_dir / f"model-v{config.version}"
    report_dir = config.out_dir / "reports" / now.strftime("%Y%m%dT%H%M%SZ")
    if artifact_dir.exists():
        raise FileExistsError(f"artifact already exists: {artifact_dir}")
    set_seed(config.seed)

    split = split_samples(load_samples(config.data_dir, config.manifest))
    if not split.train:
        raise ValueError("no usable training images")
    fold_of = assign_folds(split.train, config.n_splits, config.seed)
    features = _features(config, split)
    train_table, test_table = features.train, features.test
    if not train_table.samples:
        raise ValueError("no usable training images")
    x_train, y_train = _training_arrays(train_table)
    fold_of_row = np.array([fold_of[s.path] for s in train_table.samples])
    splits = fold_splits(fold_of_row, config.n_splits)
    folds = fold_record(
        split.train,
        fold_of,
        {s.path for s in train_table.samples},
        config.data_dir,
    )
    n_radiomic = train_table.radiomics.shape[1]
    plain = _Rows(x_train, y_train, splits)
    augmented_rows = _with_copies(
        plain,
        fold_of_row,
        copy_rows(train_table, features.copy_cnn, config.augmentation.copies),
        config.augmentation.copies,
        config.n_splits,
    )

    selection = compare_classifiers(
        x_train, y_train, n_radiomic, splits, config.seed, config.model
    )
    pca_components = config.model.pca_components
    classifier = candidate_classifiers(config.seed, config.model.classifier_params)[
        selection.best
    ]
    without = selection.scores[selection.best]
    with_copies, copies_predictions = cross_validate_predictions(
        classifier,
        augmented_rows.x,
        augmented_rows.y,
        n_radiomic,
        augmented_rows.splits,
        len(y_train),
        config.seed,
        pca_components,
    )
    augmented = keeps_augmentation(
        without.mean_macro_f1, without.std_macro_f1, with_copies.mean_macro_f1
    )
    rows, chosen = (augmented_rows, with_copies) if augmented else (plain, without)
    predictions = (
        copies_predictions if augmented else selection.predictions[selection.best]
    )
    cv_out_of_fold = {
        "augmented": augmented,
        **asdict(out_of_fold_metrics(y_train, predictions, LABELS)),
    }
    ablation = ablate_blocks(
        rows.x,
        rows.y,
        n_radiomic,
        classifier,
        rows.splits,
        chosen.mean_macro_f1,
        config.seed,
        pca_components,
    )
    model = build_model(
        classifier, n_radiomic, train_table.cnn.shape[1], config.seed, pca_components
    )
    guarantees = leakage_guarantees(split, rows.splits, len(y_train))
    baseline = metadata_baseline(train_table.samples, y_train, splits, config.seed)
    calibrated = calibrate(model, rows.x, rows.y, rows.splits)
    y_test = np.array([s.label.value for s in test_table.samples])
    test_predictions = calibrated.predict(test_table.matrix)
    test = evaluate(y_test, test_predictions, LABELS)
    shortcut_audit = {
        "metadata_baseline": baseline,
        "no_tumor_test_recall_by_source_size": recall_by_source_size(
            test_table.samples, y_test, test_predictions, NO_TUMOR
        ),
    }

    created_at = now.isoformat()
    dataset = _dataset_summary(
        train_table, test_table, features.skipped, split.removed_per_class
    )
    augmentation = _augmentation_summary(
        config.augmentation, augmented, without, with_copies
    )
    selection_dict = {
        "scores": {name: asdict(score) for name, score in selection.scores.items()},
        "best": selection.best,
    }
    validation = {
        "method": "StratifiedGroupKFold",
        "group": "group_id",
        "n_splits": config.n_splits,
        "seed": config.seed,
        "used_per_fold_and_class": used_per_fold_and_class(
            folds, config.n_splits, LABELS
        ),
    }
    resolved = resolved_config(config)
    source = asdict(config.round_source) if config.round_source else None
    report = build_report(
        created_at,
        config.version,
        config.seed,
        dataset,
        validation,
        selection_dict,
        ablation,
        augmentation,
        cv_out_of_fold,
        asdict(test),
        CALIBRATION_METHOD,
        shortcut_audit,
        guarantees,
        resolved,
        source,
    )
    write_artifact(
        artifact_dir,
        calibrated,
        config.backbone_bundle,
        feature_stats(train_table.radiomics, train_table.radiomic_names),
        {
            "version": config.version,
            "created_at": created_at,
            "seed": config.seed,
            "classifier": selection.best,
            "calibration": {
                "method": CALIBRATION_METHOD,
                "n_splits": config.n_splits,
                "ensemble": CALIBRATION_ENSEMBLE,
            },
            "preprocessing": asdict(config.preprocessing),
            "radiomics": asdict(config.radiomics),
            "quality_thresholds": asdict(config.preprocessing.thresholds),
            "radiomic_feature_names": list(train_table.radiomic_names),
            "augmentation": augmentation,
            "metrics": {
                "test": asdict(test),
                **selection_dict,
                "ablation": ablation,
                "cv_out_of_fold": cv_out_of_fold,
            },
            "shortcut_audit": shortcut_audit,
            "dataset": dataset,
            # preprocessing, radiomics and augmentation are above; config.json in the
            # report folder holds the whole resolved configuration in one piece.
            "config": {key: resolved[key] for key in ("run", "model", "classifiers")},
            "config_source": source,
        },
    )
    write_report(report_dir, report, folds)
    return TrainingResult(artifact_dir, report_dir, report)


@dataclass(frozen=True)
class _Rows:
    """Feature matrix, labels and the shared splits over its rows."""

    x: np.ndarray
    y: np.ndarray
    splits: Splits


def _with_copies(
    plain: _Rows,
    fold_of_row: np.ndarray,
    copy_x: np.ndarray,
    copies: int,
    n_splits: int,
) -> _Rows:
    """The original rows followed by their copies, each copy in its original's fold.

    Originals come first, so every fold keeps the same validation rows.
    """
    folds = np.concatenate([fold_of_row, np.repeat(fold_of_row, copies)])
    is_copy = np.arange(len(folds)) >= len(fold_of_row)
    return _Rows(
        np.vstack([plain.x, copy_x]),
        np.concatenate([plain.y, np.repeat(plain.y, copies)]),
        fold_splits(folds, n_splits, is_copy),
    )


@dataclass(frozen=True)
class _Features:
    train: FeatureTable
    test: FeatureTable
    skipped: dict[str, list[str]]
    copy_cnn: np.ndarray


def _features(config: TrainingConfig, split: Split) -> _Features:
    """Feature tables of both splits and the CNN features of the Training copies.

    Copies are made only of the Training images the gate keeps, never of Testing.
    """
    extractor = DeepFeatureExtractor.from_bundle(
        config.backbone_bundle, config.preprocessing
    )
    fingerprint = config_fingerprint(
        config.preprocessing, config.radiomics, extractor.manifest
    )
    cache_dir = config.cache_dir or config.out_dir / "feature-cache"
    digest = cache(sha256_of)
    copy_cache = CopyCache(
        cache_dir / fingerprint,
        config.augmentation,
        config.seed,
        make_copy_featurizer(extractor),
        make_cnn_input(config.preprocessing),
        digest,
    )
    featurize_train, _ = make_featurizer(
        extractor, config.preprocessing, config.radiomics, copy_cache.add
    )
    featurize_test, _ = make_featurizer(
        extractor, config.preprocessing, config.radiomics
    )
    train_table, train_skipped = FeatureCache(
        cache_dir, fingerprint, featurize_train, config.radiomics, digest
    ).table(split.train)
    test_table, test_skipped = FeatureCache(
        cache_dir, fingerprint, featurize_test, config.radiomics, digest
    ).table(split.test)
    skipped = {
        "Train": [f"{s.path.name}: {s.reason}" for s in train_skipped],
        "Test": [f"{s.path.name}: {s.reason}" for s in test_skipped],
    }
    return _Features(
        train_table, test_table, skipped, copy_cache.cnn(train_table.samples)
    )


def _training_arrays(table: FeatureTable) -> tuple[np.ndarray, np.ndarray]:
    return table.matrix, np.array([s.label.value for s in table.samples])


def _dataset_summary(
    train: FeatureTable,
    test: FeatureTable,
    skipped: dict[str, list[str]],
    removed: dict[TumorLabel, int],
) -> dict[str, Any]:
    return {
        "train_images": len(train.samples),
        "test_images": len(test.samples),
        "train_class_counts": dict(Counter(s.label.value for s in train.samples)),
        "split_removed_per_class": {
            label.value: count for label, count in removed.items()
        },
        "skipped_train": len(skipped["Train"]),
        "skipped_test": len(skipped["Test"]),
        "skipped_reasons": skipped,
    }


def _augmentation_summary(
    config: AugmentationConfig,
    on: bool,
    without: CandidateScore,
    with_copies: CandidateScore,
) -> dict[str, Any]:
    ranges = {k: v for k, v in asdict(config).items() if k != "copies"}
    return {
        "on": on,
        "copies": config.copies,
        "ranges": ranges,
        "cv_without": {"mean": without.mean_macro_f1, "std": without.std_macro_f1},
        "cv_with": {"mean": with_copies.mean_macro_f1, "std": with_copies.std_macro_f1},
        "margin": without.std_macro_f1,
    }
