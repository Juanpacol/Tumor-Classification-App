"""One training run: folds, features, selection, ablation and audit on Training; a
final run then calibrates, scores Testing once and writes the artifact."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.contracts import TumorLabel
from src.deep_learning.bundle import sha256_of
from src.deep_learning.extractor import DeepFeatureExtractor
from src.seed import set_seed
from src.training.ablation import ablate_blocks
from src.training.artifact import write_artifact
from src.training.augmentation import AugmentationConfig, keeps_augmentation
from src.training.calibration import (
    CALIBRATION_ENSEMBLE,
    calibrate,
    calibrated_out_of_fold,
)
from src.training.config import TrainingConfig, resolved_config
from src.training.dataset import Split, load_samples, split_samples
from src.training.diagnosis import error_breakdown, predictions_table
from src.training.evaluation import evaluate
from src.training.features import (
    CopyCache,
    Digest,
    FeatureCache,
    FeatureTable,
    config_fingerprint,
    copy_rows,
    make_cnn_input,
    make_copy_featurizer,
    make_featurizer,
)
from src.training.folds import (
    FoldRow,
    Splits,
    assign_folds,
    fold_record,
    fold_splits,
    used_per_fold_and_class,
)
from src.training.guarantees import leakage_guarantees
from src.training.model import (
    CandidateScore,
    SelectionResult,
    build_model,
    candidate_classifiers,
    compare_classifiers,
    cross_validate_predictions,
)
from src.training.out_of_fold import out_of_fold_metrics
from src.training.report import build_report, with_test, write_report
from src.training.shortcut import metadata_baseline, recall_by_source_size
from src.training.stats import feature_stats

LABELS = [label.value for label in TumorLabel]
NO_TUMOR = TumorLabel.NO_TUMOR.value


@dataclass(frozen=True)
class TrainingResult:
    """Where a run wrote its results; ``artifact_dir`` is ``None`` for a tuning run."""

    artifact_dir: Path | None
    report_dir: Path
    report: dict[str, Any]


def train(config: TrainingConfig, now: datetime | None = None) -> TrainingResult:
    """Run one round and write its report; a final run also writes the artifact.

    A tuning run (``config.final`` false) never reads a Testing feature. A final
    run then calibrates, scores Testing once and writes the artifact. Raises
    ``FileExistsError`` before any work if a final run's artifact exists, and
    ``BackboneBundleError`` before any image is read if the backbone bundle is
    missing or declares another backbone.
    """
    now = now or datetime.now(UTC)
    artifact_dir = config.out_dir / f"model-v{config.version}"
    report_dir = config.out_dir / "reports" / now.strftime("%Y%m%dT%H%M%SZ")
    if config.final and artifact_dir.exists():
        raise FileExistsError(f"artifact already exists: {artifact_dir}")
    set_seed(config.seed)

    extraction = _extraction(config)
    split = split_samples(load_samples(config.data_dir, config.manifest))
    if not split.train:
        raise ValueError("no usable training images")
    fold_of = assign_folds(split.train, config.n_splits, config.seed)
    tuning = _tune(config, split, fold_of, _train_features(config, extraction, split))
    report = _report(config, now, split, tuning)
    if config.final:
        test_table, skipped = _test_features(config, extraction, split)
        final = _finalize(config, tuning, test_table)
        report = _with_final(config, report, tuning, test_table, skipped, final)
        _write_artifact(config, artifact_dir, report, tuning, final)
    write_report(report_dir, report, tuning.folds, tuning.oof_rows)
    return TrainingResult(artifact_dir if config.final else None, report_dir, report)


@dataclass(frozen=True)
class _Rows:
    """Feature matrix, labels and the shared splits over its rows."""

    x: np.ndarray
    y: np.ndarray
    splits: Splits


@dataclass(frozen=True)
class _Tuning:
    """Everything a run learns from Training alone."""

    table: FeatureTable
    skipped: list[str]
    folds: list[FoldRow]
    rows: _Rows
    model: Pipeline
    selection: SelectionResult
    augmentation: dict[str, Any]
    cv_out_of_fold: dict[str, Any]
    cv_calibrated: dict[str, Any]
    oof_rows: list[dict[str, Any]]
    cv_error_breakdown: dict[str, Any]
    ablation: dict[str, float]
    metadata_baseline: dict[str, Any]
    guarantees: list[dict[str, Any]]


@dataclass(frozen=True)
class _Final:
    calibrated: BaseEstimator
    test: dict[str, Any]
    no_tumor_recall_by_source_size: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class _Extraction:
    """What both splits share to read or compute their features."""

    extractor: DeepFeatureExtractor
    fingerprint: str
    cache_dir: Path
    digest: Digest


@dataclass(frozen=True)
class _TrainFeatures:
    table: FeatureTable
    skipped: list[str]
    copy_cnn: np.ndarray


def _tune(
    config: TrainingConfig,
    split: Split,
    fold_of: dict[Path, int],
    features: _TrainFeatures,
) -> _Tuning:
    """Selection, augmentation decision, out-of-fold metrics, ablation and shortcut
    audit, from Training rows only."""
    table = features.table
    if not table.samples:
        raise ValueError("no usable training images")
    x_train, y_train = table.matrix, _labels(table)
    fold_of_row = np.array([fold_of[s.path] for s in table.samples])
    splits = fold_splits(fold_of_row, config.n_splits)
    folds = fold_record(
        split.train, fold_of, {s.path for s in table.samples}, config.data_dir
    )
    n_radiomic = table.radiomics.shape[1]
    plain = _Rows(x_train, y_train, splits)
    augmented_rows = _with_copies(
        plain,
        fold_of_row,
        copy_rows(table, features.copy_cnn, config.augmentation.copies),
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
        config.model.blocks,
    )
    augmented = keeps_augmentation(
        without.mean_macro_f1, without.std_macro_f1, with_copies.mean_macro_f1
    )
    rows, chosen = (augmented_rows, with_copies) if augmented else (plain, without)
    predictions = (
        copies_predictions if augmented else selection.predictions[selection.best]
    )
    oof_rows = predictions_table(
        table.samples, config.data_dir, fold_of_row, table.quality, predictions
    )
    model = build_model(
        classifier,
        n_radiomic,
        table.cnn.shape[1],
        config.seed,
        pca_components,
        config.model.blocks,
    )
    return _Tuning(
        table=table,
        skipped=features.skipped,
        folds=folds,
        rows=rows,
        model=model,
        selection=selection,
        augmentation=_augmentation_summary(
            config.augmentation, augmented, without, with_copies
        ),
        cv_out_of_fold={
            "augmented": augmented,
            **asdict(out_of_fold_metrics(y_train, predictions, LABELS)),
        },
        oof_rows=oof_rows,
        cv_error_breakdown=error_breakdown(oof_rows, LABELS, NO_TUMOR),
        cv_calibrated=asdict(
            calibrated_out_of_fold(
                model,
                rows.x,
                rows.y,
                rows.splits,
                len(y_train),
                config.model.calibration,
                LABELS,
            )
        ),
        ablation=ablate_blocks(
            rows.x,
            rows.y,
            n_radiomic,
            classifier,
            rows.splits,
            chosen.mean_macro_f1,
            config.seed,
            pca_components,
            config.model.blocks,
        ),
        metadata_baseline=metadata_baseline(
            table.samples, y_train, splits, config.seed
        ),
        guarantees=leakage_guarantees(split, rows.splits, len(y_train)),
    )


def _finalize(
    config: TrainingConfig, tuning: _Tuning, test_table: FeatureTable
) -> _Final:
    """Calibrate the selected pipeline on the shared folds and score Testing once."""
    rows = tuning.rows
    calibrated = calibrate(
        tuning.model, rows.x, rows.y, rows.splits, config.model.calibration
    )
    y_test = _labels(test_table)
    predictions = calibrated.predict(test_table.matrix)
    return _Final(
        calibrated,
        asdict(evaluate(y_test, predictions, LABELS)),
        recall_by_source_size(test_table.samples, y_test, predictions, NO_TUMOR),
    )


def _report(
    config: TrainingConfig, now: datetime, split: Split, tuning: _Tuning
) -> dict[str, Any]:
    table = tuning.table
    validation = {
        "method": "StratifiedGroupKFold",
        "group": "group_id",
        "n_splits": config.n_splits,
        "seed": config.seed,
        "used_per_fold_and_class": used_per_fold_and_class(
            tuning.folds, config.n_splits, LABELS
        ),
    }
    dataset = {
        "train_images": len(table.samples),
        "train_class_counts": dict(Counter(s.label.value for s in table.samples)),
        "split_removed_per_class": {
            label.value: count for label, count in split.removed_per_class.items()
        },
        "skipped_train": len(tuning.skipped),
        "skipped_reasons": {"Train": tuning.skipped},
    }
    return build_report(
        mode="final" if config.final else "tuning",
        created_at=now.isoformat(),
        version=config.version,
        seed=config.seed,
        dataset=dataset,
        validation=validation,
        selection=_selection_dict(tuning.selection),
        ablation=tuning.ablation,
        augmentation=tuning.augmentation,
        cv_out_of_fold=tuning.cv_out_of_fold,
        cv_calibrated=tuning.cv_calibrated,
        cv_error_breakdown=tuning.cv_error_breakdown,
        shortcut_audit={"metadata_baseline": tuning.metadata_baseline},
        guarantees=tuning.guarantees,
        config=resolved_config(config),
        config_source=asdict(config.round_source) if config.round_source else None,
    )


def _with_final(
    config: TrainingConfig,
    report: dict[str, Any],
    tuning: _Tuning,
    test_table: FeatureTable,
    skipped: list[str],
    final: _Final,
) -> dict[str, Any]:
    """``report`` with the Testing results of a final run."""
    dataset = {
        **report["dataset"],
        "test_images": len(test_table.samples),
        "skipped_test": len(skipped),
        "skipped_reasons": {"Train": tuning.skipped, "Test": skipped},
    }
    audit = {
        **report["shortcut_audit"],
        "no_tumor_test_recall_by_source_size": final.no_tumor_recall_by_source_size,
    }
    return with_test(
        {**report, "dataset": dataset, "shortcut_audit": audit},
        final.test,
        config.model.calibration,
    )


def _write_artifact(
    config: TrainingConfig,
    artifact_dir: Path,
    report: dict[str, Any],
    tuning: _Tuning,
    final: _Final,
) -> None:
    table = tuning.table
    resolved = report["config"]
    write_artifact(
        artifact_dir,
        final.calibrated,
        config.backbone_bundle,
        feature_stats(table.radiomics, table.radiomic_names),
        {
            "version": config.version,
            "created_at": report["created_at"],
            "seed": config.seed,
            "classifier": tuning.selection.best,
            "calibration": {
                "method": config.model.calibration,
                "n_splits": config.n_splits,
                "ensemble": CALIBRATION_ENSEMBLE,
            },
            "preprocessing": asdict(config.preprocessing),
            "radiomics": asdict(config.radiomics),
            "quality_thresholds": asdict(config.preprocessing.thresholds),
            "radiomic_feature_names": list(table.radiomic_names),
            "augmentation": tuning.augmentation,
            "metrics": {
                "test": final.test,
                **report["selection"],
                "ablation": tuning.ablation,
                "cv_out_of_fold": tuning.cv_out_of_fold,
                "cv_calibrated": tuning.cv_calibrated,
            },
            "shortcut_audit": report["shortcut_audit"],
            "dataset": report["dataset"],
            # preprocessing, radiomics and augmentation are above; config.json in the
            # report folder holds the whole resolved configuration in one piece.
            "config": {key: resolved[key] for key in ("run", "model", "classifiers")},
            "config_source": report["config_source"],
        },
    )


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


class BackboneBundleError(ValueError):
    """The weights directory has no usable bundle for the configured backbone."""


def _extraction(config: TrainingConfig) -> _Extraction:
    """The extractor of the configured backbone, checked before any image is read."""
    backbone, folder = config.model.backbone, config.backbone_bundle
    if not folder.is_dir():
        raise BackboneBundleError(
            f"no {backbone} bundle in {folder}: export it with "
            f"src.deep_learning.export_backbone --backbone {backbone} --out {folder}"
        )
    extractor = DeepFeatureExtractor.from_bundle(folder, config.preprocessing)
    if extractor.manifest.backbone != backbone:
        raise BackboneBundleError(
            f"{folder} holds a {extractor.manifest.backbone} bundle, "
            f"but the round uses {backbone}"
        )
    return _Extraction(
        extractor,
        config_fingerprint(config.preprocessing, config.radiomics, extractor.manifest),
        config.cache_dir or config.out_dir / "feature-cache",
        cache(sha256_of),
    )


def _train_features(
    config: TrainingConfig, extraction: _Extraction, split: Split
) -> _TrainFeatures:
    """Feature table of the Training split and the CNN features of its copies.

    Copies are made only of the Training images the gate keeps, never of Testing.
    """
    copy_cache = CopyCache(
        extraction.cache_dir / extraction.fingerprint,
        config.augmentation,
        config.seed,
        make_copy_featurizer(extraction.extractor),
        make_cnn_input(config.preprocessing),
        extraction.digest,
    )
    featurize, _ = make_featurizer(
        extraction.extractor, config.preprocessing, config.radiomics, copy_cache.add
    )
    table, skipped = _cache(config, extraction, featurize).table(split.train)
    return _TrainFeatures(table, _reasons(skipped), copy_cache.cnn(table.samples))


def _test_features(
    config: TrainingConfig, extraction: _Extraction, split: Split
) -> tuple[FeatureTable, list[str]]:
    """Feature table of the Testing split; only a final run reads it."""
    featurize, _ = make_featurizer(
        extraction.extractor, config.preprocessing, config.radiomics
    )
    table, skipped = _cache(config, extraction, featurize).table(split.test)
    return table, _reasons(skipped)


def _cache(
    config: TrainingConfig, extraction: _Extraction, featurize: Any
) -> FeatureCache:
    return FeatureCache(
        extraction.cache_dir,
        extraction.fingerprint,
        featurize,
        config.radiomics,
        extraction.digest,
    )


def _reasons(skipped: list[Any]) -> list[str]:
    return [f"{s.path.name}: {s.reason}" for s in skipped]


def _labels(table: FeatureTable) -> np.ndarray:
    return np.array([s.label.value for s in table.samples])


def _selection_dict(selection: SelectionResult) -> dict[str, Any]:
    return {
        "scores": {name: asdict(score) for name, score in selection.scores.items()},
        "best": selection.best,
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
