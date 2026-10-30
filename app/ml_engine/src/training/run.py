"""One training run: folds, features, selection, ablation, calibration, test and artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.classical.config import RadiomicsConfig
from src.contracts import TumorLabel
from src.deep_learning.extractor import DeepFeatureExtractor
from src.preprocessing.prepare import PreprocessingConfig
from src.seed import set_seed
from src.training.ablation import ablate_blocks
from src.training.artifact import write_artifact
from src.training.calibration import (
    CALIBRATION_ENSEMBLE,
    CALIBRATION_METHOD,
    calibrate,
)
from src.training.dataset import Split, load_samples, split_samples
from src.training.evaluation import evaluate
from src.training.features import FeatureCache, FeatureTable, make_featurizer
from src.training.folds import (
    N_SPLITS,
    assign_folds,
    fold_record,
    fold_splits,
    used_per_fold_and_class,
)
from src.training.model import (
    DEFAULT_SEED,
    build_model,
    candidate_classifiers,
    compare_classifiers,
)
from src.training.report import build_report, write_report
from src.training.stats import feature_stats

LABELS = [label.value for label in TumorLabel]


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    manifest: Path
    backbone_bundle: Path
    out_dir: Path
    seed: int = DEFAULT_SEED
    n_splits: int = N_SPLITS
    version: str = "0.0.0-dev"
    cache_dir: Path | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    radiomics: RadiomicsConfig = field(default_factory=RadiomicsConfig)


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
    train_table, test_table, skipped = _feature_tables(config, split)
    if not train_table.samples:
        raise ValueError("no usable training images")
    x_train, y_train = _training_arrays(train_table)
    splits = fold_splits(
        np.array([fold_of[s.path] for s in train_table.samples]), config.n_splits
    )
    folds = fold_record(
        split.train,
        fold_of,
        {s.path for s in train_table.samples},
        config.data_dir,
    )
    n_radiomic = train_table.radiomics.shape[1]

    selection = compare_classifiers(x_train, y_train, n_radiomic, splits, config.seed)
    ablation = ablate_blocks(
        x_train,
        y_train,
        n_radiomic,
        selection.best,
        splits,
        selection.scores[selection.best].mean_macro_f1,
        config.seed,
    )
    model = build_model(
        candidate_classifiers(config.seed)[selection.best],
        n_radiomic,
        train_table.cnn.shape[1],
        config.seed,
    )
    calibrated = calibrate(model, x_train, y_train, splits)
    test = evaluate(
        calibrated,
        test_table.matrix,
        np.array([s.label.value for s in test_table.samples]),
        LABELS,
    )

    created_at = now.isoformat()
    dataset = _dataset_summary(
        train_table, test_table, skipped, split.removed_per_class
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
    report = build_report(
        created_at,
        config.version,
        config.seed,
        dataset,
        validation,
        selection_dict,
        ablation,
        asdict(test),
        CALIBRATION_METHOD,
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
            "metrics": {"test": asdict(test), **selection_dict, "ablation": ablation},
            "dataset": dataset,
        },
    )
    write_report(report_dir, report, folds)
    return TrainingResult(artifact_dir, report_dir, report)


def _feature_tables(
    config: TrainingConfig, split: Split
) -> tuple[FeatureTable, FeatureTable, dict[str, list[str]]]:
    extractor = DeepFeatureExtractor.from_bundle(
        config.backbone_bundle, config.preprocessing
    )
    featurize, fingerprint = make_featurizer(
        extractor, config.preprocessing, config.radiomics
    )
    cache = FeatureCache(
        config.cache_dir or config.out_dir / "feature-cache",
        fingerprint,
        featurize,
        config.radiomics,
    )
    train_table, train_skipped = cache.table(split.train)
    test_table, test_skipped = cache.table(split.test)
    skipped = {
        "Train": [f"{s.path.name}: {s.reason}" for s in train_skipped],
        "Test": [f"{s.path.name}: {s.reason}" for s in test_skipped],
    }
    return train_table, test_table, skipped


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
