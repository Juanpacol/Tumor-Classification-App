"""Hybrid classifier pipeline and cross-validated classifier selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.training.folds import Splits
from src.training.out_of_fold import out_of_fold_predictions

DEFAULT_SEED = 42
PCA_COMPONENTS = 32


BLOCKS = ("radiomic", "cnn")

GUARDED_PARAMS = frozenset({"class_weight", "random_state"})
"""Classifier parameters a training round may not change: classes stay balanced and
randomness comes from the run seed."""


@dataclass(frozen=True)
class ModelConfig:
    """Tunable model settings: PCA size of the radiomic block and hyperparameter
    overrides per candidate classifier, keyed by candidate name."""

    pca_components: int = PCA_COMPONENTS
    classifier_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pca_components < 1:
            raise ValueError("pca_components must be at least 1")


def build_model(
    classifier: BaseEstimator,
    n_radiomic: int,
    n_cnn: int,
    seed: int = DEFAULT_SEED,
    pca_components: int = PCA_COMPONENTS,
    blocks: tuple[str, ...] = BLOCKS,
) -> Pipeline:
    """Scaler and PCA on the radiomic block, scaler on the CNN block, then the classifier.

    The input is the radiomic columns followed by the CNN columns, and ``blocks``
    chooses which of them the classifier sees (the others are dropped). Every
    fitted step lives inside this pipeline, so cross-validation refits them per
    fold and nothing leaks from the validation fold.
    """
    radiomic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "pca",
                PCA(n_components=min(pca_components, n_radiomic), random_state=seed),
            ),
        ]
    )
    available = {
        "radiomic": (radiomic, list(range(n_radiomic))),
        "cnn": (StandardScaler(), list(range(n_radiomic, n_radiomic + n_cnn))),
    }
    features = ColumnTransformer(
        [(name, *available[name]) for name in blocks if name in available]
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def candidate_classifiers(
    seed: int = DEFAULT_SEED, params: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, BaseEstimator]:
    """The classifiers that are compared, with ``params`` overriding hyperparameters
    per candidate. Class weights are balanced because the cleaned dataset is
    unbalanced (for example 275 'notumor' against 992 'glioma' in Train)."""
    params = params or {}
    if guarded := {k for p in params.values() for k in p} & GUARDED_PARAMS:
        raise ValueError(
            f"classifier parameters cannot be overridden: {sorted(guarded)}"
        )
    defaults = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
    }
    if unknown := set(params) - set(defaults):
        raise ValueError(f"unknown classifiers: {sorted(unknown)}")
    return {
        name: classifier.set_params(**params.get(name, {}))
        for name, classifier in defaults.items()
    }


@dataclass(frozen=True)
class CandidateScore:
    mean_macro_f1: float
    std_macro_f1: float
    fold_macro_f1: tuple[float, ...]
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    fold_balanced_accuracy: tuple[float, ...]


@dataclass(frozen=True)
class SelectionResult:
    """Scores per candidate, the selected one and each candidate's out-of-fold
    predictions of the original rows."""

    scores: dict[str, CandidateScore]
    best: str
    predictions: dict[str, np.ndarray] = field(
        default_factory=dict, compare=False, repr=False
    )


def cross_validate_scores(
    classifier: BaseEstimator,
    matrix: np.ndarray,
    y: np.ndarray,
    n_radiomic: int,
    splits: Splits,
    seed: int = DEFAULT_SEED,
    blocks: tuple[str, ...] = BLOCKS,
    pca_components: int = PCA_COMPONENTS,
) -> CandidateScore:
    """Macro-F1 and balanced accuracy of one classifier over ``splits``, refitting
    every step per fold."""
    n_cnn = matrix.shape[1] - n_radiomic
    model = build_model(classifier, n_radiomic, n_cnn, seed, pca_components, blocks)
    return _scores(_cross_validate(model, matrix, y, splits))


def cross_validate_predictions(
    classifier: BaseEstimator,
    matrix: np.ndarray,
    y: np.ndarray,
    n_radiomic: int,
    splits: Splits,
    n_originals: int,
    seed: int = DEFAULT_SEED,
    pca_components: int = PCA_COMPONENTS,
) -> tuple[CandidateScore, np.ndarray]:
    """Scores as ``cross_validate_scores``, plus the out-of-fold predictions of the
    first ``n_originals`` rows made by the same fitted fold models."""
    n_cnn = matrix.shape[1] - n_radiomic
    model = build_model(classifier, n_radiomic, n_cnn, seed, pca_components)
    folds = _cross_validate(model, matrix, y, splits, keep_models=True)
    predictions = out_of_fold_predictions(
        folds["estimator"], folds["indices"]["test"], matrix, y, n_originals
    )
    return _scores(folds), predictions


def _cross_validate(
    model: Pipeline,
    matrix: np.ndarray,
    y: np.ndarray,
    splits: Splits,
    keep_models: bool = False,
) -> dict[str, Any]:
    return cross_validate(
        model,
        matrix,
        y,
        cv=splits,
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        return_estimator=keep_models,
        return_indices=keep_models,
    )


def _scores(folds: dict[str, Any]) -> CandidateScore:
    f1, balanced = folds["test_macro_f1"], folds["test_balanced_accuracy"]
    return CandidateScore(
        float(f1.mean()),
        float(f1.std()),
        tuple(float(v) for v in f1),
        float(balanced.mean()),
        float(balanced.std()),
        tuple(float(v) for v in balanced),
    )


def compare_classifiers(
    matrix: np.ndarray,
    y: np.ndarray,
    n_radiomic: int,
    splits: Splits,
    seed: int = DEFAULT_SEED,
    config: ModelConfig | None = None,
) -> SelectionResult:
    """Cross-validate every candidate and pick the best mean macro-F1 (ties: by name).

    Balanced accuracy is reported but does not take part in the choice.
    """
    config = config or ModelConfig()
    results = {
        name: cross_validate_predictions(
            classifier,
            matrix,
            y,
            n_radiomic,
            splits,
            len(y),
            seed,
            config.pca_components,
        )
        for name, classifier in candidate_classifiers(
            seed, config.classifier_params
        ).items()
    }
    scores = {name: score for name, (score, _) in results.items()}
    best = max(sorted(scores), key=lambda name: scores[name].mean_macro_f1)
    predictions = {name: predicted for name, (_, predicted) in results.items()}
    return SelectionResult(scores, best, predictions)
