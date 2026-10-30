"""Hybrid classifier pipeline and cross-validated classifier selection."""

from __future__ import annotations

from dataclasses import dataclass

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

DEFAULT_SEED = 42
PCA_COMPONENTS = 32


BLOCKS = ("radiomic", "cnn")


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


def candidate_classifiers(seed: int = DEFAULT_SEED) -> dict[str, BaseEstimator]:
    """The classifiers that are compared. Class weights are balanced because the cleaned
    dataset is unbalanced (for example 275 'notumor' against 992 'glioma' in Train)."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
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
    scores: dict[str, CandidateScore]
    best: str


def cross_validate_scores(
    classifier: BaseEstimator,
    matrix: np.ndarray,
    y: np.ndarray,
    n_radiomic: int,
    splits: Splits,
    seed: int = DEFAULT_SEED,
    blocks: tuple[str, ...] = BLOCKS,
) -> CandidateScore:
    """Macro-F1 and balanced accuracy of one classifier over ``splits``, refitting
    every step per fold."""
    n_cnn = matrix.shape[1] - n_radiomic
    model = build_model(classifier, n_radiomic, n_cnn, seed, blocks=blocks)
    folds = cross_validate(
        model,
        matrix,
        y,
        cv=splits,
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
    )
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
) -> SelectionResult:
    """Cross-validate every candidate and pick the best mean macro-F1 (ties: by name).

    Balanced accuracy is reported but does not take part in the choice.
    """
    scores = {
        name: cross_validate_scores(classifier, matrix, y, n_radiomic, splits, seed)
        for name, classifier in candidate_classifiers(seed).items()
    }
    best = max(sorted(scores), key=lambda name: scores[name].mean_macro_f1)
    return SelectionResult(scores, best)
