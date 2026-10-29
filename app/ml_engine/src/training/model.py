"""Hybrid classifier pipeline and cross-validated classifier selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_SEED = 42
PCA_COMPONENTS = 32
N_SPLITS = 5


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


def group_folds(
    n_splits: int = N_SPLITS, seed: int = DEFAULT_SEED
) -> StratifiedGroupKFold:
    """Stratified folds in which a group of near-duplicate images never straddles two folds.

    Pass ``groups`` when splitting. The dataset has no patient id, so the groups
    from the cleaning step are the closest available proxy.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


@dataclass(frozen=True)
class CandidateScore:
    mean_macro_f1: float
    std_macro_f1: float
    fold_macro_f1: tuple[float, ...]


@dataclass(frozen=True)
class SelectionResult:
    scores: dict[str, CandidateScore]
    best: str


def cross_validate_macro_f1(
    classifier: BaseEstimator,
    matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_radiomic: int,
    seed: int = DEFAULT_SEED,
    n_splits: int = N_SPLITS,
    blocks: tuple[str, ...] = BLOCKS,
) -> CandidateScore:
    """Macro-F1 of one classifier over group-aware folds, refitting every step per fold."""
    n_cnn = matrix.shape[1] - n_radiomic
    model = build_model(classifier, n_radiomic, n_cnn, seed, blocks=blocks)
    folds = cross_val_score(
        model,
        matrix,
        y,
        groups=groups,
        cv=group_folds(n_splits, seed),
        scoring="f1_macro",
    )
    return CandidateScore(
        float(folds.mean()), float(folds.std()), tuple(float(v) for v in folds)
    )


def compare_classifiers(
    matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_radiomic: int,
    seed: int = DEFAULT_SEED,
    n_splits: int = N_SPLITS,
) -> SelectionResult:
    """Cross-validate every candidate and pick the best mean macro-F1 (ties: by name)."""
    scores = {
        name: cross_validate_macro_f1(
            classifier, matrix, y, groups, n_radiomic, seed, n_splits
        )
        for name, classifier in candidate_classifiers(seed).items()
    }
    best = max(sorted(scores), key=lambda name: scores[name].mean_macro_f1)
    return SelectionResult(scores, best)
