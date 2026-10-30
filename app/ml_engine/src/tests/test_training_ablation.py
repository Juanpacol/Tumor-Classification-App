import pytest

from src.tests.synthetic import grouped_splits, separable_features
from src.training.ablation import VARIANTS, ablate_blocks
from src.training.model import candidate_classifiers, cross_validate_scores

N_RADIOMIC = 45


def scores(informative: tuple[str, ...]) -> dict[str, float]:
    """Ablation with the hybrid score computed the way selection computes it."""
    matrix, labels, groups = separable_features(informative=informative)
    splits = grouped_splits(labels, groups, n_splits=3)
    hybrid = cross_validate_scores(
        candidate_classifiers()["logistic_regression"],
        matrix,
        labels,
        N_RADIOMIC,
        splits,
    ).mean_macro_f1
    return ablate_blocks(
        matrix, labels, N_RADIOMIC, "logistic_regression", splits, hybrid
    )


def test_every_variant_is_scored() -> None:
    assert set(scores(("radiomic", "cnn"))) == set(VARIANTS)


def test_the_hybrid_score_is_reused_not_refitted() -> None:
    matrix, labels, groups = separable_features()

    result = ablate_blocks(
        matrix,
        labels,
        N_RADIOMIC,
        "logistic_regression",
        grouped_splits(labels, groups, 3),
        0.123,
    )

    assert result["hybrid"] == 0.123


def test_cnn_only_wins_when_only_the_cnn_block_carries_signal() -> None:
    result = scores(("cnn",))

    assert result["cnn_only"] > 0.9
    assert result["radiomics_only"] < 0.6
    assert result["hybrid"] > 0.8


def test_radiomics_only_wins_when_only_the_radiomic_block_carries_signal() -> None:
    result = scores(("radiomic",))

    assert result["radiomics_only"] > 0.9
    assert result["cnn_only"] < 0.6


def test_unknown_classifier_is_rejected() -> None:
    matrix, labels, groups = separable_features(n_per_class=10)

    with pytest.raises(KeyError):
        ablate_blocks(
            matrix,
            labels,
            N_RADIOMIC,
            "xgboost",
            grouped_splits(labels, groups, 3),
            0.0,
        )
