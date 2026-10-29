import pytest

from src.tests.synthetic import separable_features
from src.training.ablation import VARIANTS, ablate_blocks

N_RADIOMIC = 45


def scores(informative: tuple[str, ...]) -> dict[str, float]:
    matrix, labels, groups = separable_features(informative=informative)
    return ablate_blocks(
        matrix, labels, groups, N_RADIOMIC, "logistic_regression", n_splits=3
    )


def test_every_variant_is_scored() -> None:
    assert set(scores(("radiomic", "cnn"))) == set(VARIANTS)


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
        ablate_blocks(matrix, labels, groups, N_RADIOMIC, "xgboost", n_splits=3)
