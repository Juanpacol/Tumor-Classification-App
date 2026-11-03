from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.classical.config import RadiomicsConfig
from src.contracts import QualityMetrics, TumorLabel
from src.deep_learning.bundle import BackboneManifest
from src.errors import QualityCheckError, UnsupportedImageError
from src.preprocessing.prepare import PreprocessingConfig
from src.preprocessing.quality import QualityThresholds
from src.tests.bundles import IMAGENET_INPUT
from src.tests.synthetic import save, synthetic_slice
from src.training.augmentation import AugmentationConfig, AugmentationParams
from src.training.dataset import Sample
from src.training.features import (
    CopyCache,
    FeatureCache,
    config_fingerprint,
    copy_rows,
)

RADIOMICS = RadiomicsConfig()


class CountingFeaturizer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(path)
        if path.name.startswith("bad"):
            raise QualityCheckError("too noisy", QualityMetrics(1.0, 1.0, False))
        if path.name.startswith("odd"):
            raise UnsupportedImageError("not an image")
        value = float(len(path.name))
        return np.full(45, value, np.float32), np.full(8, value * 2, np.float32)


def make_samples(tmp_path: Path, names: list[str]) -> list[Sample]:
    samples = []
    for index, name in enumerate(names):
        path = save(synthetic_slice(index, size=64), tmp_path / name)
        samples.append(Sample(path, "Train", TumorLabel.GLIOMA, index))
    return samples


def cache(
    tmp_path: Path, featurize: CountingFeaturizer, fingerprint: str = "fp1"
) -> FeatureCache:
    return FeatureCache(tmp_path / "cache", fingerprint, featurize, RADIOMICS)


def test_table_rows_follow_the_samples(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "bbbb.png"])

    table, skipped = cache(tmp_path, CountingFeaturizer()).table(samples)

    assert skipped == []
    assert table.samples == tuple(samples)
    assert table.radiomics.shape == (2, 45) and table.cnn.shape == (2, 8)
    assert table.matrix.shape == (2, 53)
    assert table.radiomics[0, 0] == len("a.png")
    assert len(table.radiomic_names) == 45


def test_second_run_reads_everything_from_the_cache(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png"])
    featurizer = CountingFeaturizer()
    first, _ = cache(tmp_path, featurizer).table(samples)

    second, _ = cache(tmp_path, featurizer).table(samples)

    assert len(featurizer.calls) == 2
    assert np.array_equal(first.matrix, second.matrix)


def test_only_new_images_are_computed(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png", "c.png"])
    featurizer = CountingFeaturizer()
    cache(tmp_path, featurizer).table(samples[:2])

    cache(tmp_path, featurizer).table(samples)

    assert [p.name for p in featurizer.calls] == ["a.png", "b.png", "c.png"]


def test_changing_an_image_recomputes_it(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png"])
    featurizer = CountingFeaturizer()
    cache(tmp_path, featurizer).table(samples)
    save(synthetic_slice(99, size=64), samples[0].path)

    cache(tmp_path, featurizer).table(samples)

    assert len(featurizer.calls) == 2


def test_a_new_fingerprint_does_not_reuse_the_old_cache(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png"])
    featurizer = CountingFeaturizer()
    cache(tmp_path, featurizer, "fp1").table(samples)

    cache(tmp_path, featurizer, "fp2").table(samples)

    assert len(featurizer.calls) == 2


def test_rejected_images_are_skipped_with_their_reason_and_cached(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path, ["a.png", "bad.png", "odd.png"])
    featurizer = CountingFeaturizer()

    table, skipped = cache(tmp_path, featurizer).table(samples)
    cache(tmp_path, featurizer).table(samples)

    assert [s.path.name for s in table.samples] == ["a.png"]
    assert {s.path.name: s.reason for s in skipped} == {
        "bad.png": "too noisy",
        "odd.png": "not an image",
    }
    assert len(featurizer.calls) == 3


def test_empty_input_gives_an_empty_table(tmp_path: Path) -> None:
    table, skipped = cache(tmp_path, CountingFeaturizer()).table([])

    assert table.matrix.shape[0] == 0 and skipped == []


BACKBONE = BackboneManifest(
    backbone="efficientnet_b0",
    embedding_dim=1280,
    sha256="sha",
    torchvision_version="test",
    weights_version="TEST",
    **IMAGENET_INPUT,
)


def test_fingerprint_changes_with_any_input() -> None:
    base = config_fingerprint(PreprocessingConfig(), RADIOMICS, BACKBONE)

    assert base == config_fingerprint(PreprocessingConfig(), RADIOMICS, BACKBONE)
    assert base != config_fingerprint(
        PreprocessingConfig(cnn_size=256), RADIOMICS, BACKBONE
    )
    assert base != config_fingerprint(
        PreprocessingConfig(), RadiomicsConfig(lbp_points=16), BACKBONE
    )
    assert base != config_fingerprint(
        PreprocessingConfig(), RADIOMICS, replace(BACKBONE, sha256="other")
    )
    assert base != config_fingerprint(
        PreprocessingConfig(min_mask_fraction=0.1), RADIOMICS, BACKBONE
    )
    assert base != config_fingerprint(
        PreprocessingConfig(crop_margin=0.1), RADIOMICS, BACKBONE
    )
    assert base != config_fingerprint(
        PreprocessingConfig(thresholds=QualityThresholds(metrics_version=1)),
        RADIOMICS,
        BACKBONE,
    )


def test_fingerprint_changes_with_the_backbone_input_statistics() -> None:
    base = config_fingerprint(PreprocessingConfig(), RADIOMICS, BACKBONE)

    for change in (
        {"mean": (0.5, 0.5, 0.5)},
        {"std": (0.5, 0.5, 0.5)},
        {"input_size": 256},
    ):
        assert base != config_fingerprint(
            PreprocessingConfig(), RADIOMICS, replace(BACKBONE, **change)
        )


DEFAULT_AUGMENTATION = AugmentationConfig()


def marker(path: Path) -> float:
    """A value that tells which original a CNN image came from."""
    return float(ord(path.name[0]))


class CountingCnnInput:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> np.ndarray:
        self.calls.append(path)
        return np.full((4, 4), marker(path), np.float32)


class CountingCopyFeaturizer:
    def __init__(self) -> None:
        self.calls: list[tuple[float, list[AugmentationParams]]] = []

    def __call__(
        self, cnn_image: np.ndarray, params: Sequence[AugmentationParams]
    ) -> np.ndarray:
        self.calls.append((float(cnn_image[0, 0]), list(params)))
        return np.array([[p.rotation_deg, p.scale] for p in params], np.float32)


def copy_cache(
    tmp_path: Path,
    featurize: CountingCopyFeaturizer,
    config: AugmentationConfig = DEFAULT_AUGMENTATION,
    seed: int = 42,
    cnn_input: CountingCnnInput | None = None,
) -> CopyCache:
    return CopyCache(
        tmp_path / "cache" / "fp1",
        config,
        seed,
        featurize,
        cnn_input or CountingCnnInput(),
    )


def test_copies_follow_the_samples_with_k_rows_each(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png"])
    featurizer = CountingCopyFeaturizer()

    cnn = copy_cache(tmp_path, featurizer, AugmentationConfig(copies=2)).cnn(samples)

    assert cnn.shape == (4, 2)
    assert [key for key, _ in featurizer.calls] == [marker(s.path) for s in samples]
    assert all(len(params) == 2 for _, params in featurizer.calls)


def test_copies_added_from_a_prepared_image_are_not_prepared_again(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png"])
    featurizer, cnn_input = CountingCopyFeaturizer(), CountingCnnInput()
    copies = copy_cache(tmp_path, featurizer, cnn_input=cnn_input)
    for sample in samples:
        copies.add(sample.path, np.full((4, 4), marker(sample.path), np.float32))
        copies.add(sample.path, np.full((4, 4), marker(sample.path), np.float32))

    cnn = copies.cnn(samples)

    assert cnn_input.calls == []
    assert [key for key, _ in featurizer.calls] == [marker(s.path) for s in samples]
    assert cnn.shape == (2, 2)


def test_a_repeated_run_recomputes_no_copy(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png"])
    featurizer = CountingCopyFeaturizer()
    first = copy_cache(tmp_path, featurizer).cnn(samples)

    second = copy_cache(tmp_path, featurizer).cnn(samples)

    assert len(featurizer.calls) == 2
    assert np.array_equal(first, second)


def test_changing_a_range_k_or_the_seed_recomputes_every_copy(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png"])
    originals = CountingFeaturizer()
    cache(tmp_path, originals).table(samples)
    featurizer = CountingCopyFeaturizer()
    copy_cache(tmp_path, featurizer).cnn(samples)

    for config, seed in (
        (AugmentationConfig(max_rotation_deg=15.0), 42),
        (AugmentationConfig(copies=2), 42),
        (AugmentationConfig(), 7),
    ):
        before = len(featurizer.calls)
        copy_cache(tmp_path, featurizer, config, seed).cnn(samples)
        assert len(featurizer.calls) == before + 2

    cache(tmp_path, originals).table(samples)
    assert len(originals.calls) == 2


def test_the_draw_does_not_depend_on_processing_order(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png", "c.png"])
    forward = CountingCopyFeaturizer()
    copy_cache(tmp_path / "one", forward).cnn(samples)
    backward = CountingCopyFeaturizer()
    copy_cache(tmp_path / "two", backward).cnn(samples[::-1])

    assert dict(forward.calls) == dict(backward.calls)


def test_a_copy_row_has_its_originals_radiomics(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "bbbb.png"])
    table, _ = cache(tmp_path, CountingFeaturizer()).table(samples)
    copy_cnn = np.arange(2 * 2 * 8, dtype=np.float32).reshape(4, 8)

    rows = copy_rows(table, copy_cnn, copies=2)

    assert rows.shape == (4, 53)
    assert np.array_equal(rows[:, :45], np.repeat(table.radiomics, 2, axis=0))
    assert np.array_equal(rows[:, 45:], copy_cnn)
