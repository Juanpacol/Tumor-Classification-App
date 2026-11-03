"""Feature table for training, with a per-image cache on disk."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.classical.config import RadiomicsConfig
from src.classical.extractor import extract_radiomics, feature_names
from src.deep_learning.bundle import BackboneManifest, sha256_of
from src.deep_learning.extractor import DeepFeatureExtractor
from src.errors import MLEngineError
from src.preprocessing.loading import load_mri_image
from src.preprocessing.prepare import PreprocessingConfig, prepare_image
from src.training.augmentation import (
    AugmentationConfig,
    AugmentationParams,
    augment,
    copy_rng,
    draw,
)
from src.training.dataset import Sample

Featurizer = Callable[[Path], tuple[np.ndarray, np.ndarray]]
CopyFeaturizer = Callable[[np.ndarray, Sequence[AugmentationParams]], np.ndarray]
Digest = Callable[[Path], str]


@dataclass(frozen=True)
class FeatureTable:
    """Aligned features of the images that were usable.

    ``radiomics`` is (N, R) and ``cnn`` is (N, C); row ``i`` belongs to ``samples[i]``.
    """

    samples: tuple[Sample, ...]
    radiomics: np.ndarray
    cnn: np.ndarray
    radiomic_names: tuple[str, ...]

    @property
    def matrix(self) -> np.ndarray:
        """Radiomic block followed by the CNN block, as the training pipeline expects."""
        return np.hstack([self.radiomics, self.cnn])


@dataclass(frozen=True)
class SkippedImage:
    """An image that could not be used, with the reason the engine gave."""

    path: Path
    reason: str


def config_fingerprint(
    preprocessing: PreprocessingConfig,
    radiomics: RadiomicsConfig,
    backbone: BackboneManifest,
) -> str:
    """Hash of everything that changes the features, so a stale cache is never reused.

    The backbone counts through its weights and the input it normalizes them with.
    """
    payload = json.dumps(
        {
            "preprocessing": asdict(preprocessing),
            "radiomics": asdict(radiomics),
            "backbone": {
                "sha256": backbone.sha256,
                "input_size": backbone.input_size,
                "mean": list(backbone.mean),
                "std": list(backbone.std),
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_featurizer(
    extractor: DeepFeatureExtractor,
    preprocessing: PreprocessingConfig,
    radiomics: RadiomicsConfig,
    on_prepared: Callable[[Path, np.ndarray], None] | None = None,
) -> tuple[Featurizer, str]:
    """The real per-image featurizer and the fingerprint of its configuration.

    The featurizer raises ``UnsupportedImageError`` or ``QualityCheckError`` for
    images the engine would reject at serving time. ``on_prepared`` receives the
    CNN image of every usable image it computes, so augmented copies can be made
    without preparing the image a second time.
    """

    def featurize(path: Path) -> tuple[np.ndarray, np.ndarray]:
        prepared = prepare_image(load_mri_image(path), preprocessing)
        radiomic = extract_radiomics(prepared.radiomics_image, prepared.mask, radiomics)
        cnn = extractor.extract(prepared.cnn_image)
        if on_prepared is not None:
            on_prepared(path, prepared.cnn_image)
        return radiomic.values.astype(np.float32), cnn

    return featurize, config_fingerprint(preprocessing, radiomics, extractor.manifest)


class FeatureCache:
    """One small ``.npz`` per image, keyed by file content and configuration.

    Changing a config, the backbone or an image recomputes only what is stale.
    Images the engine rejects are cached too, so they are not retried.
    """

    def __init__(
        self,
        directory: Path,
        fingerprint: str,
        featurize: Featurizer,
        radiomics: RadiomicsConfig,
        digest: Digest = sha256_of,
    ) -> None:
        self._directory = directory / fingerprint
        self._featurize = featurize
        self._names = feature_names(radiomics)
        self._digest = digest

    def table(
        self, samples: Sequence[Sample]
    ) -> tuple[FeatureTable, list[SkippedImage]]:
        kept: list[Sample] = []
        radiomic_rows: list[np.ndarray] = []
        cnn_rows: list[np.ndarray] = []
        skipped: list[SkippedImage] = []
        for sample in samples:
            entry = self._entry(sample.path)
            if "reason" in entry:
                skipped.append(SkippedImage(sample.path, str(entry["reason"])))
                continue
            kept.append(sample)
            radiomic_rows.append(entry["radiomics"])
            cnn_rows.append(entry["cnn"])

        radiomics = (
            np.vstack(radiomic_rows) if kept else np.empty((0, len(self._names)))
        )
        cnn = np.vstack(cnn_rows) if kept else np.empty((0, 0))
        return FeatureTable(tuple(kept), radiomics, cnn, self._names), skipped

    def _entry(self, path: Path) -> dict[str, np.ndarray]:
        digest = self._digest(path)
        cached = self._directory / digest[:2] / f"{digest}.npz"
        if cached.is_file():
            with np.load(cached) as data:
                return {key: data[key] for key in data.files}

        try:
            radiomics, cnn = self._featurize(path)
            entry = {"radiomics": radiomics, "cnn": cnn}
        except MLEngineError as error:
            entry = {"reason": np.array(str(error))}
        _save(cached, entry)
        return entry


def augmentation_fingerprint(config: AugmentationConfig, seed: int) -> str:
    """Hash of the transform ranges, the number of copies and the run seed."""
    payload = json.dumps({"augmentation": asdict(config), "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_copy_featurizer(extractor: DeepFeatureExtractor) -> CopyFeaturizer:
    """CNN features of the copies of one CNN image (after block B6), one row per
    parameter set. Copies vary the CNN block only."""

    def featurize(
        cnn_image: np.ndarray, params: Sequence[AugmentationParams]
    ) -> np.ndarray:
        return extractor.extract_batch(
            np.stack([augment(cnn_image, p) for p in params])
        )

    return featurize


def make_cnn_input(preprocessing: PreprocessingConfig) -> Callable[[Path], np.ndarray]:
    """The CNN image of a usable file, for copies whose original came from the cache."""
    return lambda path: prepare_image(load_mri_image(path), preprocessing).cnn_image


class CopyCache:
    """CNN features of the augmented copies of each image, one ``.npz`` per original.

    It lives in its own folder under the feature cache, named after the
    augmentation fingerprint, so copies made under other settings are never
    reused and the originals' entries are never touched. ``add`` makes the copies
    from a CNN image already prepared for the original; ``cnn`` prepares one with
    ``cnn_input`` only for an original that was read from the cache.
    """

    def __init__(
        self,
        directory: Path,
        config: AugmentationConfig,
        seed: int,
        featurize: CopyFeaturizer,
        cnn_input: Callable[[Path], np.ndarray],
        digest: Digest = sha256_of,
    ) -> None:
        self._directory = directory / f"aug-{augmentation_fingerprint(config, seed)}"
        self._config = config
        self._seed = seed
        self._featurize = featurize
        self._cnn_input = cnn_input
        self._digest = digest

    def add(self, path: Path, cnn_image: np.ndarray) -> None:
        """Make and store the copies of ``path`` from its CNN image, unless cached."""
        digest = self._digest(path)
        if not self._file(digest).is_file():
            self._compute(digest, cnn_image)

    def cnn(self, samples: Sequence[Sample]) -> np.ndarray:
        """``(len(samples) * copies, C)``: the copies of each sample, in sample order."""
        rows = [self._entry(sample.path) for sample in samples]
        return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)

    def _entry(self, path: Path) -> np.ndarray:
        digest = self._digest(path)
        cached = self._file(digest)
        if cached.is_file():
            with np.load(cached) as data:
                return data["cnn"]
        return self._compute(digest, self._cnn_input(path))

    def _compute(self, digest: str, cnn_image: np.ndarray) -> np.ndarray:
        params = [
            draw(self._config, copy_rng(self._seed, digest, index))
            for index in range(self._config.copies)
        ]
        cnn = self._featurize(cnn_image, params)
        _save(self._file(digest), {"cnn": cnn})
        return cnn

    def _file(self, digest: str) -> Path:
        return self._directory / digest[:2] / f"{digest}.npz"


def copy_rows(table: FeatureTable, copy_cnn: np.ndarray, copies: int) -> np.ndarray:
    """Matrix rows of the copies: each original's radiomic row with its copy's CNN row."""
    return np.hstack([np.repeat(table.radiomics, copies, axis=0), copy_cnn])


def _save(cached: Path, entry: dict[str, np.ndarray]) -> None:
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(".tmp.npz")
    np.savez(temporary, **entry)
    temporary.replace(cached)
