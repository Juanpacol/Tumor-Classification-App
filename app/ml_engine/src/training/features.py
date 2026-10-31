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
from src.training.dataset import Sample

Featurizer = Callable[[Path], tuple[np.ndarray, np.ndarray]]


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
) -> tuple[Featurizer, str]:
    """The real per-image featurizer and the fingerprint of its configuration.

    The featurizer raises ``UnsupportedImageError`` or ``QualityCheckError`` for
    images the engine would reject at serving time.
    """

    def featurize(path: Path) -> tuple[np.ndarray, np.ndarray]:
        prepared = prepare_image(load_mri_image(path), preprocessing)
        radiomic = extract_radiomics(prepared.radiomics_image, prepared.mask, radiomics)
        return radiomic.values.astype(np.float32), extractor.extract(prepared.cnn_image)

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
    ) -> None:
        self._directory = directory / fingerprint
        self._featurize = featurize
        self._names = feature_names(radiomics)

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
        digest = sha256_of(path)
        cached = self._directory / digest[:2] / f"{digest}.npz"
        if cached.is_file():
            with np.load(cached) as data:
                return {key: data[key] for key in data.files}

        try:
            radiomics, cnn = self._featurize(path)
            entry = {"radiomics": radiomics, "cnn": cnn}
        except MLEngineError as error:
            entry = {"reason": np.array(str(error))}
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_suffix(".tmp.npz")
        np.savez(temporary, **entry)
        temporary.replace(cached)
        return entry
