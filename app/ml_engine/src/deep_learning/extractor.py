"""Frozen pretrained CNN that turns preprocessed images into embeddings."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.deep_learning.backbones import as_feature_extractor, build_backbone, get_spec
from src.deep_learning.bundle import BackboneManifest, read_bundle
from src.errors import ModelUnavailableError
from src.preprocessing.prepare import PreprocessingConfig


class DeepFeatureExtractor:
    """Global-average-pooled embeddings from a frozen, eval-mode backbone.

    Inputs are the ``(3, S, S)`` arrays that ``prepare_image`` produces. The
    model is loaded once and reused. Results are deterministic for the same
    process and hardware, but the last decimals can differ across CPUs, so do
    not compare them with stored values.
    """

    def __init__(self, model: nn.Module, manifest: BackboneManifest) -> None:
        self._model = as_feature_extractor(model, manifest.backbone)
        self._manifest = manifest

    @classmethod
    def from_bundle(
        cls, directory: str | Path, preprocessing: PreprocessingConfig | None = None
    ) -> DeepFeatureExtractor:
        """Load a verified bundle and check it matches the preprocessing config.

        Raises ``ModelUnavailableError`` when the bundle is missing, corrupt or
        expects a different input size or normalization.
        """
        state_dict, manifest = read_bundle(Path(directory))
        _check_preprocessing(manifest, preprocessing or PreprocessingConfig())
        model = build_backbone(manifest.backbone)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ModelUnavailableError(
                f"weights do not fit {manifest.backbone}: {exc}"
            ) from exc
        return cls(model, manifest)

    @property
    def manifest(self) -> BackboneManifest:
        return self._manifest

    @property
    def embedding_dim(self) -> int:
        return get_spec(self._manifest.backbone).embedding_dim

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Embedding of one ``(3, S, S)`` image as a ``(D,)`` float32 array."""
        if image.ndim != 3:
            raise ValueError("image must have shape (3, S, S)")
        return self.extract_batch(image[None])[0]

    def extract_batch(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Embeddings of ``(N, 3, S, S)`` images as an ``(N, D)`` float32 array."""
        size = self._manifest.input_size
        if images.ndim != 4 or images.shape[1:] != (3, size, size):
            raise ValueError(f"images must have shape (N, 3, {size}, {size})")
        if not np.isfinite(images).all():
            raise ValueError("images must be finite")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        tensor = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32))
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(tensor), batch_size):
                outputs.append(self._model(tensor[start : start + batch_size]).numpy())
        if not outputs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32)


def _check_preprocessing(
    manifest: BackboneManifest, config: PreprocessingConfig
) -> None:
    """The model must see the same size and normalization it was exported with."""
    if manifest.input_size != config.cnn_size:
        raise ModelUnavailableError(
            f"backbone expects {manifest.input_size}px input, preprocessing gives {config.cnn_size}px"
        )
    same_stats = all(
        math.isclose(a, b, abs_tol=1e-6)
        for pair in ((manifest.mean, config.cnn_mean), (manifest.std, config.cnn_std))
        for a, b in zip(*pair)
    )
    if not same_stats:
        raise ModelUnavailableError(
            "backbone normalization does not match the preprocessing config"
        )
