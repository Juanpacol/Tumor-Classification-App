from pathlib import Path

import numpy as np
import pytest
import torch

from src.deep_learning.backbones import build_backbone
from src.deep_learning.bundle import write_bundle
from src.deep_learning.extractor import DeepFeatureExtractor
from src.errors import ModelUnavailableError
from src.preprocessing.prepare import PreprocessingConfig, prepare_image
from src.seed import set_seed
from src.tests.synthetic import synthetic_slice

REAL_BUNDLE = Path(__file__).resolve().parents[2] / "weights"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    set_seed(0)
    directory = tmp_path_factory.mktemp("bundle")
    write_bundle(
        directory,
        "efficientnet_b0",
        build_backbone("efficientnet_b0").state_dict(),
        "RANDOM",
    )
    return directory


@pytest.fixture(scope="module")
def extractor(bundle: Path) -> DeepFeatureExtractor:
    return DeepFeatureExtractor.from_bundle(bundle)


def images(count: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(count, 3, 224, 224)).astype(np.float32)


def test_single_image_gives_a_1280_dimensional_float32_embedding(
    extractor: DeepFeatureExtractor,
) -> None:
    embedding = extractor.extract(images(1)[0])

    assert extractor.embedding_dim == 1280
    assert embedding.shape == (1280,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()


def test_extraction_is_deterministic(extractor: DeepFeatureExtractor) -> None:
    image = images(1)[0]

    assert np.array_equal(extractor.extract(image), extractor.extract(image))


def test_batch_extraction_matches_single_extraction(
    extractor: DeepFeatureExtractor,
) -> None:
    batch = images(5)

    embeddings = extractor.extract_batch(batch, batch_size=2)

    assert embeddings.shape == (5, 1280)
    assert np.allclose(embeddings[3], extractor.extract(batch[3]), atol=1e-4)
    assert np.allclose(
        embeddings, extractor.extract_batch(batch, batch_size=5), atol=1e-4
    )


def test_empty_batch_gives_an_empty_embedding_matrix(
    extractor: DeepFeatureExtractor,
) -> None:
    assert extractor.extract_batch(images(0)).shape == (0, 1280)


def test_extraction_leaves_no_gradients_behind(bundle: Path) -> None:
    model = build_backbone("efficientnet_b0")
    manifest = write_bundle(
        bundle / "own", "efficientnet_b0", model.state_dict(), "RANDOM"
    )

    extractor = DeepFeatureExtractor(model, manifest)
    extractor.extract(images(1)[0])

    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "bad",
    [np.zeros((224, 224), np.float32), np.zeros((3, 100, 100), np.float32)],
    ids=["wrong ndim", "wrong size"],
)
def test_invalid_shapes_are_rejected(
    extractor: DeepFeatureExtractor, bad: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape"):
        extractor.extract(bad)


def test_non_finite_input_and_bad_batch_size_are_rejected(
    extractor: DeepFeatureExtractor,
) -> None:
    broken = images(1)
    broken[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        extractor.extract_batch(broken)
    with pytest.raises(ValueError, match="batch_size"):
        extractor.extract_batch(images(1), batch_size=0)


def test_preprocessing_size_mismatch_is_unavailable(bundle: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="224px"):
        DeepFeatureExtractor.from_bundle(bundle, PreprocessingConfig(cnn_size=256))


def test_preprocessing_normalization_mismatch_is_unavailable(bundle: Path) -> None:
    config = PreprocessingConfig(cnn_mean=(0.5, 0.5, 0.5))

    with pytest.raises(ModelUnavailableError, match="normalization"):
        DeepFeatureExtractor.from_bundle(bundle, config)


def test_weights_that_do_not_fit_the_backbone_are_unavailable(tmp_path: Path) -> None:
    write_bundle(tmp_path, "efficientnet_b0", {"foo": torch.zeros(1)}, "RANDOM")

    with pytest.raises(ModelUnavailableError, match="do not fit"):
        DeepFeatureExtractor.from_bundle(tmp_path)


def test_prepared_image_goes_straight_into_the_extractor(
    extractor: DeepFeatureExtractor,
) -> None:
    gray = np.asarray(synthetic_slice(3, 256), dtype=np.float32) / 255.0

    embedding = extractor.extract(prepare_image(gray).cnn_input)

    assert embedding.shape == (1280,) and np.isfinite(embedding).all()


@pytest.mark.skipif(
    not (REAL_BUNDLE / "backbone.pt").is_file(), reason="no exported bundle in weights/"
)
def test_real_imagenet_bundle_loads_and_embeds() -> None:
    real = DeepFeatureExtractor.from_bundle(REAL_BUNDLE)

    embedding = real.extract(images(1)[0])

    assert embedding.shape == (real.embedding_dim,) and np.isfinite(embedding).all()
