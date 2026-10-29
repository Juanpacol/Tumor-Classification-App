import pytest
import torch

from src.deep_learning.backbones import (
    BACKBONES,
    as_feature_extractor,
    build_backbone,
    get_spec,
)


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_backbone_outputs_the_declared_embedding_size(name: str) -> None:
    model = as_feature_extractor(build_backbone(name), name)

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))

    assert tuple(output.shape) == (1, get_spec(name).embedding_dim)


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_backbone_is_frozen_and_in_eval_mode(name: str) -> None:
    model = as_feature_extractor(build_backbone(name), name)

    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_unknown_backbone_lists_the_valid_names() -> None:
    with pytest.raises(ValueError, match="efficientnet_b0"):
        get_spec("vgg19")


def test_default_backbone_is_efficientnet_b0_with_1280_dimensions() -> None:
    assert get_spec("efficientnet_b0").embedding_dim == 1280
