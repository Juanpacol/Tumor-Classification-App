import json
from pathlib import Path

import pytest
import torch

from src.deep_learning.bundle import (
    MANIFEST_FILE,
    WEIGHTS_FILE,
    read_bundle,
    sha256_of,
    write_bundle,
)
from src.errors import ModelUnavailableError


class Unsafe:
    """An arbitrary object, which a safe load must refuse to unpickle."""


def tiny_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.arange(4, dtype=torch.float32)}


def test_round_trip_keeps_the_weights_and_describes_them(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1")

    state, loaded = read_bundle(tmp_path)

    assert torch.equal(state["weight"], tiny_state()["weight"])
    assert loaded == manifest
    assert loaded.embedding_dim == 1280
    assert loaded.sha256 == sha256_of(tmp_path / WEIGHTS_FILE)


def test_missing_bundle_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="not found"):
        read_bundle(tmp_path / "nowhere")


def test_tampered_weights_are_rejected_before_loading(tmp_path: Path) -> None:
    write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1")
    with (tmp_path / WEIGHTS_FILE).open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ModelUnavailableError, match="sha256"):
        read_bundle(tmp_path)


def test_unsafe_pickle_is_refused_even_with_a_matching_hash(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path, "efficientnet_b0", {"x": Unsafe()}, "TEST_V1")
    assert manifest.sha256 == sha256_of(tmp_path / WEIGHTS_FILE)

    with pytest.raises(ModelUnavailableError, match="cannot load weights"):
        read_bundle(tmp_path)


@pytest.mark.parametrize(
    "content", ["not json", "{}", '{"backbone": "efficientnet_b0"}']
)
def test_invalid_manifest_is_unavailable(tmp_path: Path, content: str) -> None:
    write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1")
    (tmp_path / MANIFEST_FILE).write_text(content)

    with pytest.raises(ModelUnavailableError, match="invalid manifest"):
        read_bundle(tmp_path)


def test_unknown_backbone_in_manifest_is_unavailable(tmp_path: Path) -> None:
    write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1")
    raw = json.loads((tmp_path / MANIFEST_FILE).read_text())
    raw["backbone"] = "../../etc/passwd"
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(raw))

    with pytest.raises(ModelUnavailableError, match="invalid manifest"):
        read_bundle(tmp_path)
