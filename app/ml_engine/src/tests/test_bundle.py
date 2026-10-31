import json
from pathlib import Path

import pytest
import torch

from src.deep_learning import export_backbone
from src.deep_learning.backbones import build_backbone
from src.deep_learning.bundle import (
    LICENSE_NOT_VERIFIED,
    MANIFEST_FILE,
    WEIGHTS_FILE,
    read_bundle,
    sha256_of,
    write_bundle,
)
from src.errors import ModelUnavailableError
from src.tests.bundles import IMAGENET_INPUT


class Unsafe:
    """An arbitrary object, which a safe load must refuse to unpickle."""


def tiny_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.arange(4, dtype=torch.float32)}


def test_round_trip_keeps_the_weights_and_describes_them(tmp_path: Path) -> None:
    manifest = write_bundle(
        tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1", **IMAGENET_INPUT
    )

    state, loaded = read_bundle(tmp_path)

    assert torch.equal(state["weight"], tiny_state()["weight"])
    assert loaded == manifest
    assert loaded.embedding_dim == 1280
    assert loaded.sha256 == sha256_of(tmp_path / WEIGHTS_FILE)


def test_missing_bundle_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="not found"):
        read_bundle(tmp_path / "nowhere")


def test_tampered_weights_are_rejected_before_loading(tmp_path: Path) -> None:
    write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1", **IMAGENET_INPUT)
    with (tmp_path / WEIGHTS_FILE).open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ModelUnavailableError, match="sha256"):
        read_bundle(tmp_path)


def test_unsafe_pickle_is_refused_even_with_a_matching_hash(tmp_path: Path) -> None:
    manifest = write_bundle(
        tmp_path, "efficientnet_b0", {"x": Unsafe()}, "TEST_V1", **IMAGENET_INPUT
    )
    assert manifest.sha256 == sha256_of(tmp_path / WEIGHTS_FILE)

    with pytest.raises(ModelUnavailableError, match="cannot load weights"):
        read_bundle(tmp_path)


@pytest.mark.parametrize(
    "content", ["not json", "{}", '{"backbone": "efficientnet_b0"}']
)
def test_invalid_manifest_is_unavailable(tmp_path: Path, content: str) -> None:
    write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1", **IMAGENET_INPUT)
    (tmp_path / MANIFEST_FILE).write_text(content)

    with pytest.raises(ModelUnavailableError, match="invalid manifest"):
        read_bundle(tmp_path)


def test_unknown_backbone_in_manifest_is_unavailable(tmp_path: Path) -> None:
    write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1", **IMAGENET_INPUT)
    raw = json.loads((tmp_path / MANIFEST_FILE).read_text())
    raw["backbone"] = "../../etc/passwd"
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(raw))

    with pytest.raises(ModelUnavailableError, match="invalid manifest"):
        read_bundle(tmp_path)


def test_provenance_is_recorded_and_read_back(tmp_path: Path) -> None:
    write_bundle(
        tmp_path,
        "efficientnet_b0",
        tiny_state(),
        "TEST_V1",
        source_url="https://example.org/weights.pth",
        license_note="a note",
        **IMAGENET_INPUT,
    )

    manifest = read_bundle(tmp_path)[1]

    assert manifest.source_url == "https://example.org/weights.pth"
    assert manifest.license_note == "a note"


def test_provenance_defaults_say_the_terms_are_not_verified(tmp_path: Path) -> None:
    manifest = write_bundle(
        tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1", **IMAGENET_INPUT
    )

    assert manifest.source_url == ""
    assert manifest.license_note == LICENSE_NOT_VERIFIED


def test_a_bundle_exported_before_provenance_existed_still_loads(
    tmp_path: Path,
) -> None:
    write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1", **IMAGENET_INPUT)
    raw = json.loads((tmp_path / MANIFEST_FILE).read_text())
    del raw["source_url"], raw["license_note"]
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(raw))

    manifest = read_bundle(tmp_path)[1]

    assert manifest.source_url == ""
    assert manifest.license_note == LICENSE_NOT_VERIFIED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("std", [0.229, 0.0, 0.225]),
        ("std", [0.229, -0.1, 0.225]),
        ("mean", [0.5]),
        ("mean", [0.5, float("nan"), 0.5]),
        ("input_size", 0),
        ("input_size", 224.0),
        ("std", [True, True, True]),
    ],
)
def test_unusable_input_statistics_are_refused(
    tmp_path: Path, field: str, value: object
) -> None:
    write_bundle(tmp_path, "efficientnet_b0", tiny_state(), "TEST_V1", **IMAGENET_INPUT)
    raw = json.loads((tmp_path / MANIFEST_FILE).read_text())
    raw[field] = value
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(raw))

    with pytest.raises(ModelUnavailableError, match=field):
        read_bundle(tmp_path)


def test_export_writes_the_statistics_published_with_the_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Transforms:
        crop_size = (192,)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    class Weights:
        name = "FAKE_V1"
        url = "https://example.org/fake.pth"

        def transforms(self) -> Transforms:
            return Transforms()

    monkeypatch.setitem(export_backbone._IMAGENET_WEIGHTS, "efficientnet_b0", Weights())
    model = build_backbone("efficientnet_b0")
    monkeypatch.setattr(
        export_backbone.models, "efficientnet_b0", lambda weights: model
    )

    export_backbone.main(["--backbone", "efficientnet_b0", "--out", str(tmp_path)])

    manifest = read_bundle(tmp_path)[1]
    assert manifest.input_size == 192
    assert manifest.mean == (0.5, 0.5, 0.5) and manifest.std == (0.5, 0.5, 0.5)
