"""The whole flow with real parts: PostgreSQL, a model bundle trained here on
synthetic images, the real classifier adapter and the fake LLM. It fails when
a change makes the bundle unloadable or breaks the link between two steps."""

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from scipy.ndimage import gaussian_filter
from src.deep_learning.backbones import build_backbone
from src.deep_learning.bundle import write_bundle
from src.training.run import TrainingConfig, train

from adapters.outbound.llm.fake_llm import FakeLLM
from adapters.outbound.ml.ml_engine_classifier import MLEngineClassifier
from main import create_app
from settings import Settings, get_settings

SIGMAS = {"glioma": 3.0, "meningioma": 6.0, "notumor": 12.0, "pituitary": 24.0}
PER_CLASS = 20
IMAGENET = {
    "input_size": 224,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
CLASSES = {"glioma", "meningioma", "pituitary", "no_tumor"}


def textured_png(seed: int, sigma: float, size: int = 128) -> bytes:
    texture = gaussian_filter(np.random.default_rng(seed).random((size, size)), sigma)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    yy, xx = np.mgrid[:size, :size]
    head = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.4) ** 2
    pixels = np.where(head, 40 + texture * 215, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def write_dataset(root: Path) -> tuple[Path, Path]:
    data_dir, rows = root / "data", []
    for split in ("Train", "Test"):
        for label, sigma in SIGMAS.items():
            for index in range(PER_CLASS):
                relative = f"{split}/{label}/{index}.png"
                path = data_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(textured_png(len(rows), sigma))
                rows.append([relative, split, label, len(rows), 1, 1, "kept"])
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["path", "split", "label", "group_id", "group_size", "keep", "reason"]
        )
        writer.writerows(rows)
    return data_dir, manifest


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("e2e-bundle")
    data_dir, manifest = write_dataset(root)
    backbone = root / "backbone"
    state = build_backbone("efficientnet_b0").state_dict()
    write_bundle(backbone, "efficientnet_b0", state, "RANDOM", **IMAGENET)
    config = TrainingConfig(
        data_dir=data_dir,
        manifest=manifest,
        backbone_bundle=backbone,
        out_dir=root / "out",
        n_splits=3,
        version="0.0.1",
        cache_dir=root / "cache",
    )
    return train(config).artifact_dir


@pytest.fixture
def client(
    migrated_database: Settings,
    bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bundle))
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    get_settings.cache_clear()
    app = create_app(llm=FakeLLM(), classifier=MLEngineClassifier())
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_upload_classify_report_and_chat(client: TestClient) -> None:
    assert client.get("/health").json()["model_loaded"] is True

    upload = client.post(
        "/api/studies",
        files={"file": ("scan.png", textured_png(999, SIGMAS["glioma"]), "image/png")},
        data={"clinical_notes": "invented test case"},
    )
    assert upload.status_code == 201
    study_id = upload.json()["id"]
    assert upload.json()["status"] == "completed"

    study = client.get(f"/api/studies/{study_id}").json()
    assert study["prediction"]["label"] in CLASSES
    assert client.get(f"/api/studies/{study_id}/image").status_code == 200

    report = client.post(f"/api/studies/{study_id}/report")
    assert report.status_code == 200
    assert report.json()["status"] == "completed"
    citations = report.json()["content"]["citations"]
    assert any(citation["verified"] for citation in citations)

    chat = client.post(f"/api/studies/{study_id}/chat", json={"message": "Summary?"})
    assert chat.status_code == 200
    assert chat.text.strip()
    assert len(client.get(f"/api/studies/{study_id}/chat").json()) >= 1
