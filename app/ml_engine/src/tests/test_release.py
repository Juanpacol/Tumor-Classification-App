import json
import tarfile
from pathlib import Path

import pytest

from src.errors import ModelUnavailableError
from src.release import fetch as fetch_module
from src.release.__main__ import main
from src.release.package import package, release_notes

MANIFEST = {
    "version": "0.2.0",
    "created_at": "2026-11-03T00:00:00+00:00",
    "seed": 42,
    "classifier": "logistic_regression",
    "metrics": {
        "test": {
            "accuracy": 0.8,
            "macro_f1": 0.75,
            "balanced_accuracy": 0.78,
            "per_class": {
                "glioma": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10}
            },
        }
    },
}


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    folder = tmp_path / "artifacts" / "model-v0.2.0"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(MANIFEST))
    (folder / "classifier.joblib").write_bytes(b"weights")
    return folder


def test_package_writes_archive_checksum_and_notes(
    bundle: Path, tmp_path: Path
) -> None:
    archive, checksum, notes = package(bundle, tmp_path / "out")

    assert archive.name == "model-v0.2.0.tar.gz"
    assert checksum.read_text().endswith(f"  {archive.name}\n")
    assert "macro F1: 0.750" in notes.read_text()
    with tarfile.open(archive) as tar:
        assert "model-v0.2.0/classifier.joblib" in tar.getnames()


def test_package_refuses_a_folder_that_does_not_match_the_version(
    bundle: Path, tmp_path: Path
) -> None:
    renamed = bundle.rename(bundle.with_name("model-v9.9.9"))

    with pytest.raises(ModelUnavailableError, match="does not match"):
        package(renamed, tmp_path / "out")


def test_package_never_overwrites_a_published_archive(
    bundle: Path, tmp_path: Path
) -> None:
    package(bundle, tmp_path / "out")

    with pytest.raises(ModelUnavailableError, match="already exists"):
        package(bundle, tmp_path / "out")


def test_release_notes_list_every_class() -> None:
    assert "| glioma | 0.900 | 0.800 | 0.850 | 10 |" in release_notes(MANIFEST)


def serve(monkeypatch: pytest.MonkeyPatch, archive: Path, digest: str) -> None:
    monkeypatch.setattr(
        fetch_module,
        "_download",
        lambda url, target: target.write_bytes(archive.read_bytes()),
    )
    monkeypatch.setattr(fetch_module, "_expected_digest", lambda url: digest)


def test_fetch_extracts_a_verified_bundle(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, checksum, _ = package(bundle, tmp_path / "out")
    serve(monkeypatch, archive, checksum.read_text().split()[0])

    result = fetch_module.fetch("0.2.0", "owner/repo", tmp_path / "dest")

    assert (result / "classifier.joblib").read_bytes() == b"weights"


def test_fetch_fails_on_a_checksum_mismatch_and_extracts_nothing(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _, _ = package(bundle, tmp_path / "out")
    serve(monkeypatch, archive, "0" * 64)

    with pytest.raises(ModelUnavailableError, match="checksum mismatch"):
        fetch_module.fetch("0.2.0", "owner/repo", tmp_path / "dest")

    assert not (tmp_path / "dest" / "model-v0.2.0").exists()


def test_extract_rejects_paths_outside_the_bundle(tmp_path: Path) -> None:
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("x")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../escape")

    with pytest.raises(ModelUnavailableError, match="unexpected path"):
        fetch_module.extract(evil, tmp_path / "dest", "model-v0.2.0")


def test_cli_prints_the_gh_command_to_publish(
    bundle: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["package", str(bundle), "--out", str(tmp_path / "out")])

    assert code == 0
    assert "gh release create model-v0.2.0" in capsys.readouterr().out


def test_cli_reports_an_error_for_a_missing_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["package", str(tmp_path), "--out", str(tmp_path / "out")])

    assert code == 1
    assert "error:" in capsys.readouterr().err
