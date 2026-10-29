import csv
from pathlib import Path

import numpy as np
import pytest

from src.classical.config import RadiomicsConfig
from src.deep_learning.backbones import build_backbone
from src.deep_learning.bundle import write_bundle
from src.deep_learning.extractor import DeepFeatureExtractor
from src.preprocessing.prepare import PreprocessingConfig
from src.seed import set_seed
from src.tests.synthetic import save, textured_slice
from src.training.dataset import load_samples, split_samples
from src.training.features import FeatureCache, make_featurizer
from src.training.model import compare_classifiers

SIGMAS = {"glioma": 3.0, "meningioma": 6.0, "notumor": 12.0, "pituitary": 24.0}
PER_CLASS = 20


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("synthetic-dataset")
    data_dir = root / "data"
    rows = []
    group = 0
    for split in ("Train", "Test"):
        for label, sigma in SIGMAS.items():
            for index in range(PER_CLASS):
                relative = f"{split}/{label}/{index}.png"
                save(textured_slice(group, sigma), data_dir / relative)
                rows.append([relative, split, label, group, 1, 1, "kept"])
                group += 1
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["path", "split", "label", "group_id", "group_size", "keep", "reason"]
        )
        writer.writerows(rows)
    return data_dir, manifest


def test_synthetic_dataset_goes_through_features_and_selection(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    set_seed(0)
    data_dir, manifest = dataset
    bundle = tmp_path / "bundle"
    write_bundle(
        bundle,
        "efficientnet_b0",
        build_backbone("efficientnet_b0").state_dict(),
        "RANDOM",
    )
    extractor = DeepFeatureExtractor.from_bundle(bundle)
    featurize, fingerprint = make_featurizer(
        extractor, PreprocessingConfig(), RadiomicsConfig()
    )
    train, test = split_samples(load_samples(data_dir, manifest))

    cache = FeatureCache(tmp_path / "cache", fingerprint, featurize, RadiomicsConfig())
    table, skipped = cache.table(train)
    test_table, _ = cache.table(test)

    assert skipped == []
    assert table.matrix.shape == (4 * PER_CLASS, 45 + 1280)
    assert test_table.matrix.shape[0] == 4 * PER_CLASS
    assert np.isfinite(table.matrix).all()

    y = np.array([s.label.value for s in table.samples])
    groups = np.array([s.group_id for s in table.samples])
    result = compare_classifiers(table.matrix, y, groups, n_radiomic=45, n_splits=3)

    assert result.scores[result.best].mean_macro_f1 > 0.5
