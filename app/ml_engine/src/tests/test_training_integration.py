from pathlib import Path

import numpy as np
import pytest

from src.classical.config import RadiomicsConfig
from src.deep_learning.backbones import build_backbone
from src.deep_learning.bundle import write_bundle
from src.deep_learning.extractor import DeepFeatureExtractor
from src.preprocessing.prepare import PreprocessingConfig
from src.seed import set_seed
from src.tests.synthetic import write_labeled_dataset
from src.training.dataset import load_samples, split_samples
from src.training.features import FeatureCache, make_featurizer
from src.training.folds import assign_folds, fold_splits
from src.training.model import compare_classifiers

PER_CLASS = 20


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_labeled_dataset(
        tmp_path_factory.mktemp("synthetic-dataset"), PER_CLASS
    )


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
    split = split_samples(load_samples(data_dir, manifest))

    cache = FeatureCache(tmp_path / "cache", fingerprint, featurize, RadiomicsConfig())
    table, skipped = cache.table(split.train)
    test_table, _ = cache.table(split.test)

    assert skipped == []
    assert table.matrix.shape == (4 * PER_CLASS, 45 + 1280)
    assert test_table.matrix.shape[0] == 4 * PER_CLASS
    assert np.isfinite(table.matrix).all()

    y = np.array([s.label.value for s in table.samples])
    fold_of = assign_folds(split.train, 3, 0)
    splits = fold_splits(np.array([fold_of[s.path] for s in table.samples]), 3)
    result = compare_classifiers(table.matrix, y, 45, splits)

    assert result.scores[result.best].mean_macro_f1 > 0.5
