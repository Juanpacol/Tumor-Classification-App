import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from src.preprocessing.quality import METRICS_VERSION
from src.training.__main__ import build_parser, configure
from src.training.config import (
    RoundFileError,
    TrainingConfig,
    apply_round,
    resolved_config,
)
from src.training.features import config_fingerprint

ROUNDS = Path(__file__).resolve().parents[2] / "rounds"


def base() -> TrainingConfig:
    return TrainingConfig(
        data_dir=Path("d"), manifest=Path("m"), out_dir=Path("o"), weights_dir=Path("w")
    )


def round_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "round.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_an_empty_round_file_keeps_every_default(tmp_path: Path) -> None:
    config = apply_round(base(), round_file(tmp_path, ""))

    assert replace(config, round_source=None) == base()
    assert config.round_source is not None
    assert config.round_source.path == (tmp_path / "round.toml").as_posix()


def test_a_partial_round_file_changes_only_its_keys(tmp_path: Path) -> None:
    config = apply_round(
        base(),
        round_file(
            tmp_path,
            """
            [run]
            seed = 7
            [preprocessing]
            crop_margin = 0.1
            [preprocessing.thresholds]
            min_snr = 3
            [radiomics]
            percentiles = [25, 50, 75]
            [augmentation]
            copies = 2
            [model]
            pca_components = 16
            [classifiers.svm_rbf]
            C = 3.0
            """,
        ),
    )

    defaults = base()
    assert config.seed == 7 and config.n_splits == defaults.n_splits
    assert config.preprocessing.crop_margin == 0.1
    assert config.preprocessing.min_mask_fraction == 0.05
    assert config.preprocessing.thresholds.min_snr == 3
    assert config.preprocessing.thresholds.min_entropy == 3.2
    assert config.preprocessing.thresholds.metrics_version == METRICS_VERSION
    assert config.radiomics.percentiles == (25, 50, 75)
    assert config.radiomics.gray_levels == defaults.radiomics.gray_levels
    assert config.augmentation.copies == 2
    assert config.model.pca_components == 16
    assert config.model.classifier_params == {"svm_rbf": {"C": 3.0}}


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[preprocessing]\ncrop_margn = 0.1", "preprocessing.crop_margn"),
        ("[preprocessing.thresholds]\nmin_snrr = 1.0", "preprocessing.thresholds.min_snrr"),
        ("[optimizer]\nlr = 1", "optimizer"),
        ("[run]\nfolds = 3", "run.folds"),
        ("[model]\npca = 3", "model.pca"),
        ("[classifiers.xgboost]\nmax_depth = 3", "xgboost"),
        ("[classifiers.random_forest]\nn_estimator = 500", "n_estimator"),
        ("[classifiers.random_forest]\nn_estimators = -1", "classifiers.random_forest"),
        ('[classifiers.logistic_regression]\npenalty = "l1"', "classifiers.logistic_regression"),
        ("[radiomics]\nglcm_distances = [\"a\"]", r"radiomics.glcm_distances[0]"),
        ('[radiomics]\nwavelet = "foo"', "radiomics"),
        ("[radiomics]\ngray_levels = 0", "radiomics"),
        ("[augmentation]\nflip_probability = 2.0", "augmentation"),
        ("[preprocessing]\nmin_mask_fraction = 5.0", "preprocessing"),
        ("[preprocessing.thresholds]\nmin_snr = -1.0", "preprocessing.thresholds"),
        ("[augmentation]\ncopies = 1.5", "augmentation.copies"),
        ("[preprocessing]\ncrop_margin = \"wide\"", "preprocessing.crop_margin"),
        ("[run]\nn_splits = 1", "run.n_splits"),
        ("[model]\npca_components = 0", "model"),
        ("[preprocessing]\nthresholds = 3", "preprocessing.thresholds"),
    ],
)  # fmt: skip
def test_invalid_round_files_name_the_key(tmp_path: Path, text: str, key: str) -> None:
    with pytest.raises(
        RoundFileError, match=key.replace(".", r"\.").replace("[", r"\[")
    ):
        apply_round(base(), round_file(tmp_path, text))


def test_ints_for_floats_give_the_same_config_and_cache_key(tmp_path: Path) -> None:
    as_int = apply_round(
        base(),
        round_file(
            tmp_path,
            "[preprocessing.thresholds]\nmin_snr = 3\n[radiomics]\nglcm_angles = [0, 1]",
        ),
    )
    as_float = apply_round(
        base(),
        round_file(
            tmp_path,
            "[preprocessing.thresholds]\nmin_snr = 3.0\n[radiomics]\nglcm_angles = [0.0, 1.0]",
        ),
    )

    assert json.dumps(asdict(as_int.preprocessing)) == json.dumps(
        asdict(as_float.preprocessing)
    )
    assert as_int.radiomics.glcm_angles == (0.0, 1.0)
    assert isinstance(as_int.radiomics.glcm_angles[0], float)
    manifest = _Backbone()
    assert config_fingerprint(
        as_int.preprocessing, as_int.radiomics, manifest
    ) == config_fingerprint(as_float.preprocessing, as_float.radiomics, manifest)


class _Backbone:
    sha256 = "0" * 64
    input_size = 224
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ('[classifiers.logistic_regression]\nclass_weight = "none"', "class_weight"),
        ("[classifiers.svm_rbf]\nrandom_state = 1", "random_state"),
        ("[preprocessing.thresholds]\nmetrics_version = 1", "metrics_version"),
    ],
)
def test_guarded_keys_cannot_be_overridden(tmp_path: Path, text: str, key: str) -> None:
    with pytest.raises(RoundFileError, match=f"cannot be overridden.*{key}"):
        apply_round(base(), round_file(tmp_path, text))


def test_a_value_the_dataclass_rejects_is_a_round_file_error(tmp_path: Path) -> None:
    with pytest.raises(RoundFileError, match="preprocessing"):
        apply_round(base(), round_file(tmp_path, "[preprocessing]\ncrop_margin = -1.0"))


def test_malformed_toml_is_a_round_file_error(tmp_path: Path) -> None:
    with pytest.raises(RoundFileError):
        apply_round(base(), round_file(tmp_path, "[run\nseed = 1"))


def test_the_resolved_config_lists_every_parameter(tmp_path: Path) -> None:
    config = apply_round(base(), round_file(tmp_path, "[classifiers.svm_rbf]\nC = 3.0"))

    resolved = resolved_config(config)

    assert set(resolved) == {
        "run",
        "preprocessing",
        "radiomics",
        "augmentation",
        "model",
        "classifiers",
    }
    assert resolved["classifiers"]["svm_rbf"]["C"] == 3.0
    assert resolved["classifiers"]["svm_rbf"]["class_weight"] == "balanced"
    assert resolved["preprocessing"]["thresholds"]["metrics_version"] == METRICS_VERSION


def test_the_round_source_hashes_the_bytes_that_were_parsed(tmp_path: Path) -> None:
    path = round_file(tmp_path, "[run]\nseed = 7\n")
    loaded = path.read_bytes()

    config = apply_round(base(), path)
    path.write_text("[run]\nseed = 8\n", encoding="utf-8")

    assert config.round_source is not None
    assert config.round_source.path.endswith("round.toml")
    assert config.round_source.sha256 == hashlib.sha256(loaded).hexdigest()
    assert base().round_source is None


def test_the_baseline_round_file_is_the_defaults() -> None:
    config = apply_round(base(), ROUNDS / "0.1.0.toml")

    assert resolved_config(config) == resolved_config(base())


def configured(*extra: str) -> TrainingConfig:
    argv = ["--data-dir", "d", "--manifest", "m", *extra]
    return configure(build_parser().parse_args(argv))


def test_cli_flags_override_the_round_file(tmp_path: Path) -> None:
    path = round_file(tmp_path, "[run]\nseed = 7\nn_splits = 4")

    config = configured("--config", str(path), "--seed", "9")

    assert config.seed == 9 and config.n_splits == 4


def test_without_flags_the_round_file_decides(tmp_path: Path) -> None:
    config = configured("--config", str(round_file(tmp_path, "[run]\nseed = 7")))

    assert config.seed == 7


def test_without_a_round_file_the_cli_uses_the_defaults() -> None:
    config = configured()

    assert resolved_config(config) == resolved_config(base())
    assert config.round_source is None


def test_a_round_chooses_the_backbone_and_its_bundle_folder(tmp_path: Path) -> None:
    text = '[model]\nbackbone = "resnet50"\n'
    config = apply_round(base(), round_file(tmp_path, text))

    assert config.model.backbone == "resnet50"
    assert config.backbone_bundle == Path("w") / "resnet50"
    assert resolved_config(config)["model"]["backbone"] == "resnet50"


def test_the_default_backbone_is_efficientnet_b0() -> None:
    assert base().model.backbone == "efficientnet_b0"
    assert base().backbone_bundle == Path("w") / "efficientnet_b0"


def test_an_unknown_backbone_is_rejected_naming_the_key(tmp_path: Path) -> None:
    with pytest.raises(RoundFileError, match="model.backbone"):
        apply_round(base(), round_file(tmp_path, '[model]\nbackbone = "vgg16"\n'))


def test_the_cli_takes_the_weights_folder() -> None:
    args = build_parser().parse_args(
        ["--data-dir", "d", "--manifest", "m", "--weights-dir", "bundles"]
    )

    assert configure(args).weights_dir == Path("bundles")


def test_a_round_can_drop_the_radiomic_block(tmp_path: Path) -> None:
    text = '[model]\nblocks = ["cnn"]\n'
    config = apply_round(base(), round_file(tmp_path, text))

    assert config.model.blocks == ("cnn",)
    assert resolved_config(config)["model"]["blocks"] == ("cnn",)


@pytest.mark.parametrize("blocks", ["[]", '["radiomic"]', '["cnn", "radiomic"]'])
def test_other_blocks_are_rejected_naming_the_key(tmp_path: Path, blocks: str) -> None:
    with pytest.raises(RoundFileError, match="model.blocks"):
        apply_round(base(), round_file(tmp_path, f"[model]\nblocks = {blocks}\n"))


def test_a_round_chooses_the_calibration_method(tmp_path: Path) -> None:
    text = '[model]\ncalibration = "isotonic"\n'
    config = apply_round(base(), round_file(tmp_path, text))

    assert config.model.calibration == "isotonic"
    assert base().model.calibration == "sigmoid"


def test_an_unknown_calibration_method_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RoundFileError, match="model.calibration"):
        apply_round(base(), round_file(tmp_path, '[model]\ncalibration = "platt"\n'))
