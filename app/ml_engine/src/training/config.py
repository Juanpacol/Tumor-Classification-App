"""Training configuration and the TOML round files that override its defaults."""

from __future__ import annotations

import hashlib
import tomllib
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone

from src.classical.config import RadiomicsConfig
from src.preprocessing.prepare import PreprocessingConfig
from src.training.augmentation import AugmentationConfig
from src.training.folds import N_SPLITS
from src.training.model import DEFAULT_SEED, ModelConfig, candidate_classifiers

GUARDED_KEYS = frozenset({"preprocessing.thresholds.metrics_version"})
"""Round-file keys that name an engine property rather than a tunable value."""


@dataclass(frozen=True)
class RoundSource:
    """Path and SHA-256 of a round file, hashed from the bytes that were parsed."""

    path: str
    sha256: str


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    manifest: Path
    backbone_bundle: Path
    out_dir: Path
    seed: int = DEFAULT_SEED
    n_splits: int = N_SPLITS
    version: str = "0.0.0-dev"
    cache_dir: Path | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    radiomics: RadiomicsConfig = field(default_factory=RadiomicsConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    round_source: RoundSource | None = None


class RoundFileError(ValueError):
    """A round file that names an unknown or guarded key, or holds an invalid value."""


def apply_round(config: TrainingConfig, path: Path) -> TrainingConfig:
    """``config`` with the overrides of the round file at ``path``.

    Keys the file leaves out keep the value they have in ``config``. Every key is
    checked before anything is returned, so an invalid file stops a run before any
    image is read. Raises ``RoundFileError`` naming the dotted key at fault.
    """
    data = path.read_bytes()
    try:
        table = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise RoundFileError(f"{path}: {error}") from error
    sections = {
        "run": _run,
        "preprocessing": lambda c, t: {
            "preprocessing": _overlay(c.preprocessing, t, "preprocessing")
        },
        "radiomics": lambda c, t: {"radiomics": _overlay(c.radiomics, t, "radiomics")},
        "augmentation": lambda c, t: {
            "augmentation": _overlay(c.augmentation, t, "augmentation")
        },
        "model": lambda c, t: {"model": _model(c.model, t)},
        "classifiers": lambda c, t: {"model": _classifiers(c.model, t, c.seed)},
    }
    if unknown := set(table) - set(sections):
        raise RoundFileError(f"unknown section: {min(unknown)}")
    for name, apply in sections.items():
        if name in table:
            config = replace(config, **apply(config, _table(table[name], name)))
    source = RoundSource(path.as_posix(), hashlib.sha256(data).hexdigest())
    return replace(config, round_source=source)


def resolved_config(config: TrainingConfig) -> dict[str, Any]:
    """Every tunable parameter of ``config`` with its effective value, including the
    full parameter set of each candidate classifier."""
    candidates = candidate_classifiers(config.seed, config.model.classifier_params)
    return {
        "run": {"seed": config.seed, "n_splits": config.n_splits},
        "preprocessing": asdict(config.preprocessing),
        "radiomics": asdict(config.radiomics),
        "augmentation": asdict(config.augmentation),
        "model": {"pca_components": config.model.pca_components},
        "classifiers": {
            name: classifier.get_params(deep=False)
            for name, classifier in candidates.items()
        },
    }


def _table(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RoundFileError(f"{key}: expected a table")
    return value


def _run(config: TrainingConfig, table: dict[str, Any]) -> dict[str, Any]:
    if unknown := set(table) - {"seed", "n_splits"}:
        raise RoundFileError(f"unknown key: run.{min(unknown)}")
    for key, value in table.items():
        _coerce(value, 0, f"run.{key}")
    if table.get("n_splits", config.n_splits) < 2:
        raise RoundFileError("run.n_splits: must be at least 2")
    return table


def _model(model: ModelConfig, table: dict[str, Any]) -> ModelConfig:
    if unknown := set(table) - {"pca_components"}:
        raise RoundFileError(f"unknown key: model.{min(unknown)}")
    values = {
        key: _coerce(value, getattr(model, key), f"model.{key}")
        for key, value in table.items()
    }
    try:
        return replace(model, **values)
    except ValueError as error:
        raise RoundFileError(f"model: {error}") from error


def _classifiers(model: ModelConfig, table: dict[str, Any], seed: int) -> ModelConfig:
    """Hyperparameter overrides, checked by building the candidates (which rejects
    unknown names, unknown parameters and guarded ones) and by fitting each changed
    candidate on a tiny synthetic set, so incompatible combinations fail now."""
    params = {
        name: _table(value, f"classifiers.{name}") for name, value in table.items()
    }
    try:
        candidates = candidate_classifiers(seed, params)
    except ValueError as error:
        raise RoundFileError(f"classifiers: {error}") from error
    x = np.random.default_rng(seed).normal(size=(40, 4))
    y = np.repeat(np.arange(4), 10)
    for name in params:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clone(candidates[name]).fit(x, y)
        except (ValueError, TypeError) as error:
            raise RoundFileError(f"classifiers.{name}: {error}") from error
    return replace(model, classifier_params=params)


def _overlay(default: Any, table: dict[str, Any], prefix: str) -> Any:
    """``default`` (a dataclass) with the values of ``table``, recursing into nested
    dataclasses; arrays become tuples, as the dataclasses store them."""
    known = {f.name for f in fields(default)}
    values: dict[str, Any] = {}
    for key, value in table.items():
        dotted = f"{prefix}.{key}"
        if dotted in GUARDED_KEYS:
            raise RoundFileError(f"cannot be overridden: {dotted}")
        if key not in known:
            raise RoundFileError(f"unknown key: {dotted}")
        current = getattr(default, key)
        if is_dataclass(current):
            values[key] = _overlay(current, _table(value, dotted), dotted)
            continue
        values[key] = _coerce(value, current, dotted)
    try:
        return replace(default, **values)
    except (ValueError, TypeError) as error:
        raise RoundFileError(f"{prefix}: {error}") from error


def _coerce(value: Any, current: Any, key: str) -> Any:
    """``value`` in the type of the default ``current``, or ``RoundFileError``.

    Ints become floats where the default is a float, so ``3`` and ``3.0`` give the
    same configuration and the same feature-cache key. Arrays become tuples, with
    each element checked against the default's first element.
    """
    if isinstance(current, tuple):
        if not isinstance(value, list):
            raise RoundFileError(
                f"{key}: expected an array, got {type(value).__name__}"
            )
        if not current:
            return tuple(value)
        return tuple(
            _coerce(item, current[0], f"{key}[{i}]") for i, item in enumerate(value)
        )
    if isinstance(current, bool):
        ok = isinstance(value, bool)
    elif isinstance(current, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif isinstance(current, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, type(current))
    if not ok:
        raise RoundFileError(
            f"{key}: expected {type(current).__name__}, got {type(value).__name__}"
        )
    return value
