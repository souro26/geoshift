"""Load and validate project configuration from the repository YAML files."""

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config_directory() -> Path:
    configured_directory = os.environ.get("GEOSHIFT_CONFIG_DIR")
    if configured_directory:
        return Path(configured_directory).expanduser().resolve()
    return PROJECT_ROOT / "configs"


@lru_cache(maxsize=16)
def _read_yaml(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    try:
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"GeoShift configuration file not found: {path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in GeoShift configuration {path}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(f"GeoShift configuration must be a YAML mapping: {path}")
    return config


def load_config(name: str) -> dict[str, Any]:
    """Load a named YAML mapping from configs/ (or GEOSHIFT_CONFIG_DIR)."""

    if Path(name).name != name or not name.endswith((".yml", ".yaml")):
        raise ValueError("Configuration name must be a YAML filename, not a path.")
    return deepcopy(_read_yaml(str((_config_directory() / name).resolve())))


def project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
