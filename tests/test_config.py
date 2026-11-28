"""Tests for project YAML configuration loading."""

from pathlib import Path

import pytest

from geoshift.config import load_config


@pytest.mark.parametrize(
    "filename",
    [
        "data.yml",
        "evaluation.yml",
        "inference.yml",
        "models.yml",
        "paths.yml",
        "training.yml",
    ],
)
def test_project_config_files_load_as_mappings(filename: str) -> None:
    config = load_config(filename)

    assert isinstance(config, dict)
    assert config


def test_config_directory_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yml"
    config_path.write_text("service:\n  port: 9000\n", encoding="utf-8")
    monkeypatch.setenv("GEOSHIFT_CONFIG_DIR", str(tmp_path))

    assert load_config("custom.yml") == {"service": {"port": 9000}}


def test_config_loader_rejects_paths_as_names() -> None:
    with pytest.raises(ValueError, match="filename"):
        load_config("nested/data.yml")
