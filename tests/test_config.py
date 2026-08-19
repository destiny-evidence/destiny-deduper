"""Tests for destiny_deduper.config module-level path resolution and copy logic."""

import importlib
from pathlib import Path

import pytest

from destiny_deduper import config


@pytest.fixture(autouse=True)
def _reload_config_after_test():
    """Ensure later tests see the real config module state again."""
    yield
    importlib.reload(config)


def _reload_with_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    return importlib.reload(config)


def test_user_config_copied_when_missing(monkeypatch, tmp_path):
    (tmp_path / ".config" / "destiny-dedupe").mkdir(parents=True)
    reloaded = _reload_with_home(monkeypatch, tmp_path)

    assert reloaded.USER_CONFIG_FILE_PATH.is_file()
    assert (
        reloaded.USER_CONFIG_FILE_PATH.read_text()
        == reloaded.CONFIG_FILE_PATH.read_text()
    )


def test_user_config_not_overwritten_when_present(monkeypatch, tmp_path):
    user_dir = tmp_path / ".config" / "destiny-dedupe"
    user_dir.mkdir(parents=True)
    custom_config = user_dir / ".config.yaml"
    custom_config.write_text("custom: true\n")

    reloaded = _reload_with_home(monkeypatch, tmp_path)

    assert reloaded.USER_CONFIG_FILE_PATH.read_text() == "custom: true\n"


def test_settings_yaml_file_prefers_user_config_when_present(monkeypatch, tmp_path):
    user_dir = tmp_path / ".config" / "destiny-dedupe"
    user_dir.mkdir(parents=True)
    (user_dir / ".config.yaml").write_text(config.CONFIG_FILE_PATH.read_text())

    reloaded = _reload_with_home(monkeypatch, tmp_path)

    assert reloaded.Settings.model_config["yaml_file"] == reloaded.USER_CONFIG_FILE_PATH


def test_settings_yaml_file_falls_back_on_windows(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("sys.platform", "win32")
    reloaded = importlib.reload(config)

    assert reloaded.Settings.model_config["yaml_file"] == reloaded.CONFIG_FILE_PATH


def test_mkdir_permission_denied_falls_back_gracefully(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("mkdir denied")
        ),
    )

    reloaded = importlib.reload(config)

    assert not reloaded.USER_CONFIG_FILE_PATH.is_file()
    assert reloaded.Settings.model_config["yaml_file"] == reloaded.CONFIG_FILE_PATH


def test_get_settings_returns_cached_instance():
    first = config.get_settings()
    second = config.get_settings()

    assert first is second


def test_get_settings_loads_expected_structure():
    settings = config.get_settings()

    assert settings.thresholds.paper.match is not None
    assert settings.weights.doi is not None
    assert isinstance(settings.stopwords.title, list)
