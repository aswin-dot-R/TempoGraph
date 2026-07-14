"""Tests for ``src.settings`` — PS8a contract.

All tests use ``pytest.importorskip`` + monkeypatching ``os.environ``.
No GPU required, no network, no real whisper binary.
"""

import os
import pytest
from unittest import mock

pytest.importorskip("src.settings")

import src.settings
from src.settings import Settings, get_settings


def _clean_env():
    """Remove every TEMPOGRAPH_* variable we know about."""
    for key in (
        "TEMPOGRAPH_VLM_URL",
        "TEMPOGRAPH_VLM_MODEL",
        "TEMPOGRAPH_WALKER_URL",
        "TEMPOGRAPH_VERIFIER_URL",
        "TEMPOGRAPH_WHISPER_BIN",
        "TEMPOGRAPH_WHISPER_MODELS",
        "TEMPOGRAPH_RESULTS_DIR",
    ):
        os.environ.pop(key, None)


def test_defaults_returned_with_clean_env():
    _clean_env()
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.vlm_url == "http://127.0.0.1:8085"
    assert settings.vlm_model == "ornith-1.0-9b-Q4_K_M.gguf"
    assert settings.walker_url == "http://127.0.0.1:8085"
    assert settings.verifier_url == "http://127.0.0.1:8096"
    assert settings.whisper_binary == os.path.expanduser(
        "~/whisper.cpp/build/bin/whisper-cli"
    )
    assert settings.whisper_model_dir == os.path.expanduser("~/whisper.cpp/models")
    assert settings.results_dir == "results"


def test_walker_url_follows_vlm_url_by_default():
    _clean_env()
    settings = get_settings()
    assert settings.walker_url == settings.vlm_url


def test_env_var_overrides_every_field():
    _clean_env()
    with mock.patch.dict(
        os.environ,
        {
            "TEMPOGRAPH_VLM_URL": "http://example.com:9999",
            "TEMPOGRAPH_VLM_MODEL": "custom-model.gguf",
            "TEMPOGRAPH_WALKER_URL": "http://other.com:1234",
            "TEMPOGRAPH_VERIFIER_URL": "http://verifier.example.com:5555",
            "TEMPOGRAPH_WHISPER_BIN": "/opt/custom/whisper-cli",
            "TEMPOGRAPH_WHISPER_MODELS": "/opt/custom/models",
            "TEMPOGRAPH_RESULTS_DIR": "/tmp/results",
        },
    ):
        settings = get_settings()
        assert settings.vlm_url == "http://example.com:9999"
        assert settings.vlm_model == "custom-model.gguf"
        assert settings.walker_url == "http://other.com:1234"
        assert settings.verifier_url == "http://verifier.example.com:5555"
        assert settings.whisper_binary == "/opt/custom/whisper-cli"
        assert settings.whisper_model_dir == "/opt/custom/models"
        assert settings.results_dir == "/tmp/results"


def test_whisper_paths_expanduser():
    _clean_env()
    with mock.patch.dict(
        os.environ,
        {
            "TEMPOGRAPH_WHISPER_BIN": "~/whisper.cpp/build/bin/whisper-cli",
            "TEMPOGRAPH_WHISPER_MODELS": "~/whisper.cpp/models",
        },
    ):
        settings = get_settings()
        assert settings.whisper_binary == os.path.expanduser(
            "~/whisper.cpp/build/bin/whisper-cli"
        )
        assert settings.whisper_model_dir == os.path.expanduser("~/whisper.cpp/models")
        assert "~" not in settings.whisper_binary
        assert "~" not in settings.whisper_model_dir


def test_no_caching_between_calls():
    _clean_env()
    with mock.patch.dict(os.environ, {"TEMPOGRAPH_VLM_URL": "http://first.com"}):
        s1 = get_settings()
        assert s1.vlm_url == "http://first.com"
    with mock.patch.dict(os.environ, {"TEMPOGRAPH_VLM_URL": "http://second.com"}):
        s2 = get_settings()
        assert s2.vlm_url == "http://second.com"
        assert s1.vlm_url != s2.vlm_url


def test_settings_are_immutable():
    _clean_env()
    settings = get_settings()
    with pytest.raises(AttributeError):
        settings.vlm_url = "http://nowhere.com"
    with pytest.raises(Exception):
        settings.__dict__ = {"vlm_url": "http://hacked.com"}


def test_repr_includes_all_fields():
    _clean_env()
    settings = get_settings()
    r = repr(settings)
    for field in (
        "vlm_url",
        "walker_url",
        "verifier_url",
        "whisper_binary",
        "whisper_model_dir",
        "results_dir",
        "vlm_model",
    ):
        assert field in r
