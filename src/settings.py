"""Environment-driven settings for TempoGraph. Zero deps.

All values are read from environment variables on every call to
``get_settings()`` — no caching, no ``__init__``-time evaluation. Tests
can monkeypatch ``os.environ`` to simulate overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Settings:
    vlm_url: str = "http://127.0.0.1:8085"
    vlm_model: str = "ornith-1.0-9b-Q4_K_M.gguf"
    walker_url: str = field(default="")
    verifier_url: str = "http://127.0.0.1:8096"
    whisper_binary: str = field(default="~/whisper.cpp/build/bin/whisper-cli")
    whisper_model_dir: str = field(default="~/whisper.cpp/models")
    results_dir: str = "results"

    def __post_init__(self):
        if not self.walker_url:
            object.__setattr__(self, "walker_url", self.vlm_url)

    @classmethod
    def from_env(cls):
        import os

        whisper_binary = os.environ.get("TEMPOGRAPH_WHISPER_BIN", cls.whisper_binary)
        whisper_model_dir = os.environ.get(
            "TEMPOGRAPH_WHISPER_MODELS", cls.whisper_model_dir
        )

        vlm_url = os.environ.get("TEMPOGRAPH_VLM_URL", cls.vlm_url)
        walker_url = os.environ.get("TEMPOGRAPH_WALKER_URL", cls.vlm_url)
        verifier_url = os.environ.get("TEMPOGRAPH_VERIFIER_URL", cls.verifier_url)
        whisper_binary_expanded = os.path.expanduser(whisper_binary)
        whisper_model_dir_expanded = os.path.expanduser(whisper_model_dir)
        results_dir = os.environ.get("TEMPOGRAPH_RESULTS_DIR", cls.results_dir)

        return cls(
            vlm_url=vlm_url,
            vlm_model=os.environ.get("TEMPOGRAPH_VLM_MODEL", cls.vlm_model),
            walker_url=walker_url,
            verifier_url=verifier_url,
            whisper_binary=whisper_binary_expanded,
            whisper_model_dir=whisper_model_dir_expanded,
            results_dir=results_dir,
        )

    def __repr__(self):
        return (
            f"Settings(vlm_url={self.vlm_url!r}, vlm_model={self.vlm_model!r}, "
            f"walker_url={self.walker_url!r}, verifier_url={self.verifier_url!r}, "
            f"whisper_binary={self.whisper_binary!r}, whisper_model_dir={self.whisper_model_dir!r}, "
            f"results_dir={self.results_dir!r})"
        )


def get_settings() -> Settings:
    """Read env each call (no caching — tests monkeypatch os.environ)."""
    return Settings.from_env()
