"""Backends package for TempoGraph."""

from .base import BaseVLMBackend
from .gemini_backend import GeminiBackend

try:
    from .qwen_backend import QwenBackend
except ImportError:
    QwenBackend = None  # type: ignore[assignment,misc]

__all__ = ["BaseVLMBackend", "GeminiBackend", "QwenBackend"]
