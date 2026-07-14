"""Modules package for TempoGraph v2."""

from .frame_selector import FrameSelector
from .frame_scorer import FrameScorer
from .whisper_cpp import WhisperCppTranscriber

try:
    from .detector import ObjectDetector
except ImportError:
    ObjectDetector = None  # type: ignore[assignment,misc]

try:
    from .depth import DepthEstimator
except ImportError:
    DepthEstimator = None  # type: ignore[assignment,misc]

__all__ = [
    "FrameSelector",
    "FrameScorer",
    "WhisperCppTranscriber",
    "ObjectDetector",
    "DepthEstimator",
]
