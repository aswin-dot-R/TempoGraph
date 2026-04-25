"""Modules package for TempoGraph."""

from .frame_extractor import FrameExtractor, ExtractionResult

try:
    from .detector import ObjectDetector
except ImportError:
    ObjectDetector = None  # type: ignore[assignment,misc]

try:
    from .depth import DepthEstimator
except ImportError:
    DepthEstimator = None  # type: ignore[assignment,misc]

try:
    from .audio import AudioAnalyzer
except ImportError:
    AudioAnalyzer = None  # type: ignore[assignment,misc]

from .frame_selector import FrameSelector

__all__ = [
    "FrameExtractor",
    "ExtractionResult",
    "ObjectDetector",
    "DepthEstimator",
    "AudioAnalyzer",
    "FrameSelector",
]
