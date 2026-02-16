from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
from src.models import AnalysisResult


class BaseVLMBackend(ABC):
    """Abstract base class for VLM backends."""

    @abstractmethod
    def analyze_video(
        self,
        video_path: str,
        frames: List[Path] = None,
        audio_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyze video and return structured results."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def cleanup(self):
        """Release GPU memory and resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        pass
