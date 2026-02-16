"""
Audio analysis using Whisper-small for local mode.

Model: openai/whisper-small (~461MB)
VRAM: ~1GB on GPU, or runs on CPU

Usage:
import whisper
model = whisper.load_model("small", device="cuda")
result = model.transcribe(
    "audio.wav",
    word_timestamps=True,    # gives per-word timing
    language=None,           # auto-detect
)

# result["segments"] is a list of:
# {
#   "id": 0,
#   "start": 0.0,       # seconds
#   "end": 2.5,
#   "text": " Come here boy!",
#   "words": [
#     {"word": "Come", "start": 0.0, "end": 0.3},
#     ...
#   ]
# }

Convert Whisper segments → AudioEvent objects:
- Each segment becomes a "speech" AudioEvent
- start_time/end_time in MM:SS format
- text = segment text
- speaker = "Speaker 1" (simple — no real diarization in Whisper)
- emotion = None (or use simple heuristic: exclamation = excited,
  question = questioning)

NOTE: For non-speech sounds (barking, music, etc.), Whisper won't
detect these. We can add basic audio classification using librosa
as a stretch goal, but for the hackathon, just do speech transcription.
"""

import logging
import torch
import gc
from pathlib import Path
from typing import Optional, List
from src.models import AudioEvent


class AudioAnalyzer:
    def __init__(self, model_size: str = "small", device: str = "auto"):
        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.logger = logging.getLogger(__name__)
        self._model = None

    def _resolve_device(self, device: str) -> str:
        """Resolve device string, auto-detect if needed."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _load_model(self):
        """Lazy load Whisper model."""
        if self._model is not None:
            return

        self.logger.info(f"Loading Whisper model: {self.model_size}")

        try:
            import whisper
        except ImportError as e:
            raise ImportError(
                "whisper not installed. Install with: pip install openai-whisper"
            ) from e

        self._model = whisper.load_model(self.model_size, device=self.device)
        self.logger.info("Whisper model loaded successfully")

    def analyze(self, audio_path: str) -> List[AudioEvent]:
        """
        Transcribe audio and return AudioEvent list.
        """
        self._load_model()

        self.logger.info(f"Transcribing audio: {audio_path}")

        try:
            # Transcribe with word timestamps
            result = self._model.transcribe(
                audio_path,
                word_timestamps=True,
                language=None,  # Auto-detect language
                fp16=False,  # Use float32 for better accuracy
            )

            # Convert segments to AudioEvent objects
            audio_events = self._segments_to_events(result["segments"])

            self.logger.info(f"Transcription complete: {len(audio_events)} segments")

            return audio_events

        except Exception as e:
            self.logger.error(f"Error during audio transcription: {e}")
            raise

    def _segments_to_events(self, segments: List[dict]) -> List[AudioEvent]:
        """Convert Whisper segments to AudioEvent objects."""
        from src.models import SoundType

        audio_events = []

        for segment in segments:
            start_time = self._seconds_to_mmss(segment["start"])
            end_time = self._seconds_to_mmss(segment["end"])

            event = AudioEvent(
                type=SoundType.SPEECH,
                start_time=start_time,
                end_time=end_time,
                speaker="Speaker 1",
                text=segment["text"].strip(),
                label=None,
                emotion=None,
                confidence=0.8,
            )

            audio_events.append(event)

        return audio_events

    def _seconds_to_mmss(self, seconds: float) -> str:
        """Convert 65.3 → '01:05'"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def cleanup(self):
        """Free GPU memory."""
        self.logger.info("Cleaning up audio analyzer...")
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1e9
            self.logger.info(f"VRAM after cleanup: {vram_after:.2f}GB")
