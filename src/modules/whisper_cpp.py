"""Audio transcription via whisper.cpp (Vulkan / CUDA / HIP / CPU).

Subprocess wrapper around the `whisper-cli` binary built from
https://github.com/ggml-org/whisper.cpp. Extracts mono 16kHz audio with
ffmpeg, transcribes, parses the JSON output, persists segments to the
TempoGraph SQLite store, and writes a sidecar `transcript.json`.

GPU fallback chain:
  1. Try the requested gpu_device (e.g. 1 = NVIDIA)
  2. On failure, try the other GPU (e.g. 0 = AMD)
  3. On failure, fall back to CPU (--no-gpu)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_BINARY = "/home/ashie/whisper.cpp/build/bin/whisper-cli"
DEFAULT_MODEL_DIR = "/home/ashie/whisper.cpp/models"

# Vulkan device mapping for the current workstation.
# Device 0 = AMD RX 9070 XT, Device 1 = NVIDIA RTX 3060.
ALL_GPU_DEVICES = [1, 0]  # preferred order: NVIDIA first (radv on AMD is flaky)


@dataclass
class WhisperSegment:
    start_ms: int
    end_ms: int
    text: str
    no_speech_prob: Optional[float] = None
    avg_logprob: Optional[float] = None


class WhisperCppTranscriber:
    def __init__(
        self,
        binary: str = DEFAULT_BINARY,
        model: str = "base.en",
        model_dir: str = DEFAULT_MODEL_DIR,
        gpu_device: Optional[int] = None,
        threads: Optional[int] = None,
        language: Optional[str] = None,
    ):
        self.binary = binary
        self.model = model
        self.model_dir = model_dir
        self.gpu_device = gpu_device
        self.threads = threads
        self.language = language
        self.logger = logging.getLogger(__name__)

    def model_path(self) -> Path:
        return Path(self.model_dir) / f"ggml-{self.model}.bin"

    def is_available(self) -> bool:
        return Path(self.binary).exists() and self.model_path().exists()

    def ensure_model_downloaded(self) -> None:
        if self.model_path().exists():
            return
        download_script = Path(self.model_dir).parent / "models" / "download-ggml-model.sh"
        if not download_script.exists():
            raise RuntimeError(
                f"Model {self.model} not found at {self.model_path()} and "
                f"download script missing at {download_script}. Pull whisper.cpp."
            )
        self.logger.info(f"Downloading whisper model {self.model}...")
        subprocess.run(
            ["bash", str(download_script), self.model],
            cwd=str(download_script.parent.parent),
            check=True,
        )

    @staticmethod
    def extract_audio(video_path: str, out_wav: str) -> None:
        """Extract mono 16kHz PCM audio from any input video."""
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", out_wav,
            ],
            check=True, capture_output=True, text=True, timeout=300,
        )

    def _build_device_attempts(self) -> List[Optional[int]]:
        """Build the ordered list of GPU devices to try, ending with CPU.

        Returns a list of device IDs (int) or None for CPU.
        The requested device comes first, then others, then CPU.
        """
        attempts: List[Optional[int]] = []

        if self.gpu_device is not None:
            attempts.append(self.gpu_device)

        # Add other GPUs that aren't already in the list
        for dev in ALL_GPU_DEVICES:
            if dev not in attempts:
                attempts.append(dev)

        # Always end with CPU as the ultimate fallback
        attempts.append(None)

        return attempts

    def _run_whisper_cli(
        self, wav_path: str, json_out: str, device: Optional[int],
    ) -> subprocess.CompletedProcess:
        """Run whisper-cli with the given device. Returns CompletedProcess."""
        cmd = [
            self.binary,
            "-m", str(self.model_path()),
            "-f", wav_path,
            "-oj",
            "-of", json_out,
        ]
        if device is None:
            cmd += ["--no-gpu"]
        else:
            cmd += ["-dev", str(device)]
        if self.threads is not None:
            cmd += ["-t", str(self.threads)]
        if self.language:
            cmd += ["-l", self.language]

        self.logger.info(f"Running whisper-cli: {' '.join(cmd)}")
        return subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=1800,
        )

    def transcribe_video(self, video_path: str) -> List[WhisperSegment]:
        """Transcribe a video's audio track, with automatic GPU fallback.

        Tries GPU devices in order, falling back to CPU if all GPUs fail.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of WhisperSegment with timestamps and text.

        Raises:
            FileNotFoundError: If whisper-cli binary is missing.
            RuntimeError: If all transcription attempts fail.
        """
        if not self.is_available():
            self.ensure_model_downloaded()
        if not Path(self.binary).exists():
            raise FileNotFoundError(
                f"whisper-cli binary not found at {self.binary}. "
                "Build whisper.cpp first."
            )

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "audio.wav")
            self.logger.info("Extracting audio with ffmpeg...")
            self.extract_audio(video_path, wav_path)

            attempts = self._build_device_attempts()
            last_error = None

            for device in attempts:
                device_label = f"GPU device {device}" if device is not None else "CPU"
                json_out = os.path.join(tmp, "audio")
                json_path = json_out + ".json"

                # Clean up any leftover .json from a prior failed attempt
                if os.path.exists(json_path):
                    os.remove(json_path)

                try:
                    result = self._run_whisper_cli(wav_path, json_out, device)

                    if result.returncode != 0:
                        self.logger.warning(
                            f"whisper-cli failed on {device_label} "
                            f"(exit {result.returncode}): "
                            f"{result.stderr[:500]}"
                        )
                        last_error = RuntimeError(
                            f"whisper-cli failed on {device_label} "
                            f"(exit {result.returncode})"
                        )
                        continue

                    if not os.path.exists(json_path):
                        self.logger.warning(
                            f"whisper-cli on {device_label} exited 0 "
                            f"but produced no JSON output."
                        )
                        last_error = RuntimeError(
                            f"No JSON output from {device_label}"
                        )
                        continue

                    with open(json_path) as f:
                        payload = json.load(f)

                    segments = self._parse_segments(payload)
                    self.logger.info(
                        f"Transcription succeeded on {device_label}: "
                        f"{len(segments)} segments"
                    )
                    return segments

                except subprocess.TimeoutExpired:
                    self.logger.warning(
                        f"whisper-cli timed out on {device_label} (30m limit)"
                    )
                    last_error = RuntimeError(
                        f"Timeout on {device_label}"
                    )
                    continue
                except json.JSONDecodeError as e:
                    self.logger.warning(
                        f"whisper-cli on {device_label} produced invalid JSON: {e}"
                    )
                    last_error = RuntimeError(
                        f"Invalid JSON from {device_label}: {e}"
                    )
                    continue

            # All attempts exhausted
            raise RuntimeError(
                f"All whisper transcription attempts failed. "
                f"Tried: {[f'GPU {d}' if d is not None else 'CPU' for d in attempts]}. "
                f"Last error: {last_error}"
            )

    @staticmethod
    def _parse_segments(payload: dict) -> List[WhisperSegment]:
        segments_raw = payload.get("transcription") or payload.get("segments") or []
        out: List[WhisperSegment] = []
        for s in segments_raw:
            offsets = s.get("offsets") or {}
            start_ms = int(offsets.get("from") if "from" in offsets else s.get("start", 0) * 1000)
            end_ms = int(offsets.get("to") if "to" in offsets else s.get("end", 0) * 1000)
            text = (s.get("text") or "").strip()
            if not text:
                continue
            out.append(WhisperSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                no_speech_prob=s.get("no_speech_prob"),
                avg_logprob=s.get("avg_logprob"),
            ))
        return out


def write_segments_to_db(db, segments: List[WhisperSegment]) -> None:
    for seg in segments:
        db.insert_audio_segment(
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            text=seg.text,
            no_speech_prob=seg.no_speech_prob,
            avg_logprob=seg.avg_logprob,
        )


def write_transcript_json(out_dir: Path, segments: List[WhisperSegment]) -> None:
    payload = [
        {
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "text": s.text,
            "no_speech_prob": s.no_speech_prob,
            "avg_logprob": s.avg_logprob,
        }
        for s in segments
    ]
    with open(out_dir / "transcript.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
