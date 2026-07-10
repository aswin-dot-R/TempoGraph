"""Probe video metadata and derive a pipeline plan from it.

Pure functions — no real video needed for unit tests.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class VideoFacts:
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass
class DerivedPlan:
    yolo_size: str
    yolo_seg: bool
    yolo_fps: float
    confidence: float
    depth_enabled: bool
    audio_enabled: bool
    whisper_model: str
    vlm_frame_mode: str
    vlm_fps: float
    chunk_size: int
    dynamic_chunking: bool
    vlm_dedup_threshold: float
    threshold_mult: float
    camera_mode: str
    skip_vlm: bool = False

    def to_pipeline_kwargs(self) -> dict:
        """Convert to kwargs suitable for PipelineV2.__init__ (minus config)."""
        return {
            "yolo_fps": self.yolo_fps,
            "vlm_fps": self.vlm_fps,
            "chunk_size": self.chunk_size,
            "depth_enabled": self.depth_enabled,
            "use_segmentation": self.yolo_seg,
            "yolo_size": self.yolo_size,
            "threshold_mult": self.threshold_mult,
            "confidence": self.confidence,
            "vlm_frame_mode": self.vlm_frame_mode,
            "vlm_dedup_threshold": self.vlm_dedup_threshold,
            "dynamic_chunking": self.dynamic_chunking,
            "camera_mode": self.camera_mode,
            "audio_enabled": self.audio_enabled,
            "whisper_model": self.whisper_model,
            "skip_vlm": self.skip_vlm,
        }


def probe(path: str) -> VideoFacts:
    """Run ffprobe on a video and return structured metadata."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    fmt = data.get("format", {})

    # Duration
    duration_s = float(fmt.get("duration", 0))
    if duration_s == 0 and video_stream:
        nb_frames = int(video_stream.get("nb_frames", 0))
        fps = float(video_stream.get("r_frame_rate", "30/1").split("/")[1] or 30)
        if nb_frames > 0:
            duration_s = nb_frames / fps

    # Video properties
    if video_stream:
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    else:
        width = 0
        height = 0
        fps = 0.0

    has_audio = audio_stream is not None

    return VideoFacts(
        duration_s=duration_s,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
    )


def derive_plan(facts: VideoFacts) -> DerivedPlan:
    """Derive pipeline knobs from video facts per the design doc rules.

    Rules:
    - YOLO variant: short clip (<=3 min) -> yolo26x-seg, else yolo26n-seg
    - Sweep FPS budgeted by duration
    - Confidence stays 0.5
    - Whisper: skipped when no audio track; base.en otherwise
    - Depth: on
    - VLM captioning: frame rate fitted to time budget, dynamic chunking on,
      dedup default
    """
    duration = facts.duration_s

    # YOLO variant
    if duration <= 180:
        yolo_size = "x"
        yolo_seg = True
    else:
        yolo_size = "n"
        yolo_seg = True

    # YOLO sweep FPS — budget ~30% of duration for detection
    if duration > 0:
        yolo_fps = max(
            0.5, min(4.0, int(duration * 0.3 / max(1, int(duration * 0.3)) + 1))
        )
        # Keep it reasonable: 0.5-3 fps
        yolo_fps = max(0.5, min(3.0, round(duration / 30.0, 1)))
    else:
        yolo_fps = 1.0

    confidence = 0.5
    threshold_mult = 1.0

    # Audio
    audio_enabled = facts.has_audio
    whisper_model = "base.en" if audio_enabled else "tiny.en"

    # Depth
    depth_enabled = True

    # VLM
    vlm_frame_mode = "keyframes"
    # VLM FPS — target ~1 frame per 2-3s of video
    if duration > 0:
        vlm_fps = max(0.1, min(2.0, round(duration / 120.0, 1)))
    else:
        vlm_fps = 0.5

    chunk_size = 10
    dynamic_chunking = True
    vlm_dedup_threshold = 0.92

    # Camera mode
    camera_mode = "auto"

    return DerivedPlan(
        yolo_size=yolo_size,
        yolo_seg=yolo_seg,
        yolo_fps=yolo_fps,
        confidence=confidence,
        depth_enabled=depth_enabled,
        audio_enabled=audio_enabled,
        whisper_model=whisper_model,
        vlm_frame_mode=vlm_frame_mode,
        vlm_fps=vlm_fps,
        chunk_size=chunk_size,
        dynamic_chunking=dynamic_chunking,
        vlm_dedup_threshold=vlm_dedup_threshold,
        threshold_mult=threshold_mult,
        camera_mode=camera_mode,
    )
