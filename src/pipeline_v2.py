"""TempoGraph v2 orchestrator: chunked VLM pipeline backed by SQLite."""

import gc
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from src.aggregator import CaptionAggregator
from src.backends.llama_server_backend import LlamaServerBackend
from src.graph_builder import GraphBuilder
from src.models import (
    AnalysisResult,
    CameraMode,
    PipelineConfig,
    PipelineResult,
)
from src.modules.depth import DepthEstimator
from src.modules.detector import ObjectDetector
from src.modules.frame_scorer import FrameScorer
from src.modules.frame_selector import FrameSelector
from src.modules.whisper_cpp import (
    WhisperCppTranscriber,
    write_segments_to_db,
    write_transcript_json,
)
from src.storage import TempoGraphDB


class PipelineV2:
    def __init__(
        self,
        config: PipelineConfig,
        camera_mode: CameraMode = CameraMode.STATIC,
        yolo_fps: float = 1.0,
        vlm_fps: float = 0.5,
        chunk_size: int = 10,
        depth_enabled: bool = False,
        use_segmentation: bool = False,
        yolo_size: str = "n",
        threshold_mult: float = 1.0,
        jpeg_quality: int = 80,
        frame_max_width: int = 640,
        skip_vlm: bool = False,
        vlm_url: str = "http://127.0.0.1:8082",
        vlm_model: str = "Qwen3.5-9B-Q8_0.gguf",
        vlm_autostart_service: Optional[str] = None,
        vlm_autostart_timeout_s: float = 90.0,
        vlm_autostop: bool = False,
        vlm_frame_mode: str = "scored",
        audio_enabled: bool = False,
        whisper_model: str = "base.en",
        whisper_binary: Optional[str] = None,
        whisper_gpu_device: Optional[int] = 1,  # Vulkan dev 1 = NVIDIA 3060 (radv on AMD has device-lost issues)
        whisper_language: Optional[str] = None,
        on_stage: Optional[Callable[[str, str, dict], None]] = None,
    ):
        self.config = config
        self.camera_mode = camera_mode
        self.yolo_fps = yolo_fps
        self.vlm_fps = vlm_fps
        self.chunk_size = chunk_size
        self.depth_enabled = depth_enabled
        self.use_segmentation = use_segmentation
        if yolo_size not in ("n", "s", "m", "l", "x"):
            raise ValueError(
                f"yolo_size must be one of n/s/m/l/x, got {yolo_size!r}"
            )
        self.yolo_size = yolo_size
        self.threshold_mult = threshold_mult
        self.jpeg_quality = jpeg_quality
        self.frame_max_width = frame_max_width
        self.skip_vlm = skip_vlm
        self.vlm_url = vlm_url
        self.vlm_model = vlm_model
        self.vlm_autostart_service = vlm_autostart_service
        self.vlm_autostart_timeout_s = vlm_autostart_timeout_s
        self.vlm_autostop = vlm_autostop
        if vlm_frame_mode not in ("scored", "keyframes"):
            raise ValueError(
                f"vlm_frame_mode must be 'scored' or 'keyframes', got {vlm_frame_mode!r}"
            )
        self.vlm_frame_mode = vlm_frame_mode
        self.audio_enabled = audio_enabled
        self.whisper_model = whisper_model
        self.whisper_binary = whisper_binary
        self.whisper_gpu_device = whisper_gpu_device
        self.whisper_language = whisper_language
        self._on_stage = on_stage
        self.logger = logging.getLogger(__name__)

    def _stage(self, name: str, event: str = "start", **info) -> None:
        if self._on_stage is None:
            return
        try:
            self._on_stage(name, event, info)
        except Exception as e:
            self.logger.warning(f"on_stage callback failed: {e}")

    def _ensure_vlm_ready(self, backend) -> None:
        """If the VLM server is down and a service is configured, start it and poll."""
        if backend.is_available():
            return
        if not self.vlm_autostart_service:
            self._stage(
                "VLM captioning", "error",
                error=f"llama-server not reachable at {self.vlm_url}",
            )
            raise RuntimeError(
                f"VLM server not reachable at {self.vlm_url}. "
                "Start it with: systemctl --user start qwen35-turboquant.service"
            )

        self._stage(
            "VLM autostart", "start", service=self.vlm_autostart_service
        )
        t0 = time.time()
        try:
            subprocess.run(
                ["systemctl", "--user", "start", self.vlm_autostart_service],
                check=True, capture_output=True, text=True, timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError) as e:
            self._stage("VLM autostart", "error", error=str(e))
            raise RuntimeError(
                f"Failed to start {self.vlm_autostart_service}: {e}"
            ) from e

        deadline = t0 + self.vlm_autostart_timeout_s
        while time.time() < deadline:
            if backend.is_available():
                self._stage(
                    "VLM autostart", "done",
                    elapsed_s=round(time.time() - t0, 2),
                )
                return
            time.sleep(1.0)

        self._stage(
            "VLM autostart", "error",
            error=f"timeout after {self.vlm_autostart_timeout_s}s",
        )
        raise RuntimeError(
            f"{self.vlm_autostart_service} did not become reachable at "
            f"{self.vlm_url} within {self.vlm_autostart_timeout_s}s"
        )

    def run(self) -> PipelineResult:
        start = time.time()
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        db = TempoGraphDB(out_dir / "tempograph.db")
        try:
            # Stage 1: frame selection + JPEG export
            self._stage("Frame selection", "start")
            t0 = time.time()
            selection = FrameSelector().select(
                video_path=self.config.video_path,
                camera_mode=self.camera_mode,
                sample_fps=self.yolo_fps,
                threshold_mult=self.threshold_mult,
            )
            frame_paths = self._extract_and_save_frames(
                selection.frame_indices, out_dir / "frames"
            )
            video_fps, w, h = self._video_meta(self.config.video_path)
            # YOLO ran against the saved JPEGs (which may be resized to
            # frame_max_width). Normalise bboxes against the actual saved
            # dims, not the original video dims — otherwise the boxes are
            # shrunk by the resize factor when the UI renders them.
            if frame_paths:
                _saved = cv2.imread(str(frame_paths[0]))
                saved_h, saved_w = _saved.shape[:2]
            else:
                saved_w, saved_h = w, h
            for idx, path in zip(selection.frame_indices, frame_paths):
                ts_ms = int(idx * 1000.0 / max(video_fps, 1.0))
                db.insert_frame(
                    frame_idx=idx,
                    timestamp_ms=ts_ms,
                    image_path=str(path),
                    is_keyframe=(idx in set(selection.keyframe_indices)),
                    delta_score=self._delta_for_index(selection, idx),
                )
            self._stage(
                "Frame selection", "done",
                elapsed_s=round(time.time() - t0, 2),
                frames=len(selection.frame_indices),
                keyframes=len(selection.keyframe_indices),
            )

            # Stage 1.5: Audio transcription (optional, runs on full video)
            if self.audio_enabled:
                self._stage("Audio transcription", "start",
                            model=self.whisper_model,
                            gpu_device=self.whisper_gpu_device)
                t0 = time.time()
                try:
                    binary = self.whisper_binary or "/home/ashie/whisper.cpp/build/bin/whisper-cli"
                    transcriber = WhisperCppTranscriber(
                        binary=binary,
                        model=self.whisper_model,
                        gpu_device=self.whisper_gpu_device,
                        language=self.whisper_language,
                    )
                    segments = transcriber.transcribe_video(self.config.video_path)
                    write_segments_to_db(db, segments)
                    write_transcript_json(out_dir, segments)
                    self._stage(
                        "Audio transcription", "done",
                        elapsed_s=round(time.time() - t0, 2),
                        segments=len(segments),
                        chars=sum(len(s.text) for s in segments),
                    )
                except Exception as e:
                    self.logger.warning(f"Audio transcription failed: {e}")
                    self._stage("Audio transcription", "error", error=str(e))
            else:
                self._stage("Audio transcription", "skipped")

            # Stage 2: YOLO sweep
            self._stage(
                "YOLO detection", "start",
                model=(f"yolo26{self.yolo_size}-seg.pt" if self.use_segmentation
                       else f"yolo26{self.yolo_size}.pt"),
                frames=len(frame_paths),
            )
            t0 = time.time()
            model_path = (f"yolo26{self.yolo_size}-seg.pt" if self.use_segmentation
                       else f"yolo26{self.yolo_size}.pt")
            detector = ObjectDetector(
                model_path=model_path,
                confidence=self.config.confidence,
                device="cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu",
            )
            detector.detect_to_db(
                frame_indices=selection.frame_indices,
                frame_paths=frame_paths,
                db=db,
                frame_width=saved_w,
                frame_height=saved_h,
            )
            detector.cleanup()
            gc.collect()
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            det_count = db.count_detections()
            self._stage(
                "YOLO detection", "done",
                elapsed_s=round(time.time() - t0, 2),
                detections=det_count,
            )

            # Stage 3: depth (optional)
            if self.depth_enabled:
                self._stage("Depth estimation", "start", frames=len(frame_paths))
                t0 = time.time()
                depth = DepthEstimator(
                    device="cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
                )
                depth.estimate_to_db(
                    frame_indices=selection.frame_indices,
                    frame_paths=frame_paths,
                    db=db,
                    output_dir=str(out_dir / "depth"),
                )
                depth.cleanup()
                gc.collect()
                if _TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._stage(
                    "Depth estimation", "done",
                    elapsed_s=round(time.time() - t0, 2),
                )
            else:
                self._stage("Depth estimation", "skipped")

            # Stage 4: VLM frame subset
            self._stage("Frame scoring", "start", mode=self.vlm_frame_mode)
            t0 = time.time()
            if self.vlm_frame_mode == "keyframes":
                kfs = sorted(set(selection.keyframe_indices))
                if kfs:
                    vlm_frames = kfs
                    fallback_used = False
                else:
                    self.logger.warning(
                        "vlm_frame_mode=keyframes but FrameSelector returned 0 "
                        "keyframes; falling back to all sampled frames."
                    )
                    vlm_frames = list(selection.frame_indices)
                    fallback_used = True
                self._stage(
                    "Frame scoring", "done",
                    elapsed_s=round(time.time() - t0, 2),
                    mode="keyframes",
                    vlm_frames=len(vlm_frames),
                    fallback_used=fallback_used,
                )
            else:
                video_duration = max(1.0, self._video_duration(self.config.video_path))
                k = max(1, int(round(video_duration * self.vlm_fps)))
                scorer = FrameScorer(db)
                vlm_frames = scorer.score_and_select(
                    candidate_frame_indices=selection.frame_indices,
                    keyframe_indices=set(selection.keyframe_indices),
                    k=k,
                )
                self._stage(
                    "Frame scoring", "done",
                    elapsed_s=round(time.time() - t0, 2),
                    mode="scored",
                    vlm_frames=len(vlm_frames),
                )

            if self.skip_vlm:
                self._stage("VLM captioning", "skipped", reason="skip_vlm flag")
                self._stage("Aggregation", "skipped", reason="skip_vlm flag")
                self.logger.info(
                    "Skipping VLM stages (skip_vlm=True). "
                    "Stopped after frame scoring."
                )
                elapsed = time.time() - start
                return PipelineResult(
                    analysis=None,
                    detection=None,
                    depth=None,
                    config=self.config,
                    annotated_video_path=None,
                    processing_time=elapsed,
                )

            # Stage 5: chunked VLM
            chunks: List[Tuple[int, List[int]]] = []
            for i in range(0, len(vlm_frames), self.chunk_size):
                chunks.append((len(chunks), vlm_frames[i : i + self.chunk_size]))
            self._stage(
                "VLM captioning", "start",
                chunks=len(chunks),
                vlm_url=self.vlm_url,
                vlm_model=self.vlm_model,
            )
            t0 = time.time()
            backend = LlamaServerBackend(base_url=self.vlm_url, model=self.vlm_model)
            self._ensure_vlm_ready(backend)
            chunk_caps = backend.caption_chunks(chunks=chunks, db=db)
            try:
                with open(out_dir / "chunks.json", "w") as f:
                    json.dump(
                        [
                            {
                                "chunk_id": c.chunk_id,
                                "frame_indices": list(c.frame_indices),
                                "per_frame_lines": {
                                    str(k): v for k, v in c.per_frame_lines.items()
                                },
                                "summary": c.summary,
                                "raw_response": c.raw_response,
                            }
                            for c in chunk_caps
                        ],
                        f,
                        indent=2,
                    )
            except Exception as e:
                self.logger.warning(f"failed to write chunks.json: {e}")
            self._stage(
                "VLM captioning", "done",
                elapsed_s=round(time.time() - t0, 2),
                non_empty_chunks=sum(1 for c in chunk_caps if c.summary),
            )

            # Stage 6: aggregate
            self._stage("Aggregation", "start")
            t0 = time.time()
            transcript_segments = (
                db.get_audio_segments() if self.audio_enabled else []
            )
            analysis = CaptionAggregator(
                base_url=self.vlm_url, model=self.vlm_model
            ).aggregate(chunk_caps, transcript_segments=transcript_segments)
            self._stage(
                "Aggregation", "done",
                elapsed_s=round(time.time() - t0, 2),
                entities=len(analysis.entities),
                events=len(analysis.visual_events),
            )

            # Build graph + persist
            graph = GraphBuilder()
            graph.build(analysis)
            try:
                graph.to_pyvis_html(str(out_dir / "graph.html"))
            except Exception as e:
                self.logger.warning(f"graph html failed: {e}")
            with open(out_dir / "analysis.json", "w") as f:
                f.write(analysis.model_dump_json(indent=2))

            elapsed = time.time() - start
            return PipelineResult(
                analysis=analysis,
                detection=None,
                depth=None,
                config=self.config,
                annotated_video_path=None,
                processing_time=elapsed,
            )
        finally:
            db.close()
            self._maybe_stop_vlm_service()

    def _maybe_stop_vlm_service(self) -> None:
        if not self.vlm_autostop or not self.vlm_autostart_service:
            return
        self._stage("VLM autostop", "start", service=self.vlm_autostart_service)
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", self.vlm_autostart_service],
                check=True, capture_output=True, text=True, timeout=15,
            )
            self._stage("VLM autostop", "done")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError) as e:
            self.logger.warning(f"failed to stop {self.vlm_autostart_service}: {e}")
            self._stage("VLM autostop", "error", error=str(e))

    def _video_meta(self, path: str) -> Tuple[float, int, int]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return fps, w, h

    def _video_duration(self, path: str) -> float:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n / fps if fps > 0 else 0.0

    def _delta_for_index(self, selection, idx) -> float:
        try:
            i = selection.scan_indices.index(idx)
            return float(selection.deltas[i])
        except ValueError:
            return 0.0

    def _extract_and_save_frames(self, indices, out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(self.config.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        scale = self.frame_max_width / width if width > self.frame_max_width else 1.0
        paths: List[Path] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if scale < 1.0:
                new_w = int(width * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            p = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            paths.append(p)
        cap.release()
        return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TempoGraph v2 pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/v2_run")
    parser.add_argument("--camera", default="static",
                        choices=["static", "moving", "auto"])
    parser.add_argument("--yolo-fps", type=float, default=1.0)
    parser.add_argument("--vlm-fps", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--seg", action="store_true",
                        help="Use the seg variant (yolo26<size>-seg.pt) "
                             "instead of the bbox-only model.")
    parser.add_argument("--yolo-size", choices=["n", "s", "m", "l", "x"],
                        default="n",
                        help="YOLO26 model size: n (nano, fastest) → x (xlarge, "
                             "most accurate). Larger = more VRAM and slower.")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--threshold-mult", type=float, default=1.0)
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8082")
    parser.add_argument("--vlm-model", default="Qwen3.5-9B-Q8_0.gguf")
    parser.add_argument(
        "--vlm-autostart-service",
        default=None,
        help="systemd --user unit to auto-start if VLM is unreachable "
             "(e.g. qwen35-turboquant.service)",
    )
    parser.add_argument(
        "--vlm-autostop",
        action="store_true",
        help="systemctl --user stop the autostart service when the pipeline finishes",
    )
    parser.add_argument(
        "--vlm-frame-mode",
        choices=["scored", "keyframes"],
        default="scored",
        help="scored: top-K by FrameScorer at --vlm-fps (default). "
             "keyframes: only motion-detected keyframes from FrameSelector "
             "(--vlm-fps ignored).",
    )
    parser.add_argument("--audio", action="store_true",
                        help="Transcribe audio with whisper.cpp")
    parser.add_argument("--whisper-model", default="base.en",
                        help="ggml model name (tiny.en/base.en/small.en/medium.en/large-v3)")
    parser.add_argument("--whisper-binary", default=None,
                        help="Path to whisper-cli binary (default: /home/ashie/whisper.cpp/build/bin/whisper-cli)")
    parser.add_argument("--whisper-gpu-device", type=int, default=1,
                        help="Vulkan device index. Default 1 = NVIDIA 3060 "
                             "(0 = AMD 9070 XT but radv has occasional device-lost errors).")
    parser.add_argument("--whisper-language", default=None,
                        help="Force language (e.g. 'en'); default = auto-detect")
    args = parser.parse_args()

    config = PipelineConfig(
        backend="llama-server",
        modules={"behavior": True, "detection": True, "depth": args.depth, "audio": False},
        fps=args.yolo_fps,
        max_frames=999,
        confidence=args.confidence,
        video_path=args.video,
        output_dir=args.output,
    )
    pipe = PipelineV2(
        config,
        camera_mode=CameraMode(args.camera),
        yolo_fps=args.yolo_fps,
        vlm_fps=args.vlm_fps,
        chunk_size=args.chunk_size,
        depth_enabled=args.depth,
        use_segmentation=args.seg,
        yolo_size=args.yolo_size,
        threshold_mult=args.threshold_mult,
        skip_vlm=args.skip_vlm,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        vlm_autostart_service=args.vlm_autostart_service,
        vlm_autostop=args.vlm_autostop,
        vlm_frame_mode=args.vlm_frame_mode,
        audio_enabled=args.audio,
        whisper_model=args.whisper_model,
        whisper_binary=args.whisper_binary,
        whisper_gpu_device=args.whisper_gpu_device,
        whisper_language=args.whisper_language,
    )
    result = pipe.run()
    print(f"Done in {result.processing_time:.1f}s -> {args.output}")
