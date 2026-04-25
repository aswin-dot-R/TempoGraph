"""TempoGraph v2 orchestrator: chunked VLM pipeline backed by SQLite."""

import gc
import logging
import time
from pathlib import Path
from typing import List, Tuple

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
        threshold_mult: float = 1.0,
        jpeg_quality: int = 80,
        frame_max_width: int = 640,
    ):
        self.config = config
        self.camera_mode = camera_mode
        self.yolo_fps = yolo_fps
        self.vlm_fps = vlm_fps
        self.chunk_size = chunk_size
        self.depth_enabled = depth_enabled
        self.use_segmentation = use_segmentation
        self.threshold_mult = threshold_mult
        self.jpeg_quality = jpeg_quality
        self.frame_max_width = frame_max_width
        self.logger = logging.getLogger(__name__)

    def run(self) -> PipelineResult:
        start = time.time()
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        db = TempoGraphDB(out_dir / "tempograph.db")
        try:
            # Stage 1: frame selection + JPEG export
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
            for idx, path in zip(selection.frame_indices, frame_paths):
                ts_ms = int(idx * 1000.0 / max(video_fps, 1.0))
                db.insert_frame(
                    frame_idx=idx,
                    timestamp_ms=ts_ms,
                    image_path=str(path),
                    is_keyframe=(idx in set(selection.keyframe_indices)),
                    delta_score=self._delta_for_index(selection, idx),
                )

            # Stage 2: YOLO sweep
            model_path = "yolo11n-seg.pt" if self.use_segmentation else "yolo11n.pt"
            detector = ObjectDetector(
                model_path=model_path,
                confidence=self.config.confidence,
                device="cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu",
            )
            detector.detect_to_db(
                frame_indices=selection.frame_indices,
                frame_paths=frame_paths,
                db=db,
                frame_width=w,
                frame_height=h,
            )
            detector.cleanup()
            gc.collect()
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: depth (optional)
            if self.depth_enabled:
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

            # Stage 4: VLM frame subset
            video_duration = max(1.0, self._video_duration(self.config.video_path))
            k = max(1, int(round(video_duration * self.vlm_fps)))
            scorer = FrameScorer(db)
            vlm_frames = scorer.score_and_select(
                candidate_frame_indices=selection.frame_indices,
                keyframe_indices=set(selection.keyframe_indices),
                k=k,
            )

            # Stage 5: chunked VLM
            chunks: List[Tuple[int, List[int]]] = []
            for i in range(0, len(vlm_frames), self.chunk_size):
                chunks.append((len(chunks), vlm_frames[i : i + self.chunk_size]))
            backend = LlamaServerBackend()
            chunk_caps = backend.caption_chunks(chunks=chunks, db=db)

            # Stage 6: aggregate
            analysis = CaptionAggregator().aggregate(chunk_caps)

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
                        help="Use yolo11n-seg.pt instead of yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--threshold-mult", type=float, default=1.0)
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
        threshold_mult=args.threshold_mult,
    )
    result = pipe.run()
    print(f"Done in {result.processing_time:.1f}s -> {args.output}")
