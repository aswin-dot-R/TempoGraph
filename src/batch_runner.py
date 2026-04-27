"""Batch video processing for TempoGraph dataset generation.

Processes every video in a directory through PipelineV2, skipping
already-completed runs, and optionally exports all dataset formats
after each video.

Usage (CLI):
    python -m src.batch_runner --video-dir /path/to/videos --output-dir results/ \\
        --export --yolo-size n --audio --whisper-model base.en

Usage (Python):
    from src.batch_runner import BatchRunner
    runner = BatchRunner(video_dir="videos/", output_dir="results/")
    runner.run_all()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.models import CameraMode, PipelineConfig
from src.pipeline_v2 import PipelineV2

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class BatchRunner:
    """Process a directory of videos through PipelineV2.

    Args:
        video_dir: Directory containing video files to process.
        output_dir: Root output directory (each video gets a subdirectory).
        export_datasets: If True, run dataset export after each video.
        **pipeline_kwargs: Forwarded to PipelineV2 constructor.
    """

    def __init__(
        self,
        video_dir: str,
        output_dir: str = "results",
        export_datasets: bool = False,
        camera_mode: str = "static",
        yolo_fps: float = 1.0,
        vlm_fps: float = 0.5,
        chunk_size: int = 10,
        depth_enabled: bool = False,
        use_segmentation: bool = True,
        yolo_size: str = "n",
        threshold_mult: float = 1.0,
        vlm_dedup_threshold: float = 0.92,
        vlm_url: str = "http://127.0.0.1:8082",
        vlm_model: str = "Qwen3.5-9B-Q8_0.gguf",
        vlm_frame_mode: str = "keyframes",
        vlm_autostart_service: Optional[str] = "qwen35-turboquant.service",
        vlm_autostop: bool = False,
        audio_enabled: bool = False,
        whisper_model: str = "base.en",
        whisper_gpu_device: Optional[int] = 1,
        whisper_language: Optional[str] = None,
    ):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.export_datasets = export_datasets

        # Pipeline configuration
        self.camera_mode = camera_mode
        self.yolo_fps = yolo_fps
        self.vlm_fps = vlm_fps
        self.chunk_size = chunk_size
        self.depth_enabled = depth_enabled
        self.use_segmentation = use_segmentation
        self.yolo_size = yolo_size
        self.threshold_mult = threshold_mult
        self.vlm_dedup_threshold = vlm_dedup_threshold
        self.vlm_url = vlm_url
        self.vlm_model = vlm_model
        self.vlm_frame_mode = vlm_frame_mode
        self.vlm_autostart_service = vlm_autostart_service
        self.vlm_autostop = vlm_autostop
        self.audio_enabled = audio_enabled
        self.whisper_model = whisper_model
        self.whisper_gpu_device = whisper_gpu_device
        self.whisper_language = whisper_language

    def discover_videos(self) -> List[Path]:
        """Find all video files in video_dir (non-recursive)."""
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")
        videos = sorted(
            p for p in self.video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        logger.info(f"Discovered {len(videos)} videos in {self.video_dir}")
        return videos

    def is_completed(self, video_path: Path) -> bool:
        """Check if a video has already been fully processed."""
        run_dir = self.output_dir / video_path.name
        analysis = run_dir / "analysis.json"
        return analysis.exists()

    def run_all(self, skip_completed: bool = True) -> Dict[str, dict]:
        """Process all videos in the directory.

        Args:
            skip_completed: If True, skip videos that already have
                analysis.json in their output dir.

        Returns:
            Dict mapping video filename to result info:
            {"status": "done"|"skipped"|"error", "time_s": float, "error": str|None}
        """
        videos = self.discover_videos()
        results: Dict[str, dict] = {}
        total_start = time.time()

        for i, video_path in enumerate(videos, 1):
            name = video_path.name
            logger.info(f"[{i}/{len(videos)}] Processing: {name}")

            if skip_completed and self.is_completed(video_path):
                logger.info(f"  Skipping (already completed): {name}")
                results[name] = {"status": "skipped", "time_s": 0, "error": None}
                continue

            t0 = time.time()
            try:
                self._process_one(video_path)
                elapsed = time.time() - t0
                results[name] = {"status": "done", "time_s": round(elapsed, 1), "error": None}
                logger.info(f"  Done in {elapsed:.1f}s: {name}")
            except Exception as e:
                elapsed = time.time() - t0
                results[name] = {"status": "error", "time_s": round(elapsed, 1), "error": str(e)}
                logger.error(f"  Failed ({elapsed:.1f}s): {name} — {e}")

        total_elapsed = time.time() - total_start
        done = sum(1 for r in results.values() if r["status"] == "done")
        skipped = sum(1 for r in results.values() if r["status"] == "skipped")
        errors = sum(1 for r in results.values() if r["status"] == "error")

        logger.info(
            f"Batch complete: {done} done, {skipped} skipped, {errors} errors "
            f"in {total_elapsed:.1f}s"
        )

        # Write batch summary
        import json
        summary_path = self.output_dir / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "total_videos": len(videos),
                "done": done,
                "skipped": skipped,
                "errors": errors,
                "total_time_s": round(total_elapsed, 1),
                "per_video": results,
            }, f, indent=2)
        logger.info(f"Batch summary → {summary_path}")

        return results

    def _process_one(self, video_path: Path) -> None:
        """Run the full pipeline on a single video."""
        run_dir = self.output_dir / video_path.name

        config = PipelineConfig(
            backend="llama-server",
            modules={
                "behavior": True,
                "detection": True,
                "depth": self.depth_enabled,
                "audio": False,
            },
            fps=self.yolo_fps,
            max_frames=999,
            confidence=0.5,
            video_path=str(video_path),
            output_dir=str(run_dir),
        )

        pipe = PipelineV2(
            config,
            camera_mode=CameraMode(self.camera_mode),
            yolo_fps=self.yolo_fps,
            vlm_fps=self.vlm_fps,
            chunk_size=self.chunk_size,
            depth_enabled=self.depth_enabled,
            use_segmentation=self.use_segmentation,
            yolo_size=self.yolo_size,
            threshold_mult=self.threshold_mult,
            vlm_dedup_threshold=self.vlm_dedup_threshold,
            vlm_url=self.vlm_url,
            vlm_model=self.vlm_model,
            vlm_frame_mode=self.vlm_frame_mode,
            vlm_autostart_service=self.vlm_autostart_service,
            vlm_autostop=False,  # Keep VLM running between videos in a batch
            audio_enabled=self.audio_enabled,
            whisper_model=self.whisper_model,
            whisper_gpu_device=self.whisper_gpu_device,
            whisper_language=self.whisper_language,
        )
        pipe.run()

        # Export datasets if requested
        if self.export_datasets:
            try:
                from src.dataset_exporter import export_all
                export_all(run_dir, video_name=video_path.stem)
            except Exception as e:
                logger.warning(f"Dataset export failed for {video_path.name}: {e}")


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Batch process videos through TempoGraph pipeline"
    )
    parser.add_argument("--video-dir", required=True,
                        help="Directory containing video files")
    parser.add_argument("--output-dir", default="results",
                        help="Root output directory (default: results/)")
    parser.add_argument("--export", action="store_true",
                        help="Export dataset formats (COCO, JSONL) after each video")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-process already completed videos")

    # Pipeline options
    parser.add_argument("--camera", default="static",
                        choices=["static", "moving", "auto"])
    parser.add_argument("--yolo-fps", type=float, default=1.0)
    parser.add_argument("--vlm-fps", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--seg", action="store_true", default=True)
    parser.add_argument("--yolo-size", default="n",
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--threshold-mult", type=float, default=1.0)
    parser.add_argument("--vlm-dedup-threshold", type=float, default=0.92)
    parser.add_argument("--vlm-frame-mode", default="keyframes",
                        choices=["scored", "keyframes"])
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8082")
    parser.add_argument("--vlm-model", default="Qwen3.5-9B-Q8_0.gguf")
    parser.add_argument("--vlm-autostart-service", default="qwen35-turboquant.service")
    parser.add_argument("--vlm-autostop", action="store_true",
                        help="Stop VLM service after all videos are processed")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument("--whisper-gpu-device", type=int, default=1)
    parser.add_argument("--whisper-language", default=None)

    args = parser.parse_args()

    runner = BatchRunner(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        export_datasets=args.export,
        camera_mode=args.camera,
        yolo_fps=args.yolo_fps,
        vlm_fps=args.vlm_fps,
        chunk_size=args.chunk_size,
        depth_enabled=args.depth,
        use_segmentation=args.seg,
        yolo_size=args.yolo_size,
        threshold_mult=args.threshold_mult,
        vlm_dedup_threshold=args.vlm_dedup_threshold,
        vlm_frame_mode=args.vlm_frame_mode,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        vlm_autostart_service=args.vlm_autostart_service,
        vlm_autostop=args.vlm_autostop,
        audio_enabled=args.audio,
        whisper_model=args.whisper_model,
        whisper_gpu_device=args.whisper_gpu_device,
        whisper_language=args.whisper_language,
    )

    results = runner.run_all(skip_completed=not args.no_skip)

    # Optionally stop VLM after the entire batch
    if args.vlm_autostop and args.vlm_autostart_service:
        import subprocess
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", args.vlm_autostart_service],
                check=True, timeout=30,
            )
            logger.info(f"Stopped {args.vlm_autostart_service}")
        except Exception as e:
            logger.warning(f"Could not stop VLM service: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Batch processing complete")
    print(f"{'='*60}")
    for name, info in results.items():
        icon = {"done": "✓", "skipped": "·", "error": "✗"}[info["status"]]
        line = f"  {icon} {name} — {info['status']}"
        if info["time_s"]:
            line += f" ({info['time_s']:.1f}s)"
        if info["error"]:
            line += f" — {info['error'][:80]}"
        print(line)


if __name__ == "__main__":
    main()
