import time
import logging
import yaml
import torch
import gc
from pathlib import Path
from typing import Optional
from src.models import (
    PipelineConfig,
    PipelineResult,
    AnalysisResult,
    DetectionResult,
    DepthResult,
)
from src.modules.frame_extractor import FrameExtractor, ExtractionResult
from src.json_parser import JSONParser
from src.graph_builder import GraphBuilder
from src.video_annotator import VideoAnnotator
from src.backends.gemini_backend import GeminiBackend
from src.backends.qwen_backend import QwenBackend
from src.modules.detector import ObjectDetector
from src.modules.depth import DepthEstimator
from src.modules.audio import AudioAnalyzer


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.frame_extractor = FrameExtractor()
        self.json_parser = JSONParser(min_confidence=config.confidence)
        self.graph_builder = GraphBuilder()
        self.video_annotator = VideoAnnotator()
        self._vlm_backend = None
        self._detector = None
        self._depth_estimator = None
        self._audio_analyzer = None

    def run(self) -> PipelineResult:
        """Run full pipeline."""
        start_time = time.time()
        self.logger.info(
            f"Starting pipeline: backend={self.config.backend}, "
            f"modules={self.config.modules}"
        )

        # Step 1: Extract frames and audio
        extraction = self.frame_extractor.extract(
            video_path=self.config.video_path,
            fps=self.config.fps,
            max_frames=self.config.max_frames,
            extract_audio=self.config.modules.get("audio", False),
        )
        self.logger.info(
            f"Extracted {len(extraction.frame_paths)} frames, "
            f"audio={'yes' if extraction.audio_path else 'no'}"
        )

        # Step 2: Run object detection if enabled
        detection = None
        if self.config.modules.get("detection", False):
            detection = self._run_detection(extraction)

        # Step 3: Run depth estimation if enabled
        depth = None
        if self.config.modules.get("depth", False):
            depth = self._run_depth(extraction)

        # Step 4: Unload detection + depth models
        self._unload_vision_models()

        # Step 5: Run VLM behavior analysis (always on)
        analysis = self._run_vlm_analysis(extraction)

        # Step 6: Run audio analysis if enabled (local mode only)
        if (
            self.config.modules.get("audio", False)
            and self.config.backend == "qwen"
            and extraction.audio_path
        ):
            self._merge_audio_results(analysis, extraction.audio_path)

        # Step 7: Build graph
        graph = self.graph_builder.build(analysis)

        # Step 7b: Save graph visualization
        self._save_graph(graph, analysis)

        # Step 8: Create annotated video
        annotated_path = None
        if detection or depth:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = str(output_dir / "annotated.mp4")
            self.video_annotator.annotate(
                video_path=self.config.video_path,
                output_path=annotated_path,
                detection=detection,
                depth=depth,
                analysis=analysis,
                fps=extraction.extraction_fps,
            )

        # Step 9: Save results
        self._save_results(analysis, detection, depth)

        # Cleanup
        self.frame_extractor.cleanup()
        self._cleanup_backends()

        elapsed = time.time() - start_time
        self.logger.info(f"Pipeline complete in {elapsed:.1f}s")

        return PipelineResult(
            analysis=analysis,
            detection=detection,
            depth=depth,
            config=self.config,
            annotated_video_path=annotated_path,
            processing_time=elapsed,
        )

    def _run_detection(self, extraction: ExtractionResult) -> DetectionResult:
        """Run YOLOv8-nano on extracted frames."""
        self.logger.info("Running object detection...")

        # Log VRAM before
        self._log_vram("before detection")

        # Create detector
        detector = ObjectDetector(
            confidence=self.config.confidence,
            imgsz=640,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Run detection
        detection = detector.detect_frames(extraction.frame_paths)

        # Log VRAM after
        self._log_vram("after detection")

        return detection

    def _run_depth(self, extraction: ExtractionResult) -> DepthResult:
        """Run Depth Anything V2 Small on extracted frames."""
        self.logger.info("Running depth estimation...")

        # Log VRAM before
        self._log_vram("before depth")

        # Create depth estimator
        depth_estimator = DepthEstimator(
            model_variant="vits", device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Run depth estimation
        depth = depth_estimator.estimate_frames(
            extraction.frame_paths,
            output_dir=str(Path(self.config.output_dir) / "depth"),
        )

        # Log VRAM after
        self._log_vram("after depth")

        return depth

    def _run_vlm_analysis(self, extraction: ExtractionResult) -> AnalysisResult:
        """Run VLM backend for behavior analysis."""
        self.logger.info(f"Running VLM analysis with backend: {self.config.backend}")

        # Log VRAM before
        self._log_vram("before VLM")

        # Create backend
        if self.config.backend == "gemini":
            self._vlm_backend = GeminiBackend()
        elif self.config.backend == "qwen":
            self._vlm_backend = QwenBackend(max_frames=self.config.max_frames)
        elif self.config.backend == "llama-server":
            from src.backends.llama_server_backend import LlamaServerBackend

            self._vlm_backend = LlamaServerBackend(max_frames=self.config.max_frames)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

        # Run analysis
        audio_path_str = str(extraction.audio_path) if extraction.audio_path else None
        analysis = self._vlm_backend.analyze_video(
            video_path=self.config.video_path,
            frames=extraction.frame_paths,
            audio_path=audio_path_str,
        )

        # Log VRAM after
        self._log_vram("after VLM")

        return analysis

    def _merge_audio_results(self, analysis: AnalysisResult, audio_path):
        """Merge Whisper results into analysis."""
        self.logger.info("Running audio analysis...")

        # Create audio analyzer (auto-detect GPU)
        self._audio_analyzer = AudioAnalyzer(model_size="small", device="auto")

        # Run audio analysis
        audio_events = self._audio_analyzer.analyze(audio_path)

        # Merge into analysis
        analysis.audio_events.extend(audio_events)

        self.logger.info(f"Merged {len(audio_events)} audio events into analysis")

    def _unload_vision_models(self):
        """Free GPU VRAM from detection + depth models."""
        self.logger.info("Unloading vision models...")

        # Cleanup detector
        if self._detector:
            self._detector.cleanup()

        # Cleanup depth estimator
        if self._depth_estimator:
            self._depth_estimator.cleanup()

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log VRAM after unload
        self._log_vram("after unload")

        self.logger.info("Vision models unloaded")

    def _cleanup_backends(self):
        """Cleanup all backends."""
        if self._vlm_backend:
            self._vlm_backend.cleanup()
        if self._audio_analyzer:
            self._audio_analyzer.cleanup()

    def _log_vram(self, label: str):
        """Log VRAM usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.logger.info(
                f"VRAM [{label}]: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved"
            )

    def _save_results(self, analysis, detection, depth):
        """Save results to output directory."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save analysis JSON
        with open(output_dir / "analysis.json", "w") as f:
            f.write(analysis.model_dump_json(indent=2))

        if detection:
            with open(output_dir / "detection.json", "w") as f:
                f.write(detection.model_dump_json(indent=2))

        self.logger.info(f"Results saved to {output_dir}")

    def _save_graph(self, graph, analysis):
        """Save graph as JSON and HTML visualization."""
        output_dir = Path(self.config.output_dir)

        # Save graph JSON for API/frontend
        graph_json = self.graph_builder.to_json()
        with open(output_dir / "graph.json", "w") as f:
            import json
            json.dump(graph_json, f, indent=2)

        # Save timeline JSON for quick skimming
        timeline = self._get_timeline_from_analysis(analysis)
        with open(output_dir / "timeline.json", "w") as f:
            import json
            json.dump(timeline, f, indent=2)

        # Save interactive HTML graph
        try:
            self.graph_builder.to_pyvis_html(str(output_dir / "graph.html"))
            self.logger.info(f"Graph visualization saved to {output_dir}")
        except Exception as e:
            self.logger.warning(f"Could not generate HTML graph: {e}")

    def _get_timeline_from_analysis(self, analysis) -> list:
        """Extract timeline of events sorted by time."""
        def time_to_seconds(ts: str) -> float:
            if ":" in ts:
                parts = ts.split(":")
                return int(parts[0]) * 60 + float(parts[1])
            return float(ts) if ts else 0.0

        timeline = []

        # Add visual events directly from analysis
        for event in analysis.visual_events:
            timeline.append({
                "type": "visual",
                "event_type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                "entities": event.entities,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "description": event.description,
                "confidence": event.confidence,
            })

        # Add audio events
        for audio in analysis.audio_events:
            timeline.append({
                "type": "audio",
                "event_type": audio.type.value if hasattr(audio.type, 'value') else str(audio.type),
                "start_time": audio.start_time,
                "end_time": audio.end_time,
                "description": audio.text or audio.label,
                "speaker": audio.speaker,
            })

        # Sort by start time
        timeline.sort(key=lambda x: time_to_seconds(x.get("start_time", "00:00")))

        return timeline


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TempoGraph Pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--backend", default="llama-server", choices=["gemini", "qwen", "llama-server"])
    parser.add_argument("--modules", default="behavior,detection,audio")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--output", default="results")

    args = parser.parse_args()

    # Load config
    import yaml

    with open("configs/default.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    config = PipelineConfig(
        backend=args.backend,
        modules={m: True for m in args.modules.split(",")},
        fps=args.fps,
        max_frames=args.max_frames,
        video_path=args.video,
        output_dir=args.output,
    )

    pipeline = Pipeline(config)
    result = pipeline.run()
    print(f"Done in {result.processing_time:.1f}s")
