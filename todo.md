# TempoGraph — Nemotron Build Tasks
## Local Model Workstream (Run First)

> **For:** Nemotron 30B via Claude Code
> **Scope:** Project scaffolding, data models, graph logic, UI, Docker, configs, utils
> **Why these tasks:** Standard Python, well-documented libraries, no tricky GPU/VLM wiring
> **After this completes:** Hand off to Opus for VLM backends + VRAM management

---

## Task Order (Feed these sequentially)

### TASK 1: Project Structure + Configs + Data Models

```
Create the following project structure and files for "TempoGraph" — 
a video intelligence pipeline.

Project structure:
tempograph/
├── README.md (placeholder)
├── LICENSE (MIT)
├── requirements.txt
├── setup.py
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── models.py          ← data models (this task)
│   ├── backends/
│   │   ├── __init__.py
│   │   └── base.py        ← abstract backend (this task)
│   ├── modules/
│   │   └── __init__.py
│   ├── json_parser.py
│   ├── graph_builder.py
│   ├── video_annotator.py
│   ├── pipeline.py
│   └── api.py
├── ui/
│   └── app.py
├── data/
│   └── samples/
├── results/
└── tests/
    └── test_parser.py

FILES TO CREATE NOW:

1. requirements.txt:
ultralytics
depth-anything-v2
openai-whisper
google-genai
transformers
bitsandbytes
accelerate
torch
networkx
fastapi
uvicorn
streamlit
opencv-python
pillow
pyyaml
ffmpeg-python
plotly
pyvis
streamlit-agraph
pydantic

2. configs/default.yaml:
pipeline:
  backend: "gemini"          # "gemini" or "qwen"
  modules:
    behavior: true            # always on
    detection: true           # YOLOv8-nano
    depth: false              # Depth Anything V2 Small
    audio: true               # Whisper / Gemini audio

frame_extraction:
  fps: 1.0
  max_frames: 60
  resize_width: 640

detection:
  model: "yolov8n.pt"
  confidence: 0.5
  imgsz: 640

depth:
  model: "vits"              # vits = Small variant
  device: "cuda"

audio:
  model: "small"             # whisper-small
  device: "cpu"              # always CPU to save VRAM
  language: null             # auto-detect

vlm:
  gemini:
    model: "gemini-2.0-flash"
    temperature: 0.1
    max_output_tokens: 8192
  qwen:
    model: "Qwen/Qwen2.5-VL-3B-Instruct"
    quantize: "4bit"
    device: "cuda"
    max_new_tokens: 4096

graph:
  min_confidence: 0.3

output:
  save_annotated_video: true
  save_depth_maps: false
  save_json: true
  results_dir: "results"

3. src/models.py — Pydantic data models:

from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class BehaviorType(str, Enum):
    APPROACH = "approach"
    DEPART = "depart"
    INTERACT = "interact"
    FOLLOW = "follow"
    IDLE = "idle"
    GROUP = "group"
    AVOID = "avoid"
    CHASE = "chase"
    OBSERVE = "observe"

class SoundType(str, Enum):
    SPEECH = "speech"
    MUSIC = "music"
    ANIMAL_SOUND = "animal_sound"
    VEHICLE = "vehicle"
    IMPACT = "impact"
    ENVIRONMENTAL = "environmental"
    SILENCE = "silence"

class Entity(BaseModel):
    id: str
    type: str
    description: str
    first_seen: str          # MM:SS format
    last_seen: str

class VisualEvent(BaseModel):
    type: BehaviorType
    entities: List[str]      # entity IDs
    start_time: str
    end_time: str
    description: str
    confidence: float = 0.5

class AudioEvent(BaseModel):
    type: SoundType
    start_time: str
    end_time: str
    speaker: Optional[str] = None
    text: Optional[str] = None
    label: Optional[str] = None
    emotion: Optional[str] = None
    confidence: float = 0.5

class MultimodalCorrelation(BaseModel):
    visual_event: int        # index into visual_events
    audio_event: int         # index into audio_events
    description: str

class AnalysisResult(BaseModel):
    entities: List[Entity] = []
    visual_events: List[VisualEvent] = []
    audio_events: List[AudioEvent] = []
    multimodal_correlations: List[MultimodalCorrelation] = []
    summary: str = ""

class DetectionBox(BaseModel):
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    track_id: Optional[int] = None

class DetectionResult(BaseModel):
    boxes: List[DetectionBox] = []
    fps: float = 1.0
    frame_count: int = 0

class DepthFrame(BaseModel):
    frame_idx: int
    depth_map_path: str      # path to saved .npy

class DepthResult(BaseModel):
    frames: List[DepthFrame] = []

class PipelineConfig(BaseModel):
    backend: str = "gemini"
    modules: dict = {"behavior": True, "detection": True, 
                     "depth": False, "audio": True}
    fps: float = 1.0
    max_frames: int = 60
    confidence: float = 0.5
    video_path: str = ""
    output_dir: str = "results"

class PipelineResult(BaseModel):
    analysis: Optional[AnalysisResult] = None
    detection: Optional[DetectionResult] = None
    depth: Optional[DepthResult] = None
    config: Optional[PipelineConfig] = None
    annotated_video_path: Optional[str] = None
    processing_time: float = 0.0

4. src/backends/base.py — abstract backend:

from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from src.models import AnalysisResult

class BaseVLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def analyze_video(
        self, 
        video_path: str,
        frames: List[Path] = None,
        audio_path: str = None,
        prompt: str = None
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

Make sure all __init__.py files have appropriate imports.
Use Python 3.10+ type hints throughout.
```

---

### TASK 2: Frame Extractor + Audio Extractor

```
Create src/modules/frame_extractor.py for TempoGraph.

This module extracts frames from video at a configurable FPS and 
optionally extracts the audio track.

Requirements:
- Use OpenCV (cv2) for frame extraction
- Use subprocess + ffmpeg for audio extraction
- Return list of frame paths + metadata

Implementation:

import cv2
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExtractionResult:
    frame_paths: List[Path]
    audio_path: Optional[Path]
    total_frames: int
    video_fps: float
    video_duration: float       # seconds
    extraction_fps: float       # what we sampled at
    width: int
    height: int

class FrameExtractor:
    def __init__(self, output_dir: str = "/tmp/tempograph_frames"):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
    
    def extract(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 60,
        resize_width: Optional[int] = 640,
        extract_audio: bool = True
    ) -> ExtractionResult:
        """
        Extract frames at given FPS and optionally audio.
        
        Steps:
        1. Open video with cv2.VideoCapture
        2. Get video metadata (fps, duration, dimensions)
        3. Calculate frame interval = video_fps / extraction_fps
        4. Extract frames at interval, resize if needed
        5. Save as JPEG to output_dir/frame_XXXXX.jpg
        6. If extract_audio: use ffmpeg to extract audio as 16kHz mono WAV
        7. Return ExtractionResult
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Clear any previous frames
        for f in self.output_dir.glob("frame_*.jpg"):
            f.unlink()
        
        # Implementation here...
        # Key details:
        # - frame_interval = int(video_fps / fps)
        # - For each frame, if frame_count % frame_interval == 0: save
        # - Stop at max_frames
        # - Resize: cv2.resize(frame, (resize_width, int(h * resize_width / w)))
        # - Audio: subprocess.run([
        #     "ffmpeg", "-i", video_path, "-vn", 
        #     "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        #     str(audio_path), "-y"
        #   ], capture_output=True)
        
    def cleanup(self):
        """Remove extracted frames and audio."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

Implement fully with proper error handling:
- Video file not found
- Video can't be opened
- No frames extracted  
- ffmpeg not installed (log warning, skip audio)
- Video has no audio track (log warning, return audio_path=None)

Add logging throughout with self.logger.
```

---

### TASK 3: JSON Parser with Validation & Retry Logic

```
Create src/json_parser.py for TempoGraph.

This module parses raw JSON strings from VLM output (either Gemini or Qwen)
into validated AnalysisResult objects. VLMs often return malformed JSON,
so this needs to be robust.

Requirements:
- Parse raw text that may contain JSON embedded in markdown code blocks
- Handle common VLM JSON failures:
  - Trailing commas
  - Single quotes instead of double quotes
  - Missing closing brackets
  - Extra text before/after JSON
  - Truncated output
- Validate against AnalysisResult schema
- Normalize timestamps to MM:SS format
- Filter by confidence threshold
- Return partial results rather than failing completely

Implementation plan:

import json
import re
import logging
from typing import Optional
from src.models import (AnalysisResult, Entity, VisualEvent, 
                         AudioEvent, MultimodalCorrelation)

class JSONParser:
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
    
    def parse(self, raw_text: str) -> AnalysisResult:
        """
        Parse VLM output text into AnalysisResult.
        Tries multiple strategies in order.
        """
        # Strategy 1: Extract JSON from markdown code blocks
        # Strategy 2: Find JSON object boundaries { ... }
        # Strategy 3: Try to fix common issues and re-parse
        # Strategy 4: Return empty AnalysisResult with error logged
    
    def _extract_json_string(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling markdown blocks."""
        # Try ```json ... ``` first
        # Then try finding outermost { ... }
        # Strip any text before first { and after last }
    
    def _fix_common_issues(self, json_str: str) -> str:
        """Fix common VLM JSON mistakes."""
        # Remove trailing commas before } or ]
        # Replace single quotes with double quotes (careful with apostrophes)
        # Try to close unclosed brackets
        # Remove control characters
    
    def _normalize_timestamp(self, ts: str) -> str:
        """Normalize timestamp to MM:SS format."""
        # Handle: "0:03", "00:03", "3", "0:3", "00:03.5", "3.5s"
        # Return: "00:03"
    
    def _validate_and_build(self, data: dict) -> AnalysisResult:
        """Build AnalysisResult from parsed dict with validation."""
        # Parse entities (lenient - skip invalid ones)
        # Parse visual_events (filter by confidence)
        # Parse audio_events (filter by confidence)
        # Parse correlations (validate indices exist)
        # Get summary

Implement fully with comprehensive error handling.
Include unit-testable methods.
Add a tests/test_parser.py with at least 5 test cases:
1. Clean valid JSON
2. JSON in markdown code blocks
3. JSON with trailing commas
4. Truncated JSON (missing closing brackets)
5. Completely invalid text (should return empty AnalysisResult)
```

---

### TASK 4: Graph Builder (NetworkX)

```
Create src/graph_builder.py for TempoGraph.

This module builds a temporal interaction graph from AnalysisResult.
Entities become nodes, events become edges with temporal attributes.

Requirements:
- Use NetworkX for graph construction
- Nodes = entities with attributes (type, description, first_seen, last_seen)
- Edges = visual_events connecting entity pairs, with attributes 
  (type, start_time, end_time, confidence, description)
- Audio events stored as graph-level attributes
- Support querying: "what happened between E1 and E2?", 
  "what happened at timestamp 00:30?"
- Export to JSON for frontend visualization
- Generate pyvis HTML for interactive graph display

Implementation:

import networkx as nx
import json
import logging
from typing import List, Dict, Optional, Any
from src.models import AnalysisResult, PipelineResult

class GraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # directed, allows multiple edges
        self.logger = logging.getLogger(__name__)
    
    def build(self, result: AnalysisResult) -> nx.MultiDiGraph:
        """Build graph from analysis result."""
        self.graph.clear()
        
        # Add entity nodes
        for entity in result.entities:
            self.graph.add_node(
                entity.id,
                type=entity.type,
                description=entity.description,
                first_seen=entity.first_seen,
                last_seen=entity.last_seen,
                # visual attributes for rendering
                color=self._get_color_for_type(entity.type),
                size=30
            )
        
        # Add visual event edges
        for i, event in enumerate(result.visual_events):
            if len(event.entities) >= 2:
                # Add edge between first two entities
                self.graph.add_edge(
                    event.entities[0],
                    event.entities[1],
                    key=f"ve_{i}",
                    type=event.type.value,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    description=event.description,
                    confidence=event.confidence
                )
            elif len(event.entities) == 1:
                # Self-referencing event (idle, observe)
                # Store as node attribute
                pass
        
        # Store audio events as graph attribute
        self.graph.graph['audio_events'] = [
            e.model_dump() for e in result.audio_events
        ]
        self.graph.graph['summary'] = result.summary
        self.graph.graph['correlations'] = [
            c.model_dump() for c in result.multimodal_correlations
        ]
        
        return self.graph
    
    def to_json(self) -> Dict[str, Any]:
        """Export graph to JSON for frontend."""
        # Return: {nodes: [...], edges: [...], 
        #          audio_events: [...], summary: "..."}
    
    def to_pyvis_html(self, output_path: str = "graph.html") -> str:
        """Generate interactive pyvis HTML visualization."""
        # Use pyvis or streamlit-agraph format
        # Color nodes by entity type
        # Label edges with behavior type
        # Size nodes by number of connections
    
    def query_by_entities(self, e1: str, e2: str) -> List[Dict]:
        """Get all events between two entities."""
    
    def query_by_time(self, timestamp: str) -> Dict:
        """Get all events active at a given timestamp."""
        # Need to parse MM:SS and check if timestamp falls 
        # within start_time..end_time of each event
    
    def get_timeline(self) -> List[Dict]:
        """Get all events sorted by start_time for timeline view."""
        # Merge visual_events and audio_events
        # Sort by start_time
        # Return unified timeline
    
    def get_stats(self) -> Dict:
        """Get summary statistics."""
        # Total entities, events, interactions per entity pair,
        # most active entity, dominant behavior type
    
    def _get_color_for_type(self, entity_type: str) -> str:
        """Return hex color for entity type."""
        colors = {
            "person": "#4A90D9",
            "dog": "#E8943A",
            "cat": "#9B59B6",
            "vehicle": "#2ECC71",
            "object": "#95A5A6",
        }
        return colors.get(entity_type.lower(), "#BDC3C7")
    
    def _timestamp_to_seconds(self, ts: str) -> float:
        """Convert MM:SS to seconds."""
        parts = ts.split(":")
        return int(parts[0]) * 60 + float(parts[1])

Implement fully. Include proper error handling for:
- Empty analysis results
- Events referencing non-existent entities
- Malformed timestamps
```

---

### TASK 5: Video Annotator

```
Create src/video_annotator.py for TempoGraph.

This module takes original video + detection results + depth maps + 
audio transcription and creates an annotated output video.

Requirements:
- Overlay YOLO bounding boxes with class labels and track IDs
- Overlay depth map as semi-transparent colormap (jet or inferno)
- Burn in audio transcription as subtitles at bottom
- Show current timestamp and active events as text overlay
- Output as MP4 using OpenCV VideoWriter
- Handle cases where some modules are disabled (no depth, no audio, etc.)

Implementation:

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Dict
from src.models import (DetectionResult, DetectionBox, DepthResult,
                         AnalysisResult, PipelineConfig)

class VideoAnnotator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Color palette for different classes/tracks
        self.colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 100), (150, 200, 100), (100, 150, 200),
        ]
    
    def annotate(
        self,
        video_path: str,
        output_path: str,
        detection: Optional[DetectionResult] = None,
        depth: Optional[DepthResult] = None,
        analysis: Optional[AnalysisResult] = None,
        fps: float = 1.0,
        depth_alpha: float = 0.3,
    ) -> str:
        """
        Create annotated video.
        
        Steps:
        1. Open original video
        2. For each frame:
           a. Draw YOLO bboxes if detection is provided
           b. Overlay depth colormap if depth is provided
           c. Draw subtitle text if analysis has audio_events
           d. Draw timestamp + active event labels at top
        3. Write to output MP4
        4. Return output path
        """
    
    def _draw_boxes(
        self, frame: np.ndarray, 
        boxes: List[DetectionBox]
    ) -> np.ndarray:
        """Draw bounding boxes with labels on frame."""
        # For each box:
        # - cv2.rectangle for bbox
        # - cv2.putText for "class_name track_id (conf)"
        # - Use self.colors[track_id % len(self.colors)]
    
    def _overlay_depth(
        self, frame: np.ndarray,
        depth_map: np.ndarray,
        alpha: float = 0.3
    ) -> np.ndarray:
        """Overlay depth map as semi-transparent colormap."""
        # Normalize depth_map to 0-255
        # Apply cv2.applyColorMap with COLORMAP_INFERNO
        # cv2.addWeighted(frame, 1-alpha, colored_depth, alpha, 0)
    
    def _draw_subtitles(
        self, frame: np.ndarray,
        text: str,
        position: str = "bottom"
    ) -> np.ndarray:
        """Draw subtitle text with background."""
        # Black semi-transparent background strip
        # White text centered
        # Use cv2.FONT_HERSHEY_SIMPLEX
    
    def _draw_event_labels(
        self, frame: np.ndarray,
        events: List[str],
        timestamp: str
    ) -> np.ndarray:
        """Draw timestamp and active events at top of frame."""
        # Top-left: timestamp in MM:SS
        # Top-right: active event labels (e.g. "approach", "speech")
    
    def _get_active_subtitle(
        self, 
        analysis: AnalysisResult,
        current_time_sec: float
    ) -> Optional[str]:
        """Get subtitle text active at current timestamp."""
        # Check audio_events for speech events that span current time
    
    def _get_active_events(
        self,
        analysis: AnalysisResult, 
        current_time_sec: float
    ) -> List[str]:
        """Get list of active event descriptions at current time."""

Implement fully using only OpenCV (cv2) and numpy.
Handle edge cases: video can't be opened, no detections, 
depth map shape mismatch, output dir doesn't exist.
```

---

### TASK 6: FastAPI Server

```
Create src/api.py for TempoGraph.

REST API that wraps the pipeline for programmatic access.

Endpoints:
- POST /analyze — upload video + config, return results
- GET /status/{job_id} — check processing status
- GET /results/{job_id} — get results JSON
- GET /health — health check

Implementation:

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional
from src.models import PipelineConfig, PipelineResult

app = FastAPI(
    title="TempoGraph API",
    description="Video Intelligence Pipeline",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (for hackathon scope)
jobs: dict = {}  # job_id -> {"status": "...", "result": ..., "error": ...}

@app.get("/health")
async def health():
    return {"status": "ok", "gpu_available": torch.cuda.is_available()}

@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    backend: str = Form("gemini"),
    modules: str = Form("behavior,detection,audio"),
    fps: float = Form(1.0),
    max_frames: int = Form(60),
    confidence: float = Form(0.5),
):
    """
    Upload video and start analysis.
    Returns job_id for status polling.
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded video to temp location
    video_path = f"/tmp/tempograph_{job_id}/{video.filename}"
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    # Parse modules string to dict
    module_list = [m.strip() for m in modules.split(",")]
    module_dict = {
        "behavior": "behavior" in module_list,
        "detection": "detection" in module_list,
        "depth": "depth" in module_list,
        "audio": "audio" in module_list,
    }
    
    config = PipelineConfig(
        backend=backend,
        modules=module_dict,
        fps=fps,
        max_frames=max_frames,
        confidence=confidence,
        video_path=video_path,
        output_dir=f"results/{job_id}"
    )
    
    jobs[job_id] = {"status": "processing", "result": None, "error": None}
    background_tasks.add_task(run_pipeline_job, job_id, config)
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, 
                          content={"error": "Job not found"})
    return {"job_id": job_id, "status": jobs[job_id]["status"]}

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, 
                          content={"error": "Job not found"})
    job = jobs[job_id]
    if job["status"] == "processing":
        return {"status": "processing"}
    if job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    return {"status": "complete", "result": job["result"]}

async def run_pipeline_job(job_id: str, config: PipelineConfig):
    """Background task that runs the pipeline."""
    try:
        # Import here to avoid circular imports
        from src.pipeline import Pipeline
        pipeline = Pipeline(config)
        result = pipeline.run()
        jobs[job_id] = {
            "status": "complete",
            "result": result.model_dump(),
            "error": None
        }
    except Exception as e:
        logging.error(f"Job {job_id} failed: {e}")
        jobs[job_id] = {
            "status": "error",
            "result": None,
            "error": str(e)
        }

Implement fully. Add input validation and proper error responses.
Include a simple CLI runner at the bottom:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### TASK 7: Streamlit UI

```
Create ui/app.py for TempoGraph.

Full Streamlit UI with modular controls and tabbed results.

Key features:
- Video upload (st.file_uploader, accept video/*)
- Backend selector: radio buttons "☁️ Cloud (Gemini Flash)" / "💻 Local (6GB GPU)"
- Module checkboxes:
  - "🧠 Entity Tracking & Behavior" (always checked, disabled)
  - "🔲 Object Detection (YOLOv8-nano)"
  - "🌊 Depth Estimation (Depth Anything V2)"
  - "🔊 Audio Analysis (Whisper)"
- Advanced settings in expander:
  - FPS slider (0.5 to 5.0, default 1.0)
  - Max frames slider (10 to 120, default 60)
  - Confidence slider (0.1 to 0.9, default 0.5)
- "▶ Analyze Video" button
- Processing: show spinner with status updates
- Results in tabs:
  1. 📹 Annotated Video — st.video player
  2. 📊 Timeline — plotly horizontal bar chart:
     - X axis = time (seconds)
     - Y axis = entity IDs and "audio" row
     - Color = event type (approach=blue, interact=green, etc.)
     - Audio events as separate row at bottom
  3. 🔗 Interaction Graph — embedded pyvis HTML or streamlit-agraph
  4. 📝 Summary — text from analysis + stats table
  5. 📥 Export — download buttons for JSON and annotated video

Layout structure:
- st.set_page_config(page_title="TempoGraph", layout="wide")
- Sidebar for controls OR main column layout
- Use st.session_state to persist results across reruns

IMPORTANT: 
- The pipeline import and execution will be wired later by Opus.
  For now, create the full UI with MOCK DATA so we can verify layout.
- Create a _generate_mock_result() function that returns a PipelineResult 
  with realistic fake data (3 entities, 5 visual events, 3 audio events).
- Wire the button to use mock data for now.
- Leave a clearly marked comment: # TODO: OPUS — Replace mock with real pipeline

Use plotly.express for timeline chart.
Use streamlit-agraph or components.html for graph viz.
Style with minimal custom CSS — keep it clean.

Also create a separate function for each tab's rendering so 
they can be tested independently.
```

---

### TASK 8: Docker + docker-compose

```
Create Dockerfile and docker-compose.yml for TempoGraph.

Two Docker services:
1. tempograph-gpu: Full pipeline with CUDA support (for local mode)
2. tempograph-cpu: Cloud-only mode (Gemini only, no GPU needed)

Dockerfile (GPU version):
- Base: nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
- Install: python3.11, pip, ffmpeg, git
- Copy requirements.txt and install
- Copy src/, ui/, configs/
- Expose ports: 8000 (API), 8501 (Streamlit)
- Default CMD: run both API and UI

Dockerfile.cpu (CPU version):
- Base: python:3.11-slim
- Install: ffmpeg
- Same pip install but with --extra-index-url for CPU-only torch
- No CUDA/GPU dependencies

docker-compose.yml:
services:
  tempograph:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"    # API
      - "8501:8501"    # Streamlit UI
    volumes:
      - ./results:/app/results
      - ./data:/app/data
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      bash -c "uvicorn src.api:app --host 0.0.0.0 --port 8000 &
               streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"

Also create:
- .dockerignore (ignore .git, __pycache__, results/, data/samples/, .env)
- .env.example with GEMINI_API_KEY=your_key_here

Keep Dockerfiles lean — use multi-stage if helpful.
Add health check.
```

---

### TASK 9: Pipeline Orchestrator (Skeleton)

```
Create src/pipeline.py for TempoGraph.

This is the main orchestrator that coordinates all modules.
Create the SKELETON with clear interfaces — the VLM backend 
calls will be filled in by Opus later.

Implementation:

import time
import logging
import yaml
from pathlib import Path
from typing import Optional
from src.models import (PipelineConfig, PipelineResult, AnalysisResult,
                         DetectionResult, DepthResult)
from src.modules.frame_extractor import FrameExtractor, ExtractionResult
from src.json_parser import JSONParser
from src.graph_builder import GraphBuilder
from src.video_annotator import VideoAnnotator

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
        self.logger.info(f"Starting pipeline: backend={self.config.backend}, "
                        f"modules={self.config.modules}")
        
        # Step 1: Extract frames and audio
        extraction = self.frame_extractor.extract(
            video_path=self.config.video_path,
            fps=self.config.fps,
            max_frames=self.config.max_frames,
            extract_audio=self.config.modules.get("audio", False)
        )
        self.logger.info(f"Extracted {len(extraction.frame_paths)} frames, "
                        f"audio={'yes' if extraction.audio_path else 'no'}")
        
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
        # TODO: OPUS — Implement _run_vlm_analysis
        analysis = self._run_vlm_analysis(extraction)
        
        # Step 6: Run audio analysis if enabled (local mode only)
        if (self.config.modules.get("audio", False) and 
            self.config.backend == "qwen" and extraction.audio_path):
            # TODO: OPUS — Implement _run_audio_analysis  
            self._merge_audio_results(analysis, extraction.audio_path)
        
        # Step 7: Build graph
        graph = self.graph_builder.build(analysis)
        
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
                fps=extraction.extraction_fps
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
            processing_time=elapsed
        )
    
    def _run_detection(self, extraction: ExtractionResult) -> DetectionResult:
        """Run YOLOv8-nano on extracted frames."""
        # TODO: OPUS — Wire actual YOLO module with VRAM management
        # For now, return empty
        self.logger.info("Object detection: not yet implemented")
        return DetectionResult()
    
    def _run_depth(self, extraction: ExtractionResult) -> DepthResult:
        """Run Depth Anything V2 Small on extracted frames."""
        # TODO: OPUS — Wire actual depth module with VRAM management
        self.logger.info("Depth estimation: not yet implemented")
        return DepthResult()
    
    def _run_vlm_analysis(self, extraction: ExtractionResult) -> AnalysisResult:
        """Run VLM backend for behavior analysis."""
        # TODO: OPUS — Initialize correct backend, run, parse JSON
        self.logger.info("VLM analysis: not yet implemented")
        return AnalysisResult(summary="Pipeline skeleton — VLM not yet wired")
    
    def _merge_audio_results(self, analysis: AnalysisResult, audio_path):
        """Merge Whisper results into analysis."""
        # TODO: OPUS — Run Whisper, merge into analysis.audio_events
        pass
    
    def _unload_vision_models(self):
        """Free GPU VRAM from detection + depth models."""
        # TODO: OPUS — Implement proper VRAM cleanup
        pass
    
    def _cleanup_backends(self):
        """Cleanup all backends."""
        if self._vlm_backend:
            self._vlm_backend.cleanup()
    
    def _save_results(self, analysis, detection, depth):
        """Save results to output directory."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis JSON
        with open(output_dir / "analysis.json", "w") as f:
            f.write(analysis.model_json(indent=2))
        
        if detection:
            with open(output_dir / "detection.json", "w") as f:
                f.write(detection.model_json(indent=2))
        
        self.logger.info(f"Results saved to {output_dir}")

# CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TempoGraph Pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--backend", default="gemini", choices=["gemini", "qwen"])
    parser.add_argument("--modules", default="behavior,detection,audio")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    
    module_list = [m.strip() for m in args.modules.split(",")]
    config = PipelineConfig(
        backend=args.backend,
        modules={m: True for m in module_list},
        fps=args.fps,
        max_frames=args.max_frames,
        video_path=args.video,
        output_dir=args.output
    )
    
    pipeline = Pipeline(config)
    result = pipeline.run()
    print(f"Done in {result.processing_time:.1f}s")

Mark every TODO: OPUS comment clearly so we know what needs 
to be filled in by the Opus workstream.
```

---

## Checklist

After all 9 tasks, Nemotron should have created:

- [ ] `requirements.txt`
- [ ] `configs/default.yaml`
- [ ] `src/models.py` — all Pydantic data models
- [ ] `src/backends/base.py` — abstract VLM interface
- [ ] `src/modules/frame_extractor.py` — OpenCV + ffmpeg
- [ ] `src/json_parser.py` — robust JSON parsing
- [ ] `tests/test_parser.py` — parser unit tests
- [ ] `src/graph_builder.py` — NetworkX graph
- [ ] `src/video_annotator.py` — OpenCV video overlay
- [ ] `src/api.py` — FastAPI server
- [ ] `ui/app.py` — Streamlit UI with mock data
- [ ] `Dockerfile` + `Dockerfile.cpu`
- [ ] `docker-compose.yml`
- [ ] `.dockerignore` + `.env.example`
- [ ] `src/pipeline.py` — orchestrator skeleton with TODO: OPUS markers

**Total: ~15 files, all standard Python, no GPU/VLM complexity.**

---

## Notes for the Person Running This

- Feed tasks ONE AT A TIME to Nemotron
- Let it generate, review, fix obvious issues, then move to next task
- If it hallucinates an API (wrong import, wrong method name), just fix it — the structures are all standard Python
- The `TODO: OPUS` markers will tell you exactly where the Opus workstream plugs in
- You can test everything with mock data before Opus fills in the real backends
