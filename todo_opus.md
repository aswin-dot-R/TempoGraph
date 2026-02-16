# TempoGraph — Opus Build Tasks
## Frontier Model Workstream (Run After Nemotron Completes)

> **For:** Claude Opus via claude.ai or Claude Code
> **Scope:** VLM backends (Gemini + Qwen), VRAM management, module integration, pipeline wiring
> **Prerequisite:** Nemotron workstream must be complete — all scaffolding, models, UI, etc.
> **Why Opus:** These tasks involve exact API signatures, quantization config, GPU memory management, and multi-library integration where hallucinated code = runtime crashes

---

## Context for Opus

The Nemotron workstream has already created:
- `src/models.py` — all Pydantic data models (Entity, VisualEvent, AudioEvent, AnalysisResult, etc.)
- `src/backends/base.py` — abstract BaseVLMBackend with analyze_video(), is_available(), cleanup()
- `src/modules/frame_extractor.py` — extracts frames (OpenCV) and audio (ffmpeg)
- `src/json_parser.py` — robust JSON parsing with validation
- `src/graph_builder.py` — NetworkX temporal graph
- `src/video_annotator.py` — OpenCV video annotation
- `src/pipeline.py` — orchestrator SKELETON with `TODO: OPUS` markers
- `ui/app.py` — Streamlit UI with mock data and `TODO: OPUS` markers
- `src/api.py` — FastAPI server
- `configs/default.yaml` — all config params
- Docker files

Your job: Fill in all `TODO: OPUS` markers and create the GPU-aware modules.

**Critical constraint:** Everything must fit in 6GB VRAM (RTX 3060 Ti).
Models load/unload sequentially — never all at once.

---

## Task Order

### OPUS TASK 1: Gemini Backend (`src/backends/gemini_backend.py`)

This is the cloud VLM backend using Google's Gemini Flash API.
Gemini can process full video natively (including audio) in a single API call.

**Key requirements:**
- Use `google-genai` SDK (NOT the old `google.generativeai`)
- Upload video via File API first, then reference in prompt
- Single call handles both visual and audio analysis
- Parse response through json_parser
- Handle rate limits, retries, malformed responses

```python
"""
Gemini Flash backend for TempoGraph.

Uses google-genai SDK. Key API flow:
1. client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
2. Upload video:
   video_file = client.files.upload(file=video_path)
   # Poll until file.state == "ACTIVE"
3. Generate:
   response = client.models.generate_content(
       model="gemini-2.0-flash",
       contents=[
           types.Content(parts=[
               types.Part.from_uri(
                   file_uri=video_file.uri,
                   mime_type=video_file.mime_type
               ),
               types.Part.from_text(ANALYSIS_PROMPT)
           ])
       ],
       config=types.GenerateContentConfig(
           temperature=0.1,
           max_output_tokens=8192,
           response_mime_type="application/json"  # forces JSON output
       )
   )
4. Parse: response.text → json_parser.parse()

IMPORTANT DETAILS:
- Gemini processes audio FROM the video natively — no need for separate 
  audio extraction in cloud mode
- response_mime_type="application/json" forces structured output
- Video files take time to process — poll file.state until "ACTIVE"
- Free tier: 15 RPM, 1000 RPD for Flash
- Handle google.api_core.exceptions for rate limits

Audio token cost: ~32 tokens/second of audio (very cheap)
Video token cost: ~263 tokens/second at 1 FPS
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional
from src.backends.base import BaseVLMBackend
from src.models import AnalysisResult
from src.json_parser import JSONParser

# The full VLM prompt from the plan (include the complete prompt 
# from configs or inline)
ANALYSIS_PROMPT = """..."""  # Copy from plan v3

class GeminiBackend(BaseVLMBackend):
    def __init__(self, model: str = "gemini-2.0-flash", 
                 temperature: float = 0.1,
                 max_output_tokens: int = 8192):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.logger = logging.getLogger(__name__)
        self.parser = JSONParser()
        self._client = None
    
    def _get_client(self):
        """Lazy init Gemini client."""
        if self._client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        return self._client
    
    def analyze_video(
        self,
        video_path: str,
        frames: List[Path] = None,  # ignored — Gemini uses full video
        audio_path: str = None,      # ignored — Gemini extracts from video
        prompt: str = None
    ) -> AnalysisResult:
        """
        Upload video to Gemini and get comprehensive analysis.
        Handles: upload → wait for processing → generate → parse
        """
        # IMPLEMENT:
        # 1. Upload video file
        # 2. Wait for ACTIVE state (poll with backoff)
        # 3. Generate with prompt
        # 4. Parse JSON response
        # 5. Retry up to 3 times if JSON parsing fails
        # 6. Delete uploaded file after done
        pass
    
    def _upload_and_wait(self, video_path: str, timeout: int = 120):
        """Upload video and wait for it to become ACTIVE."""
        # client.files.upload(file=video_path)
        # Poll every 5 seconds until state == "ACTIVE" or timeout
        pass
    
    def is_available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))
    
    def cleanup(self):
        pass  # No GPU resources to free
    
    @property
    def name(self) -> str:
        return "gemini"
    
    @property
    def requires_gpu(self) -> bool:
        return False
```

**Test this by:**
1. Set GEMINI_API_KEY env var
2. Run on a short sample video (15-30 seconds)
3. Verify JSON output matches AnalysisResult schema
4. Verify audio events are populated (speech, sounds)

---

### OPUS TASK 2: Qwen Local Backend (`src/backends/qwen_backend.py`)

This is the local VLM backend using Qwen2.5-VL-3B with 4-bit quantization.
This is the trickiest file in the project — VRAM management is critical.

**Key requirements:**
- Load Qwen2.5-VL-3B-Instruct with bitsandbytes 4-bit quantization
- Process extracted frames (not full video)
- Use the Qwen2.5-VL processor for image encoding
- Force JSON output via prompt engineering
- Explicit VRAM cleanup after inference
- Must use ≤2GB VRAM

```python
"""
Qwen2.5-VL-3B local backend for TempoGraph.

CRITICAL VRAM NOTES:
- This model MUST use 4-bit quantization to fit in 6GB VRAM
- Load with BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
- After inference: del model, del processor, torch.cuda.empty_cache(), gc.collect()
- Peak VRAM should be ~2.0-2.5GB

EXACT API USAGE for Qwen2.5-VL:

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Build message with multiple frames as images:
messages = [
    {
        "role": "user",
        "content": [
            # Add each frame as an image
            {"type": "image", "image": f"file://{frame_path}"}  
            for frame_path in frame_paths
        ] + [
            {"type": "text", "text": ANALYSIS_PROMPT_LOCAL}
        ]
    }
]

# Process with qwen_vl_utils
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.1)
output_ids_trimmed = output_ids[0][inputs.input_ids.shape[1]:]
response_text = processor.decode(output_ids_trimmed, skip_special_tokens=True)

IMPORTANT CAVEATS:
- Qwen2.5-VL can't process audio — only visual frames
- For local audio: separate Whisper pipeline handles that
- The local prompt should NOT reference audio analysis
- Limit to ~15-20 frames max to stay within context + VRAM budget
- If more frames than limit: subsample uniformly
- qwen_vl_utils is from the qwen-vl-utils pip package

LOCAL-ONLY PROMPT (no audio section):
The prompt should ask for entities + visual_events only.
Audio events will be merged from Whisper separately.
"""

import os
import gc
import torch
import logging
from pathlib import Path
from typing import List, Optional
from src.backends.base import BaseVLMBackend
from src.models import AnalysisResult
from src.json_parser import JSONParser

# Local-only prompt (no audio — Whisper handles that separately)
ANALYSIS_PROMPT_LOCAL = """You are a video analysis AI. Analyze these video frames 
(sampled at regular intervals) and identify:

1. All distinct entities (people, animals, objects) with unique IDs
2. All temporal behaviors and interactions between entities
3. Estimated timestamps based on frame position

Output ONLY valid JSON:
{
  "entities": [
    {"id": "E1", "type": "dog", "description": "brown labrador", 
     "first_seen": "00:02", "last_seen": "00:45"}
  ],
  "visual_events": [
    {"type": "approach", "entities": ["E1", "E2"], 
     "start_time": "00:03", "end_time": "00:05",
     "description": "Brown labrador walks toward white poodle",
     "confidence": 0.9}
  ],
  "audio_events": [],
  "multimodal_correlations": [],
  "summary": "Brief description of what happens in the video."
}

Behavior types: approach, depart, interact, follow, idle, group, avoid, chase, observe.
Be precise. Include ALL entities and interactions you observe across frames."""


class QwenBackend(BaseVLMBackend):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 4096,
        max_frames: int = 16,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames  # limit frames sent to VLM
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self.parser = JSONParser()
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Load model with 4-bit quantization. ~2GB VRAM."""
        if self._model is not None:
            return
        
        self.logger.info("Loading Qwen2.5-VL-3B with 4-bit quantization...")
        # IMPLEMENT: 
        # 1. Create BitsAndBytesConfig
        # 2. Load model with quantization_config + device_map="auto"
        # 3. Load processor
        # 4. Log VRAM usage: torch.cuda.memory_allocated() / 1e9
        pass
    
    def analyze_video(
        self,
        video_path: str,
        frames: List[Path] = None,
        audio_path: str = None,  # ignored — Whisper handles local audio
        prompt: str = None
    ) -> AnalysisResult:
        """
        Process frames through Qwen2.5-VL-3B.
        
        Steps:
        1. Load model if not loaded
        2. Subsample frames to self.max_frames
        3. Build message with images + prompt
        4. Run inference
        5. Parse JSON response
        6. Return AnalysisResult
        """
        # IMPLEMENT with proper error handling
        # Key: if len(frames) > self.max_frames, subsample uniformly
        pass
    
    def _subsample_frames(self, frames: List[Path]) -> List[Path]:
        """Uniformly subsample frames to max_frames."""
        if len(frames) <= self.max_frames:
            return frames
        indices = [int(i * len(frames) / self.max_frames) 
                   for i in range(self.max_frames)]
        return [frames[i] for i in indices]
    
    def is_available(self) -> bool:
        return torch.cuda.is_available()
    
    def cleanup(self):
        """CRITICAL: Free all GPU memory."""
        self.logger.info("Cleaning up Qwen backend...")
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1e9
            self.logger.info(f"VRAM after cleanup: {vram_after:.2f}GB")
    
    @property
    def name(self) -> str:
        return "qwen"
    
    @property
    def requires_gpu(self) -> bool:
        return True
```

**Test this by:**
1. Load on a GPU with ≥6GB VRAM
2. Check `torch.cuda.memory_allocated()` after loading — should be ~2GB
3. Run on 10 frames, verify JSON output
4. Call cleanup(), verify VRAM drops to near 0
5. Verify it can load again after cleanup (second run)

---

### OPUS TASK 3: YOLO Detection Module (`src/modules/detector.py`)

**Key requirements:**
- YOLOv8-nano for object detection
- Track objects across frames (use ultralytics built-in tracking if available, else simple IoU matching)
- Return DetectionResult with per-frame bounding boxes
- Explicit GPU cleanup after processing all frames

```python
"""
YOLOv8-nano detection module.

Model: yolov8n.pt (~6.3MB, pretrained on COCO 80 classes)
VRAM: ~0.3GB when loaded

Usage:
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Detection on single frame:
results = model(frame, conf=0.5, imgsz=640, verbose=False)
# results[0].boxes.xyxy  → tensor of [x1, y1, x2, y2]
# results[0].boxes.conf  → tensor of confidences
# results[0].boxes.cls   → tensor of class indices
# model.names[int(cls)]  → class name string

# For tracking across frames (assigns persistent IDs):
results = model.track(frame, conf=0.5, imgsz=640, persist=True, verbose=False)
# results[0].boxes.id  → tensor of track IDs (or None if tracking fails)

VRAM management:
- Load model to CUDA
- Process all frames
- del model + torch.cuda.empty_cache() 
- Must free VRAM before Qwen loads
"""

import torch
import gc
import logging
from pathlib import Path
from typing import List, Optional
from src.models import DetectionResult, DetectionBox

class ObjectDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        imgsz: int = 640,
        device: str = "cuda"
    ):
        # IMPLEMENT: Store params, lazy-load model
        pass
    
    def detect_frames(self, frame_paths: List[Path]) -> DetectionResult:
        """
        Run detection + tracking on all frames.
        Returns DetectionResult with per-frame boxes.
        """
        # IMPLEMENT:
        # 1. Load model if not loaded
        # 2. For each frame:
        #    a. Read with cv2.imread
        #    b. model.track(frame, persist=True) for tracking
        #    c. Extract boxes, classes, confidences, track IDs
        #    d. Append to results
        # 3. Return DetectionResult
        pass
    
    def cleanup(self):
        """Free GPU memory."""
        # del self._model, gc.collect(), torch.cuda.empty_cache()
        pass
```

---

### OPUS TASK 4: Depth Estimation Module (`src/modules/depth.py`)

```python
"""
Depth Anything V2 Small module.

Model: depth-anything-v2-vits (~98MB weights, ViT-Small encoder)
VRAM: ~0.5GB when loaded
Speed: ~100ms per frame on GPU

EXACT USAGE:
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 
             'out_channels': [48, 96, 192, 384]},
}

model = DepthAnythingV2(**model_configs['vits'])
model.load_state_dict(
    torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu')
)
model = model.to('cuda').eval()

# Inference:
raw_img = cv2.imread(str(frame_path))
depth = model.infer_image(raw_img)  # HxW numpy array, float32
# depth values: relative (not metric), higher = farther

# Colorize for visualization:
depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255
depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

WEIGHT DOWNLOAD:
The model weights need to be downloaded separately.
Check if file exists, if not, download from HuggingFace:
huggingface_hub.hf_hub_download(
    repo_id="depth-anything/Depth-Anything-V2-Small",
    filename="depth_anything_v2_vits.pth",
    local_dir="checkpoints"
)

ALTERNATIVE: Use transformers pipeline (simpler but slightly more VRAM):
from transformers import pipeline
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small")
result = pipe(image)  # returns {"depth": PIL.Image, "predicted_depth": tensor}

VRAM management: same pattern — load, process all frames, cleanup.
"""

import torch
import gc
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List
from src.models import DepthResult, DepthFrame

class DepthEstimator:
    def __init__(self, model_variant: str = "vits", device: str = "cuda"):
        # IMPLEMENT
        pass
    
    def estimate_frames(
        self, frame_paths: List[Path], output_dir: str = "/tmp/tempograph_depth"
    ) -> DepthResult:
        """
        Run depth estimation on all frames.
        Saves depth maps as .npy and colorized .png files.
        """
        # IMPLEMENT:
        # 1. Load model if not loaded
        # 2. For each frame: infer, save depth .npy + colorized .png
        # 3. Return DepthResult
        pass
    
    def cleanup(self):
        """Free GPU memory."""
        pass
```

---

### OPUS TASK 5: Audio Analysis Module (`src/modules/audio.py`)

```python
"""
Audio analysis using Whisper-small for local mode.

Model: openai/whisper-small (~461MB, runs on CPU)
VRAM: 0 — always CPU to preserve GPU for other models

Usage:
import whisper
model = whisper.load_model("small", device="cpu")
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
from pathlib import Path
from typing import Optional, List
from src.models import AudioEvent

class AudioAnalyzer:
    def __init__(self, model_size: str = "small", device: str = "cpu"):
        # IMPLEMENT: lazy load whisper
        pass
    
    def analyze(self, audio_path: str) -> List[AudioEvent]:
        """
        Transcribe audio and return AudioEvent list.
        """
        # IMPLEMENT:
        # 1. Load whisper model (CPU)
        # 2. Transcribe with word_timestamps=True
        # 3. Convert segments → AudioEvent objects
        # 4. Return list
        pass
    
    def _seconds_to_mmss(self, seconds: float) -> str:
        """Convert 65.3 → '01:05'"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    
    def cleanup(self):
        """Free memory (CPU model, less critical)."""
        pass
```

---

### OPUS TASK 6: Wire Everything in Pipeline (`src/pipeline.py`)

Now fill in ALL the `TODO: OPUS` markers in the pipeline skeleton.

```
The Nemotron workstream created src/pipeline.py with TODO: OPUS markers.
Fill in every marker with real implementations.

Key changes needed:

1. _run_detection():
   - Import ObjectDetector from src.modules.detector
   - Create detector, run detect_frames(), return result
   - Log VRAM usage before and after

2. _run_depth():
   - Import DepthEstimator from src.modules.depth
   - Create estimator, run estimate_frames(), return result
   - Log VRAM usage

3. _unload_vision_models():
   - Call cleanup() on detector and depth estimator
   - gc.collect() + torch.cuda.empty_cache()
   - Log VRAM freed
   - THIS IS CRITICAL — Qwen needs 2GB and must load after these unload

4. _run_vlm_analysis():
   - If backend == "gemini": 
     create GeminiBackend, call analyze_video(video_path=...)
   - If backend == "qwen":
     create QwenBackend, call analyze_video(frames=extraction.frame_paths)
   - Parse result through json_parser
   - Store backend reference for later cleanup

5. _merge_audio_results():
   - Create AudioAnalyzer
   - Run analyze(audio_path)
   - Append results to analysis.audio_events
   - Simple multimodal correlation: match audio timestamps 
     with visual event timestamps (±2 seconds window)

6. Add VRAM logging throughout:
   def _log_vram(self, label: str):
       if torch.cuda.is_available():
           allocated = torch.cuda.memory_allocated() / 1e9
           reserved = torch.cuda.memory_reserved() / 1e9
           self.logger.info(f"VRAM [{label}]: {allocated:.2f}GB allocated, "
                           f"{reserved:.2f}GB reserved")

EXECUTION ORDER IS CRITICAL:
Frame extraction → YOLO (0.3GB) → Depth (0.5GB) → 
UNLOAD BOTH → Qwen (2.0GB) → UNLOAD → Whisper (CPU, 0GB)

If at any point VRAM exceeds 4GB, something is wrong.
```

---

### OPUS TASK 7: Wire UI to Real Pipeline (`ui/app.py`)

```
The Nemotron workstream created ui/app.py with mock data.
Replace the TODO: OPUS markers with real pipeline calls.

Changes needed:

1. Replace _generate_mock_result() calls with actual Pipeline execution:
   
   from src.pipeline import Pipeline
   from src.models import PipelineConfig
   
   config = PipelineConfig(
       backend=selected_backend,
       modules={
           "behavior": True,
           "detection": detection_checkbox,
           "depth": depth_checkbox,
           "audio": audio_checkbox,
       },
       fps=fps_slider,
       max_frames=max_frames_slider,
       confidence=confidence_slider,
       video_path=saved_video_path,
       output_dir=f"results/{session_id}"
   )
   pipeline = Pipeline(config)
   result = pipeline.run()

2. Handle video file saving:
   - st.file_uploader returns UploadedFile
   - Save to temp file: 
     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
         f.write(uploaded_file.read())
         video_path = f.name

3. Add error handling:
   - Wrap pipeline.run() in try/except
   - Show st.error() on failure
   - Show VRAM warning if backend=="qwen" and no CUDA available

4. Add real-time status updates:
   - Use st.status() context manager for progress
   - Show which module is currently running

5. Handle annotated video display:
   - If result.annotated_video_path exists, show with st.video()
   - If not (no detection/depth modules), show original video

6. Make timeline chart use real data from result.analysis
7. Make graph viz use real data from GraphBuilder

Keep the mock data function as a fallback / demo mode option.
Add a checkbox: "☐ Use demo data (no processing)" for quick testing.
```

---

## VRAM Verification Script

Create this utility for testing the VRAM budget:

```python
# tests/test_vram_budget.py
"""
Verify all models fit in 6GB VRAM when loaded sequentially.
Run on actual GPU to validate.
"""
import torch
import gc
import time

def log_vram(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{label}] Allocated: {alloc:.3f}GB, Reserved: {reserved:.3f}GB")

def test_sequential_loading():
    """Test that all models fit in 6GB when loaded sequentially."""
    assert torch.cuda.is_available(), "No GPU available"
    
    total_vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"Total VRAM: {total_vram:.1f}GB")
    
    log_vram("baseline")
    
    # Phase 1: YOLO + Depth (should be ~0.8GB total)
    print("\n--- Phase 1: YOLO + Depth ---")
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    yolo.to("cuda")
    log_vram("after YOLO load")
    
    # Run one dummy inference
    import numpy as np
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    yolo(dummy, verbose=False)
    log_vram("after YOLO inference")
    
    # Load depth
    # ... (test depth model loading + inference)
    log_vram("after Depth load")
    
    # Phase 2: Unload
    print("\n--- Phase 2: Unload ---")
    del yolo
    gc.collect()
    torch.cuda.empty_cache()
    log_vram("after unload")
    
    # Phase 3: Qwen (should be ~2.0GB)
    print("\n--- Phase 3: Qwen ---")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    log_vram("after Qwen load")
    
    peak = torch.cuda.memory_allocated() / 1e9
    assert peak < 4.0, f"Peak VRAM {peak:.2f}GB exceeds budget!"
    print(f"\n✅ Peak VRAM: {peak:.2f}GB — fits in 6GB!")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_vram("final cleanup")

if __name__ == "__main__":
    test_sequential_loading()
```

---

## Checklist

After Opus completes, these files should be fully implemented:

- [ ] `src/backends/gemini_backend.py` — Gemini Flash with File API
- [ ] `src/backends/qwen_backend.py` — Qwen2.5-VL-3B with 4-bit quant
- [ ] `src/modules/detector.py` — YOLOv8-nano with tracking
- [ ] `src/modules/depth.py` — Depth Anything V2 Small
- [ ] `src/modules/audio.py` — Whisper-small on CPU
- [ ] `src/pipeline.py` — all TODO: OPUS markers filled
- [ ] `ui/app.py` — all TODO: OPUS markers filled, real pipeline wired
- [ ] `tests/test_vram_budget.py` — VRAM verification script

**After this:** The entire TempoGraph pipeline should work end-to-end.
Run `streamlit run ui/app.py`, upload a video, and get results.

---

## Testing Order

1. `test_vram_budget.py` — verify models fit in 6GB
2. Gemini backend alone — `python -c "from src.backends.gemini_backend import GeminiBackend; ..."`
3. Qwen backend alone — same pattern
4. Full pipeline cloud mode — `python -m src.pipeline --video test.mp4 --backend gemini`
5. Full pipeline local mode — `python -m src.pipeline --video test.mp4 --backend qwen`
6. Streamlit UI — `streamlit run ui/app.py`
7. Docker — `docker compose up --build`
