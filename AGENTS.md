# TempoGraph - Agent Coding Guidelines

## Overview

TempoGraph is a fully-local multimodal video analysis pipeline. It extracts
temporal behaviors, interactions, audio transcripts, and cross-modal
correlations from video using YOLO + Depth Anything V2 + Whisper.cpp +
Qwen3.5-VL (served by llama.cpp), persisted to a per-run SQLite store with
a Streamlit UI for both running the pipeline and browsing past results.

See `README.md` for a high-level overview and `docs/PIPELINE.md` for the
deep technical doc with every stage and knob.

## Project Structure

```
TempoGraph/
├── src/
│   ├── pipeline_v2.py        # orchestrator (the only pipeline)
│   ├── aggregator.py         # chunk → analysis.json
│   ├── batch_runner.py       # run pipeline over a directory of videos
│   ├── dataset_exporter.py   # COCO + JSONL dataset export
│   ├── runtime_estimator.py  # ETA model used by the UI
│   ├── storage.py            # SQLite schema + helpers
│   ├── graph_builder.py      # NetworkX → pyvis HTML
│   ├── json_parser.py        # lenient LLM-JSON extractor
│   ├── models.py             # Pydantic data models
│   ├── backends/
│   │   ├── base.py
│   │   └── llama_server_backend.py  # → llama.cpp /v1/chat/completions
│   └── modules/
│       ├── frame_selector.py        # motion-aware frame sampler
│       ├── frame_scorer.py          # top-K scorer for VLM frame pick
│       ├── detector.py              # ultralytics YOLO26
│       ├── depth.py                 # transformers Depth Anything V2
│       └── whisper_cpp.py           # subprocess wrapper for whisper.cpp
├── ui/
│   ├── app.py                # main pipeline page (Streamlit)
│   └── pages/Results.py      # results browser (Streamlit)
├── tests/
│   ├── test_parser.py     # JSON parser unit tests
│   └── test_vram_budget.py # VRAM verification test
├── configs/
│   └── default.yaml       # Default configuration
└── requirements.txt       # Python dependencies
```

## Build, Lint, and Test Commands

### Running Tests

```bash
# Run all tests with pytest
pytest tests/

# Run a single test file
pytest tests/test_parser.py

# Run a specific test
pytest tests/test_parser.py::TestJSONParser::test_clean_valid_json

# Run VRAM budget test (requires GPU)
python tests/test_vram_budget.py
```

### Code Formatting

```bash
# Format code with Black (line length 88)
black src/ tests/

# Check formatting without modifying
black --check src/ tests/
```

### Linting

```bash
# Lint with flake8
flake8 src/ tests/

# Lint specific file
flake8 src/pipeline_v2.py
```

### Running the Pipeline

```bash
# Streamlit UI (the recommended path)
make run

# CLI on a single video
make run-cli VIDEO=clip.mp4

# Bulk: process every video in a directory
python -m src.batch_runner --video-dir videos/ --output-dir results/

# Skip-VLM smoke test (synthetic 10s video)
make smoke
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development (includes dev dependencies)
pip install -r requirements.txt pytest black flake8
```

## Code Style Guidelines

### General Rules

- Python 3.9+ required
- Use type hints for all function parameters and return types
- Maximum line length: 88 characters (Black default)
- Use 4 spaces for indentation (no tabs)

### Imports

- Use absolute imports from `src` package
- Group imports in this order: stdlib, third-party, local
- Sort imports alphabetically within each group

```python
# Correct
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from pydantic import BaseModel

from src.models import PipelineConfig, AnalysisResult
from src.modules.detector import ObjectDetector
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `Pipeline`, `ObjectDetector`)
- **Functions/variables**: snake_case (e.g., `run_pipeline`, `frame_paths`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `MAX_FRAMES`)
- **Private methods**: prefix with underscore (e.g., `_run_detection`)
- **Enum values**: UPPER_SNAKE_CASE for values

### Type Hints

Always use type hints for function signatures:

```python
def run(self) -> PipelineResult:
    pass

def _run_detection(self, extraction: ExtractionResult) -> DetectionResult:
    pass

def process_frame(self, frame: np.ndarray, threshold: float = 0.5) -> Optional[DetectionBox]:
    pass
```

### Pydantic Models

Use Pydantic v2 for all data models. Define models in `src/models.py`:

```python
from pydantic import BaseModel
from typing import List, Optional

class Entity(BaseModel):
    id: str
    type: str
    description: str
    first_seen: str
    last_seen: str
```

### Error Handling

- Use explicit exception types
- Add context to error messages
- Log errors before raising
- Handle GPU/CUDA errors gracefully

```python
# Good practice
def _run_detection(self, extraction: ExtractionResult) -> DetectionResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for detection module")
    
    try:
        detector = ObjectDetector(...)
        result = detector.detect_frames(extraction.frame_paths)
    except Exception as e:
        self.logger.error(f"Detection failed: {e}")
        raise
    return result
```

### Logging

Use the standard logging module:

```python
import logging

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        self.logger.info(f"Starting pipeline: backend={self.config.backend}")
        self.logger.debug(f"Processing {len(frames)} frames")
```

### Model Cleanup

Always cleanup GPU resources after use:

```python
def cleanup(self):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Documentation

- Use docstrings for public classes and functions
- Follow Google docstring format
- Document exceptions that may be raised

```python
def analyze_video(self, video_path: str, frames: List[str]) -> AnalysisResult:
    """Analyze video frames using the VLM backend.

    Args:
        video_path: Path to input video file.
        frames: List of extracted frame image paths.

    Returns:
        AnalysisResult containing detected entities, events, and correlations.

    Raises:
        RuntimeError: If backend initialization fails.
    """
```

### Testing

- Write unit tests using `unittest` or `pytest`
- Place tests in `tests/` directory
- Test file naming: `test_*.py`
- Test class naming: `Test*`

```python
import unittest

class TestJSONParser(unittest.TestCase):
    def setUp(self):
        self.parser = JSONParser(min_confidence=0.3)

    def test_clean_valid_json(self):
        raw_text = '{"entities": [...]}'
        result = self.parser.parse(raw_text)
        self.assertIsInstance(result, AnalysisResult)
```

### VRAM Budget

The pipeline is designed to fit within 6GB VRAM. Models are loaded sequentially:

1. Frame extraction (CPU)
2. YOLO detection (~0.3GB)
3. Depth estimation (~0.5GB)
4. Unload vision models
5. Qwen VLM (~2.0GB, 4-bit quantized)
6. Unload Qwen
7. Whisper audio (CPU)

When adding new models, ensure they fit within this budget or add a test to verify.

### Configuration

- Use `configs/default.yaml` for default settings
- Use Pydantic models for config validation
- Support environment variables for API keys

```python
config = PipelineConfig(
    backend="qwen",
    modules={"behavior": True, "detection": True, "depth": False, "audio": True},
    fps=1.0,
    max_frames=60,
    confidence=0.5,
    video_path="video.mp4",
    output_dir="results"
)
```

### Git Conventions

- Create feature branches for new features
- Write meaningful commit messages
- Run lint/format before committing:

```bash
black src/ tests/
flake8 src/ tests/
pytest tests/
```

### Environment Variables

```bash
# Cloud mode
export GEMINI_API_KEY="your-api-key"

# Local mode
export CUDA_VISIBLE_DEVICES=0
```

### Dependencies

Key dependencies (see `requirements.txt`):

- `torch>=2.0.0` - Deep learning
- `pydantic>=2.0.0` - Data validation
- `ultralytics>=8.0.0` - YOLO detection
- `transformers>=4.35.0` - Qwen model
- `google-genai>=0.2.0` - Gemini API
- `streamlit>=1.28.0` - UI framework

When adding dependencies, update `requirements.txt` and verify compatibility with Python 3.9+.
