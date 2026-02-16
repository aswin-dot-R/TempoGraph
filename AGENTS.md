# TempoGraph - Agent Coding Guidelines

## Overview

TempoGraph is a multimodal video analysis pipeline that extracts temporal behaviors, interactions, and correlations from video content using AI models. It supports dual backends: cloud (Gemini) and local (Qwen2.5-VL-3B).

## Project Structure

```
TempoGraph/
├── src/
│   ├── backends/          # VLM backends (base.py, gemini_backend.py, qwen_backend.py)
│   ├── modules/           # Analysis modules (audio.py, depth.py, detector.py, frame_extractor.py)
│   ├── models.py          # Pydantic data models
│   ├── pipeline.py        # Main pipeline orchestrator
│   ├── json_parser.py     # JSON parsing utilities
│   ├── graph_builder.py   # NetworkX graph builder
│   └── video_annotator.py # Video annotation
├── ui/
│   └── app.py             # Streamlit UI
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
flake8 src/pipeline.py
```

### Running the Pipeline

```bash
# Cloud mode with Gemini
python -m src.pipeline --video sample.mp4 --backend gemini

# Local mode with Qwen
python -m src.pipeline --video sample.mp4 --backend qwen --modules behavior,detection,audio

# Streamlit UI
streamlit run ui/app.py
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
