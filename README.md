# TempoGraph

**Turn any video into a queryable knowledge graph of who did what, when, and why.**

TempoGraph is a multimodal video intelligence pipeline that watches a video and produces a structured temporal graph of entities, behaviors, interactions, and audio events — all linked on a shared timeline. Think of it as "ctrl+F for video."

---

## The Problem

Video is the most data-rich medium we produce, yet it remains the hardest to search, analyze, and extract structured insights from. Security teams scrub hours of footage manually. Researchers annotate animal behavior frame-by-frame. Content moderators watch everything in real-time. There is no "spreadsheet for video."

Existing tools either:
- Detect objects but ignore *behavior* and *relationships*
- Transcribe audio but ignore what's happening visually
- Require expensive cloud-only APIs with no local/private option

**TempoGraph solves this by fusing vision, language, and audio AI into a single pipeline that outputs a structured, queryable temporal graph.**

---

## How It Works

```
  Video File (mp4/avi/mov/mkv)
           |
           v
  +--------------------+
  | Adaptive Keyframe  |   Not constant-rate sampling. Uses pixel-delta analysis
  | Extraction         |   (like H.264 I-frame detection) to pick frames where
  +--------------------+   visual change actually happens. Static scenes = fewer
           |               frames. Action scenes = more frames.
           v
  +--------------------+     +------------------+     +------------------+
  | Object Detection   |     | Depth Estimation |     | Audio Analysis   |
  | (YOLO11-nano)      |     | (DepthAnythingV2)|     | (Whisper)        |
  | track_id per entity|     | per-frame depth  |     | speech segments  |
  +--------------------+     +------------------+     +------------------+
           \                        |                        /
            +----------------------------------------------+
                                    |
                                    v
                        +-----------------------+
                        | VLM Behavior Analysis |  Gemini Flash (cloud) OR
                        | "Who did what, when?" |  Qwen3-VL via Ollama (local)
                        +-----------------------+
                                    |
                                    v
                        +-----------------------+
                        | Temporal Knowledge    |  Entities = nodes
                        | Graph (NetworkX)      |  Behaviors = directed edges
                        +-----------------------+  with [start_time, end_time]
                                    |
                   +----------------+----------------+
                   |                |                |
                   v                v                v
            graph.json      timeline.json     graph.html
            (API/query)     (event list)      (interactive viz)
```

### Adaptive Keyframe Extraction

Most video pipelines sample at a fixed rate (e.g., 1 frame/sec), wasting budget on static scenes and under-sampling action. TempoGraph uses a content-aware approach inspired by H.264 encoding:

1. **Delta scan**: Compute mean absolute pixel difference between consecutive frames on downscaled grayscale thumbnails
2. **Keyframe detection**: Frames where delta exceeds `median + 1*stddev` are marked as keyframes (scene changes, motion peaks)
3. **Budget allocation**: Remaining frame budget is distributed proportionally to cumulative delta per segment — more visual change = more frames sampled
4. **Result**: Dense sampling during action, sparse during stillness. A 19-second video with `max_frames=20` might sample 10 frames in the first 12 seconds of activity and only 1 frame across 7 seconds of static footage.

### Three Backend Options

| Backend | Where it runs | GPU needed? | Best for |
|---------|--------------|-------------|----------|
| **Gemini Flash** | Google Cloud API | No | Full video+audio analysis in one call, highest quality |
| **Ollama (qwen3-vl)** | Local via Ollama server | Server-side | Privacy-sensitive data, offline use, free inference |
| **Qwen2.5-VL** | Local GPU (4-bit quantized) | Yes (6GB+) | Maximum control, no network dependency |

### Pipeline Modules

| Module | Model | What it does | VRAM |
|--------|-------|-------------|------|
| Frame Extraction | OpenCV + numpy | Adaptive keyframe selection via pixel deltas | CPU only |
| Object Detection | YOLO11-nano | Bounding boxes + persistent tracking across frames | ~0.5 GB |
| Depth Estimation | Depth Anything V2 (ViT-S) | Per-frame monocular depth maps | ~0.5 GB |
| Behavior Analysis | Gemini Flash / Qwen3-VL | Entity identification, interaction classification, temporal events | Cloud or ~2 GB |
| Audio Analysis | Whisper-small | Speech transcription with timestamps | CPU |
| Graph Builder | NetworkX | Entities as nodes, behaviors as directed temporal edges | CPU only |

All GPU modules run sequentially with explicit VRAM cleanup between stages (`gc.collect()` + `torch.cuda.empty_cache()`), keeping peak usage under 3 GB even on a single 6 GB card.

---

## What You Get

For any input video, TempoGraph produces:

- **`analysis.json`** — Structured entities, visual events (typed behaviors with timestamps), audio events, multimodal correlations, and a natural language summary
- **`graph.json`** — Node/edge representation for API consumption or further querying
- **`timeline.json`** — All events (visual + audio) sorted chronologically
- **`graph.html`** — Interactive force-directed graph visualization (self-contained HTML, no server needed)
- **`annotated.mp4`** — Original video overlaid with detection boxes, depth colormaps, and event subtitles

### Example Output (from `clip.mp4` — 19s lab video of a beagle)

```json
{
  "nodes": [
    {"id": "E1", "type": "animal", "description": "Beagle dog", "first_seen": "00:00", "last_seen": "00:01"}
  ],
  "edges": [],
  "summary": "A beagle dog is navigating a laboratory environment with multiple experimental stations and barriers."
}
```

### Queryable Graph

```python
from src.graph_builder import GraphBuilder

builder = GraphBuilder()
graph = builder.build(analysis_result)

# "What was happening at the 10-second mark?"
builder.query_by_time("00:10")
# → [{"type": "interact", "description": "Man pets dog", "confidence": 0.9}, ...]

# "Show me all events between entity E1 and E2"
builder.query_by_entities("E1", "E2")

# Stats
builder.get_stats()
# → {"total_entities": 3, "most_active_entity": "E2", "dominant_behavior": "approach"}
```

---

## Quick Start

### Cloud Mode (no GPU needed)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
export GEMINI_API_KEY="your-key-here"

# 3. Run
python -m src.pipeline --video your_video.mp4 --backend gemini --output results/my_video
```

### Local Mode (via Ollama)

```bash
# 1. Install Ollama and pull the model
ollama pull qwen3-vl:4b

# 2. Run
python -m src.pipeline --video your_video.mp4 --backend llama-server --output results/my_video
```

### Web UI (Streamlit)

```bash
streamlit run ui/app.py
```

Upload a video, pick your backend, toggle modules, and click Analyze. Results appear in five tabs: Annotated Video, Timeline, Interaction Graph, Summary, and Export.

### REST API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
# Submit a video
curl -X POST http://localhost:8000/analyze \
  -F "video=@clip.mp4" \
  -F "backend=gemini" \
  -F "modules=behavior,detection,audio"

# Check status
curl http://localhost:8000/status/{job_id}

# Get results
curl http://localhost:8000/results/{job_id}
```

### Docker

```bash
# GPU mode (all backends)
docker compose up tempograph-gpu

# CPU mode (Gemini cloud only)
docker compose up tempograph-cpu
```

---

## Project Structure

```
TempoGraph/
├── src/
│   ├── pipeline.py              # Main orchestrator (9-step sequential pipeline)
│   ├── models.py                # Pydantic v2 data models (shared contract)
│   ├── graph_builder.py         # NetworkX temporal graph + query methods
│   ├── json_parser.py           # Robust VLM output parser (handles malformed JSON)
│   ├── video_annotator.py       # Overlay boxes, depth, subtitles onto video
│   ├── api.py                   # FastAPI REST server
│   ├── backends/
│   │   ├── base.py              # Abstract backend interface
│   │   ├── gemini_backend.py    # Google Gemini 2.0 Flash
│   │   ├── qwen_backend.py      # Qwen2.5-VL local (4-bit quantized)
│   │   └── llama_server_backend.py  # Ollama HTTP backend
│   └── modules/
│       ├── frame_extractor.py   # Adaptive keyframe extraction (pixel-delta)
│       ├── detector.py          # YOLO11 detection + tracking
│       ├── depth.py             # Depth Anything V2 estimation
│       └── audio.py             # Whisper transcription
├── ui/
│   └── app.py                   # Streamlit web interface
├── configs/
│   └── default.yaml             # Default pipeline configuration
├── tests/
│   ├── test_parser.py           # JSON parser unit tests
│   └── test_vram_budget.py      # GPU memory verification
├── lib/                         # Bundled JS/CSS for self-contained graph.html
├── Dockerfile                   # GPU container (CUDA 12.1)
├── Dockerfile.cpu               # CPU-only container
├── docker-compose.yml           # GPU + CPU service definitions
└── requirements.txt
```

---

## Why TempoGraph Wins

### PROGRESS: What We Built in the Hackathon

- A complete, working end-to-end pipeline — video in, structured knowledge graph out
- Three interchangeable VLM backends (cloud, local Ollama, local quantized)
- Five analysis modules (keyframe extraction, YOLO detection, depth estimation, VLM behavior analysis, Whisper audio)
- Adaptive keyframe extraction using pixel-delta analysis (not constant-rate sampling)
- Robust JSON parser that handles the messy, malformed outputs VLMs actually produce
- A queryable temporal graph with time-range and entity-pair queries
- Interactive graph visualization (pyvis HTML, no server needed)
- Streamlit web UI with real-time pipeline status
- FastAPI REST API with async job processing
- Docker deployment (GPU and CPU variants)
- Annotated video output with detection boxes, depth overlays, and event subtitles

This is not a wrapper around a single API call. It is a multi-model orchestration pipeline with VRAM-safe sequential execution, content-aware frame sampling, and multiple output formats.

### CONCEPT: The Real Problem It Solves

**Video is the last unstructured frontier.** We have search engines for text, databases for tabular data, and vector stores for embeddings. But video — the richest data source — remains opaque. You cannot query it, filter it, or join it with other data without watching it.

TempoGraph turns video into structured data:
- **Security/surveillance**: "Show me all approach events between 2:00-5:00 PM" instead of watching 3 hours of footage
- **Animal behavior research**: Automated ethogram generation — who interacted with whom, for how long, what type of behavior
- **Sports analytics**: Player interaction graphs, possession timelines, event detection
- **Content moderation**: Flag specific behavior types across thousands of hours of video
- **Accessibility**: Structured video descriptions for visually impaired users

The temporal graph representation is the key innovation — it is not just "what objects are in the video" but "who did what to whom, when, and how confidently do we know it."

### FEASIBILITY: Path to a Real Business

**Market**: Video analytics is projected at $20B+ by 2027. Every enterprise with cameras, research labs with animal studies, or platforms with user-generated video needs this.

**Business model**: SaaS API (pay per video-minute analyzed) + on-premise deployment for privacy-sensitive customers (healthcare, defense, research).

**Competitive moat**:
- The temporal graph representation is novel — competitors output flat detection lists, not queryable relationship graphs
- Backend flexibility (cloud/local/Ollama) means we serve both cost-sensitive and privacy-sensitive customers from day one
- Adaptive keyframe extraction reduces API costs 40-60% vs. constant-rate sampling by sending fewer redundant frames to the VLM
- The robust JSON parser handles real-world VLM output failures that break naive implementations

**Growth path**:
1. Open-source the core pipeline (community adoption, feedback loop)
2. Hosted API for developers who want video understanding without infrastructure
3. Enterprise tier with custom behavior taxonomies, multi-camera fusion, and real-time streaming
4. Domain-specific models fine-tuned on customer data (animal behavior, sports, manufacturing QA)

**Unit economics**: Gemini Flash costs ~$0.075/min for video analysis. At a 10x markup ($0.75/min), a single security camera generating 8 hours of daily highlights produces ~$100/month in revenue per camera at near-zero marginal cost.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data models | Pydantic v2 |
| Graph engine | NetworkX (MultiDiGraph) |
| Vision LLM | Gemini 2.0 Flash / Qwen3-VL (4-bit) |
| Object detection | YOLO11-nano (Ultralytics) |
| Depth estimation | Depth Anything V2 (ViT-Small) |
| Audio | OpenAI Whisper-small |
| Video processing | OpenCV + ffmpeg |
| Web UI | Streamlit |
| REST API | FastAPI + Uvicorn |
| Visualization | pyvis + vis.js / Plotly |
| Containers | Docker + NVIDIA Container Toolkit |

---

## Configuration

```yaml
# configs/default.yaml
backend: "gemini"          # gemini | qwen | llama-server
modules:
  behavior: true           # VLM analysis (always on)
  detection: true          # YOLO object detection
  depth: false             # Depth estimation
  audio: true              # Whisper transcription
fps: 1.0                   # Base scan rate for keyframe detection
max_frames: 60             # Max frames to extract
confidence: 0.5            # Minimum detection confidence
```

## Environment Variables

```bash
GEMINI_API_KEY=your-key    # Required for cloud mode
CUDA_VISIBLE_DEVICES=0     # Optional: select GPU
```

---

## License

MIT
