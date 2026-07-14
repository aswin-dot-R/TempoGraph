# TempoGraph

**The fully-local video understanding pipeline: drop in a video, get transcripts, entities, events, a knowledge graph, searchable moments, clips, and highlight reels — no cloud, no API keys, no per-minute fees.**

Open-source Twelve Labs / Azure Video Indexer on your own GPU.

---

## Quickstart

```bash
make install && make run
```

That's it. Drag a video in, click **Run full pipeline**, and get structured
output in ~5 minutes.

For a **5-minute no-GPU path** (skip VLM, use tiny YOLO):

```bash
make install && make smoke
```

The smoke test synthesizes a 10-second video and runs the pipeline end-to-end
without requiring a local LLM — perfect for verifying your install.

---

## What you get — output contract

Every run writes `analysis.json` with this schema:

```json
{
  "entities": [
    { "id": "E1", "type": "person", "description": "...", "first_seen": "MM:SS", "last_seen": "MM:SS" }
  ],
  "visual_events": [
    { "type": "approach", "entities": ["E1", "E2"], "start_time": "MM:SS", "end_time": "MM:SS", "description": "...", "confidence": 0.7 }
  ],
  "audio_events": [
    { "type": "speech", "start_time": "MM:SS", "end_time": "MM:SS", "speaker": "...", "text": "...", "label": "...", "emotion": "...", "confidence": 0.8 }
  ],
  "multimodal_correlations": [
    { "visual_event": 0, "audio_event": 0, "description": "..." }
  ],
  "summary": "..."
}
```

Plus: per-run SQLite (`tempograph.db`), saved frames, depth maps,
`transcript.json`, `chunks.json`, `graph.html`.

---

## Comparison: open vs. cloud video understanding

| Feature | TempoGraph | Twelve Labs | Azure Video Indexer | NVIDIA VSS blueprint | byjlw/video-analyzer |
|---|---|---|---|---|---|
| Local? | ✅ Yes | ❌ Cloud API | ❌ Cloud API | ❌ Cloud API | ❌ Cloud API |
| Open-source? | ✅ MIT | ❌ Closed | ❌ Closed | ❌ Closed | ❌ Closed |
| Transcripts | ✅ Whisper.cpp | ✅ | ✅ | ✅ | ✅ |
| Entities / events | ✅ VLM + aggregation | ✅ | ✅ | ✅ | ✅ |
| Knowledge graph | ✅ pyvis HTML | ❌ | ✅ | ❌ | ❌ |
| Searchable moments | ✅ Plotly timeline | ✅ | ✅ | ✅ | ✅ |
| Clips / highlight reels | ✅ Streamlit UI | ✅ | ✅ | ✅ | ✅ |
| Install effort | `make install` (~5 min) | API key + quota | Azure account | Azure account | API key + quota |

**Bottom line:** TempoGraph gives you the same analytical surface area as
enterprise solutions, runs entirely offline, costs nothing per minute, and
lets you inspect every stage.

---

## Configuration

Set these environment variables to customize defaults:

| Variable | Default | Description |
|---|---|---|
| `TEMPOGRAPH_VLM_URL` | `http://127.0.0.1:8085` | llama-server address for Qwen3.5-VL |
| `TEMPOGRAPH_VLM_MODEL` | `ornith-1.0-9b-Q4_K_M.gguf` | Model name in the model directory |
| `TEMPOGRAPH_WALKER_URL` | (inherits VLM_URL) | Optional walker service |
| `TEMPOGRAPH_VERIFIER_URL` | `http://127.0.0.1:8096` | Optional verifier service |
| `TEMPOGRAPH_WHISPER_BIN` | `~/whisper.cpp/build/bin/whisper-cli` | Path to whisper binary (expanded) |
| `TEMPOGRAPH_WHISPER_MODELS` | `~/whisper.cpp/models` | Whisper model directory (expanded) |
| `TEMPOGRAPH_RESULTS_DIR` | `results` | Output directory |

See [`docs/HARDWARE.md`](docs/HARDWARE.md) for per-stage VRAM/runtime
breakdown.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                              │
│              ui/app.py + ui/pages/Results.py                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼────────┐
                    │  pipeline_v2.py  │
                    │  (chunked runner)│
                    └─────────┬────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                     ▼
    ┌─────────────┐  ┌─────────────┐  ┌────────────────┐
    │ Stage 1.5   │  │ Stage 2     │  │ Stage 3         │
    │ Audio (CPU) │  │ YOLO26     │  │ Depth (opt.)    │
    └──────┬──────┘  └──────┬──────┘  └───────┬─────────┘
            │              │                   │
            │  ┌───────────┼───────────┐       │
            │  ▼           ▼           ▼       │
            │  Stage 4     Stage 4     Stage 4  │
            │  Frame      Frame       Frame     │
            │  Scoring     Scoring    Scoring    │
            └──────────────┼──────────────┘     │
                           ▼                     │
                    ┌─────────────┐              │
                    │  Stage 5   │  ◄── Qwen3.5-VL│
                    │  VLM Chunk│  llama-server  │
                    └──────┬─────┘               │
                           │                     │
                           ▼                     │
                    ┌─────────────┐               │
                    │  Stage 6   │  ◄── aggregation│
                    │  Aggregate │  (Qwen3.5 text)│
                    └──────┬─────┘               │
                           │                      │
                    ┌──────▼──────┐                │
                    │  results/   │                │
                    │  tempograph.db│              │
                    │  analysis.json│              │
                    │  graph.html  │              │
                    └─────────────┘                │
                                                   │
                                                   ▼
                                         ┌─────────────┐
                                         │  Results UI  │
                                         │  8 tabs     │
                                         │  Overview   │
                                         │  Frame insp │
                                         │  Entity insp│
                                         │  VLM outputs│
                                         │  Captions   │
                                         │  Timeline   │
                                         │  Annotated  │
                                         │  Files      │
                                         └─────────────┘
```
