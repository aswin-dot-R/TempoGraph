# PS8b — Ship packaging: README, Docker, CI, settings tests (9B lane)

The public face of the release, written in parallel with PS8a's
hardening. Two kinds of work: (A) `tests/test_settings.py` against
PS8a's frozen `Settings` contract, (B) release collateral. Do not touch
PS8a's files.

## Scope fence

- **Files you may create:** `tests/test_settings.py`, `LICENSE`,
  `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`,
  `docs/HARDWARE.md`
- **Files you may modify:** `README.md`
- Touch NOTHING else.

## Forbidden

`sudo`, `systemctl`, `pip install`, `docker build/run` (write the
Dockerfile; the human builds it), git push. Interpreter:
`/home/ashie/anaconda3/bin/python3`.

## Task A — `tests/test_settings.py`

Against PS8a's contract (`pytest.importorskip("src.settings")`):
defaults returned with a clean env (monkeypatch.delenv each var,
raising=False); each env var overrides its field; whisper paths
expanduser'd (no literal `~` in the returned value); `get_settings()`
reflects env changes between calls (no caching).

## Task B — release collateral

1. **LICENSE** — MIT, copyright 2026 the repository owner.
2. **README.md** — rewrite, GIF-first structure (leave a
   `![demo](docs/demo.gif)` placeholder + `TODO(human): record 20-30s
   drop→watch→explore GIF`):
   - Tagline: "The fully-local video understanding pipeline: drop in a
     video, get transcripts, entities, events, a knowledge graph,
     searchable moments, clips, and highlight reels — no cloud, no API
     keys, no per-minute fees."
   - "Open-source Twelve Labs / Azure Video Indexer on your own GPU"
     comparison table (vs Twelve Labs, Azure VI, NVIDIA VSS blueprint,
     byjlw/video-analyzer): local?, open?, transcript, entities/events,
     graph, search, clips, install effort.
   - Quickstart: `make install && make run`, plus the 5-minute no-GPU
     path (`make smoke`, skip-VLM mode).
   - Output contract: the `analysis.json` schema summarized from
     `src/models.py` (entities, visual_events, audio_events,
     correlations, dense_timeline).
   - Configuration table: the TEMPOGRAPH_* env vars from PS8a's
     contract.
   - Architecture diagram (ASCII, from CLAUDE.md's stage list).
3. **docs/HARDWARE.md** — per-stage VRAM/runtime table: YOLO26 n/x
   (CUDA, ~1-4 GB), Depth (transformers, optional), whisper.cpp
   (Vulkan/CPU), VLM + dense captioning (any OpenAI-compatible
   llama-server; sizes 9B/35B as tested), CPU-only fallback matrix
   (skip-VLM + yolo-n). State honestly what was tested on: RTX 3060 6GB
   + 2 AMD GPUs.
4. **Dockerfile** — python:3.12-slim base; system deps ffmpeg + build
   tools; pip requirements; build whisper.cpp (CPU) at
   /opt/whisper.cpp; env TEMPOGRAPH_WHISPER_BIN pointed there; expose
   8501; entrypoint streamlit. VLM stays external
   (TEMPOGRAPH_VLM_URL env at `docker run`) — document that in a
   comment header. `.dockerignore`: results/, *.pt, .git, __pycache__.
5. **.github/workflows/ci.yml** — on push/PR: python 3.12, pip install
   -r requirements.txt + pytest, `pytest -q` (the suite is
   mock-based and needs no GPU; ffmpeg via apt).

## ACCEPTANCE (paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_settings.py --collect-only -q
/home/ashie/anaconda3/bin/python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ci yml ok')"
head -30 README.md
/home/ashie/anaconda3/bin/python3 -m black tests/test_settings.py
git status --porcelain   # only the scoped files
```

(do not commit — gate reviewer merges both lanes)
