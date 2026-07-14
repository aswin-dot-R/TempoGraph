# Changelog

All notable changes to TempoGraph are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
adapted for this project.

## [1.0.0] — 2026-07-14

### Added

- Fully-local multimodal video-analysis pipeline: YOLO26 detection, Depth Anything V2, whisper.cpp ASR, dense per-frame VLM captioning with an EscalationVerifier that escalates to a larger model when needed, chunked VLM captioning, LLM aggregation into entities/events/correlations, and a pyvis knowledge-graph export.
- Per-run SQLite store with resume-from-DB stage guards; Streamlit Drop -> Watch -> Explore UI with a results browser (frame/entity inspector, timeline, annotated video, graph, clips, ask, ethogram, dataset export).
- Parallel dense captioning with a live processing view; the DenseCaptionWalker auto-detects server slots from `/props`.
- Graph-driven clip export via `src/clip_export.py`, highlight reel generation (most interesting 60 s, auto-cut) via `src/highlight_reel.py`, natural-language search (FTS5 over everything a human would look up) in `src/search.py`, mask RLE persistence via the `mask_rle` column in the `detections` table, and click-to-play timestamps.
- `src/models.py` Pydantic v2 models: `Entity`, `VisualEvent`, `AudioEvent`, `AnalysisResult`, `DetectionBox`.
- `src/storage.py` `TempoGraphDB` — SQLite schema + helpers (WAL + busy_timeout).
- `src/auto_profile.py` — `probe(path) → VideoFacts` + `derive_plan(facts) → DerivedPlan`.
- `src/summarizer.py` — injectable LLM callable for narratives.

### Changed

- Dense stage now fails on total caption failure rather than marking the stage complete (retry on resume).
- Dense stage no longer treats a single total-caption failure as stage-complete — stage is retried on resume.
- Aggregation prompt now fits the server context (previously overflowed small slots and produced zero entities).
- Optional stages (depth, VLM, aggregation) degrade gracefully instead of killing a real-video run.
- Retargeted VLM captioning and summarizer to always-on Ornith 9B at `http://127.0.0.1:8085`.
- Purged retired qwen35 service from error hints; batch/CLI VLM defaults now read from settings.
- Walker auto-detects server slots via GET `/props` (kwarg > env > probe > default).
- Dark-first CSS theme with hero drop zone, styled cards, and checklist UI facelift.
- Env settings moved into `src/settings.py` (zero-dep, re-reads each call).
- Portable paths and `LICENSE` added.

### Fixed

- Resume path now returns a real `AnalysisResult` instead of a duck-typed stand-in.
- First real-run failures fixed: switch_page path, dense captions wired into UI, aggregation retry.
- Aggregation prompt context overflow fixed — no more zero-entity results on small-context servers.

### Testing

- Test suite: 265 passing tests (`pytest tests/`).
