# PS4 — Wire dense captioning into pipeline, UI, and aggregation

Step 4 of 4 of Dense Temporal Captioning
(design: `docs/superpowers/specs/2026-07-12-dense-temporal-captioning-design.md`).
Requires PS1–PS3 merged. Paste every ACCEPTANCE output into your final
summary.

## Scope fence

- **Files you may modify:** `src/pipeline_v2.py`, `src/aggregator.py`,
  `ui/pages/Results.py`
- **Files you may create:** `tests/test_dense_pipeline.py`
- Touch NOTHING else.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server, allocating GPU memory. Tests mock `run_dense_captioning`
entirely — no HTTP. Interpreter: `/home/ashie/anaconda3/bin/python3`.

## Task 1 — pipeline stage

`PipelineV2.__init__` gains three kwargs, placed next to the existing
`vlm_url` group and stored the same way:

```python
        dense_captions: bool = False,
        walker_url: str = "http://127.0.0.1:8085",
        verifier_url: str = "http://127.0.0.1:8096",
```

In `run()`, add a stage named `"Dense captions"` immediately AFTER the
"YOLO detection" stage block and BEFORE the depth block, gated on
`self.dense_captions`. Follow the existing stage pattern EXACTLY (look at
how "YOLO detection" does it — `_stage("...", "start")`, resume guard via
`db.is_stage_complete(...)` + `_stage(..., "skipped", reason="resumed from
DB")`, work, `_stage(..., "done", **counts)`, `db.mark_stage_complete(...)`
with elapsed seconds and n_units = frames captioned). When the flag is off,
emit `_stage("Dense captions", "skipped")` like the depth stage does.

The work call:

```python
from src.modules.dense_captioner import run_dense_captioning
counts = run_dense_captioning(
    db_path=self.db_path,            # use the same variable the other stages use
    walker_url=self.walker_url,
    verifier_url=self.verifier_url,
    on_progress=lambda info: self._stage("Dense captions", "progress", **info),
    cancel_event=self.cancel_event,  # same attribute the other stages check
)
```

(Verify the real attribute names for the DB path and cancel event by
reading the neighbouring stages first; match them, don't invent new ones.)

## Task 2 — aggregator pass

In `src/aggregator.py`, add to `CaptionAggregator`:

```python
def load_dense_timeline(self, db_path: Path, max_lines: int = 120) -> list:
    """Condensed dense-caption timeline for aggregation and analysis.json.

    Reads frame_captions joined with frames (for timestamp_ms). Keeps every
    escalated row (verifier_caption preferred over caption when
    verifier_agrees == 0) and evenly subsamples the rest so the total is
    <= max_lines. Returns [{"timestamp_ms": int, "text": str,
    "escalated": bool, "verified": bool}] sorted by timestamp.
    """
```

Wire it in wherever the aggregator builds its final `AnalysisResult` dict /
`analysis.json` payload: when the DB has any `frame_captions` rows, include
the list under a top-level key `"dense_timeline"`, and prepend a compact
text block (`"MM:SS text"` lines, escalated lines prefixed with `*`) to the
aggregation LLM prompt context so events/entities can draw on it. If the
table is empty, behavior is byte-for-byte unchanged — every existing test
must pass untouched.

## Task 3 — UI surfacing

`ui/pages/Results.py`:

1. **Frame inspector**: where per-frame details render (near the transcript
   join), if `frame_captions` has a row for the selected frame show:
   caption line; change line (italic); and when `verifier_caption` is not
   NULL a small block "35B second opinion: <verifier_caption>" with an
   agree ✅ / disagree ⚠️ marker from `verifier_agrees`. Guard with
   `db.has_table("frame_captions")` so pre-feature run DBs render exactly
   as before.
2. **Timeline tab**: add dense-caption entries (one per escalated row, text
   = verifier caption if it disagreed else walker caption) alongside the
   existing timeline items, labeled with their MM:SS timestamp. Same
   has_table guard.

## Task 4 — tests (`tests/test_dense_pipeline.py`)

1. **Stage wiring** — monkeypatch
   `src.pipeline_v2.run_dense_captioning` (the name imported into
   pipeline_v2) with a fake that inserts 3 caption rows and returns counts;
   run the pipeline the way `tests/test_pipeline_v2.py` does (copy its
   mocking setup for the other stages) with `dense_captions=True`; assert
   the fake was called once, `run_stages` contains "Dense captions", and a
   second run skips it (resume guard) without calling the fake again.
2. **Flag off** — `dense_captions=False` (default): fake never called, no
   "Dense captions" row in `run_stages` beyond the skipped event.
3. **Aggregator** — fixture DB with 10 caption rows (3 escalated, 1
   disagreed): `load_dense_timeline` keeps all 3 escalated, respects
   max_lines subsampling (set max_lines=5), prefers verifier text on the
   disagreement; and the analysis payload contains `dense_timeline`.
4. **AppTest** — extend the pattern from `tests/test_results_apptest.py`:
   fixture DB with dense captions renders the frame inspector and Timeline
   without exceptions; a fixture DB WITHOUT the table also still renders.

## ACCEPTANCE (run, paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_dense_pipeline.py -q
/home/ashie/anaconda3/bin/python3 -m pytest -q      # full suite green
timeout 12 /home/ashie/anaconda3/bin/python3 -m streamlit run ui/app.py \
  --server.headless true --server.port 8599 & sleep 8; curl -sf localhost:8599 >/dev/null && echo UI_BOOTS
git status --porcelain                               # only the 4 scoped files
```

## Commit message

```
Wire dense captioning into pipeline, Results UI, and aggregation (PS4)
```
