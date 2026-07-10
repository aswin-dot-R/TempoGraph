# Follow-up PS: finish resume-from-DB + wire the summarizer

You are working unattended in this git worktree on branch `ui-v3-dropflow`,
continuing your previous session's work (see `SUMMARY.md`). Two items from
the original PS (`docs/ps/2026-07-10-ui-dropflow-ps.md`) were left
unfinished. Independent review confirmed everything else passed. Same
environment facts and FORBIDDEN list as the original PS apply — reread that
section now.

This time every requirement has an integration-level test. A component that
exists but is not called does not count as done.

## Task 1 — Resume-from-DB stage guards

`src/storage.py` already provides `is_stage_complete` / `mark_stage_complete`
and the `run_stages` table. `src/pipeline_v2.py:run()` never calls them.

Required:
- At the start of each stage in `PipelineV2.run()`, if
  `is_stage_complete(stage_name)` for this run DB, log a skip line and move
  on; after each successful stage, `mark_stage_complete(stage_name)`.
- Works with the cancel path: a cancelled run resumed later skips only the
  stages that finished.

Tests (`tests/test_resume.py`):
- Build a run DB where stage A is marked complete; run the pipeline with all
  stage functions mocked; assert stage A's mock was NOT called and stage B's
  was.
- Fresh DB: all stage mocks called, all stages marked complete afterwards.

## Task 2 — Wire `src/summarizer.py` into the Results Overview tab

The module exists with unit tests but nothing calls it. Required:
- Results Overview tab renders a "Summary" section: call
  `generate_summary()` over the run's aggregated captions + transcript.
- LLM callable resolution: if the configured llama backend
  (`src/backends/llama_server_backend.py`) is reachable, pass its completion
  callable; otherwise pass none so the heuristic path renders. NEVER start a
  server yourself; reachability = a 2-second HTTP health probe, nothing more.
- Cache the generated summary in the run DB (new `run_meta` key or similar)
  so revisiting a run does not regenerate it.

Tests (append to `tests/test_ui_dropflow.py` or new file):
- AppTest with a fixture run DB and a FAKE injected LLM callable: assert the
  Overview tab renders the fake's summary text verbatim.
- Same fixture, no LLM: assert the heuristic summary text renders (non-empty,
  contains at least one entity name from the fixture DB).
- Unit: summary caching — second render does not call the LLM callable again.

## ACCEPTANCE — all must pass, run them yourself

```bash
cd /home/ashie/TempoGraph-ui-v3
PY=/home/ashie/anaconda3/bin/python3
$PY -m pytest -q                       # whole suite green (57 existing + new)
$PY -m pytest tests/test_resume.py -q
grep -q "is_stage_complete" src/pipeline_v2.py   # guards actually wired
grep -qE "summarizer|generate_summary" ui/pages/Results.py  # actually called
timeout 12 $PY -m streamlit run ui/app.py --server.headless true \
  --server.port 8599 & sleep 8 && curl -sf localhost:8599 >/dev/null && echo UI_BOOTS
```

## Deliverables

- Commits on this branch, one per task.
- Append a "## Follow-up (resume + summarizer)" section to `SUMMARY.md`
  with pasted acceptance output and any known gaps.
