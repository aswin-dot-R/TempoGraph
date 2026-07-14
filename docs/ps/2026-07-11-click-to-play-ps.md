# Feature PS: every timestamp is a link — click transcript, watch the moment

You are working unattended in this git worktree on branch `ui-v3-dropflow`.
Same environment facts and FORBIDDEN list as
`docs/ps/2026-07-10-ui-dropflow-ps.md` — reread that section now. The
suite currently has 115 passing tests; it must stay green. Commit per
stage; a component that exists but is not called does not count as done;
finish with `git status --porcelain` empty and `git log` showing your
commits.

## The feature

TempoGraph finds the moment; the user still can't watch it. Today only
the events timeline is clickable, and it seeks a 4 fps annotated strip
with proportional time mapping. Required end state — one rule across the
whole Results page:

> **Anything that shows a timestamp plays the video at that timestamp,
> in one click.**

That means: transcript segment rows, caption entries, visual-event rows
(timeline AND any event tables), and the frame inspector's current frame
all become play links into the **original source video**, falling back to
the annotated strip when the source is missing.

## Design

1. `ui/video_player.py` (NEW, streamlit-free logic + thin render fn):
   - `resolve_video(run_dir, db) -> VideoSource`: original path from
     `run_meta.video_path` (persisted since the clip-export work) if the
     file still exists; else `annotated_strip.mp4` with a
     `time_mapper(source_s) -> strip_s` (reuse the proportional mapping
     already in `_render_interactive_timeline`); else None.
   - `render_player(source, start_s)`: `st.video(..., start_time=int(s))`
     plus a caption stating what is playing ("source video @ 03:12" /
     "annotated strip (source file missing)").
2. A single **sticky player pattern**: the Results page holds one player
   slot near the top of the active tab (session_state keys
   `player_start_s`, `player_requested`). Any click sets state and
   `st.rerun()`; the slot renders the player. No duplicate players per
   row.
3. Make the timestamps clickable:
   - Transcript (`_render_captions`): replace the static segments
     dataframe with rows that each carry a ▶ button (columns layout:
     ▶ | start–end | text). Keep the full-text block and metrics.
   - Captions/keyframe entries and visual-event listings: same ▶ pattern
     wherever a start time is displayed.
   - Frame inspector: a "▶ Play from here" button using the frame's
     `timestamp_ms`.
   - Events plotly timeline: keep existing behaviour but route its click
     through the same state (and prefer the original video now).
4. Long-transcript ergonomics: paginate or virtualise segment rows past
   200 segments (simple "show more" expander pages of 100) so the page
   stays responsive.

## Stages (commit after each)

1. `video_player.py` + unit tests: resolve order (original present →
   original; missing → strip + mapper; neither → None); mapper math
   matches the existing proportional logic on a fixture.
2. Transcript rows clickable + AppTest: on a fixture DB with audio
   segments, clicking segment N's ▶ sets `player_start_s` to that
   segment's start and a video element renders (fixture original video =
   tiny generated mp4 whose path is written into run_meta).
3. Captions, events, frame inspector wired to the same state + AppTests
   for each surface (one test per surface minimum).
4. Pagination for long transcripts + test (250-segment fixture → first
   page shows 100, "show more" reveals next page).
5. SUMMARY.md append ("## Click-to-play") with pasted acceptance output.

## ACCEPTANCE — all must pass

```bash
cd /home/ashie/TempoGraph-ui-v3
PY=/home/ashie/anaconda3/bin/python3
$PY -m pytest -q                                  # 115 existing + new, green
$PY -m pytest tests/test_click_to_play.py -q      # the new surface tests
grep -q "video_player" ui/pages/Results.py        # actually wired
grep -qE "start_time" ui/video_player.py
timeout 12 $PY -m streamlit run ui/app.py --server.headless true \
  --server.port 8599 & sleep 8 && curl -sf localhost:8599 >/dev/null && echo UI_BOOTS
git status --porcelain | wc -l | grep -qx 0
```

## Notes

- `st.video` `start_time` seeks on load — that is sufficient; do NOT
  reach for custom JS/components.
- Original-video resolution must tolerate the file having moved: never
  crash, always fall back with the honest caption.
- Do not regress the Clips section or the existing timeline click path.
