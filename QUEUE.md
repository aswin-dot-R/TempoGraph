# QUEUE — Ornith-executed work items → v1.0 ship

Process: Fable (Claude) writes PS files into `docs/ps/`; the user runs
opencode sessions on the Ornith models; Fable gate-reviews diffs and
commits. Parallel-lane items (Na/Nb) launch BOTH at once — 35B lane on
the architect, 9B lane pinned via
`opencode run -m ornith-local/ornith-1.0-9b-Q4_K_M.gguf "..."`. Disjoint
files + frozen contracts mean they cannot conflict. Neither lane
commits; bring both diffs to Fable, the combined suite runs once, both
lanes land in one gate-reviewed commit.

States: `spec'd` → `in-progress` → `gate review (Fable)` → `merged`

## Shipped so far

| # | Item | PS | State |
|---|---|---|---|
| 1 | frame_captions schema + WAL + DB helpers | ps1 | **merged** (2026-07-12) |
| 2 | DenseCaptionWalker (9B per-frame captions) | ps2 | **merged** (2026-07-12) |
| 3 | EscalationVerifier (35B parallel verdicts) | ps3 | **merged** (2026-07-12) |
| 4 | Dense captioning → pipeline, UI, aggregation | ps4 | **merged** (2026-07-12) |
| H1 | Hotfix: optional stages degrade gracefully | hotfix1a/1b | **merged** (2026-07-12) |
| — | VLM retarget to always-on Ornith 9B (:8085) | (Fable direct) | **merged** (2026-07-12) |

## The road to v1.0 (in order; a/b lanes run in parallel)

| # | Item | PS files | State |
|---|---|---|---|
| 5 | **Click-to-play** — every timestamp plays the video | `ps5a.md` (impl, 35B) + `ps5b.md` (tests, 9B) | spec'd |
| 6 | **Natural-language search** — FTS5 over transcript + dense captions + detections + events; show-frame / play-span actions. Depends on 5 (player contract) | `ps6a.md` + `ps6b.md` | spec'd |
| 7 | **Highlight reel** — top-delta spans → 60s summary clip | `ps7a.md` + `ps7b.md` | spec'd |
| 8 | **Ship hardening + packaging** — settings/env config, no hardcoded paths, run-dir collisions fixed, requirements fixed, LICENSE (MIT), GIF-first README with comparison table, HARDWARE.md, Dockerfile, CI | `ps8a.md` (hardening, 35B) + `ps8b.md` (tests + collateral, 9B) | spec'd |
| 9 | **SHIP v1.0** — Fable-led release checklist: full suite + smoke on real footage, README GIF recorded by human, repo scrub (`results/`, `*.pt`, personal paths), squash-review of branch, merge to main, tag v1.0.0, push to public GitHub | (no PS — gate session) | blocked on 5–8 |

Definition of shipped: a stranger with one GPU clones the repo, runs
`make install && make run`, drops a video, and gets transcript, dense
captions, entities/events graph, click-to-play, search, clips, and a
highlight reel — fully local. Positioning (from 2026-07-12 market
research): the only fully-local open-source video→insights pipeline;
"open-source Twelve Labs on your own GPU".

## Post-ship backlog (from TODO.md; PS files written when reached)

| Item | Source | Note |
|---|---|---|
| Cross-run entity registry ("the archive remembers") | TODO 3 | biggest capability leap; unblocks the next two |
| Archive-wide Ask (GraphRAG-lite) | TODO 4 | depends on registry |
| Ethogram v2 (time budgets, transition matrix, bouts) | TODO 5 | the research-niche wedge |
| Activity recognition (rules + sklearn) | TODO 9 | |
| Smart alerts / webhooks | TODO 10 | user-supervised parts flagged |
| Cross-video trends page | TODO 12 | |
| Sound classification (non-speech audio events) | TODO 13 | |
| Daily reports generator | TODO 14 | |
| Shareable HTML widgets | TODO 15 | |
| Live mode + Hermes alerts | TODO 6 | **USER-SUPERVISED — design doc with user first** |

Rules for every item: work only inside the PS scope fence; suite green
after every merge (`/home/ashie/anaconda3/bin/python3 -m pytest -q`);
paste ACCEPTANCE output verbatim; parallel lanes never commit — the gate
reviewer does; SUMMARY.md gets a dated entry per merge.

Design docs: `docs/superpowers/specs/2026-07-12-dense-temporal-captioning-design.md`
(items 1–4, NL-search scope, example queries);
`docs/ps/2026-07-11-click-to-play-ps.md` (item 5);
`docs/ps/hotfix1.md` (root-cause record for H1).
