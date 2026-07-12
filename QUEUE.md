# QUEUE — Ornith-executed work items

Process: Fable (Claude) writes one PS per item into `docs/ps/`; the user
runs an opencode session on **Ornith 35B** with the PS as the task; then a
quick **Ornith 9B** session runs `docs/ps/review-checklist.md` against the
diff; then the user brings the diff + pasted ACCEPTANCE output back to
Fable for gate review before committing with the PS's commit message.

States: `spec'd` → `in-progress` → `pre-review (9B)` → `gate review (Fable)` → `merged`

| # | Item | PS file | State |
|---|---|---|---|
| 1 | frame_captions schema + WAL + DB helpers | `docs/ps/ps1.md` | **merged** (`52e71f5`, 2026-07-12) |
| 2 | DenseCaptionWalker (9B, per-frame captions + change lines) | `docs/ps/ps2.md` | **merged** (2026-07-12) |
| 3 | EscalationVerifier (35B, parallel second opinions) | `docs/ps/ps3.md` | **merged** (2026-07-12) |
| 4 | Pipeline stage + Results UI + aggregator pass | `docs/ps/ps4.md` | **merged** (2026-07-12) |
| 5a | Click-to-play IMPLEMENTATION (35B lane) | `docs/ps/ps5a.md` | spec'd — run in parallel with 5b |
| 5b | Click-to-play TEST SUITE (9B lane) | `docs/ps/ps5b.md` | spec'd — run in parallel with 5a |
| 6 | NL search — FTS5 over transcript + dense captions + detections + events, Ornith-assisted semantic layer, show-frame / play-span result actions | — | queued |
| 7 | Highlight reel auto-generation | — | queued |
| 8 | Open-source prep — LICENSE, GIF-first README, Docker, VRAM table, positioning "open-source Twelve Labs on your own GPU" | — | queued |

**Parallel-lane items (5a/5b):** launch BOTH at once — one opencode
session on the 35B architect with ps5a, one session pinned to the 9B
(`opencode run -m ornith-local/ornith-1.0-9b-Q4_K_M.gguf "..."`) with
ps5b. Disjoint files + frozen shared interfaces, so they cannot conflict.
Neither lane commits; when both finish, bring both diffs to Fable — the
combined suite runs once and both lanes land in one gate-reviewed commit.
(Item 5's original design doc: `docs/ps/2026-07-11-click-to-play-ps.md`.)

Rules that apply to every item (also stated in each PS): work only inside
the PS scope fence; suite must be green after every item
(`/home/ashie/anaconda3/bin/python3 -m pytest -q`); paste ACCEPTANCE output
verbatim; one commit per item with the given message; append a dated entry
to `SUMMARY.md`.

Design doc for items 1–4:
`docs/superpowers/specs/2026-07-12-dense-temporal-captioning-design.md`.
Market research (positioning for item 8) is summarized in that doc's
"Why" and "Queue" sections.
