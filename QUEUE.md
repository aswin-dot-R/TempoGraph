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
| 2 | DenseCaptionWalker (9B, per-frame captions + change lines) | `docs/ps/ps2.md` | spec'd |
| 3 | EscalationVerifier (35B, parallel second opinions) | `docs/ps/ps3.md` | spec'd |
| 4 | Pipeline stage + Results UI + aggregator pass | `docs/ps/ps4.md` | spec'd |
| 5 | Click-to-play (timestamps become video links) | `docs/ps/2026-07-11-click-to-play-ps.md` (needs Ornith-sizing pass by Fable) | queued |
| 6 | NL search — FTS5 over transcript + dense captions + detections + events, Ornith-assisted semantic layer, show-frame / play-span result actions | — | queued |
| 7 | Highlight reel auto-generation | — | queued |
| 8 | Open-source prep — LICENSE, GIF-first README, Docker, VRAM table, positioning "open-source Twelve Labs on your own GPU" | — | queued |

Rules that apply to every item (also stated in each PS): work only inside
the PS scope fence; suite must be green after every item
(`/home/ashie/anaconda3/bin/python3 -m pytest -q`); paste ACCEPTANCE output
verbatim; one commit per item with the given message; append a dated entry
to `SUMMARY.md`.

Design doc for items 1–4:
`docs/superpowers/specs/2026-07-12-dense-temporal-captioning-design.md`.
Market research (positioning for item 8) is summarized in that doc's
"Why" and "Queue" sections.
