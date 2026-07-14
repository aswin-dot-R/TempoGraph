# Pre-review checklist (run this with Ornith 9B after every PS)

You are reviewing a completed work item, NOT implementing it. You get: the
PS file, `git diff`, `git status --porcelain`, and the pasted ACCEPTANCE
output. Answer each check with PASS or FAIL plus one line of evidence.
Any FAIL = send the work back before it reaches gate review.

1. **Scope fence** — every file in `git status --porcelain` is listed in
   the PS's "Scope fence". No stray files, no unrelated reformatting in
   the diff.
2. **Acceptance ran** — the summary contains real pasted output for EVERY
   command in the PS's ACCEPTANCE block (not paraphrased, not "it passed").
3. **Suite green** — the full-suite pytest line shows `0 failed` and at
   least as many passed as before the item started.
4. **Tests exist** — every test file the PS names was created and appears
   in the diff with real test functions (not empty stubs, no
   `pytest.skip` on everything).
5. **No forbidden actions** — the summary/diff shows no `pip install`,
   `systemctl`, `sudo`, git push, or new dependencies in
   `requirements.txt`.
6. **Docstrings + type hints** — new public functions/classes in the diff
   have Google-style docstrings and type hints (spot-check three).
7. **SUMMARY.md** — a dated entry for this item was appended.
8. **Commit message** — matches the one at the bottom of the PS.

Output format:

```
1 scope-fence: PASS — files: ...
2 acceptance: ...
...
VERDICT: READY FOR GATE REVIEW | SEND BACK (reasons)
```
