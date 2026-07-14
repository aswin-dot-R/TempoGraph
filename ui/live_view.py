"""Live captioning view: real-time frame transcript + 35B verdicts.

Provides:
- ``fetch_live_state(db_path, n)`` — pure-data fetch (testable without Streamlit)
- ``render_live_view(db_path)`` — Streamlit UI for the live view

The live view shows the most recent ``n`` captioned frames in reverse order,
with their 35B verdicts (from the PS3 verifier).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


def fetch_live_state(db_path: Path, n: int = 5) -> Optional[dict]:
    """Fetch the most recent ``n`` captioned frames from the database.

    Opens the DB read-only (sqlite3 URI mode). Missing file / missing table
    / empty table → None.

    Returns:
        {
            "current": {frame_idx, timestamp_ms, image_path, caption,
                         change_line, prompt, escalated},
            "recent":  [same-shape dicts, newest first, up to n],
            "transcript": [ {start_ms, end_ms, text} ... ]   # segments
                          overlapping [current.timestamp_ms - 1000,
                                       current.timestamp_ms + 1000]
            "verdicts": [ {frame_idx, verifier_caption, verifier_agrees,
                           verifier_model} ... ]  # last 5 verified,
                        newest verified_at first
            "counts": {"captioned": int, "escalated": int, "verified": int},
        }

    Returns None if the DB file is missing, the table is missing, or the
    table is empty.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check required tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('frames', 'frame_captions')"
        )
        if not cursor.fetchone():
            conn.close()
            return None

        # Get current row (highest created_at, newest verified_at first)
        current_row = None
        try:
            cursor.execute(
                "SELECT fc.frame_idx, fc.caption, fc.change_line, fc.escalated, "
                "fc.walker_model, fc.prompt, "
                "fc.verifier_caption, fc.verifier_agrees, fc.verifier_model, "
                "fc.verified_at, f.timestamp_ms, f.image_path "
                "FROM frame_captions fc "
                "JOIN frames f ON f.frame_idx = fc.frame_idx "
                "ORDER BY fc.created_at DESC, fc.verified_at DESC "
                "LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                current_row = dict(row)
        except sqlite3.OperationalError:
            pass

        if not current_row:
            conn.close()
            return None

        # Build current dict
        current = {
            "frame_idx": current_row["frame_idx"],
            "timestamp_ms": current_row["timestamp_ms"],
            "image_path": current_row["image_path"],
            "caption": current_row["caption"] or "(no caption)",
            "change_line": current_row["change_line"] or "(no change)",
            "prompt": current_row["prompt"] or "",
            "escalated": (
                bool(current_row["escalated"]) if current_row["escalated"] else False
            ),
        }

        # Transcript: audio segments overlapping ±1 s around current timestamp
        try:
            start_ms = current_row["timestamp_ms"] - 1000
            end_ms = current_row["timestamp_ms"] + 1000
            cursor.execute(
                "SELECT start_ms, end_ms, text FROM audio_segments "
                "WHERE start_ms < ? AND end_ms > ? "
                "ORDER BY start_ms ASC",
                (end_ms, start_ms),
            )
            transcript = [
                {
                    "start_ms": r["start_ms"],
                    "end_ms": r["end_ms"],
                    "text": r["text"],
                }
                for r in cursor.fetchall()
            ]
        except sqlite3.OperationalError:
            transcript = []

        # Recent frames (newest first, up to n)
        try:
            cursor.execute(
                "SELECT fc.frame_idx, fc.caption, fc.change_line, fc.escalated, "
                "fc.walker_model, fc.prompt, "
                "fc.verifier_caption, fc.verifier_agrees, fc.verifier_model, "
                "fc.verified_at, f.timestamp_ms, f.image_path "
                "FROM frame_captions fc "
                "JOIN frames f ON f.frame_idx = fc.frame_idx "
                "ORDER BY fc.created_at DESC, fc.verified_at DESC "
                "LIMIT ?",
                (n,),
            )
            recent = []
            for row in cursor.fetchall():
                recent.append(
                    {
                        "frame_idx": row["frame_idx"],
                        "timestamp_ms": row["timestamp_ms"],
                        "image_path": row["image_path"],
                        "caption": row["caption"] or "(no caption)",
                        "change_line": row["change_line"] or "(no change)",
                        "prompt": row["prompt"] or "",
                        "escalated": (
                            bool(row["escalated"]) if row["escalated"] else False
                        ),
                        "verifier_caption": row["verifier_caption"] or "",
                        "verifier_agrees": (
                            bool(row["verifier_agrees"])
                            if row["verifier_agrees"]
                            else False
                        ),
                        "verifier_model": row["verifier_model"] or "",
                        "verified_at": row["verified_at"] or "",
                    }
                )
        except sqlite3.OperationalError:
            recent = []

        # Last 5 verified rows, newest verified_at first
        verdicts = []
        try:
            cursor.execute(
                "SELECT fc.frame_idx, fc.verifier_caption, fc.verifier_agrees, "
                "fc.verifier_model, fc.verified_at "
                "FROM frame_captions fc "
                "WHERE fc.verifier_agrees IS NOT NULL "
                "ORDER BY fc.verified_at DESC "
                "LIMIT 5"
            )
            for row in cursor.fetchall():
                verdicts.append(
                    {
                        "frame_idx": row["frame_idx"],
                        "verifier_caption": row["verifier_caption"] or "",
                        "verifier_agrees": (
                            bool(row["verifier_agrees"])
                            if row["verifier_agrees"]
                            else False
                        ),
                        "verifier_model": row["verifier_model"] or "",
                        "verified_at": row["verified_at"] or "",
                    }
                )
        except sqlite3.OperationalError:
            verdicts = []

        # Counts
        try:
            cursor.execute("SELECT COUNT(*) FROM frame_captions")
            total = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            total = 0

        try:
            cursor.execute("SELECT COUNT(*) FROM frame_captions WHERE escalated = 1")
            escalated_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            escalated_count = 0

        try:
            cursor.execute(
                "SELECT COUNT(*) FROM frame_captions WHERE verified_at IS NOT NULL"
            )
            verified_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            verified_count = 0

        counts = {
            "captioned": total,
            "escalated": escalated_count,
            "verified": verified_count,
        }

        conn.close()

        return {
            "current": current,
            "recent": recent,
            "transcript": transcript,
            "verdicts": verdicts,
            "counts": counts,
        }
    except Exception:
        return None


def render_live_view(db_path: Path) -> None:
    """Streamlit rendering of fetch_live_state.

    Shows the current frame image (st.image), caption + change line, an expander
    with the exact prompt, transcript ±1 s under the image, a second column
    streaming the last verifier verdicts (✓ agree / ✗ override + 35B caption),
    and a small trailing feed of the previous captions. Handles
    fetch_live_state(...) is None with st.caption("waiting for first
    caption…"). Never raises on a mid-write row.
    """
    import streamlit as st

    try:
        state = fetch_live_state(db_path)
    except Exception:
        st.caption("Waiting for first caption…")
        return

    if state is None:
        st.caption("Waiting for first caption…")
        return

    # Summary stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Captioned", state["counts"]["captioned"])
    c2.metric("Escalated", state["counts"]["escalated"])
    c3.metric("Verified", state["counts"]["verified"])

    # Current frame image and caption
    st.subheader("Current frame")
    if state["current"]["image_path"] and Path(state["current"]["image_path"]).exists():
        st.image(state["current"]["image_path"])

    with st.expander("Caption"):
        st.write(f"**Caption:** {state['current']['caption']}")
        if state["current"]["change_line"]:
            st.write(f"**Change:** {state['current']['change_line']}")

    # Transcript ±1 s under the image
    with st.expander("Audio transcript"):
        for seg in state["transcript"]:
            st.write(f"**{seg['start_ms']:.0f}–{seg['end_ms']:.0f} ms:** {seg['text']}")

    # Verdicts column
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("35B Verdicts")
        for v in state["verdicts"]:
            if v["verifier_agrees"]:
                st.success(f"**Agrees** — {v['verifier_caption']}")
            else:
                st.warning(f"**Override** — {v['verifier_caption']}")
    with c2:
        # Recent captions feed
        st.subheader("Recent captions")
        for r in state["recent"]:
            st.write(f"**Frame {r['frame_idx']}:** {r['caption']}")
            if r["change_line"]:
                st.caption(f"Change: {r['change_line']}")
