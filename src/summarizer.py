"""Summarize video analysis results using an injectable LLM callable.

In production, the callable points at the llama-server backend.
In tests, inject a fake function.
"""

from __future__ import annotations

from typing import Callable, Optional

SUMMARY_PROMPT = """You are given a video analysis summary. Write exactly 5 lines:
Line 1: One-sentence overall description of what the video shows.
Line 2: List of key entities detected (names/types).
Line 3: Summary of major events or actions.
Line 4: Notable spatial or temporal patterns (e.g., "most activity in the second half").
Line 5: Any audio/speech highlights (or "No audio" if silent).

Input:
Entities: {entities}
Visual events: {events}
Audio events: {audio}
Summary: {summary}

Output exactly 5 lines, nothing else."""


def generate_summary(
    entities: list[dict],
    visual_events: list[dict],
    audio_events: list[dict],
    summary_text: str,
    llm_callable: Optional[Callable] = None,
) -> str:
    """Generate a 5-line narrative summary.

    Args:
        entities: List of entity dicts from analysis.json.
        visual_events: List of visual event dicts.
        audio_events: List of audio event dicts.
        summary_text: The raw LLM summary from analysis.json.
        llm_callable: Optional callable that takes a prompt string
            and returns a summary string. If None, uses a default
            heuristic-based summary.

    Returns:
        A 5-line narrative summary string.
    """
    if llm_callable is not None:
        prompt = SUMMARY_PROMPT.format(
            entities=_fmt_entities(entities),
            events=_fmt_events(visual_events),
            audio=_fmt_audio(audio_events),
            summary=summary_text,
        )
        return llm_callable(prompt)

    # Default: heuristic-based summary from existing data
    lines = []

    # Line 1: Overall description
    if summary_text:
        lines.append(summary_text)
    else:
        lines.append("Video analysis completed.")

    # Line 2: Entities
    if entities:
        entity_types = set(e.get("type", "unknown") for e in entities)
        lines.append(f"Entities: {', '.join(sorted(entity_types))}")
    else:
        lines.append("Entities: none detected")

    # Line 3: Events
    if visual_events:
        event_types = set(e.get("type", "unknown") for e in visual_events)
        lines.append(f"Events: {', '.join(sorted(event_types))}")
    else:
        lines.append("Events: none recorded")

    # Line 4: Patterns
    if len(entities) > 1:
        lines.append(f"Multi-entity interaction across {len(entities)} entities.")
    elif len(entities) == 1:
        lines.append("Single entity tracked throughout the video.")
    else:
        lines.append("No significant spatial or temporal patterns detected.")

    # Line 5: Audio
    if audio_events:
        speech_count = sum(1 for a in audio_events if a.get("type") == "speech")
        if speech_count > 0:
            lines.append(f"Audio: {speech_count} speech segment(s) detected.")
        else:
            lines.append(f"Audio: {len(audio_events)} non-speech event(s) detected.")
    else:
        lines.append("No audio track or audio not transcribed.")

    return "\n".join(lines)


def _fmt_entities(entities: list[dict]) -> str:
    if not entities:
        return "(none)"
    parts = []
    for e in entities:
        eid = e.get("id", "?")
        etype = e.get("type", "?")
        desc = e.get("description", "")
        parts.append(f"{eid} ({etype}): {desc[:80]}")
    return "\n".join(parts)


def _fmt_events(events: list[dict]) -> str:
    if not events:
        return "(none)"
    parts = []
    for ev in events[:20]:
        etype = ev.get("type", "?")
        ents = ", ".join(ev.get("entities", []))
        desc = ev.get("description", "")
        parts.append(
            f"[{ev.get('start_time', '?')}–{ev.get('end_time', '?')}] "
            f"{etype}: {ents} — {desc[:60]}"
        )
    return "\n".join(parts)


def _fmt_audio(audio_events: list[dict]) -> str:
    if not audio_events:
        return "(none)"
    parts = []
    for ae in audio_events[:10]:
        text = ae.get("text", "")
        st = ae.get("start_time", "?")
        et = ae.get("end_time", "?")
        parts.append(f"[{st}–{et}] {ae.get('type', '?')}: {text[:80]}")
    return "\n".join(parts)
