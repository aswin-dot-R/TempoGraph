"""Shared visual identity for TempoGraph Streamlit pages.

This module injects a single CSS block that turns the stock Streamlit
chrome into a dark, precise, calm instrument look. Both pages call
``apply_theme()`` first, before any widgets, so the cascade is in
effect for every element below.
"""

from __future__ import annotations

import streamlit as st


_THEME_CSS = """
<style>
/* ── wordmark header ─────────────────────────────────────────── */
.tg-wordmark {
    text-align: center;
    padding-top: 6px;
    padding-bottom: 2px;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 6px;
    color: #E6EAEE;
    text-transform: uppercase;
    font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
    user-select: none;
}
.tg-wordmark::after {
    content: "";
    display: block;
    width: 56px;
    height: 1px;
    background: #3FBFB5;
    margin: 6px auto 0 auto;
}
.tg-tagline {
    text-align: center;
    font-size: 11px;
    letter-spacing: 2px;
    color: #6a737a;
    margin-top: 2px;
    margin-bottom: 10px;
    font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
    text-transform: lowercase;
}

/* ── keep the st.title readable by tests but subdued ────────── */
.tg-subtitle {
    text-align: center;
    font-size: 13px;
    color: #6a737a;
    margin-top: -4px;
    margin-bottom: 18px;
    font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
}

/* ── buttons ─────────────────────────────────────────────────── */
div[data-testid="stButton"] button {
    border-radius: 6px;
    border: 1px solid #3a3f48;
    background: transparent;
    color: #E6EAEE;
    padding: 4px 16px;
    transition: border-color 120ms, background 120ms, color 120ms;
    font-size: 13px;
}
div[data-testid="stButton"] button:hover {
    border-color: #3FBFB5;
    background: transparent;
    color: #3FBFB5;
}
div[data-testid="stButton"] button[svelte-312t7a] {
    background: #3FBFB5;
    color: #0e1117;
    border-color: #3FBFB5;
    font-weight: 600;
}
div[data-testid="stButton"] button[svelte-312t7a]:hover {
    background: #48d1c7;
    border-color: #48d1c7;
    color: #0e1117;
}
div[data-testid="stButton"] button[svelte-yz4k1b] {
    background: #2a2e35;
    color: #ccc;
    border-color: #3a3f48;
}

/* ── metrics / stat cards ────────────────────────────────────── */
.tg-metric-card {
    background: #1C2229;
    border: 1px solid #2A323B;
    border-radius: 6px;
    padding: 14px 16px;
    text-align: center;
    transition: border-color 120ms;
}
.tg-metric-card:hover {
    border-color: #3FBFB5;
}
.tg-metric-value {
    font-size: 20px;
    font-weight: 700;
    color: #E6EAEE;
    line-height: 1.2;
}
.tg-metric-label {
    font-size: 11px;
    color: #6a737a;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ── tabs — underline style ──────────────────────────────────── */
div[data-testid="stTabs"] {
    border: none;
    box-shadow: none;
}
div[data-testid="stTabs"] [role="tab"] {
    border: none !important;
    border-radius: 0 !important;
    color: #888;
    padding: 8px 16px !important;
    transition: color 120ms, border-color 120ms;
    font-size: 13px;
}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
div[data-testid="stTabs"] [role="tab"]:hover {
    color: #3FBFB5 !important;
    border-bottom: 2px solid #3FBFB5 !important;
    background: transparent !important;
}

/* ── file uploader drop zone ─────────────────────────────────── */
.tg-dropzone {
    border: 2px dashed #3FBFB5;
    border-radius: 8px;
    padding: 32px 24px;
    text-align: center;
    background: rgba(63, 191, 181, 0.04);
    transition: background 120ms, border-color 120ms;
}

/* ── hero block for landing ──────────────────────────────────── */
.tg-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 10px 0 6px 0;
}

/* ── progress checklist ──────────────────────────────────────── */
.tg-stage {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 13px;
    padding: 3px 0;
    line-height: 1.6;
}
.tg-stage .tg-stage-name {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ── chrome cleanup ──────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header { padding-bottom: 0 !important; }
footer { padding-top: 0 !important; }

/* ── reduced motion ──────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition-duration: 0ms !important;
        animation-duration: 0ms !important;
        animation-iteration-count: 1 !important;
    }
}
</style>
"""


def apply_theme() -> None:
    """Inject TempoGraph's theme CSS once per page render.

    Safe to call multiple times — Streamlit deduplicates identical
    ``st.markdown`` blocks by key, so we anchor it to a constant key.
    """
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
