"""Tests for aggregator utility functions and prompt-fitting logic.

Mocks all HTTP — no server may be contacted.  Covers:

  * ``estimate_tokens``, ``subsample_lines``, ``truncate_middle``
  * ``CaptionAggregator.get_n_ctx()`` (probe, fallback, caching)
  * ``_fit_blocks`` (shrinks dense → transcript → captions to fit)
  * End-to-end ``aggregate()`` regression: never sends a prompt exceeding
    the budget (the regression was a 174-frame dense timeline blowing a
    8192-slot server with a 400).
  * Large-context path (n_ctx=262144) → no trimming.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.aggregator import (
    CaptionAggregator,
    SINGLE_PASS_PROMPT,
    DEFAULT_N_CTX,
    estimate_tokens,
    subsample_lines,
    truncate_middle,
)
from src.models import ChunkCaption


# ── Utility functions ──────────────────────────────────────────────────


class TestEstimateTokens:
    """estimate_tokens: rough token count (~3 chars per token)."""

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short(self):
        assert estimate_tokens("a") == 0
        assert estimate_tokens("ab") == 0
        assert estimate_tokens("abc") == 1

    def test_exact_multiple(self):
        assert estimate_tokens("abcabc") == 2

    def test_non_multiple(self):
        # 11 chars → 11 // 3 = 3
        assert estimate_tokens("hello world") == 3


class TestSubsampleLines:
    """subsample_lines: thin to ~keep entries, first+last always kept."""

    def test_identity_on_large_keep(self):
        lines = [f"line {i}" for i in range(20)]
        assert subsample_lines(lines, 20) == lines
        assert subsample_lines(lines, 100) == lines

    def test_identity_on_nonpositive_keep(self):
        lines = [f"line {i}" for i in range(10)]
        assert subsample_lines(lines, 0) == lines
        assert subsample_lines(lines, -1) == lines

    def test_keep_one(self):
        lines = [f"line {i}" for i in range(10)]
        assert subsample_lines(lines, 1) == ["line 0"]

    def test_keep_two(self):
        lines = [f"line {i}" for i in range(10)]
        assert subsample_lines(lines, 2) == ["line 0", "line 9"]

    def test_keeps_first_and_last(self):
        lines = [f"line {i}" for i in range(50)]
        result = subsample_lines(lines, 7)
        assert result[0] == "line 0"
        assert result[-1] == "line 49"

    def test_returns_correct_count(self):
        lines = [f"line {i}" for i in range(100)]
        result = subsample_lines(lines, 10)
        assert len(result) == 10


class TestTruncateMiddle:
    """truncate_middle: drop the middle, keeping head+tail."""

    def test_short_text_unchanged(self):
        assert truncate_middle("hi", 100) == "hi"
        assert truncate_middle("exactly fifty characters here!!!!!!!!!!", 50) == (
            "exactly fifty characters here!!!!!!!!!!"
        )

    def test_long_text_keeps_head_tail(self):
        long_text = "a" * 100
        marker = "\n[... trimmed to fit context ...]\n"
        result = truncate_middle(long_text, 50)
        assert len(result) <= 50
        assert result.startswith("a")
        assert result.endswith("a")
        assert marker in result

    def test_zero_max_chars(self):
        assert truncate_middle("hello", 0) == ""


# ── get_n_ctx ─────────────────────────────────────────────────────────


class TestGetNCtx:
    """CaptionAggregator.get_n_ctx() — probe, fallback, caching."""

    def test_probed_returns_server_value(self):
        agg = CaptionAggregator(base_url="http://127.0.0.1:8082")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"default_generation_settings": {"n_ctx": 8192}}
        mock_resp.raise_for_status.return_value = None

        with patch("src.aggregator.requests.get", return_value=mock_resp) as mock_get:
            n_ctx = agg.get_n_ctx()

        assert n_ctx == 8192
        assert agg._n_ctx == 8192
        mock_get.assert_called_once()

    def test_connection_error_falls_back_to_default(self):
        agg = CaptionAggregator(base_url="http://127.0.0.1:8082")

        with patch(
            "src.aggregator.requests.get",
            side_effect=ConnectionError("refused"),
        ) as mock_get:
            n_ctx = agg.get_n_ctx()

        assert n_ctx == DEFAULT_N_CTX
        mock_get.assert_called_once()

    def test_cached_on_second_call(self):
        """Second call returns cached value without hitting the network."""
        agg = CaptionAggregator(base_url="http://127.0.0.1:8082")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"default_generation_settings": {"n_ctx": 8192}}
        mock_resp.raise_for_status.return_value = None

        with patch("src.aggregator.requests.get", return_value=mock_resp) as mock_get:
            agg.get_n_ctx()
            agg.get_n_ctx()

        assert mock_get.call_count == 1


# ── _fit_blocks ──────────────────────────────────────────────────────


class TestFitBlocks:
    """CaptionAggregator._fit_blocks — shrinks until prompt fits."""

    def test_fits_within_budget(self):
        """Huge dense timeline + transcript → prompt fits; dense shrunk most.

        First and last dense lines must survive.
        """
        agg = CaptionAggregator(base_url="http://127.0.0.1:8082")

        with patch.object(agg, "get_n_ctx", return_value=8192):
            dense_lines = [
                f"frame {i}: a detailed description of what is happening in this scene "
                f"with lots of detail about the objects and people visible"
                for i in range(200)
            ]
            dense_text = "\n".join(dense_lines)
            transcript = "word " * 10_000
            captions = "text " * 10_000

            captions_out, dense_out, transcript_out = agg._fit_blocks(
                captions, dense_text, transcript
            )

            budget = agg._prompt_budget()
            overhead = estimate_tokens(SINGLE_PASS_PROMPT)
            total = (
                overhead
                + estimate_tokens(dense_out)
                + estimate_tokens(captions_out)
                + estimate_tokens(transcript_out)
            )
            assert total <= budget, f"Prompt tokens {total} exceeds budget {budget}"

            # Dense block should have been shrunk.
            assert len(dense_out.splitlines()) < len(dense_lines)

            # First and last dense lines must survive.
            out_lines = dense_out.splitlines()
            assert out_lines[0] == dense_lines[0]
            assert out_lines[-1] == dense_lines[-1]

    def test_no_trim_on_huge_context(self):
        """n_ctx=262144 → nothing is trimmed; blocks pass through unchanged."""
        agg = CaptionAggregator(base_url="http://127.0.0.1:8082")

        with patch.object(agg, "get_n_ctx", return_value=262_144):
            dense = "line " * 1_000
            transcript = "word " * 1_000
            captions = "text " * 1_000

            captions_out, dense_out, transcript_out = agg._fit_blocks(
                captions, dense, transcript
            )

            assert captions_out == captions
            assert dense_out == dense
            assert transcript_out == transcript


# ── End-to-end aggregate regression ─────────────────────────────────


class TestAggregateEndToEnd:
    """aggregate() never sends a prompt whose estimate_tokens exceeds the budget."""

    def test_aggregate_fits_budget_on_8192_ctx_server(self):
        """Regression test: a 174-frame dense timeline used to blow the 8192
        slot and return 400.  Mock a post that captures the prompt and
        verify it fits."""
        agg = CaptionAggregator(
            base_url="http://127.0.0.1:8082",
            single_pass_max_chunks=30,
            group_size=10,
        )

        captured_prompts: list[str] = []

        def mock_post(*args, **kwargs):
            prompt = kwargs["json"]["messages"][0]["content"][0]["text"]
            captured_prompts.append(prompt)
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [
                        {"message": {"content": '{"entities":[],"summary":"ok"}'}}
                    ]
                },
                raise_for_status=MagicMock(),
            )

        mock_props = MagicMock()
        mock_props.json.return_value = {"default_generation_settings": {"n_ctx": 8192}}
        mock_props.raise_for_status.return_value = None

        with patch("src.aggregator.requests.post", side_effect=mock_post), patch(
            "src.aggregator.requests.get", return_value=mock_props
        ):
            chunks = [
                ChunkCaption(
                    chunk_id=i,
                    frame_indices=[i],
                    per_frame_lines={
                        i: f"frame {i}: a long description with many words to fill space "
                        * 5
                    },
                    summary=f"summary {i}: another long summary with many words to fill space "
                    * 5,
                    raw_response="",
                )
                for i in range(50)
            ]

            result = agg.aggregate(chunks)

            budget = agg._prompt_budget()
            for prompt in captured_prompts:
                prompt_tokens = estimate_tokens(prompt)
                assert (
                    prompt_tokens <= budget
                ), f"Prompt tokens {prompt_tokens} exceeds budget {budget}"

        # The aggregation should have produced a result (even if empty).
        assert result is not None
