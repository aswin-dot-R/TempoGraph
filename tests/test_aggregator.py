from unittest.mock import patch, MagicMock

from src.aggregator import CaptionAggregator
from src.models import ChunkCaption


VALID_JSON = """{
  "entities": [{"id":"E1","type":"person","description":"a man","first_seen":"00:00","last_seen":"00:10"}],
  "visual_events": [{"type":"walking","entities":["E1"],"start_time":"00:00","end_time":"00:05","description":"walks left","confidence":0.8}],
  "audio_events": [],
  "multimodal_correlations": [],
  "summary": "A man walks across the scene."
}"""


def test_aggregator_single_pass_returns_analysis_result():
    chunks = [
        ChunkCaption(0, [0, 1], {0: "man enters", 1: "man walks"}, "a man enters and walks", ""),
        ChunkCaption(1, [10], {10: "man exits"}, "the man exits", ""),
    ]
    with patch("src.aggregator.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"choices": [{"message": {"content": VALID_JSON}}]}),
            raise_for_status=MagicMock(),
        )
        agg = CaptionAggregator()
        result = agg.aggregate(chunks)

    assert len(result.entities) == 1
    assert result.entities[0].id == "E1"
    assert "man" in result.summary.lower()


def test_aggregator_falls_back_on_bad_json():
    chunks = [ChunkCaption(0, [0], {0: "blah"}, "", "")]
    with patch("src.aggregator.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"choices": [{"message": {"content": "not json at all"}}]}),
            raise_for_status=MagicMock(),
        )
        agg = CaptionAggregator()
        result = agg.aggregate(chunks)

    # Lenient parser returns empty AnalysisResult rather than crashing
    assert result.entities == []
    assert result.visual_events == []


def test_hierarchical_path_used_for_long_chunk_lists():
    long_chunks = [
        ChunkCaption(i, [i], {i: f"line {i}"}, f"sum {i}", "")
        for i in range(45)
    ]
    with patch("src.aggregator.requests.post") as mock_post:
        # First N calls compress meta-captions, last call returns final JSON
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"choices": [{"message": {"content": VALID_JSON}}]}),
            raise_for_status=MagicMock(),
        )
        agg = CaptionAggregator(single_pass_max_chunks=30, group_size=10)
        result = agg.aggregate(long_chunks)

    assert mock_post.call_count >= 2  # at least one meta + one final
    assert result.summary != ""
