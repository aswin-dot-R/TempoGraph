from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from src.storage import TempoGraphDB
from src.backends.llama_server_backend import LlamaServerBackend


def _jpg(p: Path):
    cv2.imwrite(str(p), np.full((100, 100, 3), 200, dtype=np.uint8))


def test_caption_chunks_propagates_seed(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    paths = []
    for i in range(20):
        p = tmp_path / f"f{i}.jpg"
        _jpg(p)
        db.insert_frame(i, i * 500, str(p), False, 0.0)
        paths.append(p)

    fake_responses = [
        {"choices": [{"message": {"content": "FRAME_0: a starts\nFRAME_1: a continues\nSUMMARY: agent A walking right"}}]},
        {"choices": [{"message": {"content": "FRAME_10: b appears\nSUMMARY: agent B has joined"}}]},
    ]
    with patch("src.backends.llama_server_backend.requests.post") as mock_post:
        mock_post.side_effect = [MagicMock(json=MagicMock(return_value=r), raise_for_status=MagicMock()) for r in fake_responses]

        backend = LlamaServerBackend()
        chunks = [
            (0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (1, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        ]
        results = backend.caption_chunks(chunks=chunks, db=db)

    assert len(results) == 2
    assert results[0].summary.startswith("agent A")
    assert results[1].summary.startswith("agent B")

    # Second call must include the first call's summary as the seed
    second_call_payload = mock_post.call_args_list[1].kwargs["json"]
    second_content = second_call_payload["messages"][0]["content"]
    second_prompt = next(item["text"] for item in second_content if item["type"] == "text")
    assert "agent A walking right" in second_prompt


def test_caption_chunks_handles_failure_gracefully(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    p = tmp_path / "f.jpg"
    _jpg(p)
    db.insert_frame(0, 0, str(p), False, 0.0)

    with patch("src.backends.llama_server_backend.requests.post") as mock_post:
        mock_post.side_effect = Exception("boom")
        backend = LlamaServerBackend()
        results = backend.caption_chunks(chunks=[(0, [0])], db=db)

    assert len(results) == 1
    assert results[0].summary == ""  # empty seed on failure
    assert results[0].per_frame_lines == {}
