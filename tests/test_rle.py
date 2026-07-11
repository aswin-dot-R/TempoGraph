"""Round-trip and semantics tests for src/rle.py (COCO-style RLE)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rle import decode, decode_from_string, encode, encode_to_string


class TestRoundTripProperty:
    def test_random_masks_round_trip(self):
        """Property test: decode(encode(m)) == m for random binary masks."""
        rng = np.random.default_rng(1234)
        for _ in range(200):
            h = int(rng.integers(1, 48))
            w = int(rng.integers(1, 48))
            density = rng.uniform(0.0, 1.0)
            mask = (rng.random((h, w)) < density).astype(np.uint8)
            rle = encode(mask)
            out = decode(rle)
            assert out.shape == (h, w)
            assert out.dtype == np.uint8
            np.testing.assert_array_equal(out, mask)

    def test_random_masks_string_round_trip(self):
        rng = np.random.default_rng(99)
        for _ in range(50):
            mask = (rng.random((17, 23)) < 0.3).astype(np.uint8)
            s = encode_to_string(mask)
            assert isinstance(s, str)
            json.loads(s)  # valid JSON
            np.testing.assert_array_equal(decode_from_string(s), mask)

    def test_all_zeros_and_all_ones(self):
        zeros = np.zeros((7, 5), dtype=np.uint8)
        ones = np.ones((7, 5), dtype=np.uint8)
        np.testing.assert_array_equal(decode(encode(zeros)), zeros)
        np.testing.assert_array_equal(decode(encode(ones)), ones)
        assert encode(zeros)["counts"] == [35]
        assert encode(ones)["counts"] == [0, 35]

    def test_single_pixel_masks(self):
        for h, w in [(1, 1), (1, 4), (4, 1)]:
            m = np.zeros((h, w), dtype=np.uint8)
            m[h // 2, w // 2] = 1
            np.testing.assert_array_equal(decode(encode(m)), m)


class TestCocoSemantics:
    def test_column_major_counts(self):
        """counts follow column-major order and start with the zero-run."""
        mask = np.array([[1, 0],
                         [0, 1]], dtype=np.uint8)
        rle = encode(mask)
        # Fortran-flattened: [1, 0, 0, 1] -> zeros run 0, ones 1, zeros 2, ones 1
        assert rle == {"size": [2, 2], "counts": [0, 1, 2, 1]}

    def test_size_is_height_width(self):
        mask = np.zeros((3, 9), dtype=np.uint8)
        assert encode(mask)["size"] == [3, 9]

    def test_counts_sum_equals_area(self):
        rng = np.random.default_rng(7)
        mask = (rng.random((20, 30)) < 0.5).astype(np.uint8)
        assert sum(encode(mask)["counts"]) == 600

    def test_nonbinary_input_thresholded_at_nonzero(self):
        mask = np.array([[0, 2], [255, 0]], dtype=np.uint8)
        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(decode(encode(mask)), expected)


class TestErrors:
    def test_bad_counts_sum_raises(self):
        with pytest.raises(ValueError):
            decode({"size": [2, 2], "counts": [3]})

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            decode({"size": [2, 2], "counts": [-1, 5]})

    def test_non_2d_mask_raises(self):
        with pytest.raises(ValueError):
            encode(np.zeros((2, 2, 3), dtype=np.uint8))
