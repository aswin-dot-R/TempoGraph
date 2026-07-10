"""Unit tests for src/auto_profile.py."""

from src.auto_profile import VideoFacts, derive_plan


# ── probe() tests ───────────────────────────────────────────────────
# We can't run ffprobe on real videos in unit tests, so we test
# derive_plan() with fixture VideoFacts dicts instead.


# ── derive_plan() fixture tests ─────────────────────────────────────


def _make_facts(duration_s, width=1920, height=1080, fps=30.0, has_audio=True):
    return VideoFacts(
        duration_s=duration_s,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
    )


class TestDerivePlanShortClip:
    """Short clip (<=3 min) should get yolo26x-seg."""

    def test_short_clip_with_audio(self):
        facts = _make_facts(duration_s=60.0, has_audio=True)
        plan = derive_plan(facts)

        assert plan.yolo_size == "x"
        assert plan.yolo_seg is True
        assert plan.confidence == 0.5
        assert plan.depth_enabled is True
        assert plan.audio_enabled is True
        assert plan.whisper_model == "base.en"
        assert plan.vlm_frame_mode == "keyframes"
        assert plan.dynamic_chunking is True
        assert plan.vlm_dedup_threshold == 0.92

    def test_short_clip_without_audio(self):
        facts = _make_facts(duration_s=120.0, has_audio=False)
        plan = derive_plan(facts)

        assert plan.yolo_size == "x"
        assert plan.yolo_seg is True
        assert plan.audio_enabled is False
        assert plan.whisper_model == "tiny.en"

    def test_exact_3min(self):
        """Exactly 180s should still be considered short."""
        facts = _make_facts(duration_s=180.0, has_audio=True)
        plan = derive_plan(facts)

        assert plan.yolo_size == "x"

    def test_3min_plus_1sec(self):
        """181s should switch to yolo26n."""
        facts = _make_facts(duration_s=181.0, has_audio=True)
        plan = derive_plan(facts)

        assert plan.yolo_size == "n"


class TestDerivePlanLongVideo:
    """Long videos (>3 min) should get yolo26n-seg."""

    def test_40min_720p_with_audio(self):
        facts = _make_facts(duration_s=2400.0, width=1280, height=720, has_audio=True)
        plan = derive_plan(facts)

        assert plan.yolo_size == "n"
        assert plan.yolo_seg is True
        assert plan.audio_enabled is True
        assert plan.whisper_model == "base.en"
        assert plan.depth_enabled is True

    def test_1hour_silent(self):
        facts = _make_facts(duration_s=3600.0, has_audio=False)
        plan = derive_plan(facts)

        assert plan.yolo_size == "n"
        assert plan.audio_enabled is False
        assert plan.whisper_model == "tiny.en"

    def test_yolo_fps_is_reasonable_for_long(self):
        """YOLO FPS should stay in [0.5, 3.0] range."""
        facts = _make_facts(duration_s=3600.0, has_audio=True)
        plan = derive_plan(facts)

        assert 0.5 <= plan.yolo_fps <= 3.0

    def test_vlm_fps_is_reasonable(self):
        """VLM FPS should stay in [0.1, 2.0] range."""
        facts = _make_facts(duration_s=3600.0, has_audio=True)
        plan = derive_plan(facts)

        assert 0.1 <= plan.vlm_fps <= 2.0


class TestDerivePlanEdgeCases:
    """Edge cases."""

    def test_zero_duration(self):
        facts = _make_facts(duration_s=0.0, has_audio=False)
        plan = derive_plan(facts)

        assert plan.audio_enabled is False
        assert plan.yolo_fps == 1.0
        assert plan.vlm_fps == 0.5

    def test_very_short_clip(self):
        facts = _make_facts(duration_s=5.0, has_audio=True)
        plan = derive_plan(facts)

        assert plan.yolo_size == "x"
        assert plan.audio_enabled is True

    def test_all_defaults(self):
        """Verify all DerivedPlan fields have expected defaults."""
        facts = _make_facts(duration_s=100.0, has_audio=True)
        plan = derive_plan(facts)

        assert plan.threshold_mult == 1.0
        assert plan.camera_mode == "auto"
        assert plan.chunk_size == 10
        assert not plan.skip_vlm


class TestDerivedPlanToPipelineKwargs:
    """DerivedPlan.to_pipeline_kwargs() should return a flat dict."""

    def test_returns_all_fields(self):
        facts = _make_facts(duration_s=100.0, has_audio=True)
        plan = derive_plan(facts)
        kwargs = plan.to_pipeline_kwargs()

        expected_keys = {
            "yolo_fps",
            "vlm_fps",
            "chunk_size",
            "depth_enabled",
            "use_segmentation",
            "yolo_size",
            "threshold_mult",
            "confidence",
            "vlm_frame_mode",
            "vlm_dedup_threshold",
            "dynamic_chunking",
            "camera_mode",
            "audio_enabled",
            "whisper_model",
            "skip_vlm",
        }
        assert expected_keys.issubset(set(kwargs.keys()))

    def test_values_match_plan(self):
        facts = _make_facts(duration_s=100.0, has_audio=True)
        plan = derive_plan(facts)
        kwargs = plan.to_pipeline_kwargs()

        assert kwargs["yolo_fps"] == plan.yolo_fps
        assert kwargs["depth_enabled"] == plan.depth_enabled
        assert kwargs["use_segmentation"] == plan.yolo_seg
        assert kwargs["yolo_size"] == plan.yolo_size
