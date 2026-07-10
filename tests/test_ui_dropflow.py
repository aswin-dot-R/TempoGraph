"""UI contract tests for the dropflow UI (ui/app.py).

Verifies:
- Landing: zero sidebar control widgets before a video is chosen
- Plan screen: Analyze button + Adjust expander exist
- Expander contains the legacy knobs pre-filled from derive_plan()
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestLandingScreenContract:
    """The landing screen should have zero sidebar widgets before a video is chosen."""

    def test_landing_has_zero_control_widgets(self):
        """
        Read the app.py source and verify that in the landing path
        (when no plan is in session state), no sidebar control widgets
        are created before the video is chosen.
        """
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        # The main() function first checks KEY_RUNNING, then KEY_PLAN.
        # If KEY_PLAN is None, _render_landing_screen() is called.
        # That function only calls st.sidebar.markdown() and st.sidebar.caption().
        # It should NOT call sidebar.slider, sidebar.checkbox, sidebar.selectbox,
        # sidebar.radio, sidebar.text_input, etc.

        # Find the _render_landing_screen function
        landing_start = source.find("def _render_landing_screen")
        assert landing_start >= 0, "_render_landing_screen not found"

        # Find the next function definition
        next_func = source.find("\ndef _render_recent_runs", landing_start)
        if next_func < 0:
            next_func = source.find("\ndef _render_plan_screen", landing_start)
        assert next_func > landing_start, "Could not find end of _render_landing_screen"

        landing_code = source[landing_start:next_func]

        # Check that NO sidebar control widgets are used in landing
        forbidden = [
            "sidebar.slider",
            "sidebar.checkbox",
            "sidebar.selectbox",
            "sidebar.radio",
            "sidebar.text_input",
            "st.sidebar.slider",
            "st.sidebar.checkbox",
            "st.sidebar.selectbox",
            "st.sidebar.radio",
            "st.sidebar.text_input",
        ]
        for f in forbidden:
            assert f not in landing_code, (
                f"Landing screen uses '{f}' which is a control widget. "
                "Landing should have zero sidebar control widgets."
            )


class TestPlanScreenContract:
    """The plan screen should have an Analyze button and Adjust expander."""

    def test_analyze_button_in_plan_screen(self):
        """Plan screen should render an 'Analyze' button."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        # Find _render_plan_screen function
        plan_start = source.find("def _render_plan_screen")
        assert plan_start >= 0, "_render_plan_screen not found"

        # Find the next function definition
        next_func = source.find("\ndef _render_knobs", plan_start)
        assert next_func > plan_start, "Could not find end of _render_plan_screen"

        plan_code = source[plan_start:next_func]

        # Should have an Analyze button
        assert "Analyze" in plan_code, "Plan screen should have an 'Analyze' button"
        # The button should use button() with label "Analyze"
        assert (
            "button(" in plan_code and '"Analyze"' in plan_code
        ), "Plan screen should call st.button with 'Analyze'"

    def test_adjust_expander_in_plan_screen(self):
        """Plan screen should have an 'Adjust plan' expander."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        plan_start = source.find("def _render_plan_screen")
        assert plan_start >= 0
        next_func = source.find("\ndef _render_knobs", plan_start)
        plan_code = source[plan_start:next_func]

        assert (
            "Adjust plan" in plan_code or "expander(" in plan_code
        ), "Plan screen should have an expander for adjusting the plan"


class TestKnobsPreFilled:
    """The Adjust expander should contain legacy knobs pre-filled from derive_plan()."""

    def test_knobs_function_exists(self):
        """There should be a _render_knobs function."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        assert (
            "def _render_knobs" in source
        ), "_render_knobs function not found in app.py"

    def test_knobs_uses_override_values(self):
        """Knobs should use override values from session state."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        # Find _render_knobs function
        knobs_start = source.find("def _render_knobs")
        assert knobs_start >= 0

        next_func = source.find("\ndef _render_progress_screen", knobs_start)
        if next_func < 0:
            next_func = source.find("\ndef main()", knobs_start)
        if next_func < 0:
            next_func = len(source)

        knobs_code = source[knobs_start:next_func]

        # Should reference DEFAULT_KNOBS or override values
        assert (
            "DEFAULT_KNOBS" in knobs_code or "override" in knobs_code.lower()
        ), "Knobs should reference default values or overrides for pre-filling"

    def test_knobs_contains_yolo_controls(self):
        """Knobs should contain YOLO-related controls."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        knobs_start = source.find("def _render_knobs")
        assert knobs_start >= 0

        next_func = source.find("\ndef _render_progress_screen", knobs_start)
        if next_func < 0:
            next_func = source.find("\ndef main()", knobs_start)
        if next_func < 0:
            next_func = len(source)

        knobs_code = source[knobs_start:next_func]

        # Should have YOLO controls
        assert (
            "yolo" in knobs_code.lower() or "YOLO" in knobs_code
        ), "Knobs should contain YOLO controls"

    def test_knobs_contains_audio_controls(self):
        """Knobs should contain audio/whisper controls."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        knobs_start = source.find("def _render_knobs")
        assert knobs_start >= 0

        next_func = source.find("\ndef _render_progress_screen", knobs_start)
        if next_func < 0:
            next_func = source.find("\ndef main()", knobs_start)
        if next_func < 0:
            next_func = len(source)

        knobs_code = source[knobs_start:next_func]

        assert (
            "whisper" in knobs_code.lower() or "audio" in knobs_code.lower()
        ), "Knobs should contain audio/whisper controls"


class TestProgressScreenContract:
    """The progress screen should show stage checklist and cancel button."""

    def test_cancel_button_in_progress(self):
        """Progress screen should have a Cancel button."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        progress_start = source.find("def _render_progress_screen")
        assert progress_start >= 0

        next_func = source.find("\ndef _on_stage_progress", progress_start)
        assert next_func > progress_start
        progress_code = source[progress_start:next_func]

        assert "Cancel" in progress_code, "Progress screen should have a Cancel button"

    def test_stage_checklist(self):
        """Progress screen should show stage status for each pipeline stage."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        progress_start = source.find("def _render_progress_screen")
        assert progress_start >= 0

        next_func = source.find("\ndef _on_stage_progress", progress_start)
        progress_code = source[progress_start:next_func]

        # Should have a stage checklist or status display
        assert (
            "Stage status" in progress_code or "stage" in progress_code.lower()
        ), "Progress screen should show stage status"


class TestAutoProfileIntegration:
    """Verify auto_profile is properly integrated into app.py."""

    def test_app_imports_auto_profile(self):
        """app.py should import from auto_profile."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        assert "auto_profile" in source, "app.py should import from auto_profile"

    def test_app_uses_derive_plan(self):
        """app.py should call derive_plan()."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        assert "derive_plan" in source, "app.py should call derive_plan()"

    def test_app_calls_probe(self):
        """app.py should call probe() on video."""
        app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
        source = app_path.read_text()

        assert "probe(" in source, "app.py should call probe()"


class TestResultsPageContract:
    """Verify Results.py has the new tabs."""

    def test_ask_tab_exists(self):
        """Results page should have an Ask tab."""
        results_path = (
            Path(__file__).resolve().parents[1] / "ui" / "pages" / "Results.py"
        )
        source = results_path.read_text()

        assert (
            '"Ask"' in source or "'Ask'" in source
        ), "Results page should have an Ask tab"

    def test_dataset_export_tab_exists(self):
        """Results page should have a Dataset export tab."""
        results_path = (
            Path(__file__).resolve().parents[1] / "ui" / "pages" / "Results.py"
        )
        source = results_path.read_text()

        assert "Dataset" in source, "Results page should have a Dataset export tab"

    def test_render_ask_function_exists(self):
        """Results page should have _render_ask_tab function."""
        results_path = (
            Path(__file__).resolve().parents[1] / "ui" / "pages" / "Results.py"
        )
        source = results_path.read_text()

        assert (
            "_render_ask_tab" in source
        ), "Results page should have _render_ask_tab function"
