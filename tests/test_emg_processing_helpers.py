"""Module-level EMG processing functions (`m3resp.emg.processing.core`) and the
argument checks and helpers they share (`m3resp.emg.processing.shared`).
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.core.exceptions import (
    InvalidM3respModuleError,
    MutuallyExclusiveArgsError,
    OptionalDependencyError,
    UnresolvedChannelError,
    UnsupportedWorkflowError,
)
from m3resp.data.events import BreathEvent
from m3resp.emg.processing import core, shared


class TestCoreFunctions:
    """Module-level functions in `m3resp.emg.processing.core`."""

    def test_available_postprocessing_lists_every_function(self):
        """Every registered postprocessing category and function is listed."""
        available = core.available_postprocessing()
        assert available.keys() == shared.POSTPROCESSING_FUNCTIONS.keys()
        assert "time_product" in available["features"]

    def test_preprocess_uses_a_supplied_callable(self):
        """`preprocess=` replaces default preprocessing and receives the kwargs."""
        recording = {"array": np.zeros((1, 4))}

        def custom(signal, **kwargs):
            return {"signal": signal, **kwargs}

        result = core.preprocess(recording, preprocess=custom, channel=0)
        assert result == {"signal": recording, "channel": 0}

    def test_detect_breaths_uses_a_supplied_detector(self):
        """A custom detector's (start, end) pairs become EMG `BreathEvent`s."""

        def detector(signal, **kwargs):
            return [(0.0, 1.0), (1.5, 2.5)]

        events = core.detect_breaths({}, detector=detector)

        assert [e.start_time for e in events] == [0.0, 1.5]
        assert all(isinstance(e, BreathEvent) for e in events)
        assert {e.modality for e in events} == {"emg"}
        assert {e.source for e in events} == {"resurfemg"}

    def test_compute_features_needs_a_callable(self):
        """`compute_features` without `compute=` raises UnsupportedWorkflowError."""
        with pytest.raises(UnsupportedWorkflowError, match="compute=callable"):
            core.compute_features({}, [])

    def test_compute_features_calls_the_callable(self):
        """`compute=` receives the signal, the events and the extra kwargs."""
        result = core.compute_features(
            {"fs": 1.0}, ["b"], compute=lambda s, e, **k: (s, e, k), scale=2
        )
        assert result == ({"fs": 1.0}, ["b"], {"scale": 2})

    @pytest.mark.parametrize("processed", [None, [], {"channel": 0}])
    def test_to_signals_needs_preprocessed_emg(self, processed):
        """`to_signals` rejects input that is not preprocessed EMG."""
        with pytest.raises(UnsupportedWorkflowError, match="preprocess_emg"):
            core.to_signals(processed)

    def test_to_signals_skips_missing_arrays(self):
        """Missing arrays are skipped; the time axis comes from the sampling rate."""
        signals = core.to_signals(
            {"fs": 10.0, "channel": None, "raw_channel": np.arange(5.0)}
        )
        assert [s.name for s in signals] == ["raw_channel"]
        assert signals[0].channel is None
        np.testing.assert_allclose(signals[0].time, np.arange(5) / 10.0)

    def test_to_parameters_converts_features(self):
        """Computed features become `ParameterResult`s; scalars become floats."""
        parameters = core.to_parameters(
            {"computed": {"features": {"amplitude": [1.0, 2.0], "rr": 15}}}
        )
        by_name = {p.name: p for p in parameters}
        np.testing.assert_array_equal(by_name["amplitude"].value, [1.0, 2.0])
        assert by_name["rr"].value == 15.0
        assert by_name["rr"].method == "resurfemg.rr"

    def test_to_quality_flags_ignores_non_dicts(self):
        """`to_quality_flags` returns no flags for non-dict input."""
        assert core.to_quality_flags(None) == []

    def test_to_quality_flags_includes_skipped_functions(self):
        """Skipped functions become warning flags after the computed ones."""
        flags = core.to_quality_flags(
            {
                "computed": {"quality_assessment": {"snr_pseudo": 3.2}},
                "skipped": {"quality_assessment.pocc_quality": "no occlusions"},
            }
        )
        assert [f.name for f in flags] == [
            "snr_pseudo",
            "quality_assessment.pocc_quality",
        ]
        assert flags[1].severity == "warning"

    def test_postprocess_uses_a_supplied_callable(self):
        """`postprocess=` replaces default postprocessing and receives the events."""

        def custom(processed, *, events, **kwargs):
            return {"processed": processed, "events": events, **kwargs}

        result = core.postprocess({"fs": 1}, ["e"], postprocess=custom, peep=5)
        assert result == {"processed": {"fs": 1}, "events": ["e"], "peep": 5}

    def test_run_postprocessing_function_rejects_unknown_names(self):
        """An unregistered function name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ReSurfEMG postprocessing"):
            core.run_postprocessing_function("features", "not_a_function")

    def test_run_postprocessing_function_calls_a_module_function(self):
        """A function registered by module path is resolved and called."""
        peaks = np.array([0, 100, 200, 300])
        median_rate, _ = core.run_postprocessing_function(
            "features", "respiratory_rate", peaks, 100.0
        )
        assert median_rate == pytest.approx(60.0)

    def test_run_postprocessing_function_resolves_a_class_method(self):
        """A function registered as a class method is resolved and called."""
        method = shared.POSTPROCESSING_FUNCTIONS["baseline"]["moving_baseline"]
        assert method.module.endswith("._BaselineMixin")
        envelope = np.abs(np.sin(np.linspace(0, 20, 2000)))
        result = core.run_postprocessing_function(
            "baseline",
            "moving_baseline",
            envelope,
            window_samples=200,
            step_samples=50,
            percentile=33,
        )
        assert len(result) == len(envelope)

    def test_run_postprocessing_function_reports_unresolvable_modules(
        self, monkeypatch
    ):
        """An unresolvable path surfaces as OptionalDependencyError with the cause
        chained."""
        monkeypatch.setitem(
            shared.POSTPROCESSING_FUNCTIONS["features"],
            "broken",
            shared.PostProcessingFunction("no_such_package.module", "f"),
        )
        # InvalidM3respModuleError subclasses ImportError, so it is reported
        # as a missing `resurfemg` install; the real cause stays chained.
        with pytest.raises(OptionalDependencyError) as error:
            core.run_postprocessing_function("features", "broken")
        assert isinstance(error.value.__cause__, InvalidM3respModuleError)


class TestArrayChecks:
    """Array and number checks in `m3resp.emg.processing.shared`."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(3.0, [3.0]), ([[1.0, 2.0]], [1.0, 2.0]), ([1.0, 2.0], [1.0, 2.0])],
        ids=["scalar", "row", "1d"],
    )
    def test_convert_to_1d_array(self, value, expected):
        """Scalars, rows and 1-D input all become 1-D arrays."""
        np.testing.assert_array_equal(shared._convert_to_1d_array(value), expected)

    def test_require_1d_array(self):
        """1-D input becomes a float array; 2-D input raises ValueError."""
        assert shared._require_1d_array("x", [1, 2]).dtype == float
        with pytest.raises(ValueError, match="x must be a 1D array"):
            shared._require_1d_array("x", [[1, 2], [3, 4]])

    def test_require_index_array(self):
        """1-D input becomes an int array; 2-D input raises ValueError."""
        np.testing.assert_array_equal(
            shared._require_index_array("idx", [1.0, 2.0]), [1, 2]
        )
        with pytest.raises(ValueError, match="1D array of indices"):
            shared._require_index_array("idx", [[1]])

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_require_positive_int_rejects(self, value):
        """Zero, negatives, floats and booleans are not positive integers."""
        with pytest.raises(ValueError, match="positive integer"):
            shared._require_positive_int("n", value)

    def test_require_positive_int_accepts(self):
        """A positive int passes."""
        shared._require_positive_int("n", 3)

    @pytest.mark.parametrize("value", [-0.1, 100.1])
    def test_require_percentile_rejects(self, value):
        """Percentiles outside 0-100 raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            shared._require_percentile("p", value)

    @pytest.mark.parametrize("value", [0.0, np.inf, np.nan, -2.0])
    def test_require_finite_positive_rejects(self, value):
        """Zero, negatives, inf and NaN raise ValueError."""
        with pytest.raises(ValueError, match="finite, positive"):
            shared._require_finite_positive("v", value)

    def test_require_equal_length(self):
        """Arrays of equal length pass; different lengths raise ValueError."""
        shared._require_equal_length(("a", [1, 2]), ("b", [3, 4]))
        with pytest.raises(ValueError, match="equal length"):
            shared._require_equal_length(("a", [1, 2]), ("b", [3]))

    def test_mask_invalid_sets_nan_and_copies(self):
        """Invalid entries become NaN in a copy; the input is unchanged."""
        values = np.array([1.0, 2.0, 3.0])
        masked = shared._mask_invalid(values, [True, False, True])
        np.testing.assert_array_equal(masked, [1.0, np.nan, 3.0])
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])

    def test_mask_invalid_requires_matching_lengths(self):
        """Values and validity of different lengths raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            shared._mask_invalid([1.0, 2.0], [True])


class TestIntegerSampleFrequency:
    """`_require_integer_valued_sample_frequency`."""

    @pytest.mark.parametrize("fs", [2048, 2048.0, np.float64(1000.0)])
    def test_whole_numbers_become_int(self, fs):
        """Whole-number rates, of any numeric type, are returned as int."""
        result = shared._require_integer_valued_sample_frequency(fs)
        assert result == int(fs)
        assert isinstance(result, int)

    def test_fractional_rate_is_rejected(self):
        """A fractional rate raises ValueError."""
        with pytest.raises(ValueError, match="exact integer value"):
            shared._require_integer_valued_sample_frequency(2047.5)

    @pytest.mark.parametrize("fs", [0, -1.0, np.inf, np.nan])
    def test_non_positive_or_non_finite_rate_is_rejected(self, fs):
        """Zero, negative, inf and NaN rates are rejected."""
        # Currently raised as MutuallyExclusiveArgsError; see the xfail below.
        with pytest.raises(
            (ValueError, MutuallyExclusiveArgsError), match="finite and positive"
        ):
            shared._require_integer_valued_sample_frequency(fs)

    def test_invalid_rate_raises_value_error(self):
        """Known bug (xfail): invalid rates should raise ValueError as documented."""
        with pytest.raises(ValueError):
            shared._require_integer_valued_sample_frequency(0)


class TestPostprocessingHelpers:
    """Postprocessing helpers in `m3resp.emg.processing.shared`."""

    def test_computed_category(self):
        """Returns one category's computed results, or {} when absent."""
        postprocessed = {"computed": {"features": {"amplitude": 1.0}}}
        assert shared._computed_category(postprocessed, "features") == {
            "amplitude": 1.0
        }
        assert shared._computed_category(postprocessed, "baseline") == {}
        assert shared._computed_category(None, "features") == {}

    @pytest.mark.parametrize(("value", "expected"), [(True, 1.0), (3, 3.0), (2.5, 2.5)])
    def test_as_parameter_value_scalars(self, value, expected):
        """Booleans, ints and floats become floats."""
        result = shared._as_parameter_value(value)
        assert result == expected
        assert isinstance(result, float)

    def test_as_parameter_value_arrays(self):
        """Sequences become numpy arrays."""
        np.testing.assert_array_equal(shared._as_parameter_value([1, 2]), [1, 2])

    def test_normalize_selected_postprocessing_defaults_to_everything(self):
        """No selection means every registered function."""
        selected = shared._normalize_selected_postprocessing(None)
        assert ("features", "amplitude") in selected
        assert len(selected) == sum(
            len(f) for f in shared.POSTPROCESSING_FUNCTIONS.values()
        )

    def test_normalize_selected_postprocessing_keeps_enabled_known_functions(self):
        """Only enabled, registered functions are selected."""
        selected = shared._normalize_selected_postprocessing(
            {
                "features": {"amplitude": True, "time_product": False},
                "baseline": {"unknown_function": True},
            }
        )
        assert selected == {("features", "amplitude")}

    def test_category_for_function(self):
        """A function name maps to its category, or None if unknown."""
        assert shared._category_for_function("snr_pseudo") == "quality_assessment"
        assert shared._category_for_function("nope") is None

    def test_missing_dependency_reported(self, monkeypatch):
        """A failing import is reported as a missing `resurfemg` install."""

        def _fail(name):
            raise ImportError(name)

        monkeypatch.setattr(shared, "import_module", _fail)
        assert "resurfemg" in shared._missing_postprocessing_dependency()

    def test_no_missing_dependency(self):
        """With `resurfemg` installed nothing is reported missing."""
        assert shared._missing_postprocessing_dependency() is None

    def test_unavailable_postprocessing_result_skips_every_selected_function(self):
        """Every selected function is skipped with the given reason."""
        peaks = np.array([1, 2])
        result = shared._unavailable_postprocessing_result(
            selected={("features", "amplitude"), ("baseline", "moving_baseline")},
            peak_indices=peaks,
            computed={},
            reason="missing",
            settings={"a": 1},
        )
        assert result["skipped"] == {
            "baseline.moving_baseline": "missing",
            "features.amplitude": "missing",
        }
        assert result["available"] == {}
        assert result["peak_indices"] is peaks

    def test_emg_optional_dependency_error_message(self):
        """The error message names the `m3resp[emg]` extra."""
        assert "m3resp[emg]" in str(shared._emg_optional_dependency_error())

    @pytest.mark.parametrize(
        "recording",
        [None, {"metadata": {}}, {"array": np.zeros((1, 2))}],
        ids=["not-dict", "no-array", "no-metadata"],
    )
    def test_require_emg_recording(self, recording):
        """Input without an array or metadata raises TypeError."""
        with pytest.raises(TypeError, match="EMG preprocessing expects"):
            shared._require_emg_recording(recording)


class TestChannelChoice:
    """Automatic EMG and ECG channel choice from channel labels."""

    def test_ecg_channels_by_label(self):
        """Channels labelled ECG/EKG are found; none gives None."""
        assert shared._choose_ecg_channel(["EMG", " ECG ", "ekg"]) == [1, 2]
        assert shared._choose_ecg_channel(["EMG1", "EMG2"]) is None

    def test_single_channel_is_used(self):
        """A single-channel recording uses channel 0."""
        assert shared._choose_emg_channel(1, None) == 0

    def test_only_non_ecg_channel_is_used(self):
        """The only non-ECG channel is chosen."""
        assert shared._choose_emg_channel(2, ["ECG", "dia"]) == 1

    @pytest.mark.parametrize(
        "labels", [None, ["EMG1", "EMG2"], ["a"]], ids=["none", "two-emg", "short"]
    )
    def test_ambiguous_channels_raise(self, labels):
        """Ambiguous or missing labels raise UnresolvedChannelError."""
        with pytest.raises(
            UnresolvedChannelError, match="preprocess_emg\\(channel=1\\)"
        ):
            shared._choose_emg_channel(2, labels)
