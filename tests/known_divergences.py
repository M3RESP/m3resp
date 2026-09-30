import pytest


def _known_divergence(
    reason: str,
    *,
    raises: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    strict: bool = False,
) -> pytest.MarkDecorator:
    return pytest.mark.xfail(reason=reason, raises=raises, strict=strict, run=True)


MISSING_TEST_DATA_FILE = _known_divergence(
    "The test data file is missing from the repository, so these tests cannot be run.",
    raises=FileNotFoundError,
)
GATE_CLIP_TO_0 = _known_divergence(
    "When using method 1 for gating, if the starting index of the gate is negative,"
    "resurfemg doesn't clip it to 0, folding it into the end of the signal."
    "The adapter does clip to 0, so this is a known divergence."
)

GATE_INTERP_AT_END = _known_divergence(
    "native `_gate_fill_interp` clips the post-gate index to `max_samples` "
    "instead of `max_samples - 1` (IndexError when a gate reaches the end of "
    "the signal) and treats index 0 as out of bounds via `np.where(pre, ...)`."
)
WAVELET_THRESHOLD = _known_divergence(
    "native `wavelet_denoise_ecg` computes `fixed_threshold & std_estimate` "
    "(bitwise and, TypeError) instead of `*`, and never fills `thresholds`."
)
# FIXED
SLOPESUM_BASELINE = _known_divergence(
    "native `slopesum_baseline` omits upstream's 1.2 scale factor on the "
    "baseline and computes the running mean/std over `moving_average_samples` "
    "instead of `window_samples`."
)
# FIXED
SLOPESUM_DEFAULT_PERCENTILE_WINDOW = _known_divergence(
    "native `slopesum_baseline` defaults `percentile_window_samples` to "
    "`fs // 2`; upstream defaults `perc_window` to `fs` (plus the "
    "SLOPESUM_BASELINE divergence)."
)
# FIXED
BELL_CURVE_RETURN_ORDER = _known_divergence(
    "`evaluate_bell_curve_error` return slots 1 and 2 are swapped: upstream "
    "(and hence the adapter) returns (valid, bell_error, percentage, ...), the "
    "native port returns (valid, percentage, bell_error, ...)."
)
AUB_REFERENCE_SIGNAL = _known_divergence(
    "upstream `percentage_under_baseline` ignores `ref_signal` (it passes "
    "`ref_signal=signal` to `area_under_baseline`); the native port honours "
    "`reference_signal`."
)
EVENT_TIMING_DEFAULTS = _known_divergence(
    "`evaluate_event_timing` defaults differ: adapter (min_delta=-0.5, "
    "max_delta=2.0), native (min_delta=0.0, max_delta=None)."
)
VENTILATOR_SIGNALS_HELPER = _known_divergence(
    "the adapter's `ventilator_signals` delegates to "
    "`ventilator_adapter.split_channels` (label-based resolution, extra "
    "`origins` key); the native copy still indexes fixed columns."
)


__all__ = [
    "AUB_REFERENCE_SIGNAL",
    "BELL_CURVE_RETURN_ORDER",
    "EVENT_TIMING_DEFAULTS",
    "GATE_CLIP_TO_0",
    "GATE_INTERP_AT_END",
    "MISSING_TEST_DATA_FILE",
    "SLOPESUM_BASELINE",
    "SLOPESUM_DEFAULT_PERCENTILE_WINDOW",
    "VENTILATOR_SIGNALS_HELPER",
    "WAVELET_THRESHOLD",
]
