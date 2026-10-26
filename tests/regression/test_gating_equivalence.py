"""Equivalence tests: M3RESP `gate_ecg` vs. ReSurfEMG `gating`.

Each helper is tested in isolation: the refactored helper gets a `_GateContext`
built from the *original's* gated indices/mask, so a fill test only fails if the
fill logic differs. Mask construction has its own tests.

The original's helpers are nested inside `gating()`, so `_nested()` rebuilds them
from the code object (closures over sibling helpers are resolved recursively).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import resurfemg.preprocessing.ecg_removal as orig
import resurfemg.preprocessing.envelope as evl

# Adjust to wherever the refactored module lives.
import m3resp.emg.processing.ecg as new
from m3resp.processing.windows import rolling_rms

FS = 2048
N = 20 * FS
WIDTHS = [204, 205, 11]  # even, odd (default), small odd


# --------------------------------------------------------------------------
# Pull nested helpers out of the original `gating`
# --------------------------------------------------------------------------
def _nested(outer, name, _cache=None):
    cache = {} if _cache is None else _cache
    if name in cache:
        return cache[name]
    code = next(
        c
        for c in outer.__code__.co_consts
        if isinstance(c, types.CodeType) and c.co_name == name
    )
    closure = tuple(
        types.CellType(_nested(outer, fv, cache)) for fv in code.co_freevars
    )
    fn = types.FunctionType(code, outer.__globals__, name, None, closure or None)
    cache[name] = fn
    return fn


def O(name):  # noqa: E743 - short alias for "original helper"
    return _nested(orig.gating, name)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _signal(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, N)
    x[1000:1010] = np.nan  # a few NaNs to exercise NaN-aware paths
    return x


def _signal_finite(seed=0):
    return np.random.default_rng(seed).normal(0.0, 1.0, N)


PEAK_SETS = {
    "interior": np.arange(2000, N - 2000, 1600),
    "edges": np.array([0, 3, 50, 5000, N - 60, N - 4, N - 1]),
    "overlap": np.array([3000, 3100, 3150, 9000, 9050]),
    "unsorted": np.array([12000, 4000, 8000]),
}


@pytest.fixture(params=list(PEAK_SETS), ids=list(PEAK_SETS))
def peaks(request):
    return np.sort(PEAK_SETS[request.param]).astype(np.int64)


@pytest.fixture(params=WIDTHS, ids=[f"w{w}" for w in WIDTHS])
def width(request):
    return request.param


def _ctx(signal, peaks, width, half, gated_idxs, gate_mask=None):
    """_GateContext built from externally supplied gated indices."""
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    peaks = np.sort(peaks)
    gated_idxs = np.unique(gated_idxs)
    if gate_mask is None:
        gate_mask = np.zeros(n, dtype=bool)
        gate_mask[gated_idxs] = True
    left = (width - 1) // 2
    right = width // 2
    return new._GateContext(
        array_original=sig,
        array_gated=sig.copy(),
        peaks=peaks,
        max_samples=n,
        gate_width_samples=width,
        half_gate_width=half,
        gate_start_indexes=np.clip(peaks - left, 0, n).astype(np.int64),
        gate_end_indexes=np.clip(peaks + right + 1, 0, n).astype(np.int64),
        gate_mask=gate_mask,
        gated_indexes=gated_idxs,
    )


def assert_same(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------
# Mask construction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", [0, 1, 2])
def test_gated_indices_methods_012(peaks, width, method):
    expected = np.unique(O("_compute_gated_indices")(peaks, width, N))
    ctx = new._build_gate_context(_signal(), peaks, width)
    np.testing.assert_array_equal(ctx.gated_indexes, expected)


def test_gate_mask_method_3(peaks, width):
    exp_mask, exp_idxs = O("_compute_gate_mask")(peaks, width, N)
    ctx = new._build_gate_context(_signal(), peaks, width)
    np.testing.assert_array_equal(ctx.gate_mask, exp_mask)
    np.testing.assert_array_equal(ctx.gated_indexes, exp_idxs)


@pytest.mark.parametrize("width_", [11, 204, 205])
def test_windowed_mean(width_):
    sig = _signal()
    starts = np.array([-500, -5, 0, 995, 1003, 5000, N - width_, N - 3, N + 10])
    ctx = _ctx(sig, np.array([5000]), width_, width_ // 2, np.array([5000]))
    assert_same(
        new._windowed_mean(ctx, starts), O("_masked_window_means")(sig, starts, width_)
    )


@pytest.mark.parametrize("width_", [11, 204, 205])
def test_rolling_rms(width_):
    """Covers the `# TODO: check correctness` on the `rolling_rms` import."""
    sig = _signal()
    sig[5000:5205] = np.nan  # gated region, as fed by _gate_fill_rms
    assert_same(
        rolling_rms(values=sig.copy(), window_length=width_),
        evl.full_rolling_rms(sig.copy(), width_),
    )


def test_interpolate_missing_samples(peaks, width):
    sig = _signal_finite()
    gated = np.unique(O("_compute_gated_indices")(peaks, width, N))
    rng = np.random.default_rng(1)
    mask = rng.random(gated.size) < 0.5

    expected = O("_interp_missing_samples")(sig.copy(), mask, gated)
    ctx = _ctx(sig, peaks, width, width // 2, gated)
    new._interpolate_missing_samples(ctx, mask)
    assert_same(ctx.array_gated, expected)


@pytest.mark.parametrize("which", ["first", "last", "both_ends"])
def test_interpolate_missing_samples_boundaries(which):
    sig = _signal_finite()
    gated = {
        "first": np.arange(0, 20),
        "last": np.arange(N - 20, N),
        "both_ends": np.r_[0:20, N - 20 : N],
    }[which]
    mask = np.ones(gated.size, dtype=bool)
    expected = O("_interp_missing_samples")(sig.copy(), mask, gated)
    ctx = _ctx(sig, np.array([10]), 21, 10, gated)
    new._interpolate_missing_samples(ctx, mask)
    assert_same(ctx.array_gated, expected)


def test_interpolate_missing_samples_no_anchors():
    """Every sample missing: original returns unchanged, no exception."""
    sig = _signal_finite()[:50]
    gated = np.arange(50)
    mask = np.ones(50, dtype=bool)
    expected = O("_interp_missing_samples")(sig.copy(), mask, gated)
    ctx = _ctx(sig, np.array([25]), 51, 25, gated)
    new._interpolate_missing_samples(ctx, mask)
    assert_same(ctx.array_gated, expected)


# --------------------------------------------------------------------------
# Fill helpers (fed the original's gated indices)
# --------------------------------------------------------------------------
def test_fill_zeros(peaks, width):
    sig = _signal()
    gated = O("_compute_gated_indices")(peaks, width, N)
    expected = sig.copy()
    expected[gated] = 0  # inline in original `gating`
    got = new._gate_fill_zeros(_ctx(sig, peaks, width, width // 2, gated))
    assert_same(got, expected)


def test_fill_interp(peaks, width):
    sig = _signal()
    gated = O("_compute_gated_indices")(peaks, width, N)
    fill_idxs = O("_compute_gate_peak_mapping")(peaks, gated, width, N)
    expected = O("_gate_fill_interpolation")(
        fill_idxs=fill_idxs,
        emg_raw_gated=sig.copy(),
        gated_idxs=gated,
        peak_idxs=peaks,
        gate_width=width,
    )
    got = new._gate_fill_interp(_ctx(sig, peaks, width, width // 2, gated))
    assert_same(got, expected)


def test_fill_prior_mean(peaks, width):
    sig = _signal()
    gated = O("_compute_gated_indices")(peaks, width, N)
    expected = O("_gate_fill_prior_mean")(
        emg_raw=sig,
        emg_raw_gated=sig.copy(),
        gated_idxs=gated,
        peak_idxs=peaks,
        gate_width=width,
    )
    got = new._gate_fill_prior_mean(_ctx(sig, peaks, width, width // 2, gated))
    assert_same(got, expected)


def test_fill_rms(peaks, width):
    sig = _signal()
    mask, gated = O("_compute_gate_mask")(peaks, width, N)
    expected = O("_gate_fill_rms")(sig.copy(), mask, gated, width)
    got = new._gate_fill_rms(_ctx(sig, peaks, width, width // 2, gated, mask))
    assert_same(got, expected)


# --------------------------------------------------------------------------
# Public function
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", [0, 1, 2, 3])
def test_gate_ecg_end_to_end(peaks, width, method):
    sig = _signal()
    expected = orig.gating(sig.copy(), peaks, gate_width=width, method=method)
    got = new.gate_ecg(sig.copy(), peaks, gate_width_samples=width, fill_method=method)
    assert_same(got, expected)


def test_gate_ecg_default_method():
    sig = _signal()
    peaks = PEAK_SETS["interior"]
    assert_same(new.gate_ecg(sig.copy(), peaks), orig.gating(sig.copy(), peaks))


def test_gate_ecg_does_not_mutate_input():
    sig = _signal()
    before = sig.copy()
    for m in range(4):
        new.gate_ecg(sig, PEAK_SETS["interior"], fill_method=m)
    assert_same(sig, before)


def test_gate_ecg_capture_mask():
    store = {}
    new.gate_ecg(_signal(), PEAK_SETS["interior"], fill_method=3, capture_mask=store)
    exp_mask, _ = O("_compute_gate_mask")(PEAK_SETS["interior"], 205, N)
    np.testing.assert_array_equal(store["gate_mask"], exp_mask)
