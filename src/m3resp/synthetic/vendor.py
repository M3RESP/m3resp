"""ReSurfEMG-backed synthetic generation and reproducibility guards."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from typing import Any

import numpy as np

DEFAULT_ZERO = 0.0
DEFAULT_FLOAT32_DTYPE = np.float32
DEFAULT_RETRY_ECG_ACCELERATION = 1.0


def _load_resurfemg_synthetic() -> Any:
    try:
        return importlib.import_module("resurfemg.pipelines.synthetic_data")
    except ImportError as exc:
        raise RuntimeError(
            "EMG and ventilator synthetic generation requires the optional "
            "dependency `resurfemg`. Install m3resp with the EMG/all extra or "
            "disable generate_emg and generate_ventilator."
        ) from exc


@contextlib.contextmanager
def _seeded_vendor_randomness(seed: int) -> Iterator[None]:
    """Make `resurfemg`/`neurokit2` calls reproducible for `seed`, for calls
    that do not expose their own seed parameter.

    `resurfemg.pipelines.synthetic_data.simulate_raw_emg`/
    `simulate_ventilator_data` do not forward `config.seed` into either
    library: their own noise (`np.random.normal`, e.g.
    `resurfemg.data_connector.synthetic_data` lines 182/256/260/264) reads
    NumPy's legacy global random state, seeded here via `np.random.seed`;
    `neurokit2.ecg_simulate`'s ECG placement is called with
    `random_state=None` (`resurfemg.pipelines.synthetic_data` line ~84),
    which resolves to an *unseeded* `np.random.default_rng()` regardless of
    `np.random.seed` (see `neurokit2.misc.check_random_state`) - so this
    also temporarily patches `np.random.default_rng` to fall back to `seed`
    when called with no explicit seed, restoring the original afterward.
    """

    np.random.seed(seed)
    original_default_rng = np.random.default_rng

    def _seeded_default_rng(seed_arg: Any = None) -> np.random.Generator:
        return original_default_rng(seed if seed_arg is None else seed_arg)

    np.random.default_rng = _seeded_default_rng  # type: ignore[assignment]
    try:
        yield
    finally:
        np.random.default_rng = original_default_rng


def _simulate_raw_emg_length_safe(
    synth: Any,
    expected_samples: int,
    **kwargs: Any,
) -> np.ndarray:
    """Call ReSurfEMG and guard against its ECG/EMG length mismatch."""

    try:
        signal = synth.simulate_raw_emg(**kwargs)
    except ValueError as exc:
        if not _is_resurfemg_length_mismatch(exc):
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["ecg_acceleration"] = DEFAULT_RETRY_ECG_ACCELERATION
        signal = synth.simulate_raw_emg(**retry_kwargs)

    return _normalize_signal_length(signal, expected_samples)


def _is_resurfemg_length_mismatch(exc: ValueError) -> bool:
    message = str(exc)
    return "operands could not be broadcast together" in message


def _normalize_signal_length(signal: Any, expected_samples: int) -> np.ndarray:
    values = np.asarray(signal, dtype=DEFAULT_FLOAT32_DTYPE).reshape(-1)
    if len(values) == expected_samples:
        return values
    if len(values) > expected_samples:
        return values[:expected_samples]
    if len(values) == int(DEFAULT_ZERO):
        return np.zeros(expected_samples, dtype=DEFAULT_FLOAT32_DTYPE)
    return np.pad(
        values,
        (int(DEFAULT_ZERO), expected_samples - len(values)),
        mode="edge",
    )
