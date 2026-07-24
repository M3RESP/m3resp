"""ECG-artifact-handling methods of `ReSurfEMGAdapter`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._shared import (
    _emg_optional_dependency_error,
    _require_1d_array,
    _require_index_array,
    _require_integer_valued_sample_frequency,
)


class _EcgMixin:
    def detect_ecg_peaks(
        self,
        signal: Any,
        *,
        sample_frequency: float,
        peak_fraction: float = 0.4,
        peak_width_samples: int | None = None,
        peak_distance_samples: int | None = None,
        bandpass_filter: bool = True,
    ) -> np.ndarray:
        """Detect ECG peak sample indices in `signal` (ECG or an
        ECG-contaminated EMG channel)."""

        try:
            from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("signal", signal)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        return np.asarray(
            detect_ecg_peaks(
                array,
                fs,
                peak_fraction=peak_fraction,
                peak_width_s=peak_width_samples,
                peak_distance=peak_distance_samples,
                bp_filter=bandpass_filter,
            ),
            dtype=int,
        )

    def gate_ecg(
        self,
        signal: Any,
        peak_indices: Any,
        *,
        gate_width_samples: int = 205,
        fill_method: int = 1,
    ) -> np.ndarray:
        """Gate (remove) ECG peaks from `signal` around `peak_indices`.

        `fill_method`: 0 zeros, 1 interpolation (default upstream), 2 mean
        of a neighboring segment, 3 running-RMS-based replacement.
        """

        try:
            from resurfemg.preprocessing.ecg_removal import gating
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("signal", signal)
        peaks = _require_index_array("peak_indices", peak_indices)
        if fill_method not in (0, 1, 2, 3):
            raise ValueError(
                f"fill_method must be one of 0, 1, 2, 3; got {fill_method!r}."
            )
        if gate_width_samples <= 0:
            raise ValueError(
                f"gate_width_samples must be positive; got {gate_width_samples!r}."
            )
        return np.asarray(
            gating(
                array,
                peaks,
                gate_width=gate_width_samples,
                method=fill_method,
            )
        )

    def wavelet_denoise_ecg(
        self,
        signal: Any,
        peak_indices: Any,
        *,
        sample_frequency: float,
        hard_thresholding: bool = True,
        levels: int = 4,
        wavelet_type: str = "db2",
        fixed_threshold: float = 4.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Remove ECG artifacts from `signal` via a-trous wavelet shrinkage.

        Returns `(cleaned_signal, decomposition, thresholds, gate_mask)`.
        """

        try:
            from resurfemg.preprocessing.ecg_removal import wavelet_denoising
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("signal", signal)
        peaks = _require_index_array("peak_indices", peak_indices)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        if levels <= 0:
            raise ValueError(f"levels must be positive; got {levels!r}.")
        cleaned, decomposition, thresholds, gate_mask = wavelet_denoising(
            array,
            peaks,
            fs,
            hard_thresholding=hard_thresholding,
            n=levels,
            wavelet_type=wavelet_type,
            fixed_threshold=fixed_threshold,
        )
        return (
            np.asarray(cleaned),
            np.asarray(decomposition),
            np.asarray(thresholds),
            np.asarray(gate_mask),
        )
