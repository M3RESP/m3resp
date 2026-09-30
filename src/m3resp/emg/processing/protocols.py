"""Structural protocols describing what one `ReSurfEMGAdapter` mixin calls on
another, so mypy can type-check cross-mixin `self` calls without each mixin
depending on the concrete (later-composed) `ReSurfEMGAdapter` class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class _DefaultsProtocol(Protocol):  # noqa: PYI046 -- used via `cast()` in _core.py
    """What `_CoreMixin` calls on the defaults mixin."""

    def _preprocess_default(self, signal: Any, **kwargs: Any) -> Any: ...

    def _detect_breaths_default(self, signal: Any, **kwargs: Any) -> Any: ...

    def _postprocess_default(
        self, processed_emg: Any, events: Any = None, **kwargs: Any
    ) -> dict[str, Any]: ...


class _PreprocessingOpsProtocol(Protocol):  # noqa: PYI046 -- used as a `self` annotation in _defaults.py
    """What `_DefaultsMixin._preprocess_default` calls on the ecg mixin."""

    def gate_ecg(
        self,
        signal: np.ndarray,
        peak_indices: np.ndarray,
        *,
        gate_width_samples: int = 205,
        fill_method: int = 1,
        capture_mask: dict[str, Any] | None = None,
    ) -> np.ndarray: ...

    def wavelet_denoise_ecg(
        self,
        signal: np.ndarray,
        peak_indices: np.ndarray,
        *,
        sample_frequency: float,
        hard_thresholding: bool = True,
        levels: int = 4,
        wavelet_type: str = "db2",
        fixed_threshold: float = 4.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

    def detect_ecg_peaks(
        self,
        signal: np.ndarray,
        *,
        sample_frequency: float,
        peak_fraction: float = 0.4,
        peak_width_samples: int | None = None,
        peak_distance_samples: int | None = None,
        apply_bandpass_filter: bool = True,
    ) -> np.ndarray: ...


class _PostprocessingOpsProtocol(Protocol):  # noqa: PYI046 -- used as a `self` annotation in _defaults.py
    """What `_DefaultsMixin._postprocess_default` calls on the core/baseline/
    quality mixins.
    """

    def run_postprocessing_function(
        self, category: str, function_name: str, *args: Any, **kwargs: Any
    ) -> Any: ...

    def available_postprocessing(self) -> dict[str, list[str]]: ...

    def moving_baseline(self, *args: Any, **kwargs: Any) -> Any: ...

    def slopesum_baseline(self, *args: Any, **kwargs: Any) -> Any: ...

    def snr_pseudo(self, *args: Any, **kwargs: Any) -> Any: ...

    def percentage_under_baseline(self, *args: Any, **kwargs: Any) -> Any: ...

    def detect_local_high_aub(self, *args: Any, **kwargs: Any) -> Any: ...

    def detect_extreme_time_products(self, *args: Any, **kwargs: Any) -> Any: ...

    def detect_non_consecutive_manoeuvres(self, *args: Any, **kwargs: Any) -> Any: ...

    def evaluate_bell_curve_error(self, *args: Any, **kwargs: Any) -> Any: ...

    def evaluate_event_timing(self, *args: Any, **kwargs: Any) -> Any: ...

    def evaluate_respiratory_rates(self, *args: Any, **kwargs: Any) -> Any: ...
