"""Structural protocols describing what one `ReSurfEMGAdapter` mixin calls on
another, so mypy can type-check cross-mixin `self` calls without each mixin
depending on the concrete (later-composed) `ReSurfEMGAdapter` class."""

from __future__ import annotations

from typing import Any, Protocol


class _DefaultsProtocol(Protocol):  # noqa: PYI046 -- used via `cast()` in _core.py
    """What `_CoreMixin` calls on the defaults mixin."""

    def _preprocess_default(self, signal: Any, **kwargs: Any) -> Any: ...

    def _detect_breaths_default(self, signal: Any, **kwargs: Any) -> Any: ...

    def _postprocess_default(
        self, processed_emg: Any, events: Any = None, **kwargs: Any
    ) -> dict[str, Any]: ...


class _PostprocessingOpsProtocol(Protocol):  # noqa: PYI046 -- used as a `self` annotation in _defaults.py
    """What `_DefaultsMixin._postprocess_default` calls on the core/baseline/
    quality mixins."""

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
