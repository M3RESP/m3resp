"""Adapter boundary for the upstream `resurfemg` package.

`ReSurfEMGAdapter` is composed from mixins split by responsibility
(load/postprocess orchestration, ECG handling, baseline estimation, quality
assessment, and native fallback implementations) purely for file
navigability - the class behaves exactly as it did as a single module.
`ventilator_signals`/`peak_indices_from_events` and
`POSTPROCESSING_FUNCTIONS` are re-exported here so
`from m3resp.adapters.resurfemg_adapter import <name>` keeps working
unchanged.
"""

from __future__ import annotations

from .baseline import _BaselineMixin
from .core import _CoreMixin
from .defaults import _DefaultsMixin
from .ecg import _EcgMixin
from .quality import _QualityMixin
from .shared import POSTPROCESSING_FUNCTIONS
from .signals import peak_indices_from_events, ventilator_signals


class ReSurfEMG(
    _CoreMixin,
    _EcgMixin,
    _BaselineMixin,
    _QualityMixin,
    _DefaultsMixin,
):
    """Ported methods from the `ReSurfEMG` package."""


__all__ = [
    "POSTPROCESSING_FUNCTIONS",
    "ReSurfEMG",
    "peak_indices_from_events",
    "ventilator_signals",
]
