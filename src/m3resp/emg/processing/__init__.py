"""M3resp-native EMG processing module."""

from __future__ import annotations

from m3resp.emg.processing.shared import POSTPROCESSING_FUNCTIONS

from .baseline import _BaselineMixin
from .core import _CoreMixin
from .defaults import _DefaultsMixin
from .ecg import _EcgMixin
from .protocols import _DefaultsProtocol, _PostprocessingOpsProtocol
from .quality import _QualityMixin
from .signals import peak_indices_from_events, ventilator_signals

__all__ = [
    "POSTPROCESSING_FUNCTIONS",
    "_BaselineMixin",
    "_CoreMixin",
    "_DefaultsMixin",
    "_DefaultsProtocol",
    "_EcgMixin",
    "_PostprocessingOpsProtocol",
    "_QualityMixin",
    "peak_indices_from_events",
    "ventilator_signals",
]
