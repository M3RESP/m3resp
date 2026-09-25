"""M3resp-native EMG processing module.

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
