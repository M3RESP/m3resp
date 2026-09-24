"""`ReSurfEMGAdapter`: how m3resp reads and processes sEMG with ReSurfEMG.

The adapter's methods are spread over several files by topic: loading,
preprocessing and postprocessing (``_core.py``), ECG removal (``_ecg.py``),
baselines (``_baseline.py``), quality checks (``_quality.py``), and the
default preprocessing, breath detection and postprocessing (``_defaults.py``).
They all belong to the one `ReSurfEMGAdapter` class.
`ventilator_signals`, `peak_indices_from_events` and
`POSTPROCESSING_FUNCTIONS` can also be imported from here.
"""

from __future__ import annotations

from ._baseline import _BaselineMixin
from ._core import _CoreMixin
from ._defaults import _DefaultsMixin
from ._ecg import _EcgMixin
from ._quality import _QualityMixin
from ._shared import POSTPROCESSING_FUNCTIONS
from ._signals import peak_indices_from_events, ventilator_signals


class ReSurfEMGNative(
    _CoreMixin,
    _EcgMixin,
    _BaselineMixin,
    _QualityMixin,
    _DefaultsMixin,
):
    """Read and process sEMG with ReSurfEMG, and turn results into m3resp objects.

    Each method takes plain arrays or the recording dictionary from `load`
    (samples under ``"array"``, one row per channel; sampling rate in Hz
    under ``metadata["fs"]``) and returns arrays, `BreathEvent`s, `Signal`s,
    `ParameterResult`s or `QualityFlag`s. Values keep the units of the
    recording (often uV or mV); times are in seconds, positions in samples.

    Most methods call one ReSurfEMG function with the same settings and
    defaults (filters, ECG removal, baselines, quality checks). Some are
    computed in m3resp itself: the notch filter, the envelope, breath
    detection and the per-breath measures; see
    ``docs/developer/adapters.md`` for the full list.

    `M3Session` holds one adapter as ``session.emg_adapter``, and the
    ``emg.*`` workflow steps call it. Pass ``loader=`` to read a file format
    ReSurfEMG does not know; the loader returns the same recording
    dictionary.
    """


__all__ = [
    "POSTPROCESSING_FUNCTIONS",
    "ReSurfEMGNative",
    "peak_indices_from_events",
    "ventilator_signals",
]
