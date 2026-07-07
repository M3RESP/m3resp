"""Layer 1: shared runtime scientific data model (plan_stage2.md Milestone 2.1).

These are the objects EIT, EMG, and ventilator processing code should create,
pass around, and return - replacing the untyped dicts adapters currently
produce. See ``plan/plan_stage2.md`` (Sec 8-13) for the design rationale and
``plan/stage2_consolidation.md`` for how these relate to the persisted
entities in ``m3resp.datamodel``.

``Event``/``Breath`` are intentionally re-exported from ``m3resp.core.events``
rather than redefined here: that module already implements them and is used
throughout Stage 1 (``M3Session``, adapters, the pipeline engine). Duplicating
them would fork the type Stage 1 code already depends on.
"""

from __future__ import annotations

from m3resp.core.events import BreathEvent as Breath
from m3resp.core.events import Event
from m3resp.data.collections import (
    ParameterResultCollection,
    QualityReport,
    SignalCollection,
)
from m3resp.data.linked_breath import LinkedBreath
from m3resp.data.parameters import ParameterResult
from m3resp.data.processing import ProcessingHistory, ProcessingStep
from m3resp.data.quality import QualityFlag
from m3resp.data.signals import Signal
from m3resp.data.timeseries import TimeSeries

__all__ = [
    "Breath",
    "Event",
    "LinkedBreath",
    "ParameterResult",
    "ParameterResultCollection",
    "ProcessingHistory",
    "ProcessingStep",
    "QualityFlag",
    "QualityReport",
    "Signal",
    "SignalCollection",
    "TimeSeries",
]
