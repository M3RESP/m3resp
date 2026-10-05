"""Layer 1: the data types used while a session runs.

EIT, EMG and ventilator processing code creates, passes around and returns
these objects. The saved records in ``m3resp.datamodel`` (Layer 2) are built
from them; ``docs/developer/architecture.md`` explains how the two layers fit
together.
"""

from __future__ import annotations

from m3resp.data.categories import (
    KNOWN_CATEGORIES,
    Category,
    load_category_aliases,
    normalize_category,
    register_category_alias,
)
from m3resp.data.collections import (
    ParameterResultCollection,
    QualityReport,
    SignalCollection,
)
from m3resp.data.event_data import EventData, IntervalData
from m3resp.data.events import (
    BreathEvent,
    Event,
    Interval,
    coerce_breath_event,
    coerce_breath_events,
    coerce_event,
    coerce_events,
    coerce_interval,
    coerce_intervals,
    event_to_dict,
)
from m3resp.data.linked_breath import LinkedBreath
from m3resp.data.parameters import ParameterResult
from m3resp.data.processing import ProcessingHistory, ProcessingStep
from m3resp.data.quality import QualityFlag
from m3resp.data.signals import Signal
from m3resp.data.timeseries import TimeSeries

__all__ = [
    "KNOWN_CATEGORIES",
    "BreathEvent",
    "Category",
    "Event",
    "EventData",
    "Interval",
    "IntervalData",
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
    "coerce_breath_event",
    "coerce_breath_events",
    "coerce_event",
    "coerce_events",
    "coerce_interval",
    "coerce_intervals",
    "event_to_dict",
    "load_category_aliases",
    "normalize_category",
    "register_category_alias",
]
