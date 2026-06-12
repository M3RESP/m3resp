"""M3Resp public API."""

from m3resp.core.config import (
    RotarcWorkflowConfig,
    WorkflowConfig,
    load_workflow_config,
)
from m3resp.core.events import (
    BreathEvent,
    Event,
    coerce_breath_event,
    coerce_breath_events,
    coerce_event,
    event_to_dict,
)
from m3resp.core.session import M3Session
from m3resp.modalities.eit import load as load_eit
from m3resp.modalities.emg import load as load_emg

__version__ = "0.1.0"

__all__ = [
    "BreathEvent",
    "Event",
    "M3Session",
    "RotarcWorkflowConfig",
    "WorkflowConfig",
    "coerce_breath_event",
    "coerce_breath_events",
    "coerce_event",
    "event_to_dict",
    "load_eit",
    "load_emg",
    "load_workflow_config",
]
