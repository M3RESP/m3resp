"""Adapter boundary for the upstream `resurfemg` package."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError


class ReSurfEMGAdapter:
    """Thin wrapper around `resurfemg`."""

    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader

    def load(self, path: str, **kwargs: Any) -> Any:
        """Load EMG data through `resurfemg` or an injected loader."""

        if self._loader is not None:
            return self._loader(path, **kwargs)

        try:
            import resurfemg  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError(
                "EMG support requires the optional dependency `resurfemg`. "
                'Install with `pip install "m3resp[emg]"` or inject a loader.'
            ) from exc

        raise UnsupportedWorkflowError(
            "No default ReSurfEMG loader is configured yet. Pass an injected "
            "`loader=callable` when creating `ReSurfEMGAdapter`."
        )

    def preprocess(self, signal: Any, **kwargs: Any) -> Any:
        """Preprocess EMG data when an upstream callable is provided."""

        preprocess = kwargs.pop("preprocess", None)
        if preprocess is None:
            raise UnsupportedWorkflowError(
                "EMG preprocessing needs an upstream callable in Stage 1. "
                "Pass `preprocess=callable`."
            )
        return preprocess(signal, **kwargs)

    def detect_breaths(self, signal: Any, **kwargs: Any) -> list[BreathEvent]:
        """Normalize upstream EMG breath detections into `BreathEvent` objects."""

        detector = kwargs.pop("detector", None)
        if detector is None:
            raise UnsupportedWorkflowError(
                "No default EMG breath detector is configured yet. Pass "
                "`detector=callable` to normalize detections in Stage 1."
            )

        detections = detector(signal, **kwargs)
        return _coerce_breath_events(detections, modality="emg", source="resurfemg")

    def compute_features(self, signal: Any, events: Sequence[BreathEvent], **kwargs: Any) -> Any:
        """Compute EMG features when an upstream callable is provided."""

        compute = kwargs.pop("compute", None)
        if compute is None:
            raise UnsupportedWorkflowError(
                "EMG feature extraction needs an upstream callable in Stage 1. "
                "Pass `compute=callable`."
            )
        return compute(signal, events, **kwargs)


def _coerce_breath_events(
    detections: Sequence[Any], modality: str, source: str
) -> list[BreathEvent]:
    events: list[BreathEvent] = []
    for item in detections:
        if isinstance(item, BreathEvent):
            events.append(item)
            continue

        if isinstance(item, dict):
            events.append(
                BreathEvent(
                    modality=modality,
                    start_time=float(item["start_time"]),
                    end_time=float(item["end_time"]),
                    peak_time=(
                        None
                        if item.get("peak_time") is None
                        else float(item["peak_time"])
                    ),
                    source=item.get("source", source),
                    confidence=item.get("confidence"),
                    metadata=item.get("metadata", {}),
                )
            )
            continue

        start_time, end_time, *rest = item
        peak_time = rest[0] if rest else None
        events.append(
            BreathEvent(
                modality=modality,
                start_time=float(start_time),
                end_time=float(end_time),
                peak_time=None if peak_time is None else float(peak_time),
                source=source,
            )
        )
    return events
