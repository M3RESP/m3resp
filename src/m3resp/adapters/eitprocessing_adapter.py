"""Adapter boundary for the upstream `eitprocessing` package."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError


class EITProcessingAdapter:
    """Thin wrapper around `eitprocessing`.

    Stage 1 keeps this adapter deliberately small. It imports `eitprocessing`
    only when used so `m3resp` can be installed without optional EIT support.
    """

    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader

    def load(self, path: str, vendor: str | None = None, **kwargs: Any) -> Any:
        """Load EIT data through `eitprocessing` or an injected loader."""

        if self._loader is not None:
            return self._loader(path, vendor=vendor, **kwargs)

        try:
            from eitprocessing.datahandling.loading import load_eit_data
        except ImportError as exc:
            raise OptionalDependencyError(
                "EIT support requires the optional dependency `eitprocessing`. "
                'Install with `pip install "m3resp[eit]"` or inject a loader.'
            ) from exc

        return load_eit_data(path, vendor=vendor, **kwargs)

    def get_global_impedance(self, sequence: Any, label: str = "raw") -> Any:
        """Return global impedance from a loaded `eitprocessing` sequence."""

        eit_data = sequence.eit_data[label]
        return eit_data.get_summed_impedance()

    def detect_breaths(self, data: Any, **kwargs: Any) -> list[BreathEvent]:
        """Normalize upstream EIT breath detections into `BreathEvent` objects."""

        detector = kwargs.pop("detector", None)
        if detector is None:
            raise UnsupportedWorkflowError(
                "No default EIT breath detector is configured yet. Pass "
                "`detector=callable` to normalize detections in Stage 1."
            )

        detections = detector(data, **kwargs)
        return _coerce_breath_events(detections, modality="eit", source="eitprocessing")

    def compute_tiv(self, sequence: Any, **kwargs: Any) -> Any:
        """Compute tidal impedance variation when an upstream function is provided."""

        compute = kwargs.pop("compute", None)
        if compute is None:
            raise UnsupportedWorkflowError(
                "TIV computation needs an upstream callable in Stage 1. "
                "Pass `compute=callable`."
            )
        return compute(sequence, **kwargs)


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
