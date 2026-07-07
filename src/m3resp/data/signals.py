"""``Signal``: the modality-tagged runtime signal type (plan_stage2.md Sec 9).

Adapters convert whatever ``eitprocessing``/``resurfemg`` return into
``Signal`` instances at the public boundary (Milestone 2.3); everything
downstream - session storage, pipeline steps, export - operates on this type
instead of vendor-specific objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from m3resp.data.timeseries import TimeSeries

#: Controlled vocabulary for ``Signal.modality`` (data model doc Sec 2.5).
Modality = Literal["eit", "emg", "ventilator", "pressure", "flow", "unknown"]
_VALID_MODALITIES = frozenset(get_args(Modality))

#: Controlled vocabulary for ``Signal.processing_state``.
ProcessingState = Literal["raw", "filtered", "processed", "derived"]
_VALID_PROCESSING_STATES = frozenset(get_args(ProcessingState))


@dataclass
class Signal(TimeSeries):
    """A :class:`TimeSeries` tagged with modality and provenance context."""

    modality: Modality = "unknown"
    channel: str | None = None
    source: str | None = None
    processing_state: ProcessingState = "raw"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.modality not in _VALID_MODALITIES:
            raise ValueError(
                f"Unknown Signal.modality {self.modality!r}; expected one of "
                f"{sorted(_VALID_MODALITIES)}"
            )
        if self.processing_state not in _VALID_PROCESSING_STATES:
            raise ValueError(
                f"Unknown Signal.processing_state {self.processing_state!r}; "
                f"expected one of {sorted(_VALID_PROCESSING_STATES)}"
            )

    def to_manifest_row(self) -> dict[str, object]:
        row = super().to_manifest_row()
        row.update(
            {
                "modality": self.modality,
                "channel": self.channel,
                "source": self.source,
                "processing_state": self.processing_state,
            }
        )
        return row
