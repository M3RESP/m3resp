"""EIT modality containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EITRecording:
    """Loaded EIT recording with source metadata."""

    data: Any
    path: Path
    vendor: str | None = None
    raw: Any = None
    global_impedance: Any = None


def load(
    path: str | Path,
    vendor: str | None = None,
    *,
    adapter: Any = None,
    **kwargs: Any,
) -> EITRecording:
    """Load an EIT recording through the Stage 1 adapter."""

    from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter

    eit_adapter = adapter or EITProcessingAdapter()
    sequence = eit_adapter.load(str(path), vendor=vendor, **kwargs)
    raw = None
    global_impedance = None
    if hasattr(sequence, "eit_data"):
        raw = eit_adapter.get_raw_eit(sequence)
    if hasattr(sequence, "continuous_data") and raw is not None:
        global_impedance = eit_adapter.get_global_impedance(sequence)
    return EITRecording(
        data=sequence,
        path=Path(path),
        vendor=vendor,
        raw=raw,
        global_impedance=global_impedance,
    )
