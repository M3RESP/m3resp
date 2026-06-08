"""EMG modality containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EMGRecording:
    """Loaded EMG recording with source metadata."""

    data: Any
    path: Path
    raw: Any = None
    dataframe: Any = None
    metadata: dict[str, Any] | None = None
    filtered: Any = None
    envelope: Any = None
    channel: int | None = None
    fs: float | None = None


def load(
    path: str | Path,
    *,
    adapter: Any = None,
    **kwargs: Any,
) -> EMGRecording:
    """Load an EMG recording through the Stage 1 adapter."""

    from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter

    emg_adapter = adapter or ReSurfEMGAdapter()
    recording = emg_adapter.load(str(path), **kwargs)
    raw = recording.get("array") if isinstance(recording, dict) else None
    dataframe = recording.get("dataframe") if isinstance(recording, dict) else None
    metadata = recording.get("metadata") if isinstance(recording, dict) else None
    return EMGRecording(
        data=recording,
        path=Path(path),
        raw=raw,
        dataframe=dataframe,
        metadata=metadata,
    )
