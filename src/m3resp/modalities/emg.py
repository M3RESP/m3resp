"""EMG modality containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from m3resp.modalities.time_window import TimeWindow


@dataclass
class EMGRecording:
    """Loaded EMG recording with source metadata."""

    data: Any
    path: Path
    raw: Any = None
    dataframe: Any = None
    metadata: dict[str, Any] | None = None
    #: Band-passed signal, as produced by preprocessing.
    filtered: Any = None
    #: Band-passed signal with the ECG removed, when an ECG-removal step has
    #: run. Kept separate from `filtered` so both stages stay available.
    ecg_cleaned: Any = None
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


def sample_window(
    recording: EMGRecording, start_seconds: float, end_seconds: float | None
) -> TimeWindow:
    """The samples of an EMG recording between two times.

    Times are in seconds from the recording's first sample; ``end_seconds``
    None means up to the end. Raises ValueError when the window does not lie
    inside the recording.
    """

    data = recording.data
    fs = float(data["metadata"]["fs"])
    n_samples = np.asarray(data["array"]).shape[-1]
    duration = n_samples / fs
    end = duration if end_seconds is None else float(end_seconds)
    start = float(start_seconds)
    if not 0.0 <= start < end <= duration:
        raise ValueError(
            f"slice_emg: need 0 <= start < end <= {duration:.3f} s (the "
            f"recording's length); got start={start_seconds!r}, "
            f"end={end_seconds!r}."
        )
    start_index = round(start * fs)
    return TimeWindow(
        start_seconds=start,
        end_seconds=end,
        start_index=start_index,
        end_index=round(end * fs),
        n_samples=n_samples,
        shift_seconds=start_index / fs,
        sample_frequency=fs,
    )


def keep_samples(recording: EMGRecording, start_index: int, end_index: int) -> None:
    """Keep samples `start_index` up to (not including) `end_index` of an EMG
    recording, in place.

    The samples are the last axis of the recording's array (channels by
    samples). A list stays a list.
    """

    data = recording.data
    array = data["array"]
    kept = np.asarray(array)[..., start_index:end_index]
    data["array"] = kept.tolist() if isinstance(array, list) else kept
    recording.raw = data["array"]
    recording.metadata = data.get("metadata")
