"""Ventilator modality containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.modalities.names import VENTILATOR, normalize_modality
from m3resp.modalities.time_window import TimeWindow

if TYPE_CHECKING:
    from m3resp.core.session import M3Session

#: Name a ventilator recording is filed under in `M3Session.ventilators` when
#: the caller gives none.
DEFAULT_VENTILATOR_NAME = "default"


@dataclass
class VentilatorRecording:
    """Loaded ventilator samples, source metadata, and selected channel arrays.

    Attributes:
        data (Any): Loader payload containing array and metadata.
        path (Path): Source file path.
        raw (Any): Numeric samples, arranged as channels by samples.
        dataframe (Any): Vendor table when supplied by the reader, including
            original column names and any event or time columns.
        metadata (dict[str, Any] | None): Recording labels, units and file details.
        fs (float | None): Sampling rate in Hz.
        airway_pressure (Any): Main airway-pressure array selected during
            `M3Session.preprocess_ventilator`, or None when unavailable.
        flow (Any): Main flow array after preprocessing, or None when unavailable.
        volume (Any): Main volume array after preprocessing, or None when
            unavailable. Channel units come from the preprocessing bundle.
        source_modality (str): Clock shared by these samples: "eit" or "emg" for
            channels carried in those files, or "ventilator" for a standalone
            recording. Defaults to "ventilator".

    Additional pressures and repeated measurements are stored with distinct keys
    in the preprocessing bundle. The airway_pressure field holds the channel
    selected by that bundle's primary mapping.
    """

    data: Any
    path: Path
    raw: Any = None
    dataframe: Any = None
    metadata: dict[str, Any] | None = None
    fs: float | None = None
    airway_pressure: Any = None
    flow: Any = None
    volume: Any = None
    #: Which modality's file these waveforms arrived in, and therefore whose
    #: clock they share: ``"eit"`` or ``"emg"`` when they were carried inside
    #: that recording, ``"ventilator"`` for a standalone export. Aligning the
    #: host modality aligns these channels with it, so synchronization must not
    #: also shift them by a ventilator offset - see `ventilator_clock` below.
    source_modality: str = "ventilator"


def load(
    path: str | Path,
    *,
    adapter: Any = None,
    **kwargs: Any,
) -> VentilatorRecording:
    """Load a ventilator recording through the Stage 1 adapter.

    ``adapter`` defaults to
    :class:`~m3resp.adapters.ventilator_adapter.VentilatorAdapter`, which reads
    both sources ventilator data arrives from and picks between them by file
    suffix: the multi-channel file shared with the sEMG (e.g. a Biopac export,
    delegated to `ReSurfEMGAdapter`), and the EIT ``*.bin``, which stores
    ventilator waveforms beside its impedance frames.
    """

    from m3resp.adapters.ventilator_adapter import (
        VentilatorAdapter,
        resolve_ventilator_source,
    )

    ventilator_adapter = adapter or VentilatorAdapter()
    source_modality = resolve_ventilator_source(path, kwargs.get("source"))
    recording = ventilator_adapter.load(str(path), **kwargs)
    is_dict = isinstance(recording, dict)
    metadata = recording.get("metadata") if is_dict else None
    sample_frequency = metadata.get("fs") if isinstance(metadata, dict) else None
    return VentilatorRecording(
        data=recording,
        path=Path(path),
        raw=recording.get("array") if is_dict else None,
        dataframe=recording.get("dataframe") if is_dict else None,
        metadata=metadata,
        fs=float(sample_frequency) if sample_frequency is not None else None,
        source_modality=source_modality,
    )


def ventilator_raw(session: M3Session) -> Any:
    """The loaded ventilator recording, under either key.

    ``session.raw`` stores it under both ``"ventilator"`` (canonical) and
    ``"vent"`` (the key Stage 1 shipped), pointing at the same object. Reading
    through here means code stays correct whichever key a caller populated -
    including a user who assigned ``session.raw["vent"]`` directly.
    """

    recording = session.raw.get(VENTILATOR)
    if recording is None:
        recording = session.raw.get("vent")
    return recording


def ventilator_payload(recording: Any) -> dict[str, Any] | None:
    """The ``{"array", "metadata"}`` payload of a ventilator recording.

    Accepts either a :class:`~m3resp.modalities.ventilator.VentilatorRecording`
    (which keeps it under ``.data``, like every other modality) or a bare dict,
    which is what Stage 1 stored directly in ``session.raw["vent"]``.
    """

    if isinstance(recording, dict) and "array" in recording:
        return recording
    data = getattr(recording, "data", None)
    if isinstance(data, dict) and "array" in data:
        return data
    return None


def ventilator_clock(recording: Any) -> str:
    """Which modality's clock a ventilator recording shares.

    Ventilator waveforms carried inside the EIT ``*.bin`` or the multi-channel
    sEMG export are samples of that recording's own time base, so aligning that
    modality aligns them too. They are read into a separate array rather than a
    view onto the host, so they must be cut alongside it - and must *not*
    also be shifted by a ventilator offset, which would move them twice.

    Only a standalone ventilator or monitor export has a clock of its own; that
    is what the ventilator offset in `M3Session.synchronize_raw_modalities` is
    for. Returns ``"eit"``, ``"emg"`` or :data:`VENTILATOR`.
    """

    source = getattr(recording, "source_modality", None)
    if source is None:
        payload = ventilator_payload(recording)
        metadata = (payload or {}).get("metadata") or {}
        source = metadata.get("source")
    if source is None:
        return VENTILATOR
    normalized = normalize_modality(source)
    return normalized if normalized in {"eit", "emg"} else VENTILATOR


def ventilator_recordings(session: M3Session) -> list[Any]:
    """Every loaded ventilator recording, however it was stored.

    Prefers `session.ventilators` (which can hold several instruments' data at
    once) and falls back to `session.raw`, so a caller that assigned
    ``session.raw["vent"]`` directly is still covered.
    """

    recordings = list(getattr(session, "ventilators", {}).values())
    if recordings:
        return recordings
    recording = ventilator_raw(session)
    return [recording] if recording is not None else []


def ventilator_recording(
    session: M3Session, name: str | None = None
) -> tuple[Any, str]:
    """A loaded ventilator recording and its name; the primary one when
    `name` is None. Returns ``(None, name)`` when there is none."""

    recordings = getattr(session, "ventilators", {}) or {}
    if not recordings:
        # A recording assigned directly to `session.raw["vent"]` has no name.
        return ventilator_raw(session), name or DEFAULT_VENTILATOR_NAME
    if name is None:
        name = session.primary_ventilator_name() or DEFAULT_VENTILATOR_NAME
    return recordings.get(name), name


def sample_window(
    recording: Any, start_seconds: float, end_seconds: float | None, *, name: str
) -> TimeWindow:
    """The samples of a ventilator recording between two times.

    Times are in seconds from the recording's first sample; ``end_seconds``
    None means up to the end. `name` is the recording's name, used in error
    messages. Raises ValueError when the recording holds no samples or the
    window does not lie inside it.
    """

    payload = ventilator_payload(recording)
    if payload is None:
        raise ValueError(
            f"slice_ventilator: ventilator recording {name!r} holds no sample array."
        )
    fs = float(payload["metadata"]["fs"])
    n_samples = np.asarray(payload["array"]).shape[-1]
    duration = n_samples / fs
    end = duration if end_seconds is None else float(end_seconds)
    start = float(start_seconds)
    if not 0.0 <= start < end <= duration:
        raise ValueError(
            f"slice_ventilator: need 0 <= start < end <= {duration:.3f} s "
            f"(the length of ventilator recording {name!r}); got "
            f"start={start_seconds!r}, end={end_seconds!r}."
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


def keep_samples(recording: Any, start_index: int, end_index: int) -> None:
    """Keep samples `start_index` up to (not including) `end_index` of a
    ventilator recording, in place.

    The samples are the last axis of the array (channels by samples); a list
    stays a list. The payload dict is changed rather than replaced, so both
    `session.raw` keys (which point at the same object) see the cut. A time
    axis stored in the metadata is cut too.
    """

    payload = ventilator_payload(recording)
    if payload is None:
        return
    array = payload["array"]
    kept = np.asarray(array)[..., start_index:end_index]
    payload["array"] = kept.tolist() if isinstance(array, list) else kept
    metadata = payload.get("metadata") or {}
    if metadata.get("time") is not None:
        metadata["time"] = np.asarray(metadata["time"])[start_index:end_index]
    data = getattr(recording, "data", None)
    if isinstance(data, dict):
        recording.raw = data.get("array")
        recording.metadata = data.get("metadata")


def cut_seconds_off_ends(
    recording: Any, front_seconds: float, back_seconds: float
) -> None:
    """Remove `front_seconds` from the start and `back_seconds` from the end
    of a ventilator recording, in place, counted at its own sampling rate.

    Used for ventilator data that came inside the EMG file when the EMG is
    cut: it may be sampled at another rate, so the same times are converted
    to its own sample numbers. A recording without a sampling rate is left
    as it is.
    """

    payload = ventilator_payload(recording)
    if payload is None:
        return
    fs = (payload.get("metadata") or {}).get("fs")
    if fs is None:
        return
    n_samples = np.asarray(payload["array"]).shape[-1]
    start_index = round(front_seconds * float(fs))
    end_index = n_samples - round(back_seconds * float(fs))
    if end_index <= start_index:
        raise ValueError(
            "Cutting would remove every sample of a ventilator recording: "
            f"{front_seconds!r} s off the start and {back_seconds!r} s off the "
            f"end of {n_samples} samples at {fs!r} Hz."
        )
    keep_samples(recording, start_index, end_index)
