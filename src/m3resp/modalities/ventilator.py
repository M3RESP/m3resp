"""Ventilator modality containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VentilatorRecording:
    """Loaded ventilator recording with source metadata.

    Mirrors :class:`~m3resp.modalities.emg.EMGRecording`: ``data`` holds the
    loader's raw ``{"array", "metadata"}`` payload, and the remaining fields are
    conveniences unpacked from it.

    ``raw`` and ``dataframe`` are the same measured values in two forms.
    ``raw`` is a plain numeric grid of channels by samples; ``dataframe`` is
    that grid as a table whose columns carry the vendor's own channel names
    (``Paw``, ``EMGdi``, ... as written in the file header). Only recordings
    read from the file shared with the sEMG carry the table; for ventilator
    waveforms read out of an EIT ``*.bin`` it is empty.

    It is held as the reader returned it, unchanged. No m3resp code reads it:
    channel identification works from ``metadata["labels"]`` instead, so the
    table is not on the path from a file to a `Signal`. It is kept for two
    reasons. A Biopac ``*.txt`` is read *as* a table and the numeric grid is
    derived from it, so the table is the closer record of the file. And a
    table can hold what a float grid cannot - per-column names and types,
    non-numeric columns such as vendor event markers, a time column that is
    not evenly spaced. Discarding it would mean re-reading the file to
    recover any of that.

    ``pressure``, ``flow`` and ``volume`` are the three channels a ventilator
    always reports - airway pressure, airway flow and tidal volume - populated
    by ``M3Session.preprocess_ventilator`` (the ventilator counterpart of
    ``EMGRecording.filtered``/``envelope``). ``pressure`` is the ventilator's
    own airway pressure and nothing else: a recording carrying an esophageal,
    transpulmonary or gastric pressure, or a second airway pressure from a
    Draeger pressure pod, keeps each of those under its own name in the
    preprocessing result, where they are also tagged with the quantity they
    measure. Only the airway pressure appears here.
    """

    data: Any
    path: Path
    raw: Any = None
    dataframe: Any = None
    metadata: dict[str, Any] | None = None
    fs: float | None = None
    pressure: Any = None
    flow: Any = None
    volume: Any = None
    #: Which modality's file these waveforms arrived in, and therefore whose
    #: clock they share: ``"eit"`` or ``"emg"`` when they were carried inside
    #: that recording, ``"ventilator"`` for a standalone export. Aligning the
    #: host modality aligns these channels with it, so synchronization must not
    #: also shift them by a ventilator offset - see
    #: `m3resp.synchronization.cropping.ventilator_clock`.
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
