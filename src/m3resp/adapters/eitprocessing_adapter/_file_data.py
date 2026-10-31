"""Build an eitprocessing ``Sequence`` from the data of one EIT file.

The m3resp readers (``m3resp.eit.io``) return an ``EITFileData``. The EIT
steps after loading read an eitprocessing ``Sequence``.
``file_data_to_sequence`` builds the ``Sequence`` that eitprocessing 1.8.7's
``load_eit_data`` gives for the same file, field by field.
``load_eit_file_as_sequence`` reads one file and builds its ``Sequence``.
``sequence_to_file_data`` goes the other way, for a ``Sequence`` from a
custom loader or a cut ``Sequence``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from m3resp.data.event_data import EventData, IntervalData
from m3resp.data.events import BreathEvent, Event, Interval
from m3resp.data.signals import Signal
from m3resp.eit.io.file_data import EITFileData
from m3resp.eit.io.read_file import read_eit_file

from ._shared import _lazy_import


def sequence_to_file_data(sequence: Any) -> EITFileData | None:
    """Return the m3resp data of an eitprocessing ``Sequence``.

    The result has the fields and labels the m3resp readers give for the same
    file: the raw images as ``pixel_impedance``, each ``ContinuousData`` as a
    ``Signal``, each ``SparseData`` as ``EventData`` and each
    ``IntervalData`` as ``IntervalData`` (of ``BreathEvent``s when its values
    are breaths).

    Args:
        sequence (Any): An eitprocessing ``Sequence``, or what a custom loader
            returns.

    Returns:
        EITFileData | None: The m3resp data, or None when ``sequence`` has no
            raw EIT data with images and time (for example the result of a
            custom loader that returns another kind of object).
    """
    eit_collection = getattr(sequence, "eit_data", None)
    raw = None
    if eit_collection is not None and "raw" in eit_collection:
        raw = eit_collection["raw"]
    if raw is None or not all(
        hasattr(raw, field) for field in ("pixel_impedance", "time", "vendor")
    ):
        return None

    vendor = str(getattr(raw.vendor, "value", raw.vendor))
    path = getattr(raw, "path", "")
    path = path[0] if isinstance(path, list | tuple) else path
    pixel_impedance = Signal(
        values=raw.pixel_impedance,
        time=raw.time,
        sample_frequency=raw.sample_frequency,
        name=raw.label,
        modality="eit",
        category="impedance",
        channel="pixel_impedance",
        source=vendor,
    )

    signals = {}
    for key, data in getattr(sequence, "continuous_data", {}).items():
        is_global_impedance = key.startswith("global_impedance")
        description = getattr(data, "description", "")
        signals[key] = Signal(
            values=data.values,
            time=data.time,
            sample_frequency=data.sample_frequency,
            unit=data.unit,
            name=data.name,
            metadata={"description": description} if description else {},
            modality="eit" if is_global_impedance else "ventilator",
            category=data.category,
            channel="global_impedance" if is_global_impedance else None,
            source=vendor,
        )

    events = {}
    for key, data in getattr(sequence, "sparse_data", {}).items():
        values = None if data.values is None else list(data.values)
        events[key] = EventData(
            name=key,
            modality="eit",
            events=[
                Event(
                    name=data.category,
                    modality="eit",
                    time=float(time),
                    label=None if values is None else getattr(values[i], "text", None),
                )
                for i, time in enumerate(data.time)
            ],
            values=None
            if values is None
            else [getattr(value, "marker", value) for value in values],
            category=data.category,
            metadata={"description": data.name},
        )

    intervals = {}
    for key, data in getattr(sequence, "interval_data", {}).items():
        values = [] if data.values is None else data.values
        items: list[Interval] = []
        for index, interval in enumerate(data.intervals):
            value = values[index] if index < len(values) else None
            if value is not None and hasattr(value, "middle_time"):
                items.append(
                    BreathEvent(
                        "eit",
                        float(value.start_time),
                        float(value.end_time),
                        extremum_time=float(value.middle_time),
                    )
                )
            else:
                items.append(
                    Interval(
                        "eit",
                        float(interval.start_time),
                        float(interval.end_time),
                        name=data.category,
                    )
                )
        metadata: dict[str, Any] = {"description": data.name}
        if hasattr(data, "default_partial_inclusion"):
            metadata["default_partial_inclusion"] = data.default_partial_inclusion
        intervals[key] = IntervalData(
            name=key,
            modality="eit",
            intervals=items,
            values=None
            if data.values is None
            or all(hasattr(value, "middle_time") for value in values)
            else copy.deepcopy(data.values),
            category=data.category,
            metadata=metadata,
        )

    return EITFileData(
        vendor=vendor,
        path=os.fspath(path),
        sample_frequency=raw.sample_frequency,
        pixel_impedance=pixel_impedance,
        signals=signals,
        events=events,
        intervals=intervals,
    )


def load_eit_file_as_sequence(
    path: str | os.PathLike[str],
    vendor: Any,
    label: str | None = None,
    name: str | None = None,
    description: str = "",
    sample_frequency: float | None = None,
    first_frame: int | None = 0,
    max_frames: int | None = None,
) -> Any:
    """Read one EIT file with the m3resp reader and return its ``Sequence``.

    It takes the arguments of eitprocessing 1.8.7's ``load_eit_data`` for
    one file and gives the same ``Sequence``.

    Args:
        path (str | os.PathLike[str]): The file.
        vendor (Any): ``"draeger"``, ``"sentec"`` or ``"timpel"``.
        label (str | None): The label of the ``Sequence``.
        name (str | None): The name of the ``Sequence``.
        description (str): The description of the ``Sequence``.
        sample_frequency (float | None): The sample frequency in Hz, see
            ``m3resp.eit.io.read_eit_file``.
        first_frame (int | None): The index of the first frame to read.
        max_frames (int | None): The largest number of frames to read.

    Returns:
        Sequence: The eitprocessing ``Sequence``.

    Raises:
        OptionalDependencyError: If eitprocessing is not installed.
        NotImplementedError: If there is no reader for ``vendor``.
        FileNotFoundError: If the file does not exist.
    """
    file_data = read_eit_file(
        path,
        vendor,
        sample_frequency=sample_frequency,
        first_frame=first_frame,
        max_frames=max_frames,
    )
    return file_data_to_sequence(
        file_data, label=label, name=name, description=description
    )


#: The eitprocessing category of each waveform, by vendor and label, where it
#: differs from the label. m3resp writes categories in its own names (for
#: example "airflow" for "flow"), so the eitprocessing names are kept here.
#: The other Draeger waveforms (the Medibus fields) have their label as
#: category.
_UPSTREAM_CATEGORIES = {
    ("draeger", "global_impedance_(raw)"): "impedance",
    ("timpel", "global_impedance_(raw)"): "global_impedance",
    ("timpel", "airway_pressure_(timpel)"): "pressure",
    ("timpel", "flow_(timpel)"): "flow",
    ("timpel", "volume_(timpel)"): "volume",
}


def file_data_to_sequence(
    file_data: EITFileData,
    label: str | None = None,
    name: str | None = None,
    description: str = "",
) -> Any:
    """Return the eitprocessing ``Sequence`` of one EIT file.

    The result is equal to ``load_eit_data(path, vendor=...)`` of
    eitprocessing 1.8.7 for the same file and options: the same labels,
    names, units, categories, values and times, the same order, the Draeger
    Medibus waveforms locked, and the same ``derived_from`` links.

    Args:
        file_data (EITFileData): The data of one file, from an m3resp reader.
        label (str | None): The label of the ``Sequence``. ``None`` gives
            ``"Sequence_<number>"``.
        name (str | None): The name of the ``Sequence``. ``None`` gives the
            label.
        description (str): The description of the ``Sequence``.

    Returns:
        Sequence: The eitprocessing ``Sequence``.

    Raises:
        OptionalDependencyError: If eitprocessing is not installed.
    """
    (
        Breath,
        ContinuousData,
        DataCollection,
        EITData,
        Vendor,
        Event,
        IntervalData,
        Sequence,
        SparseData,
    ) = _lazy_import(
        "eitprocessing.datahandling.breath.Breath",
        "eitprocessing.datahandling.continuousdata.ContinuousData",
        "eitprocessing.datahandling.datacollection.DataCollection",
        "eitprocessing.datahandling.eitdata.EITData",
        "eitprocessing.datahandling.eitdata.Vendor",
        "eitprocessing.datahandling.event.Event",
        "eitprocessing.datahandling.intervaldata.IntervalData",
        "eitprocessing.datahandling.sequence.Sequence",
        "eitprocessing.datahandling.sparsedata.SparseData",
    )

    vendor = file_data.vendor
    pixels = file_data.pixel_impedance
    eit_data = EITData(
        vendor=Vendor(vendor),
        path=Path(os.fspath(file_data.path)),
        sample_frequency=file_data.sample_frequency,
        nframes=pixels.n_samples,
        time=pixels.time,
        label="raw",
        pixel_impedance=pixels.values,
    )

    continuous = DataCollection(ContinuousData)
    for key, signal in file_data.signals.items():
        is_global_impedance = signal.channel == "global_impedance"
        data = ContinuousData(
            label=key,
            name=signal.name,
            unit=signal.unit,
            category=_UPSTREAM_CATEGORIES.get((vendor, key), key),
            description=signal.metadata.get("description", ""),
            derived_from=(
                [eit_data] if is_global_impedance and vendor == "draeger" else []
            ),
            time=signal.time,
            values=np.array(signal.values),
            sample_frequency=signal.sample_frequency,
        )
        if vendor == "draeger" and not is_global_impedance:
            data.lock()
        continuous.add(data)

    sparse = DataCollection(SparseData)
    for key, marks in file_data.events.items():
        values = None
        if marks.values is not None:
            values = [
                Event(int(marker), event.label or "")
                for marker, event in zip(marks.values, marks.events, strict=True)
            ]
        sparse.add(
            SparseData(
                label=key,
                name=marks.metadata.get("description", key),
                unit=None,
                category=marks.category,
                derived_from=[eit_data],
                time=np.array([event.time for event in marks.events]),
                values=values,
            )
        )

    intervals = DataCollection(IntervalData)
    for key, breaths in file_data.intervals.items():
        breath_events = cast("list[BreathEvent]", breaths.intervals)
        intervals.add(
            IntervalData(
                label=key,
                name=breaths.metadata.get("description", key),
                unit=None,
                category=breaths.category,
                intervals=[(b.start_time, b.end_time) for b in breath_events],
                values=[
                    Breath(b.start_time, b.extremum_time, b.end_time)
                    for b in breath_events
                ],
                default_partial_inclusion=breaths.metadata.get(
                    "default_partial_inclusion", False
                ),
            )
        )

    return Sequence(
        label=label,
        name=name,
        description=description,
        eit_data=DataCollection(EITData, raw=eit_data),
        continuous_data=continuous,
        sparse_data=sparse,
        interval_data=intervals,
    )
