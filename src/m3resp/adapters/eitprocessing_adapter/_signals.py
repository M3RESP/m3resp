"""Convert an m3resp ``Signal`` to an eitprocessing ``EITData`` or ``ContinuousData``, and back.

The EIT steps of m3resp pass ``Signal`` objects to each other. A step that
calls eitprocessing turns its ``Signal`` inputs into eitprocessing objects
just before the call, and the eitprocessing results back into ``Signal``
objects. The values and the time stay the same arrays or exact copies, so no
number changes.

eitprocessing objects have some fields that a ``Signal`` does not have: the
label, the description, the eitprocessing names of the category and the unit
(m3resp writes some in its own names, for example ``"a.u."`` for ``"AU"``),
the parameters, the file path, the device maker and whether the values may be
changed (locked).
``eit_data_to_signal`` and ``continuous_data_to_signal`` keep these fields in
``Signal.metadata["eitprocessing_fields"]``, so ``signal_to_eit_data`` and
``signal_to_continuous_data`` give an object equal to the one the ``Signal``
was made from. For a ``Signal`` made by m3resp code, the conversion takes the
label and name from ``Signal.name``, the category and unit from
``Signal.category`` and ``Signal.unit``, and the device maker from
``Signal.source``.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np

from m3resp.data.signals import Modality, ProcessingState, Signal

from ._shared import _lazy_import

#: The key of ``Signal.metadata`` that holds the fields of the eitprocessing
#: object the signal was made from.
EITPROCESSING_FIELDS = "eitprocessing_fields"


def eitprocessing_fields(signal: Signal) -> dict[str, Any]:
    """The eitprocessing fields kept in a signal, or an empty dict."""

    fields = signal.metadata.get(EITPROCESSING_FIELDS)
    return dict(fields) if isinstance(fields, dict) else {}


def eit_data_to_signal(
    eit_data: Any,
    *,
    processing_state: ProcessingState = "raw",
    name: str | None = None,
    source: str | None = "eitprocessing",
    method: str | None = None,
) -> Signal:
    """Return the pixel impedance of an eitprocessing ``EITData`` as a ``Signal``.

    The signal holds the same arrays: ``values`` the pixel impedance with shape
    (frames, rows, columns) and ``time`` the frame times in seconds.

    Args:
        eit_data (Any): An eitprocessing ``EITData``, or an object with
            ``pixel_impedance`` and ``time``.
        processing_state (ProcessingState): ``"raw"`` for loaded data; other
            states need ``name``.
        name (str | None): The name of the signal. For raw data, ``None``
            gives the ``name`` (else the ``label``) of ``eit_data``.
        source (str | None): Where the data came from.
        method (str | None): The function that made the data.

    Returns:
        Signal: Modality ``"eit"``, category ``"impedance"``, channel
            ``"pixel_impedance"``, with the fields of ``eit_data`` that a
            ``Signal`` lacks in ``metadata["eitprocessing_fields"]`` (see
            ``eit_data_fields``).

    Raises:
        ValueError: If ``processing_state`` is not ``"raw"`` and ``name`` is
            None. eitprocessing operations copy their input and keep its
            name, so the name of a changed object can be the name of the raw
            data.
    """

    if name is None:
        if processing_state != "raw":
            raise ValueError(
                "eit_data_to_signal: give `name` for "
                f"processing_state={processing_state!r}; the name of a changed "
                "eitprocessing object can still be the name of the raw data."
            )
        name = getattr(eit_data, "name", None) or getattr(eit_data, "label", None)
    return Signal(
        values=eit_data.pixel_impedance,
        time=eit_data.time,
        sample_frequency=getattr(eit_data, "sample_frequency", None),
        name=name,
        metadata={EITPROCESSING_FIELDS: eit_data_fields(eit_data)},
        modality="eit",
        category="impedance",
        channel="pixel_impedance",
        processing_state=processing_state,
        source=source,
        method=method,
    )


def eit_data_fields(obj: Any) -> dict[str, Any]:
    """The fields of an eitprocessing ``EITData`` that a ``Signal`` lacks.

    Args:
        obj (Any): An eitprocessing ``EITData``, or an object with some of
            its fields.

    Returns:
        dict[str, Any]: ``type``, ``label``, ``name``, ``description``,
            ``vendor`` (as text) and ``path`` (as text, or a list of texts),
            for the fields ``obj`` has.
    """

    fields: dict[str, Any] = {"type": "EITData"}
    for field in ("label", "name", "description"):
        if hasattr(obj, field):
            fields[field] = getattr(obj, field)
    if hasattr(obj, "vendor"):
        fields["vendor"] = getattr(obj.vendor, "value", obj.vendor)
    if hasattr(obj, "path"):
        path = obj.path
        fields["path"] = (
            [os.fspath(p) for p in path]
            if isinstance(path, list | tuple)
            else os.fspath(path)
        )
    return fields


def signal_to_eit_data(signal: Signal) -> Any:
    """Return an eitprocessing ``EITData`` with the pixel impedance of a ``Signal``.

    The ``EITData`` holds the arrays of the signal (no copy).

    Args:
        signal (Signal): Values with shape (frames, rows, columns), time in
            seconds and the frames per second in ``sample_frequency``.

    Returns:
        EITData: The eitprocessing object. Its label, name, description,
            device maker and file path come from
            ``signal.metadata["eitprocessing_fields"]``; when that is missing,
            the label and name are ``signal.name``, the device maker is
            ``signal.source`` and the path is empty.

    Raises:
        OptionalDependencyError: If eitprocessing is not installed.
        ValueError: If the values do not have three axes, or if the device
            maker is unknown.
    """

    EITData, Vendor = _lazy_import(
        "eitprocessing.datahandling.eitdata.EITData",
        "eitprocessing.datahandling.eitdata.Vendor",
    )
    values = np.asarray(signal.values)
    if values.ndim != 3:
        raise ValueError(
            "signal_to_eit_data needs values with shape (frames, rows, columns); "
            f"{signal.name!r} has shape {values.shape}."
        )
    fields = eitprocessing_fields(signal)
    vendor = fields.get("vendor", signal.source)
    try:
        vendor = Vendor(vendor)
    except ValueError as exc:
        raise ValueError(
            f"signal_to_eit_data: the device maker of {signal.name!r} is "
            f"{vendor!r}; eitprocessing needs one of "
            f"{sorted({v.value for v in Vendor})}."
        ) from exc
    return EITData(
        path=fields.get("path", ""),
        nframes=values.shape[0],
        time=signal.time,
        sample_frequency=signal.sample_frequency,
        vendor=vendor,
        label=fields.get("label", signal.name),
        name=fields.get("name", signal.name),
        description=fields.get("description", ""),
        pixel_impedance=signal.values,
    )


def continuous_data_fields(obj: Any) -> dict[str, Any]:
    """The fields of an eitprocessing ``ContinuousData`` that a ``Signal`` lacks.

    Args:
        obj (Any): An eitprocessing ``ContinuousData``, or an object with
            some of its fields.

    Returns:
        dict[str, Any]: ``type``, ``label``, ``name``, ``description``,
            ``category`` and ``unit`` (the eitprocessing names),
            ``parameters`` (a copy) and ``locked``, for the fields ``obj``
            has.
    """

    fields: dict[str, Any] = {"type": "ContinuousData"}
    for field in ("label", "name", "description", "category", "unit"):
        if hasattr(obj, field):
            fields[field] = getattr(obj, field)
    if hasattr(obj, "parameters"):
        fields["parameters"] = copy.deepcopy(obj.parameters)
    locked = getattr(obj, "locked", None)
    if isinstance(locked, bool):
        fields["locked"] = locked
    return fields


def continuous_data_to_signal(
    obj: Any,
    *,
    modality: Modality,
    channel: str | None,
    processing_state: ProcessingState,
    source: str | None = None,
    method: str | None = None,
    category: str | None = "impedance",
    name: str | None = None,
) -> Signal:
    """Return an eitprocessing ``ContinuousData`` as a ``Signal``.

    The signal holds the same ``values`` and ``time`` arrays. The fields of
    ``obj`` that a ``Signal`` lacks go into
    ``metadata["eitprocessing_fields"]`` (see ``continuous_data_fields``).

    Args:
        obj (Any): An eitprocessing ``ContinuousData``, or an object with
            ``values`` and ``time``.
        modality (Modality): The device or technique, for example ``"eit"``.
        channel (str | None): The channel, for example
            ``"global_impedance"``.
        processing_state (ProcessingState): ``"raw"`` for loaded data; other
            states need ``name``.
        source (str | None): Where the data came from.
        method (str | None): The function that made the data.
        category (str | None): The quantity. eitprocessing gives EIT
            impedances here, so the default is ``"impedance"``.
        name (str | None): The name of the signal. For raw data, ``None``
            gives the ``name`` (else the ``label``) of ``obj``.

    Returns:
        Signal: The signal.

    Raises:
        ValueError: If ``processing_state`` is not ``"raw"`` and ``name`` is
            None. eitprocessing operations (for example ``MDNFilter.apply``)
            copy their input and change only the fields they are given, so
            the name of a changed object can be the name of the raw data.
    """

    if name is None:
        if processing_state != "raw":
            raise ValueError(
                "continuous_data_to_signal: give `name` for "
                f"processing_state={processing_state!r}; the name of a changed "
                "eitprocessing object can still be the name of the raw data."
            )
        name = getattr(obj, "name", None) or getattr(obj, "label", None)

    return Signal(
        values=obj.values,
        time=obj.time,
        sample_frequency=getattr(obj, "sample_frequency", None),
        unit=getattr(obj, "unit", None),
        name=name,
        metadata={EITPROCESSING_FIELDS: continuous_data_fields(obj)},
        modality=modality,
        category=category,
        channel=channel,
        processing_state=processing_state,
        source=source,
        method=method,
    )


def signal_to_continuous_data(
    signal: Signal, *, derived_from: list[Any] | None = None
) -> Any:
    """Return an eitprocessing ``ContinuousData`` with the values of a ``Signal``.

    The ``ContinuousData`` holds copies of the values and the time, because
    eitprocessing makes the arrays it locks read-only.

    Args:
        signal (Signal): One value per sample, time in seconds.
        derived_from (list[Any] | None): The eitprocessing objects the data
            was made from. ``None`` gives an empty list.

    Returns:
        ContinuousData: The eitprocessing object. Its label, name,
            description, category, unit, parameters and lock come from
            ``signal.metadata["eitprocessing_fields"]``; when that is
            missing, the label and name are ``signal.name``, the category and
            unit are ``signal.category`` and ``signal.unit``, the parameters
            are empty and eitprocessing sets the lock.

    Raises:
        OptionalDependencyError: If eitprocessing is not installed.
    """

    (ContinuousData,) = _lazy_import(
        "eitprocessing.datahandling.continuousdata.ContinuousData"
    )
    fields = eitprocessing_fields(signal)
    data = ContinuousData(
        label=fields.get("label", signal.name),
        name=fields.get("name", signal.name),
        unit=fields.get("unit", signal.unit),
        category=fields.get("category", signal.category),
        description=fields.get("description", ""),
        parameters=copy.deepcopy(fields.get("parameters", {})),
        derived_from=list(derived_from or []),
        time=np.array(signal.time),
        values=np.array(signal.values),
        sample_frequency=signal.sample_frequency,
    )
    locked = fields.get("locked")
    if locked is True and not data.locked:
        data.lock()
    elif locked is False and data.locked:
        data.unlock()
    return data


def as_eit_data(obj: Any) -> Any:
    """Return ``obj`` as an eitprocessing ``EITData``.

    Args:
        obj (Any): A ``Signal`` with EIT images, or an eitprocessing object.

    Returns:
        Any: ``signal_to_eit_data(obj)`` for a ``Signal``; any other object
            as it is.
    """

    return signal_to_eit_data(obj) if isinstance(obj, Signal) else obj


def as_continuous_data(obj: Any) -> Any:
    """Return ``obj`` as an eitprocessing ``ContinuousData``.

    Args:
        obj (Any): A ``Signal`` with one value per sample, or an
            eitprocessing object.

    Returns:
        Any: ``signal_to_continuous_data(obj)`` for a ``Signal``; any other
            object as it is.
    """

    return signal_to_continuous_data(obj) if isinstance(obj, Signal) else obj


def as_eitprocessing(obj: Any) -> Any:
    """Return ``obj`` as the eitprocessing object of its kind.

    Args:
        obj (Any): A ``Signal``, or an eitprocessing object.

    Returns:
        Any: An ``EITData`` for a ``Signal`` with values of shape (frames,
            rows, columns), a ``ContinuousData`` for any other ``Signal``,
            and any other object as it is.
    """

    if not isinstance(obj, Signal):
        return obj
    if np.ndim(obj.values) == 3:
        return signal_to_eit_data(obj)
    return signal_to_continuous_data(obj)
