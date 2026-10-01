"""Place each loaded recording on one shared clock by its start time.

`M3Session.synchronize_raw_modalities` used to line recordings up by cutting
samples off them. That lost data: a pressure trace recorded for three hours
beside a thirty-minute EIT recording lost everything before the EIT started.
Now every recording keeps all its samples, and the session stores one number
per recording instead: its **start time**, the time in seconds on the shared
clock at which the recording's first sample was taken.

Each modality's own results (breath times, signals) stay on that recording's
own clock. The start time is added only where modalities are compared - when
breaths are shifted or linked across modalities, and in the before/after
synchronization plot:

    time on the shared clock = own time - own first time + start time

"Own first time" is the time of the recording's first sample on its own
clock. It is 0 s for EMG and ventilator data, whose times are counted from
the first sample, but EIT files carry the time of day (a Draeger file can
start at 36528.6 s), so it has to be taken off before the start time is
added.

A ventilator recording that arrived inside the EIT or EMG file was taken on
that file's clock, so it uses that modality's start time rather than a start
time of its own.

Standalone ventilator recordings (each with a clock of its own) use the
``"ventilator"`` start time, unless one has its own entry under
``"ventilator:<name>"`` - for example when a ventilator export and a
separate monitor export were started at different moments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.modalities.names import VENTILATOR, normalize_modality
from m3resp.modalities.ventilator import (
    ventilator_clock,
    ventilator_recording,
    ventilator_recordings,
)
from m3resp.synchronization.alignment import ventilator_start_key

if TYPE_CHECKING:
    from m3resp.core.session import M3Session

#: The modalities a start time can be set for.
MODALITIES = ("eit", "emg", VENTILATOR)


def loaded_modalities(session: M3Session) -> list[str]:
    """The modalities that have a recording loaded in `session`."""

    loaded = []
    if session.eit is not None:
        loaded.append("eit")
    if session.emg is not None:
        loaded.append("emg")
    if ventilator_recordings(session):
        loaded.append(VENTILATOR)
    return loaded


def recording_start_time(
    session: M3Session, modality: str, name: str | None = None
) -> float:
    """The start time (s) that applies to a recording, from `session.start_times`.

    EIT and EMG use their own entry. A ventilator recording that came inside
    the EIT or EMG file uses that modality's entry. A standalone ventilator
    recording uses its own ``"ventilator:<name>"`` entry when there is one,
    and the ``"ventilator"`` entry otherwise. `name` picks the ventilator
    recording; None means the primary one. A missing entry means 0.
    """

    start_times = getattr(session, "start_times", {}) or {}
    modality = normalize_modality(modality)
    if modality != VENTILATOR:
        return float(start_times.get(modality, 0.0))
    recording, name = ventilator_recording(session, name)
    clock = ventilator_clock(recording) if recording is not None else VENTILATOR
    if clock != VENTILATOR:
        return float(start_times.get(clock, 0.0))
    own_key = ventilator_start_key(name)
    if own_key in start_times:
        return float(start_times[own_key])
    return float(start_times.get(VENTILATOR, 0.0))


def own_first_time(session: M3Session, modality: str) -> float:
    """Time of the recording's first sample, on the recording's own clock (s).

    0.0 for EMG and ventilator data. For EIT, the first value of the loaded
    time axis, which is often the time of day rather than 0.
    """

    if normalize_modality(modality) != "eit" or session.eit is None:
        return 0.0
    for obj in (session.eit.global_impedance, session.eit.raw):
        time = getattr(obj, "time", None)
        if time is None:
            continue
        values = np.asarray(time, dtype=float)
        if values.size:
            return float(values.flat[0])
    return 0.0


def shared_clock_shift(
    session: M3Session, modality: str, name: str | None = None
) -> float:
    """Seconds to add to `modality`'s own times to put them on the shared clock.

    Equals the recording's start time (`recording_start_time`) minus its own
    first time. Without a call to `synchronize_raw_modalities` every start
    time is 0, which means all recordings are taken to have started together.
    `name` picks the ventilator recording; None means the primary one.
    """

    start = recording_start_time(session, modality, name)
    return start - own_first_time(session, modality)


def shared_clock_shifts(session: M3Session) -> dict[str, float]:
    """`shared_clock_shift` for every modality, keyed by modality name."""

    return {modality: shared_clock_shift(session, modality) for modality in MODALITIES}


def shift_trace(trace: dict[str, Any], shift: float) -> dict[str, Any]:
    """A copy of a plot trace (``{"time": [...], ...}``) moved by `shift` seconds."""

    moved = dict(trace)
    moved["time"] = (np.asarray(trace["time"], dtype=float) + shift).tolist()
    return moved
