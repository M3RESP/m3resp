"""Which recordings have been placed on the shared clock, and how.

A session is meant to hold recordings of the same stretch of real time. That
only holds once each recording has been synchronized: given a start time on
the shared clock (`M3Session.synchronize_raw_modalities`), or knowingly used
as it is (`M3Session.skip_synchronization`). `session.sync_methods` records
this per recording - per source, not per modality, since one device can
record several modalities and one modality can come from several devices:

- ``"eit"`` and ``"emg"``: the loaded EIT and EMG recordings.
- ``"ventilator:<name>"``: each standalone ventilator recording (one loaded
  with ``source="ventilator"``, with a clock of its own).

Ventilator data that came inside the EIT or EMG file was recorded on that
file's clock, so it is always in step with it and uses its record.

The values come from the data model's sync vocabulary
(`m3resp.datamodel.entities.SyncMethod`): ``"manual"`` for a hand-entered
offset, ``"none"`` for recordings used as they are.

Steps that compare recordings on different clocks call
`warn_if_not_synchronized`, which warns (it does not stop the step) when any
of those recordings has no record.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING

from m3resp.core.exceptions import UnsynchronizedDataWarning
from m3resp.modalities.names import VENTILATOR, normalize_modality
from m3resp.modalities.ventilator import (
    DEFAULT_VENTILATOR_NAME,
    ventilator_clock,
    ventilator_raw,
    ventilator_recording,
)
from m3resp.synchronization.alignment import ventilator_start_key

if TYPE_CHECKING:
    from m3resp.core.session import M3Session

#: A hand-entered offset (`synchronize_raw_modalities`,
#: `synchronize_multimodal_breaths`).
MANUAL = "manual"
#: Used as it is, taken to have started with the others (`skip_synchronization`).
NONE = "none"


def recording_keys(session: M3Session) -> list[str]:
    """The `session.sync_methods` keys of every loaded recording that has a
    clock of its own: ``"eit"``, ``"emg"``, and ``"ventilator:<name>"`` for
    each standalone ventilator recording."""

    keys = []
    if session.eit is not None:
        keys.append("eit")
    if session.emg is not None:
        keys.append("emg")
    named = session.ventilators or (
        {DEFAULT_VENTILATOR_NAME: ventilator_raw(session)}
        if ventilator_raw(session) is not None
        else {}
    )
    for name, recording in named.items():
        if ventilator_clock(recording) == VENTILATOR:
            keys.append(ventilator_start_key(name))
    return keys


def clock_key(session: M3Session, modality: str, name: str | None = None) -> str:
    """The `session.sync_methods` key whose record applies to a modality's
    recording. `name` picks the ventilator recording; None means the primary
    one. Ventilator data from the EIT or EMG file gives that modality's key."""

    modality = normalize_modality(modality)
    if modality != VENTILATOR:
        return modality
    recording, name = ventilator_recording(session, name)
    clock = ventilator_clock(recording) if recording is not None else VENTILATOR
    return clock if clock != VENTILATOR else ventilator_start_key(name)


def warn_if_not_synchronized(
    session: M3Session, modalities: Iterable[str], *, action: str
) -> None:
    """Warn when `action` compares recordings that were never synchronized.

    `modalities` are the modalities whose data `action` compares. Nothing is
    warned when they all share one clock (for example EMG and the airway
    pressure recorded in the same file), or when every recording involved
    has a record in `session.sync_methods` - including ``"none"`` from
    `skip_synchronization`, which is a deliberate choice.
    """

    keys = sorted({clock_key(session, modality) for modality in modalities})
    if len(keys) < 2:
        return
    missing = [key for key in keys if key not in session.sync_methods]
    if not missing:
        return
    if missing == keys:
        not_synchronized = "they were never synchronized"
    else:
        verb = "was" if len(missing) == 1 else "were"
        not_synchronized = f"{_describe(missing)} {verb} never synchronized"
    warnings.warn(
        f"{action} compares {_describe(keys)} as if they started at the same "
        f"moment, but {not_synchronized}. Set their start times with "
        "session.synchronize_raw_modalities(...), or call "
        "session.skip_synchronization() if they really did start together.",
        UnsynchronizedDataWarning,
        stacklevel=3,
    )


def _describe(keys: list[str]) -> str:
    labels = []
    for key in keys:
        if key.startswith(ventilator_start_key("")):
            labels.append(f"ventilator recording {key.split(':', 1)[1]!r}")
        else:
            labels.append(f"the {key.upper()} recording")
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]
