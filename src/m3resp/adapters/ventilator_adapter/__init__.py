"""Adapter boundary for ventilator data.

Unlike `EITProcessingAdapter` and `ReSurfEMGAdapter`, this adapter does not wrap
an upstream library: no ventilator preprocessing exists in either `eitprocessing`
or `resurfemg`, which is why ventilator channels were used unfiltered and why
ventilator data reached a session only as a passenger of the EMG path. Its
defaults are native, built on `m3resp.processing.filters` and
`m3resp.processing.peaks`.

Loading is the one exception, and it has two sources rather than one:
ventilator channels either arrive in the multi-channel file shared with the
sEMG - where `load` delegates to `ReSurfEMGAdapter` - or inside the EIT `*.bin`
itself, where the device stores them beside the impedance frames and `load`
goes through `EITProcessingAdapter` (see `_eit_source`). Dispatch is by file
suffix; either side can be replaced with an injected loader.

Composed from mixins split by responsibility, matching the layout of
`m3resp.adapters.resurfemg_adapter`.
"""

from __future__ import annotations

from ._channels import (
    CHANNEL_CATEGORIES,
    DEFAULT_CHANNEL_UNITS,
    recording_payload,
    split_channels,
)
from ._core import _CoreMixin
from ._defaults import DEFAULT_FILTER_ORDER, DEFAULT_LOWPASS_HZ, _DefaultsMixin
from ._eit_source import (
    DEFAULT_EIT_CHANNELS,
    VENTILATOR_CHANNEL_ALIASES,
    available_ventilator_channels,
    ventilator_payload_from_sequence,
)


class VentilatorAdapter(_CoreMixin, _DefaultsMixin):
    """Adapter boundary for ventilator pressure/flow/volume data."""


__all__ = [
    "CHANNEL_CATEGORIES",
    "DEFAULT_CHANNEL_UNITS",
    "DEFAULT_EIT_CHANNELS",
    "DEFAULT_FILTER_ORDER",
    "DEFAULT_LOWPASS_HZ",
    "VENTILATOR_CHANNEL_ALIASES",
    "VentilatorAdapter",
    "available_ventilator_channels",
    "recording_payload",
    "split_channels",
    "ventilator_payload_from_sequence",
]
