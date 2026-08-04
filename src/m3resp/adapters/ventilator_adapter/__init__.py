"""Adapter boundary for ventilator data.

Unlike `EITProcessingAdapter` and `ReSurfEMGAdapter`, this adapter does not wrap
an upstream library: no ventilator preprocessing exists in either `eitprocessing`
or `resurfemg`, which is why ventilator channels were used unfiltered and why
ventilator data reached a session only as a passenger of the EMG path. Its
defaults are native, built on `m3resp.processing.filters` and
`m3resp.processing.peaks`.

Loading is the one exception: ventilator channels usually arrive in the same
multi-channel file as the sEMG, so `load` delegates to `ReSurfEMGAdapter` unless
a loader is injected.

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


class VentilatorAdapter(_CoreMixin, _DefaultsMixin):
    """Adapter boundary for ventilator pressure/flow/volume data."""


__all__ = [
    "CHANNEL_CATEGORIES",
    "DEFAULT_CHANNEL_UNITS",
    "DEFAULT_FILTER_ORDER",
    "DEFAULT_LOWPASS_HZ",
    "VentilatorAdapter",
    "recording_payload",
    "split_channels",
]
