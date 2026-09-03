"""Adapter boundary for ventilator data.

Unlike `EITProcessingAdapter` and `ReSurfEMGAdapter`, this adapter does not wrap
an upstream library: no ventilator preprocessing exists in either `eitprocessing`
or `resurfemg`, which is why ventilator channels were used unfiltered and why
ventilator data reached a session only as a passenger of the EMG path. Its
defaults are native, built on `m3resp.processing.filters` and
`m3resp.processing.peaks`.

Loading is the one exception, and it has three sources rather than one:
ventilator channels arrive in the multi-channel file shared with the sEMG -
where `load` delegates to `ReSurfEMGAdapter` - inside the EIT `*.bin` itself,
where the device stores them beside the impedance frames and `load` goes
through `EITProcessingAdapter` (see `_eit_source`) - or in a third-party format
neither of those knows about, read by a function registered via
`register_ventilator_loader` (see `_loaders`), so a new format does not need a
code change here. Dispatch is by file suffix for the first two; either can be
replaced with an injected loader per instance, or a registered extension takes
over automatically for every instance.

Composed from mixins split by responsibility, matching the layout of
`m3resp.adapters.resurfemg_adapter`.
"""

from __future__ import annotations

from ._channels import (
    CHANNEL_CATEGORIES,
    DEFAULT_CHANNEL_POSITIONS,
    DEFAULT_CHANNEL_UNITS,
    DEFAULT_CHANNELS,
    ChannelSpec,
    channel_aliases,
    load_channel_aliases,
    normalize_channel_label,
    primary_channel,
    recording_payload,
    register_channel_alias,
    reset_channel_aliases,
    resolve_channel_name,
    resolve_channels,
    save_channel_aliases,
    split_channels,
)
from ._core import _CoreMixin, resolve_ventilator_source
from ._defaults import DEFAULT_FILTER_ORDER, SUGGESTED_LOWPASS_HZ, _DefaultsMixin
from ._eit_source import (
    DEFAULT_EIT_CHANNELS,
    EIT_ORIGIN,
    SENTINEL_CUTOFF,
    available_ventilator_channels,
    ventilator_payload_from_sequence,
)
from ._loaders import (
    register_ventilator_loader,
    reset_ventilator_loaders,
    unregister_ventilator_loader,
    ventilator_loaders,
)


class VentilatorAdapter(_CoreMixin, _DefaultsMixin):
    """Adapter boundary for ventilator pressure/flow/volume data."""


__all__ = [
    "CHANNEL_CATEGORIES",
    "DEFAULT_CHANNELS",
    "DEFAULT_CHANNEL_POSITIONS",
    "DEFAULT_CHANNEL_UNITS",
    "DEFAULT_EIT_CHANNELS",
    "DEFAULT_FILTER_ORDER",
    "EIT_ORIGIN",
    "SENTINEL_CUTOFF",
    "SUGGESTED_LOWPASS_HZ",
    "ChannelSpec",
    "VentilatorAdapter",
    "available_ventilator_channels",
    "channel_aliases",
    "load_channel_aliases",
    "normalize_channel_label",
    "primary_channel",
    "recording_payload",
    "register_channel_alias",
    "register_ventilator_loader",
    "reset_channel_aliases",
    "reset_ventilator_loaders",
    "resolve_channel_name",
    "resolve_channels",
    "resolve_ventilator_source",
    "save_channel_aliases",
    "split_channels",
    "unregister_ventilator_loader",
    "ventilator_loaders",
    "ventilator_payload_from_sequence",
]
