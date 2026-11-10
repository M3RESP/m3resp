"""Adapter for ventilator data.

The ventilator adapter processes ventilator channels with m3resp code
(`m3resp.processing.filters` and `m3resp.processing.peaks`).

`load` reads ventilator channels from three sources:

- the multi-channel file that also holds the sEMG, read with
  `ReSurfEMGAdapter`;
- the EIT `*.bin` file, where the device stores the ventilator channels next
  to the impedance frames, read with the m3resp EIT loader (see
  `_eit_source`);
- a file in another format, read by a function registered with
  `register_ventilator_loader` (see `_loaders`).

The file extension selects the source. A loader given to one adapter replaces
the built-in loader of that adapter. A registered loader applies to every
adapter.

The adapter is made of mixins, one per task, with the same layout as
`m3resp.adapters.resurfemg_adapter`.
"""

from __future__ import annotations

from ._breaths import (
    infer_ventilator_duration,
    infer_ventilator_fs,
    iter_ventilator_detections,
    normalize_ventilator_breath,
)
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
    ventilator_payload_from_eit,
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
    "infer_ventilator_duration",
    "infer_ventilator_fs",
    "iter_ventilator_detections",
    "load_channel_aliases",
    "normalize_channel_label",
    "normalize_ventilator_breath",
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
    "ventilator_payload_from_eit",
]
