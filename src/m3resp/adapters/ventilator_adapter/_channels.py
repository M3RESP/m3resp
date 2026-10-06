"""Identify ventilator channels and extract their samples and metadata.

A recording can contain airway, esophageal, transpulmonary and gastric
pressures, including multiple measurements of one quantity from different
sensors. Each channel keeps its physical quantity, origin and unique key in
:class:`~m3resp.data.Signal`:

======== ====================================== ==========================
axis     meaning                                ``Signal`` field
======== ====================================== ==========================
quantity what the numbers physically are        ``category``
origin   the instrument/file that recorded it   ``modality`` / ``source``
key      unique handle within one recording     ``channel``
======== ====================================== ==========================

The key is the short channel name (``"airway_pressure"``, ``"esophageal_pressure"``).
When two channels of the same quantity are present, the first keeps the bare
name and the others are suffixed with what distinguishes them
(``"airway_pressure__pod"``), keeping each measurement separately.

Vendor naming is open-ended, so the ``label -> channel`` map is a registry with
the same shape as :mod:`m3resp.data.categories`: built-in defaults plus
`register_channel_alias`/`load_channel_aliases`. These allow a site's vendor
labels to be supplied in configuration.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from m3resp.core.exceptions import UnresolvedChannelError
from m3resp.data.categories import normalize_category

#: Ventilator channel name -> the physical quantity it carries. The keys are
#: the names used throughout the ventilator bundle dicts and as
#: ``Signal.channel``; the values are ``Signal.category`` (see
#: :mod:`m3resp.data.categories`).
#:
#: Each pressure channel retains the name of the physical quantity measured.
CHANNEL_CATEGORIES: dict[str, str] = {
    "airway_pressure": "airway_pressure",
    "flow": "airflow",
    "volume": "volume",
    "esophageal_pressure": "esophageal_pressure",
    "transpulmonary_pressure": "transpulmonary_pressure",
    "gastric_pressure": "gastric_pressure",
    "tidal_volume": "tidal_volume",
}

#: Fallback units per channel, used only when the recording's metadata does not
#: carry a unit for that channel. Vendors disagree (Draeger reports volume in
#: mL and flow in L/min, Timpel in L and L/s), so a unit read from metadata
#: always wins and is never converted - see `to_signals`, which passes the
#: recorded unit through unchanged.
DEFAULT_CHANNEL_UNITS: dict[str, str] = {
    "airway_pressure": "cmH2O",
    "flow": "L/min",
    "volume": "L",
    "esophageal_pressure": "cmH2O",
    "transpulmonary_pressure": "cmH2O",
    "gastric_pressure": "cmH2O",
    "tidal_volume": "mL",
}

#: Frozen snapshots of the two dicts above, taken before anything can mutate
#: them. `register_channel_alias` can define a channel that was not one of the
#: seven built-ins; `reset_channel_aliases` uses these to undo that alongside
#: resetting the alias map itself, so a channel registered in one process/test
#: does not outlive `reset_channel_aliases()`.
_DEFAULT_CHANNEL_CATEGORIES: dict[str, str] = dict(CHANNEL_CATEGORIES)
_DEFAULT_CHANNEL_UNIT_DEFAULTS: dict[str, str] = dict(DEFAULT_CHANNEL_UNITS)

#: The channels read unless the caller asks for others, in the array order the
#: positional fallback below assumes.
DEFAULT_CHANNELS: tuple[str, ...] = ("airway_pressure", "flow", "volume")

#: Default column indices when every label is absent or unrecognized.
DEFAULT_CHANNEL_POSITIONS: dict[str, int] = {
    "airway_pressure": 0,
    "flow": 1,
    "volume": 2,
}

#: Built-in normalized vendor labels mapped to channel names. Matching
#: columns are resolved in the recording's column order.
_DEFAULT_CHANNEL_ALIASES: dict[str, str] = {
    # airway pressure: the ventilator's own, plus the pressure pod's copy
    "airway pressure": "airway_pressure",
    "airway pressure (pod)": "airway_pressure",
    "paw": "airway_pressure",
    "pressure": "airway_pressure",
    "pvent": "airway_pressure",
    # flow
    "flow": "flow",
    "airway flow": "flow",
    # volume
    "volume": "volume",
    # esophageal pressure (balloon catheter, via the pod)
    "esophageal pressure (pod)": "esophageal_pressure",
    "esophageal pressure": "esophageal_pressure",
    "pes": "esophageal_pressure",
    # transpulmonary pressure. Loaded when the device reports it directly; it
    # is not computed from airway minus esophageal here.
    "transpulmonary pressure (pod)": "transpulmonary_pressure",
    "transpulmonary pressure": "transpulmonary_pressure",
    "pl": "transpulmonary_pressure",
    # gastric/auxiliary pressure
    "gastric pressure/auxiliary pressure (pod)": "gastric_pressure",
    "gastric pressure": "gastric_pressure",
    "auxiliary pressure": "gastric_pressure",
    "pga": "gastric_pressure",
    # tidal volume, when reported as a waveform rather than per breath
    "tidal volume": "tidal_volume",
    "vt": "tidal_volume",
}

#: The active map. Starts as a copy of the defaults; `register_channel_alias`/
#: `load_channel_aliases` mutate it, `reset_channel_aliases` restores it.
_CHANNEL_ALIASES: dict[str, str] = dict(_DEFAULT_CHANNEL_ALIASES)

#: Parenthesised tags naming the *file's* origin rather than a distinct sensor,
#: dropped before matching. ``(pod)`` is deliberately absent: it marks a
#: physically separate transducer, so it must survive normalization and is what
#: distinguishes a pod airway pressure from the ventilator's own.
_VENDOR_TAGS = frozenset({"raw", "timpel", "draeger", "dräger", "sentec"})

_WHITESPACE = re.compile(r"\s+")
_TRAILING_TAG = re.compile(r"\(([^()]*)\)$")


def normalize_channel_label(label: Any) -> str:
    """Normalize a vendor channel label for alias matching.

    Lowercases, treats underscores as spaces, and strips the provenance tag
    vendors append (``"airway_pressure_(timpel)"`` -> ``"airway pressure"``)
    while keeping tags that name a separate sensor, such as ``(pod)``.
    """

    if label is None:
        return ""
    text = str(label).replace("_", " ").strip().lower()
    text = _WHITESPACE.sub(" ", text)
    while True:
        match = _TRAILING_TAG.search(text)
        if match is None or match.group(1).strip() not in _VENDOR_TAGS:
            return text
        text = text[: match.start()].strip()


def label_qualifier(label: Any) -> str | None:
    """The retained tag that distinguishes a label, e.g. ``"pod"``.

    ``None`` when the label carries no distinguishing tag. This is what names a
    channel apart when two of the same quantity are present.
    """

    match = _TRAILING_TAG.search(normalize_channel_label(label))
    if match is None:
        return None
    return _WHITESPACE.sub("_", match.group(1).strip()) or None


def resolve_channel_name(label: Any) -> str | None:
    """The canonical channel a vendor label denotes, or ``None`` if unknown."""

    return _CHANNEL_ALIASES.get(normalize_channel_label(label))


def _check_alias_target(alias: str, channel: str) -> None:
    """Raise ValueError for the obsolete target "pressure"; use "airway_pressure"."""

    if channel == "pressure":
        raise ValueError(
            f"Channel alias {alias!r} points to 'pressure'. The airway "
            "pressure channel is now called 'airway_pressure'."
        )


def register_channel_alias(
    alias: str,
    channel: str,
    *,
    category: str | None = None,
    unit: str | None = None,
) -> None:
    """Register a vendor label for a ventilator channel in the current process.

    Args:
        alias (str): Vendor label, normalized for matching.
        channel (str): Channel name, such as "airway_pressure" or "flow". A new
            name defines a custom channel.
        category (str | None): Physical quantity. For a new channel, defaults to
            its name. Supplying this updates an existing channel's category.
        unit (str | None): Default unit when a recording omits its own unit.
            Supplying this updates an existing channel's default.

    `save_channel_aliases` saves label mappings. Custom category and unit settings
    must be supplied again when reloading those mappings.

    Raises:
        ValueError: If channel is "pressure", the former name of airway_pressure.
    """

    _check_alias_target(alias, channel)
    if channel not in CHANNEL_CATEGORIES:
        CHANNEL_CATEGORIES[channel] = (
            normalize_category(category) or category or channel
        )
        if unit is not None:
            DEFAULT_CHANNEL_UNITS[channel] = unit
    else:
        if category is not None:
            CHANNEL_CATEGORIES[channel] = normalize_category(category) or category
        if unit is not None:
            DEFAULT_CHANNEL_UNITS[channel] = unit

    _CHANNEL_ALIASES[normalize_channel_label(alias)] = channel


def channel_aliases() -> dict[str, str]:
    """A copy of the currently active ``label -> channel`` map."""

    return dict(_CHANNEL_ALIASES)


def reset_channel_aliases() -> None:
    """Restore the active state to the built-in defaults.

    Undoes every `register_channel_alias` call since import or the last reset:
    the alias map, and any channel/category/unit it defined that was not one
    of the seven built-ins.
    """

    _CHANNEL_ALIASES.clear()
    _CHANNEL_ALIASES.update(_DEFAULT_CHANNEL_ALIASES)
    CHANNEL_CATEGORIES.clear()
    CHANNEL_CATEGORIES.update(_DEFAULT_CHANNEL_CATEGORIES)
    DEFAULT_CHANNEL_UNITS.clear()
    DEFAULT_CHANNEL_UNITS.update(_DEFAULT_CHANNEL_UNIT_DEFAULTS)


def save_channel_aliases(path: str | Path, *, only_custom: bool = True) -> Path:
    """Write the active channel map to a YAML/JSON file for later reuse.

    By default only additions/overrides relative to the built-in defaults are
    written, matching `m3resp.data.categories.save_category_aliases`.
    """

    resolved = Path(path).expanduser().resolve()
    if only_custom:
        payload = {
            alias: channel
            for alias, channel in _CHANNEL_ALIASES.items()
            if _DEFAULT_CHANNEL_ALIASES.get(alias) != channel
        }
    else:
        payload = dict(_CHANNEL_ALIASES)

    if resolved.suffix.lower() == ".json":
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        import yaml

        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def load_channel_aliases(path: str | Path, *, replace: bool = False) -> dict[str, str]:
    """Load and register vendor labels from a YAML or JSON file.

    Args:
        path (str | Path): File containing a mapping from labels to channel names.
            A .json suffix selects JSON; other suffixes select YAML.
        replace (bool): Reset aliases, categories and units to the built-in defaults
            before registering the file's mappings. Defaults to False, which adds
            to or updates the current mappings.

    Returns:
        dict[str, str]: Copy of the active normalized-label-to-channel mapping.

    Raises:
        TypeError: If the file's contents are a value other than a mapping.
        ValueError: If any alias targets the former channel name "pressure".
            All targets are checked before updating the active mappings.
    """

    resolved = Path(path).expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    raw: Any = (
        json.loads(text) if resolved.suffix.lower() == ".json" else _yaml_load(text)
    )
    if not isinstance(raw, dict):
        raise TypeError(
            f"Channel alias file {resolved} must contain a mapping of "
            "label -> channel name."
        )

    # Validate all targets before changing the active mappings.
    for alias, channel in raw.items():
        _check_alias_target(str(alias), str(channel))
    if replace:
        reset_channel_aliases()
    for alias, channel in raw.items():
        register_channel_alias(str(alias), str(channel))
    return channel_aliases()


def _yaml_load(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


@dataclass(frozen=True)
class ChannelSpec:
    """What identifies one ventilator channel within a recording.

    ``key`` is unique inside a bundle and becomes ``Signal.channel``;
    ``category`` becomes ``Signal.category``; ``origin`` records the
    instrument/file the channel came from and becomes ``Signal.source``.
    ``label`` keeps the vendor's own wording verbatim so nothing is lost, and
    ``unit`` is as reported - never converted.
    """

    key: str
    channel: str
    category: str | None
    origin: str | None = None
    label: str | None = None
    unit: str | None = None
    index: int | None = None


def resolve_channels(
    labels: Sequence[Any],
    *,
    requested: Iterable[str] = DEFAULT_CHANNELS,
    origin: str | None = None,
    units: Sequence[Any] = (),
    positions: dict[str, int] | None = None,
    fallback_positions: bool = True,
    qualify: bool = False,
) -> list[ChannelSpec]:
    """Identify requested ventilator channels from column labels or indices.

    Explicit positions take priority over label matches. Default column positions
    are used when fallback_positions is True and all labels are unrecognized or
    absent. A recognized label for any channel disables that fallback. Using
    positions with unrecognized nonempty labels emits a UserWarning.

    Args:
        labels (Sequence[Any]): Vendor labels in column order.
        requested (Iterable[str]): Channel names to find. Defaults to
            airway_pressure, flow and volume.
        origin (str | None): Instrument or file name to record with each channel.
        units (Sequence[Any]): Recorded units in column order. Missing units use
            DEFAULT_CHANNEL_UNITS.
        positions (dict[str, int] | None): Explicit zero-based column indices.
        fallback_positions (bool): Allow default column positions. Defaults to True.
        qualify (bool): Add origin to channel keys when supplied. Defaults to False.

    Returns:
        list[ChannelSpec]: Resolved channels in requested order. Missing channels
            are omitted. Multiple matching columns are retained: the earliest
            keeps the base key, and later columns get a sensor, origin or column
            suffix. Array bounds are checked when `split_channels` reads the data.

    Raises:
        ValueError: If a requested channel name is unknown.
    """

    requested = list(requested)
    unknown = [name for name in requested if name not in CHANNEL_CATEGORIES]
    if unknown:
        raise ValueError(
            f"Unknown ventilator channel(s) {unknown}. Known channels: "
            f"{sorted(CHANNEL_CATEGORIES)}."
        )

    positions = dict(positions or {})
    units = list(units)

    # Which columns each requested channel could come from, in column order.
    # A label is claimed by at most one channel, so nothing is counted twice.
    candidates: dict[str, list[int]] = {name: [] for name in requested}
    for index, label in enumerate(labels):
        name = resolve_channel_name(label)
        if name in candidates:
            candidates[name].append(index)

    labels_name_a_channel = any(resolve_channel_name(label) for label in labels)
    use_positions = fallback_positions and not labels_name_a_channel

    specs: list[ChannelSpec] = []
    used_keys: set[str] = set()
    by_position: dict[str, int] = {}
    for name in requested:
        if name in positions:
            indices = [int(positions[name])]
        elif candidates[name]:
            indices = candidates[name]
        elif use_positions and name in DEFAULT_CHANNEL_POSITIONS:
            indices = [DEFAULT_CHANNEL_POSITIONS[name]]
            by_position[name] = indices[0]
        else:
            continue

        for rank, index in enumerate(indices):
            label = labels[index] if 0 <= index < len(labels) else None
            key = f"{name}__{origin}" if qualify and origin else name
            if rank or key in used_keys:
                qualifier = label_qualifier(label) or origin or f"col{index}"
                key = f"{name}__{qualifier}"
                suffix = 2
                while key in used_keys:
                    key = f"{name}__{qualifier}{suffix}"
                    suffix += 1
            used_keys.add(key)
            unit = units[index] if 0 <= index < len(units) and units[index] else None
            specs.append(
                ChannelSpec(
                    key=key,
                    channel=name,
                    category=normalize_category(CHANNEL_CATEGORIES[name]),
                    origin=(label_qualifier(label) or origin),
                    label=str(label) if label is not None else None,
                    unit=unit or DEFAULT_CHANNEL_UNITS.get(name),
                    index=index,
                )
            )
    if by_position and any(label is not None and str(label) for label in labels):
        warnings.warn(
            f"None of the column labels {[str(label) for label in labels]} "
            "names a known ventilator channel, so these channels are read by "
            f"column position: {by_position}. Register the labels with "
            "`register_channel_alias` to read them by name instead.",
            UserWarning,
            stacklevel=2,
        )
    return specs


def primary_channel(bundle: Any, quantity: str) -> str | None:
    """Return the main channel key for a physical quantity, or None.

    Args:
        bundle (Any): Channel bundle returned by `split_channels` or preprocessing.
        quantity (str): Category, such as "airflow", or channel name, such as
            "flow" or "airway_pressure".

    Returns:
        str | None: Key from the bundle's primary mapping. The first resolved
            channel is the default primary channel. If that mapping lacks the
            quantity, returns quantity when it is a top-level bundle key.
            Returns None when a key is unavailable or bundle has another type.
    """

    if not isinstance(bundle, dict):
        return None
    primary = bundle.get("primary") or {}
    category = normalize_category(quantity) or CHANNEL_CATEGORIES.get(quantity)
    key = primary.get(category) if category else None
    if key is not None:
        return str(key)
    # A bundle that predates `primary` (or a hand-built one) still answers to
    # the plain channel name.
    return quantity if quantity in bundle else None


def recording_payload(recording: Any) -> dict[str, Any] | None:
    """The ``{"array", "metadata"}`` payload of a ventilator recording.

    Accepts a :class:`~m3resp.modalities.ventilator.VentilatorRecording`, a bare
    payload dict, or a raw array-like.
    """

    if isinstance(recording, dict) and "array" in recording:
        return recording
    data = getattr(recording, "data", None)
    if isinstance(data, dict) and "array" in data:
        return data
    return None


def split_channels(
    recording: Any,
    *,
    channels: Iterable[str] = DEFAULT_CHANNELS,
    airway_pressure_channel: int | None = None,
    flow_channel: int | None = None,
    volume_channel: int | None = None,
    channel_indices: dict[str, int] | None = None,
    origin: str | None = None,
    qualify: bool = False,
    fs: float | None = None,
) -> dict[str, Any]:
    """Extract named channels and their metadata from a ventilator recording.

    Channels are resolved by vendor labels, with default positions used when all
    labels are absent or unrecognized. Explicit column indices override label
    matches. Values retain their recorded units.

    Args:
        recording (Any): VentilatorRecording, a dict with array and metadata, or a
            numeric array. Arrays have shape (channels, samples); a one-dimensional
            array holds one channel. Metadata may supply labels, units and fs.
        channels (Iterable[str]): Names to extract. Defaults to airway_pressure,
            flow and volume. All requested channels must resolve.
        airway_pressure_channel (int | None): Zero-based airway-pressure index.
        flow_channel (int | None): Zero-based flow index.
        volume_channel (int | None): Zero-based volume index.
        channel_indices (dict[str, int] | None): Indices for any requested channels.
            The three individual index arguments take priority over this mapping.
        origin (str | None): Instrument or file name; defaults to metadata source.
        qualify (bool): Add origin to channel keys when supplied. Defaults to False.
        fs (float | None): Sampling rate in Hz; overrides metadata fs.

    Returns:
        dict[str, Any]: One-dimensional arrays at their channel keys and in
            channels, ChannelSpecs in specs, and fs in Hz. Also contains metadata,
            units, labels, categories, channel_indices, origins, and a primary
            mapping from physical quantities to their first resolved channel key.
            Multiple channels for the same quantity are retained with unique keys.

    Raises:
        TypeError: If the sample array or sampling rate is missing.
        ValueError: If a channel name is unknown or an explicit index is supplied
            for a channel outside the requested selection.
        UnresolvedChannelError: If a requested channel has no matching label or
            usable positional fallback.
        IndexError: If a resolved index exceeds the available channels.
    """

    payload = recording_payload(recording)
    if payload is not None:
        array = payload["array"]
        metadata = payload.get("metadata") or {}
    else:
        # A bare array-like, with the sample rate supplied by the caller.
        array = recording
        metadata = {}
    if array is None:
        raise TypeError("Ventilator preprocessing input needs an array.")

    sample_frequency = fs if fs is not None else metadata.get("fs")
    if sample_frequency is None:
        raise TypeError(
            "Ventilator preprocessing input needs a sampling rate. Pass `fs=` "
            "or include metadata['fs'] in the recording."
        )

    array = np.asarray(array, dtype=float)

    overrides = dict(channel_indices or {})
    for name, index in (
        ("airway_pressure", airway_pressure_channel),
        ("flow", flow_channel),
        ("volume", volume_channel),
    ):
        if index is not None:
            overrides[name] = int(index)

    requested = list(channels)
    # Explicit indices must belong to the requested channel selection.
    unused = sorted(set(overrides) - set(requested))
    if unused:
        raise ValueError(
            f"Column indices were given for ventilator channel(s) {unused}, "
            f"which are not among the channels being read ({requested}). "
            f"Known channels: {sorted(CHANNEL_CATEGORIES)}."
        )

    specs = resolve_channels(
        metadata.get("labels") or [],
        requested=requested,
        origin=origin or metadata.get("source"),
        units=metadata.get("units") or [],
        positions=overrides,
        qualify=qualify,
    )

    missing = {name for name in requested} - {spec.channel for spec in specs}
    if missing:
        raise UnresolvedChannelError(
            f"Ventilator channel(s) {sorted(missing)} could not be found in "
            f"this recording. Labels present: {list(metadata.get('labels') or [])}. "
            "Ask only for the channels this recording has (`channels=`), "
            "register the vendor's naming with `register_channel_alias`, or "
            "pass an explicit column index."
        )

    # A 1D array carries a single channel; treat it as `channel_count == 1`
    # rather than silently aliasing every requested index onto the whole
    # array (that previously made pressure/flow/volume identical whenever a
    # non-multi-row recording came in, with no error).
    channel_count = array.shape[0] if array.ndim > 1 else 1

    resolved: dict[str, Any] = {}
    for spec in specs:
        index = spec.index if spec.index is not None else 0
        if index >= channel_count:
            raise IndexError(
                f"Ventilator {spec.channel} channel index {index} is out of "
                f"range for a recording with {channel_count} channel"
                f"{'s' if channel_count != 1 else ''}."
            )
        resolved[spec.key] = np.asarray(
            array[index] if array.ndim > 1 else array, dtype=float
        )

    by_key = {spec.key: spec for spec in specs}
    primary: dict[str, str] = {}
    for spec in specs:
        if spec.category is not None:
            primary.setdefault(spec.category, spec.key)

    return {
        **resolved,
        "channels": resolved,
        "specs": by_key,
        "primary": primary,
        "fs": float(sample_frequency),
        "metadata": metadata,
        "channel_indices": {key: spec.index for key, spec in by_key.items()},
        "units": {key: spec.unit for key, spec in by_key.items()},
        "labels": {key: spec.label for key, spec in by_key.items()},
        "categories": {key: spec.category for key, spec in by_key.items()},
        "origins": {key: spec.origin for key, spec in by_key.items()},
    }
