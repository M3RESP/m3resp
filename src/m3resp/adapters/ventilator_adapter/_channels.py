"""Identifying ventilator channels and splitting a recording into them.

A ventilator channel cannot be identified by physical quantity alone. A single
study can record several pressures at once - a Draeger pressure pod reports
airway, esophageal, transpulmonary and gastric/auxiliary pressure alongside the
ventilator's own airway pressure - and the same quantity can arrive from more
than one instrument, so two airway pressures may be present, measured by
different devices. Keying channels by quantity would silently collapse those.

Three things therefore identify a channel, and each lands on its own field of
:class:`~m3resp.data.Signal`:

======== ====================================== ==========================
axis     meaning                                ``Signal`` field
======== ====================================== ==========================
quantity what the numbers physically are        ``category``
origin   the instrument/file that recorded it   ``modality`` / ``source``
key      unique handle within one recording     ``channel``
======== ====================================== ==========================

The key is the short channel name (``"pressure"``, ``"esophageal_pressure"``).
When two channels of the same quantity are present, the first keeps the bare
name and the others are suffixed with what distinguishes them
(``"pressure__pod"``), so both stay addressable and neither is dropped.

Vendor naming is open-ended, so the ``label -> channel`` map is a registry with
the same shape as :mod:`m3resp.data.categories`: built-in defaults plus
`register_channel_alias`/`load_channel_aliases`, so a site whose ventilator
export uses different names declares them in configuration rather than in a
patch to this module.
"""

from __future__ import annotations

import json
import re
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
#: The pressures are deliberately separate entries. Collapsing them onto one
#: ``"pressure"`` channel would label an esophageal or transpulmonary trace as
#: an airway pressure, which is a measurement error, not a naming one.
CHANNEL_CATEGORIES: dict[str, str] = {
    "pressure": "airway_pressure",
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
    "pressure": "cmH2O",
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
DEFAULT_CHANNELS: tuple[str, ...] = ("pressure", "flow", "volume")

#: Column index used for a channel when the recording carries no labels to
#: match against. This is the layout of the multi-channel sEMG export the
#: ventilator path originally assumed; it stays supported for unlabelled
#: arrays, but only as an explicit fallback rather than the primary mechanism.
DEFAULT_CHANNEL_POSITIONS: dict[str, int] = {"pressure": 0, "flow": 1, "volume": 2}

#: Built-in ``normalized label -> channel name`` pairs. Ordered most specific
#: first within each channel, which decides which candidate keeps the bare key
#: when a recording carries several channels of one quantity.
_DEFAULT_CHANNEL_ALIASES: dict[str, str] = {
    # airway pressure: the ventilator's own, plus the pressure pod's copy
    "airway pressure": "pressure",
    "airway pressure (pod)": "pressure",
    "paw": "pressure",
    "pressure": "pressure",
    "pvent": "pressure",
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


def register_channel_alias(
    alias: str,
    channel: str,
    *,
    category: str | None = None,
    unit: str | None = None,
) -> None:
    """Register a new ``label -> channel`` mapping, or a wholly new channel.

    For a vendor's naming of an *existing* channel (pressure, flow, ...), this
    is only ever a label mapping: pass just `alias`/`channel`.

    `channel` does not need to already be one of the seven built-ins. A
    physical quantity this vocabulary has no name for yet - a new instrument,
    say - can be registered directly, the same way an unrecognized string
    passed to `m3resp.data.categories.normalize_category` is kept as a custom
    label rather than rejected: `category` defaults to `channel` itself when
    not given, so the channel is always resolvable even with no category
    supplied. `unit` is optional and has no default for a new channel.

    Passing `category=`/`unit=` for an *already-known* channel updates its
    stored category/unit rather than defining a new one.

    The registration lasts for the current process; use `save_channel_aliases`
    to persist the alias mapping (channel/category/unit metadata for a newly
    registered channel is not currently persisted - pass them again on reload,
    or extend `save_channel_aliases` if that becomes a common need).
    """

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
    """Load a ``label -> channel`` map from YAML/JSON and register it.

    The intended route for adopting a site's ventilator naming without
    changing this module. By default the file is merged on top of the active
    map; ``replace=True`` first resets to the built-in defaults.
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
    """Identify the requested channels among a recording's column labels.

    Resolution per channel, in order:

    1. an explicit column index in ``positions`` (a caller override);
    2. a label match through the active alias map;
    3. the positional fallback in :data:`DEFAULT_CHANNEL_POSITIONS`, for
       unlabelled recordings, when ``fallback_positions`` is set.

    When several labels denote the same channel - a pressure pod reporting an
    airway pressure alongside the ventilator's own - the earliest column keeps
    the bare key and later ones are suffixed by what distinguishes them, so all
    of them survive rather than the later ones being dropped.

    ``qualify=True`` suffixes *every* key with ``origin``, for a recording that
    must not collide with one already loaded - a second ventilator file in the
    same session, whose airway pressure is a different measurement from the
    first file's.
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

    specs: list[ChannelSpec] = []
    used_keys: set[str] = set()
    for name in requested:
        if name in positions:
            indices = [int(positions[name])]
        elif candidates[name]:
            indices = candidates[name]
        elif fallback_positions and name in DEFAULT_CHANNEL_POSITIONS:
            indices = [DEFAULT_CHANNEL_POSITIONS[name]]
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
    return specs


def primary_channel(bundle: Any, quantity: str) -> str | None:
    """The channel key representing `quantity` in a bundle, or ``None``.

    `quantity` may be a category (``"airway_pressure"``) or a channel name
    (``"pressure"``), so callers can ask in whichever vocabulary is natural.
    When a recording carries several channels of one quantity this is the one
    downstream steps operate on - the first resolved, unless overridden.
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
    pressure_channel: int | None = None,
    flow_channel: int | None = None,
    volume_channel: int | None = None,
    channel_indices: dict[str, int] | None = None,
    origin: str | None = None,
    qualify: bool = False,
    fs: float | None = None,
) -> dict[str, Any]:
    """Split a ventilator recording into its named channels.

    Channels are found by matching the recording's own labels against the
    alias registry, falling back to fixed column positions only for an
    unlabelled array. ``channels=`` selects which quantities to extract;
    ``pressure_channel``/``flow_channel``/``volume_channel`` (and the general
    ``channel_indices``) override the resolution for one channel with an
    explicit column.

    The returned bundle carries every resolved channel under ``"channels"``
    with its :class:`ChannelSpec` under ``"specs"``, plus ``fs``, ``metadata``,
    and per-channel ``units``/``labels``/``categories``/``channel_indices``.
    Each channel's array is also exposed under its own key, so a bundle with
    the default selection still answers to ``bundle["pressure"]`` as before.
    ``"primary"`` maps each physical quantity to the channel representing it,
    which is what downstream steps use when several channels share a quantity.
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
        ("pressure", pressure_channel),
        ("flow", flow_channel),
        ("volume", volume_channel),
    ):
        if index is not None:
            overrides[name] = int(index)

    specs = resolve_channels(
        metadata.get("labels") or [],
        requested=channels,
        origin=origin or metadata.get("source"),
        units=metadata.get("units") or [],
        positions=overrides,
        qualify=qualify,
    )

    requested = list(channels)
    missing = {name for name in requested} - {spec.channel for spec in specs}
    if missing:
        raise UnresolvedChannelError(
            f"Ventilator channel(s) {sorted(missing)} could not be found in "
            f"this recording. Labels present: {list(metadata.get('labels') or [])}. "
            "Register the vendor's naming with `register_channel_alias`, or "
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
