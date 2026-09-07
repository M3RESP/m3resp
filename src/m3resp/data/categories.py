"""Canonical data-*category* vocabulary: what a signal physically measures.

``Signal.modality`` answers "which device/technique produced this?" (``"eit"``,
``"emg"``, ``"ventilator"``). ``Signal.category`` answers the orthogonal
question "what physical quantity are these numbers?" (an impedance, an airway
pressure, a flow, a volume).

Keeping these apart matters because they do not nest. Airway pressure can come
from a ventilator or a standalone monitor; a ventilator emits pressure *and*
flow *and* volume. Collapsing both axes into one string - which is what
``modality`` alone used to do, with ``"pressure"``/``"flow"`` sitting in the
same vocabulary as ``"eit"``/``"emg"`` - makes some combinations
inexpressible. The persisted Layer 2 model never had this problem: it already
separates ``Device.device_type`` from ``SignalStream.signal_type``, so
``DataModelRecorder`` was left heuristically splitting the single Layer 1
string back into two and guessing wrong (see ``_signal_type_for``).

The vocabulary here is deliberately *modality-agnostic*, following the same
principle as ``eitprocessing``'s ``categories-compact.yaml``: it is a taxonomy
of physical quantities and their physiological relationships, with no notion of
which device measured them. That is what lets the two fields compose instead of
overlap.

Like `m3resp.data.metrics`, an unrecognized name returns ``None`` rather than
the input unchanged, so a custom/experimental category is never silently
promoted to a standard one. The map is open: extend it with
`register_category_alias`, or load an external catalogue with
`load_category_aliases` - which is how a shared vocabulary (e.g. one exported
from ``eitprocessing``) can be adopted without this module hard-coding it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, get_args

#: Common values for ``Signal.category``. Open vocabulary, not an enum - listed
#: for documentation/IDE-completion convenience and not enforced at runtime,
#: exactly like :data:`m3resp.data.signals.Modality`.
#:
#: This starter set covers the quantities m3resp currently emits, plus the
#: near-neighbours needed to keep pressure sub-types distinguishable. It is
#: provisional: the intent is to align these leaf names with ``eitprocessing``'s
#: shared catalogue rather than maintain a competing taxonomy.
_KNOWN_CATEGORIES_LITERAL = Literal[
    # electrical
    "impedance",
    "electrical_potential",
    # pressure
    "airway_pressure",
    "esophageal_pressure",
    "gastric_pressure",
    "transpulmonary_pressure",
    # flow / volume
    "airflow",
    "volume",
    "tidal_volume",
]
Category = _KNOWN_CATEGORIES_LITERAL | str
KNOWN_CATEGORIES = frozenset(get_args(_KNOWN_CATEGORIES_LITERAL))

#: Built-in ``alias -> canonical category`` pairs, kept immutable so the active
#: map can always be reset to (or diffed against) the defaults. Aliases are
#: matched case-insensitively after stripping surrounding whitespace.
_DEFAULT_CATEGORY_ALIASES: dict[str, str] = {
    # impedance (EIT)
    "impedance": "impedance",
    "global_impedance": "impedance",
    "pixel_impedance": "impedance",
    # electrical potential (EMG). The envelope of an sEMG trace is still a
    # potential - "envelope" is a processing_state, not a different quantity.
    "electrical_potential": "electrical_potential",
    "emg": "electrical_potential",
    "emg_raw": "electrical_potential",
    "emg_envelope": "electrical_potential",
    # pressure. Bare "pressure" resolves to airway pressure because that is the
    # only pressure m3resp currently records; esophageal/gastric must be named
    # explicitly so they are never silently mislabelled as airway.
    "pressure": "airway_pressure",
    "airway_pressure": "airway_pressure",
    "paw": "airway_pressure",
    "esophageal_pressure": "esophageal_pressure",
    "pes": "esophageal_pressure",
    "gastric_pressure": "gastric_pressure",
    "pga": "gastric_pressure",
    "transpulmonary_pressure": "transpulmonary_pressure",
    "pl": "transpulmonary_pressure",
    # flow
    "flow": "airflow",
    "airflow": "airflow",
    # volume
    "volume": "volume",
    "tidal_volume": "tidal_volume",
    "vt": "tidal_volume",
}

#: The active map. Starts as a copy of the defaults and is what
#: `normalize_category` consults; `register_category_alias`/
#: `load_category_aliases` mutate this, `reset_category_aliases` restores it.
_CATEGORY_ALIASES: dict[str, str] = dict(_DEFAULT_CATEGORY_ALIASES)


def normalize_category(name: str | None) -> str | None:
    """Return the canonical category for `name`, or ``None`` if unknown.

    Matches case-insensitively after stripping whitespace. Returning ``None``
    for an unrecognized name is intentional: it means "this is a custom label,
    not a known standard category", so callers don't tag it as one.
    """

    if name is None:
        return None
    return _CATEGORY_ALIASES.get(name.strip().lower())


def register_category_alias(alias: str, canonical: str) -> None:
    """Register a new ``alias -> canonical category`` mapping.

    For steps/loaders emitting a category not already known here, rather than
    editing this module. The registration lasts for the current process; use
    `save_category_aliases` to persist it.
    """

    _CATEGORY_ALIASES[alias.strip().lower()] = canonical


def category_aliases() -> dict[str, str]:
    """A copy of the currently active ``alias -> canonical category`` map."""

    return dict(_CATEGORY_ALIASES)


def reset_category_aliases() -> None:
    """Restore the active map to the built-in defaults.

    Drops every `register_category_alias`/`load_category_aliases` registration
    made since import.
    """

    _CATEGORY_ALIASES.clear()
    _CATEGORY_ALIASES.update(_DEFAULT_CATEGORY_ALIASES)


def save_category_aliases(path: str | Path, *, only_custom: bool = True) -> Path:
    """Write the active category map to a YAML/JSON file for later reuse.

    ``.json`` files are written as JSON; everything else as YAML (matching
    `m3resp.workflows.spec.load_spec`'s format detection).

    By default only the additions/overrides relative to the built-in defaults
    are written, so the file stays a small, readable "custom map based on the
    defaults" and automatically picks up future changes to the built-ins. Pass
    ``only_custom=False`` to snapshot the full active map instead.

    Returns the resolved path written to.
    """

    resolved = Path(path).expanduser().resolve()
    if only_custom:
        payload = {
            alias: canonical
            for alias, canonical in _CATEGORY_ALIASES.items()
            if _DEFAULT_CATEGORY_ALIASES.get(alias) != canonical
        }
    else:
        payload = dict(_CATEGORY_ALIASES)

    if resolved.suffix.lower() == ".json":
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        import yaml

        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def load_category_aliases(path: str | Path, *, replace: bool = False) -> dict[str, str]:
    """Load a category map from a YAML/JSON file and register it.

    This is the intended route for adopting an externally-maintained taxonomy
    (such as ``eitprocessing``'s shared catalogue) without this package taking a
    hard dependency on it or vendoring a copy that drifts.

    By default the file's aliases are merged on top of the currently active
    map. Pass ``replace=True`` to first reset to the built-in defaults, so the
    result is exactly "defaults + this file". The built-in defaults are always
    retained either way - a custom map extends them, it doesn't replace them.

    Returns a copy of the active map after loading.
    """

    resolved = Path(path).expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() == ".json":
        raw: Any = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(
            f"Category alias file {resolved} must be a mapping of "
            f"alias -> canonical category, got {type(raw).__name__}."
        )

    if replace:
        reset_category_aliases()
    for alias, canonical in raw.items():
        register_category_alias(str(alias), str(canonical))
    return category_aliases()
