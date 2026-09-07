"""Canonical unit-string normalization.

Free-form unit strings drift ("uV" vs "µV", "au"/"a.u"/"A.U." for arbitrary
units), which makes them hard to compare across signals/parameters produced
by different loaders. This isn't a closed enum - `normalize_unit` maps a
small set of known aliases to one canonical spelling and passes anything
else through unchanged, so an unrecognized unit is never rejected, only left
un-normalized until an alias for it is registered.

Registrations made with `register_unit_alias` only live for the current
process. To persist a custom vocabulary, `save_unit_aliases`/
`load_unit_aliases` round-trip the map through a YAML or JSON file, using the
same dual-format convention as the pipeline-spec loader
(`m3resp.workflows.spec.load_spec`), so a project can keep its unit map
alongside its pipeline specs and load it before a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Built-in aliases, kept immutable so the active map can always be reset to
#: (or diffed against) the defaults. A "custom map" is understood as this
#: baseline plus a user's additions/overrides.
_DEFAULT_UNIT_ALIASES: dict[str, str] = {
    "uv": "µV",
    "microvolt": "µV",
    "microvolts": "µV",
    "mv": "mV",
    "millivolt": "mV",
    "millivolts": "mV",
    "au": "a.u.",
    "a.u": "a.u.",
    "arbitrary unit": "a.u.",
    "arbitrary units": "a.u.",
}

#: The active map. Starts as a copy of the defaults and is what
#: `normalize_unit` consults; `register_unit_alias`/`load_unit_aliases` mutate
#: this, `reset_unit_aliases` restores it from `_DEFAULT_UNIT_ALIASES`.
_UNIT_ALIASES: dict[str, str] = dict(_DEFAULT_UNIT_ALIASES)


def normalize_unit(unit: str | None) -> str | None:
    """Return the canonical spelling for `unit` if a known alias matches.

    Falls back to `unit` unchanged (including its original casing) when no
    alias is registered for it.
    """

    if unit is None:
        return None
    return _UNIT_ALIASES.get(unit.strip().lower(), unit)


def register_unit_alias(alias: str, canonical: str) -> None:
    """Register a new unit spelling -> canonical mapping.

    For loader/adapter implementations that produce a unit spelling not
    already known here, rather than editing this module. The registration
    lasts for the current process only; use `save_unit_aliases` to persist it.
    """

    _UNIT_ALIASES[alias.strip().lower()] = canonical


def unit_aliases() -> dict[str, str]:
    """A copy of the currently active ``alias -> canonical`` map."""

    return dict(_UNIT_ALIASES)


def reset_unit_aliases() -> None:
    """Restore the active map to the built-in defaults.

    Drops every `register_unit_alias`/`load_unit_aliases` registration made
    since import.
    """

    _UNIT_ALIASES.clear()
    _UNIT_ALIASES.update(_DEFAULT_UNIT_ALIASES)


def save_unit_aliases(path: str | Path, *, only_custom: bool = True) -> Path:
    """Write the active alias map to a YAML/JSON file for later reuse.

    ``.json`` files are written as JSON; everything else is written as YAML
    (matching `m3resp.workflows.spec.load_spec`'s format detection).

    By default (``only_custom=True``) only the additions/overrides relative to
    the built-in defaults are written, so the file stays a small, readable
    "custom map based on the defaults" and automatically picks up future
    changes to the built-ins. Pass ``only_custom=False`` to snapshot the full
    active map instead.

    Returns the resolved path written to.
    """

    resolved = Path(path).expanduser().resolve()
    if only_custom:
        payload = {
            alias: canonical
            for alias, canonical in _UNIT_ALIASES.items()
            if _DEFAULT_UNIT_ALIASES.get(alias) != canonical
        }
    else:
        payload = dict(_UNIT_ALIASES)

    if resolved.suffix.lower() == ".json":
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        import yaml

        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def load_unit_aliases(path: str | Path, *, replace: bool = False) -> dict[str, str]:
    """Load an ``alias -> canonical`` map from a YAML/JSON file and register it.

    ``.json`` files are parsed as JSON; everything else with ``yaml.safe_load``
    (a superset of JSON), mirroring `m3resp.workflows.spec.load_spec`.

    By default the file's aliases are merged on top of the currently active
    map. Pass ``replace=True`` to first reset to the built-in defaults, so the
    result is exactly "defaults + this file" with any earlier ad-hoc
    registrations dropped. The built-in defaults are always retained either
    way - a custom map extends them, it doesn't replace them.

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
            f"Unit alias file {resolved} must be a mapping of "
            f"alias -> canonical, got {type(raw).__name__}."
        )

    if replace:
        reset_unit_aliases()
    for alias, canonical in raw.items():
        register_unit_alias(str(alias), str(canonical))
    return unit_aliases()
