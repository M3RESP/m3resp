"""Canonical unit-string normalization.

Free-form unit strings drift ("uV" vs "µV", "au"/"a.u"/"A.U." for arbitrary
units), which makes them hard to compare across signals/parameters produced
by different loaders. This isn't a closed enum - `normalize_unit` maps a
small set of known aliases to one canonical spelling and passes anything
else through unchanged, so an unrecognized unit is never rejected, only left
un-normalized until an alias for it is registered.
"""

from __future__ import annotations

_UNIT_ALIASES: dict[str, str] = {
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
    already known here, rather than editing this module.
    """

    _UNIT_ALIASES[alias.strip().lower()] = canonical
