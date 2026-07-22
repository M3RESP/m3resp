"""Canonical metric-name normalization.

The same drift that affects unit strings (see `m3resp.data.units`) affects
metric names: the same quantity gets called ``"rr"``, ``"resp_rate"``,
``"respiratory_rate"`` by different steps/loaders, which makes results hard to
group or compare across runs. This module maps a small set of known aliases
to a canonical *metric type*.

Unlike `normalize_unit`, an unrecognized name returns ``None`` rather than the
input unchanged: `ParameterResult.name` stays a free-form, human-readable
label, and `metric_type` is set only when the name is confidently a known
standardized metric - so a custom/experimental metric is never mislabelled as
a standard one.

The vocabulary is deliberately small and open: register project-specific
metrics with `register_metric_alias`, and persist a custom vocabulary with
`save_metric_aliases`/`load_metric_aliases`, which round-trip the map through a
YAML or JSON file using the same dual-format convention as the pipeline-spec
loader (`m3resp.workflows.spec.load_spec`) - mirroring `m3resp.data.units`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Built-in ``alias -> canonical metric type`` pairs, kept immutable so the
#: active map can always be reset to (or diffed against) the defaults. A
#: "custom map" is this baseline plus a user's additions/overrides. Aliases
#: are matched case-insensitively after stripping surrounding whitespace.
_DEFAULT_METRIC_ALIASES: dict[str, str] = {
    # respiratory rate
    "rr": "respiratory_rate",
    "resp_rate": "respiratory_rate",
    "respiratory_rate": "respiratory_rate",
    "breathing_rate": "respiratory_rate",
    "breath_rate": "respiratory_rate",
    # tidal impedance variation (EIT)
    "tiv": "tidal_impedance_variation",
    "tidal_impedance_variation": "tidal_impedance_variation",
    # end-expiratory lung impedance (EIT)
    "eeli": "eeli",
    "end_expiratory_lung_impedance": "eeli",
    # tidal volume
    "vt": "tidal_volume",
    "tv": "tidal_volume",
    "tidal_volume": "tidal_volume",
    # minute ventilation
    "ve": "minute_ventilation",
    "minute_ventilation": "minute_ventilation",
    # EMG amplitude
    "emg_amplitude": "emg_amplitude",
}

#: The active map. Starts as a copy of the defaults and is what
#: `normalize_metric_name` consults; `register_metric_alias`/
#: `load_metric_aliases` mutate this, `reset_metric_aliases` restores it.
_METRIC_ALIASES: dict[str, str] = dict(_DEFAULT_METRIC_ALIASES)


def normalize_metric_name(name: str | None) -> str | None:
    """Return the canonical metric type for `name`, or ``None`` if unknown.

    Matches case-insensitively after stripping whitespace. Returning ``None``
    for an unrecognized name is intentional: it means "this is a custom label,
    not a known standardized metric", so callers don't tag it as one.
    """

    if name is None:
        return None
    return _METRIC_ALIASES.get(name.strip().lower())


def register_metric_alias(alias: str, canonical: str) -> None:
    """Register a new metric-name ``alias -> canonical metric type`` mapping.

    For steps/loaders that emit a metric name not already known here, rather
    than editing this module. The registration lasts for the current process;
    use `save_metric_aliases` to persist it.
    """

    _METRIC_ALIASES[alias.strip().lower()] = canonical


def metric_aliases() -> dict[str, str]:
    """A copy of the currently active ``alias -> canonical metric type`` map."""

    return dict(_METRIC_ALIASES)


def reset_metric_aliases() -> None:
    """Restore the active map to the built-in defaults.

    Drops every `register_metric_alias`/`load_metric_aliases` registration made
    since import.
    """

    _METRIC_ALIASES.clear()
    _METRIC_ALIASES.update(_DEFAULT_METRIC_ALIASES)


def save_metric_aliases(path: str | Path, *, only_custom: bool = True) -> Path:
    """Write the active metric-alias map to a YAML/JSON file for later reuse.

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
            for alias, canonical in _METRIC_ALIASES.items()
            if _DEFAULT_METRIC_ALIASES.get(alias) != canonical
        }
    else:
        payload = dict(_METRIC_ALIASES)

    if resolved.suffix.lower() == ".json":
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        import yaml

        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def load_metric_aliases(path: str | Path, *, replace: bool = False) -> dict[str, str]:
    """Load a metric-alias map from a YAML/JSON file and register it.

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
        raise ValueError(
            f"Metric alias file {resolved} must be a mapping of "
            f"alias -> canonical metric type, got {type(raw).__name__}."
        )

    if replace:
        reset_metric_aliases()
    for alias, canonical in raw.items():
        register_metric_alias(str(alias), str(canonical))
    return metric_aliases()
