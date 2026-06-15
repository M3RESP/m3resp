"""Parsing and validation of declarative pipeline specs (YAML or JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from m3resp.core.exceptions import PipelineSpecError


@dataclass(frozen=True)
class StepSpec:
    """One step invocation in a pipeline spec."""

    uses: str
    #: parameter name -> context key, overriding the step's default ``reads``.
    inputs: dict[str, str] = field(default_factory=dict)
    #: static parameters (``@name`` values reference pipeline inputs).
    params: dict[str, Any] = field(default_factory=dict)
    #: natural output name -> context key to store it under.
    outputs: dict[str, str] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True)
class PipelineSpec:
    """A parsed, ordered pipeline spec."""

    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    steps: tuple[StepSpec, ...] = ()


def load_spec(spec: str | Path | dict[str, Any] | PipelineSpec) -> PipelineSpec:
    """Load a pipeline spec from a path, a raw mapping, or a ``PipelineSpec``.

    YAML and JSON are both accepted: ``.json`` files are parsed with ``json``;
    everything else is parsed with ``yaml.safe_load`` (a superset of JSON).
    """

    if isinstance(spec, PipelineSpec):
        return spec
    if isinstance(spec, dict):
        return _parse_spec(spec)

    path = Path(spec).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise PipelineSpecError(f"Pipeline spec at {path} must be a mapping.")
    return _parse_spec(raw)


def _parse_spec(raw: dict[str, Any]) -> PipelineSpec:
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise PipelineSpecError("Pipeline spec must define a non-empty 'steps' list.")

    inputs = raw.get("inputs", {})
    if not isinstance(inputs, dict):
        raise PipelineSpecError("Pipeline 'inputs' must be a mapping.")

    return PipelineSpec(
        name=str(raw.get("name", "pipeline")),
        inputs=dict(inputs),
        steps=tuple(_parse_step(index, item) for index, item in enumerate(steps_raw)),
    )


def _parse_step(index: int, item: Any) -> StepSpec:
    if not isinstance(item, dict):
        raise PipelineSpecError(f"Step #{index} must be a mapping.")
    uses = item.get("uses")
    if not isinstance(uses, str) or not uses:
        raise PipelineSpecError(f"Step #{index} must define a 'uses' name.")

    return StepSpec(
        uses=uses,
        inputs=_str_mapping(item.get("in", {}), f"step '{uses}' 'in'"),
        params=dict(item.get("with", {}) or {}),
        outputs=_str_mapping(item.get("out", {}), f"step '{uses}' 'out'"),
        id=item.get("id"),
    )


def _str_mapping(value: Any, label: str) -> dict[str, str]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise PipelineSpecError(f"{label} must be a mapping.")
    return {str(key): str(val) for key, val in value.items()}
