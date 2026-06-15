"""Parsing and validation of declarative pipeline specs (YAML or JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from m3resp.core.exceptions import PipelineSpecError
from m3resp.core.path_helper import resolve_optional_path


@dataclass(frozen=True)
class SpecOutputsConfig:
    """Controls where and what the pipeline runner exports after execution.

    ``dir`` is resolved to an absolute path relative to the spec file.
    If ``timestamped`` is true, a ``YYYYMMDD_HHMMSS`` subfolder is appended.
    """

    dir: Path | None = None
    timestamped: bool = True
    summary_json: bool = True
    event_csvs: bool = True
    parameters_csv: bool = False
    postprocessing: bool = False
    figures: bool = False


@dataclass(frozen=True)
class SpecExperimentConfig:
    """Study-level metadata used by ROTARC-style export steps.

    These fields drive output file naming (e.g.
    ``subject_results/<run_id>/subject-mode-tp-selection.txt``).
    """

    subject_id: str | None = None
    mode: str | None = None
    timepoint: str | None = None
    run_identifier: str | None = None
    selection: str = "selected"


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
    outputs: SpecOutputsConfig = field(default_factory=SpecOutputsConfig)
    experiment: SpecExperimentConfig = field(default_factory=SpecExperimentConfig)


def load_spec(
    spec: str | Path | dict[str, Any] | PipelineSpec,
    *,
    root: str | Path | None = None,
) -> PipelineSpec:
    """Load a pipeline spec from a path, a raw mapping, or a ``PipelineSpec``.

    YAML and JSON are both accepted: ``.json`` files are parsed with ``json``;
    everything else is parsed with ``yaml.safe_load`` (a superset of JSON).

    ``root`` sets the base directory for resolving relative file paths in the
    spec (e.g. ``outputs.dir``). Defaults to the spec file's parent directory
    when loading from a path, or the current working directory otherwise.
    """

    if isinstance(spec, PipelineSpec):
        return spec

    resolved_root: Path | None = Path(root).expanduser().resolve() if root else None

    if isinstance(spec, dict):
        return _parse_spec(spec, root=resolved_root or Path.cwd())

    path = Path(spec).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise PipelineSpecError(f"Pipeline spec at {path} must be a mapping.")
    return _parse_spec(raw, root=resolved_root or path.parent)


def _parse_spec(raw: dict[str, Any], *, root: Path) -> PipelineSpec:
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
        outputs=_parse_outputs(raw.get("outputs", {}), root=root),
        experiment=_parse_experiment(raw.get("experiment", {})),
    )


def _parse_outputs(raw: Any, *, root: Path) -> SpecOutputsConfig:
    if not raw:
        return SpecOutputsConfig()
    if not isinstance(raw, dict):
        raise PipelineSpecError("Pipeline 'outputs' must be a mapping.")
    resolved_dir = resolve_optional_path(root, raw.get("dir"))
    return SpecOutputsConfig(
        dir=resolved_dir,
        timestamped=bool(raw.get("timestamped", True)),
        summary_json=bool(raw.get("summary_json", True)),
        event_csvs=bool(raw.get("event_csvs", True)),
        parameters_csv=bool(raw.get("parameters_csv", False)),
        postprocessing=bool(raw.get("postprocessing", False)),
        figures=bool(raw.get("figures", False)),
    )


def _parse_experiment(raw: Any) -> SpecExperimentConfig:
    if not raw:
        return SpecExperimentConfig()
    if not isinstance(raw, dict):
        raise PipelineSpecError("Pipeline 'experiment' must be a mapping.")
    return SpecExperimentConfig(
        subject_id=raw.get("subject_id"),
        mode=raw.get("mode"),
        timepoint=raw.get("timepoint"),
        run_identifier=raw.get("run_identifier"),
        selection=str(raw.get("selection", "selected")),
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
