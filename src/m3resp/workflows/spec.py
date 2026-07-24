"""Parsing and validation of declarative pipeline specs (YAML or JSON).

Specs come in two modes (Stage 2 pipeline-structure plan, Phase 2):

* **Legacy** -- no top-level ``schema_version`` key. Parsed permissively for
  backward compatibility (e.g. ``"false"`` still coerces to ``True`` via
  ``bool()``), but a :class:`FutureWarning` is raised for behavior that
  becomes a hard error in versioned specs.
* **Versioned** (``schema_version: 1``) -- strict. Unknown top-level/step
  keys are errors, booleans must be real YAML/JSON booleans, and an
  unsupported ``schema_version`` value is rejected with a clear message.

Both modes build the same public frozen dataclasses below, so callers and
the engine do not need to know which mode produced a given ``PipelineSpec``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from m3resp.core.exceptions import PipelineSpecError
from m3resp.core.path_helper import resolve_optional_path

#: schema_version values this release of m3resp understands.
_SUPPORTED_SCHEMA_VERSIONS = (1,)


#: Phase 6.2: replaces the old "any explicit export step present" heuristic.
#: ``None`` means "not stated" - only ever valid for a legacy (unversioned)
#: spec, where the effective mode is inferred from the spec's steps with a
#: diagnostic; a versioned spec must state it whenever ``dir`` is set.
OutputMode = Literal["automatic", "explicit", "none"]


@dataclass(frozen=True)
class SpecOutputsConfig:
    """Controls where and what the pipeline runner exports after execution.

    ``dir`` is resolved to an absolute path relative to the spec file.
    If ``timestamped`` is true, a ``YYYYMMDD_HHMMSS`` subfolder is appended.
    """

    dir: Path | None = None
    mode: OutputMode | None = None
    timestamped: bool = True
    summary_json: bool = True
    event_csvs: bool = True
    parameters_csv: bool = False
    postprocessing: bool = False
    structured_export: bool = False
    figures: bool = False
    #: Phase 6.3: compute sha256 checksums for input/output files in the run
    #: manifest. Off by default - hashing large recordings is not free.
    checksums: bool = False


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
class SpecExecutionConfig:
    """Reproducibility controls (Phase 2.6). Stage 2 supports only
    ``error_policy: fail_fast``; ``seed`` is recorded but not yet threaded
    into any step - a step that uses randomness must accept its own seed
    parameter explicitly."""

    error_policy: Literal["fail_fast"] = "fail_fast"
    seed: int | None = None


@dataclass(frozen=True)
class StepSpec:
    """One step invocation in a pipeline spec.

    ``id`` is always populated: an explicit spec ``id:`` is used as-is
    (and validated for uniqueness across the spec), otherwise a stable
    ``step_{position:03d}_{operation}`` id is generated. Reordering a spec
    changes generated ids; a caller that needs a stable external reference
    should supply an explicit ``id``.
    """

    uses: str
    id: str
    #: parameter name -> context key, overriding the step's default ``reads``.
    inputs: dict[str, str] = field(default_factory=dict)
    #: static parameters (``@name`` values reference pipeline inputs).
    params: dict[str, Any] = field(default_factory=dict)
    #: natural output name -> context key to store it under.
    outputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    """A parsed, ordered pipeline spec."""

    name: str
    schema_version: int | None = None
    description: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    #: Free, JSON-serializable study/site/session notes (Phase 2.6). Not part
    #: of file naming - see ``experiment`` for that.
    metadata: dict[str, Any] = field(default_factory=dict)
    execution: SpecExecutionConfig = field(default_factory=SpecExecutionConfig)
    steps: tuple[StepSpec, ...] = ()
    outputs: SpecOutputsConfig = field(default_factory=SpecOutputsConfig)
    experiment: SpecExperimentConfig = field(default_factory=SpecExperimentConfig)
    #: Normalized base directory every relative path in this spec resolves
    #: against (Phase 2.5) - the spec file's parent directory when loaded from
    #: a path, or the caller-supplied/current-working directory otherwise.
    root: Path = field(default_factory=Path.cwd)

    @property
    def is_legacy(self) -> bool:
        """Whether this spec was parsed without a ``schema_version``."""

        return self.schema_version is None


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


# --------------------------------------------------------------------------- #
# Versioned (strict) document model                                          #
# --------------------------------------------------------------------------- #


class _StepDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str | None = None
    uses: str
    in_: dict[str, str] = Field(default_factory=dict, alias="in")
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    out: dict[str, str] = Field(default_factory=dict)


class _OutputsDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str | None = None
    mode: Literal["automatic", "explicit", "none"] | None = None
    timestamped: StrictBool = True
    summary_json: StrictBool = True
    event_csvs: StrictBool = True
    parameters_csv: StrictBool = False
    postprocessing: StrictBool = False
    structured_export: StrictBool = False
    figures: StrictBool = False
    checksums: StrictBool = False

    @model_validator(mode="after")
    def _check_mode_and_dir_agree(self) -> _OutputsDocument:
        # Phase 6.2: "reject contradictory combinations rather than guessing."
        if self.mode in ("automatic", "explicit") and self.dir is None:
            raise ValueError(
                f"outputs.mode={self.mode!r} requires outputs.dir to be set "
                "(nothing to write into otherwise)."
            )
        if self.dir is not None and self.mode is None:
            raise ValueError(
                "outputs.mode must be set ('automatic', 'explicit', or "
                "'none') whenever outputs.dir is set in a versioned spec."
            )
        return self


class _ExperimentDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str | None = None
    mode: str | None = None
    timepoint: str | None = None
    run_identifier: str | None = None
    selection: str = "selected"


class _ExecutionDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_policy: Literal["fail_fast"] = "fail_fast"
    seed: StrictInt | None = None


class _SpecDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str = "pipeline"
    description: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution: _ExecutionDocument = Field(default_factory=_ExecutionDocument)
    outputs: _OutputsDocument = Field(default_factory=_OutputsDocument)
    experiment: _ExperimentDocument = Field(default_factory=_ExperimentDocument)
    steps: list[_StepDocument] = Field(min_length=1)


def _parse_spec(raw: dict[str, Any], *, root: Path) -> PipelineSpec:
    schema_version = raw.get("schema_version")
    if schema_version is not None:
        return _parse_versioned_spec(raw, root=root)
    return _parse_legacy_spec(raw, root=root)


def _parse_versioned_spec(raw: dict[str, Any], *, root: Path) -> PipelineSpec:
    schema_version = raw.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise PipelineSpecError(
            f"Unsupported schema_version {schema_version!r}; this version of "
            f"m3resp supports schema_version {_SUPPORTED_SCHEMA_VERSIONS!r}."
        )

    try:
        document = _SpecDocument.model_validate(raw)
    except ValidationError as exc:
        raise PipelineSpecError(
            f"Pipeline spec failed strict validation:\n{exc}"
        ) from exc

    step_ids = _finalize_step_ids(
        [step.id for step in document.steps], [step.uses for step in document.steps]
    )
    steps = tuple(
        StepSpec(
            uses=step.uses,
            id=step_id,
            inputs=dict(step.in_),
            params=dict(step.with_),
            outputs=dict(step.out),
        )
        for step, step_id in zip(document.steps, step_ids)
    )

    resolved_dir = resolve_optional_path(root, document.outputs.dir)
    return PipelineSpec(
        name=document.name,
        schema_version=schema_version,
        description=document.description,
        inputs=dict(document.inputs),
        metadata=dict(document.metadata),
        execution=SpecExecutionConfig(
            error_policy=document.execution.error_policy,
            seed=document.execution.seed,
        ),
        steps=steps,
        outputs=SpecOutputsConfig(
            dir=resolved_dir,
            mode=document.outputs.mode,
            timestamped=document.outputs.timestamped,
            summary_json=document.outputs.summary_json,
            event_csvs=document.outputs.event_csvs,
            parameters_csv=document.outputs.parameters_csv,
            postprocessing=document.outputs.postprocessing,
            structured_export=document.outputs.structured_export,
            figures=document.outputs.figures,
            checksums=document.outputs.checksums,
        ),
        experiment=SpecExperimentConfig(
            subject_id=document.experiment.subject_id,
            mode=document.experiment.mode,
            timepoint=document.experiment.timepoint,
            run_identifier=document.experiment.run_identifier,
            selection=document.experiment.selection,
        ),
        root=root,
    )


# --------------------------------------------------------------------------- #
# Legacy (permissive) parsing                                                 #
# --------------------------------------------------------------------------- #


def _parse_legacy_spec(raw: dict[str, Any], *, root: Path) -> PipelineSpec:
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise PipelineSpecError("Pipeline spec must define a non-empty 'steps' list.")

    inputs = raw.get("inputs", {})
    if not isinstance(inputs, dict):
        raise PipelineSpecError("Pipeline 'inputs' must be a mapping.")

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PipelineSpecError("Pipeline 'metadata' must be a mapping.")

    raw_steps = [
        _parse_legacy_step(index, item) for index, item in enumerate(steps_raw)
    ]
    step_ids = _finalize_step_ids(
        [raw_id for raw_id, _uses, _inputs, _params, _outputs in raw_steps],
        [uses for _raw_id, uses, _inputs, _params, _outputs in raw_steps],
    )
    steps = tuple(
        StepSpec(
            uses=uses, id=step_id, inputs=step_inputs, params=params, outputs=outputs
        )
        for (_raw_id, uses, step_inputs, params, outputs), step_id in zip(
            raw_steps, step_ids
        )
    )

    return PipelineSpec(
        name=str(raw.get("name", "pipeline")),
        schema_version=None,
        description=str(raw.get("description", "")),
        inputs=dict(inputs),
        metadata=dict(metadata),
        execution=_parse_legacy_execution(raw.get("execution", {})),
        steps=steps,
        outputs=_parse_legacy_outputs(raw.get("outputs", {}), root=root),
        experiment=_parse_legacy_experiment(raw.get("experiment", {})),
        root=root,
    )


def _legacy_bool(value: Any, default: bool, *, field_name: str) -> bool:
    """Coerce ``value`` to ``bool`` like Stage 1 did, warning when it was not
    already a real boolean (Phase 2.2: this becomes a hard error in a
    versioned spec)."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    warnings.warn(
        f"Pipeline outputs field {field_name!r} got non-boolean value "
        f"{value!r}; coercing via bool() is deprecated for unversioned specs "
        "and is a hard error once 'schema_version' is set.",
        FutureWarning,
        stacklevel=3,
    )
    return bool(value)


_OUTPUT_MODES: tuple[OutputMode, ...] = ("automatic", "explicit", "none")


def _parse_legacy_output_mode(raw: Any) -> OutputMode | None:
    """A legacy spec may opt in to an explicit ``outputs.mode`` (Phase 6.2);
    when omitted, the engine infers it from the spec's steps with a
    diagnostic. Unlike a versioned spec, this is never required."""

    if raw is None:
        return None
    if raw not in _OUTPUT_MODES:
        raise PipelineSpecError(
            f"Pipeline 'outputs.mode' must be one of {_OUTPUT_MODES}, got {raw!r}."
        )
    return cast(OutputMode, raw)


def _parse_legacy_outputs(raw: Any, *, root: Path) -> SpecOutputsConfig:
    if not raw:
        return SpecOutputsConfig()
    if not isinstance(raw, dict):
        raise PipelineSpecError("Pipeline 'outputs' must be a mapping.")
    resolved_dir = resolve_optional_path(root, raw.get("dir"))
    mode = _parse_legacy_output_mode(raw.get("mode"))
    if mode in ("automatic", "explicit") and resolved_dir is None:
        raise PipelineSpecError(
            f"Pipeline 'outputs.mode' {mode!r} requires 'outputs.dir' to be "
            "set (nothing to write into otherwise)."
        )
    return SpecOutputsConfig(
        dir=resolved_dir,
        mode=mode,
        timestamped=_legacy_bool(
            raw.get("timestamped"), True, field_name="timestamped"
        ),
        summary_json=_legacy_bool(
            raw.get("summary_json"), True, field_name="summary_json"
        ),
        event_csvs=_legacy_bool(raw.get("event_csvs"), True, field_name="event_csvs"),
        parameters_csv=_legacy_bool(
            raw.get("parameters_csv"), False, field_name="parameters_csv"
        ),
        postprocessing=_legacy_bool(
            raw.get("postprocessing"), False, field_name="postprocessing"
        ),
        structured_export=_legacy_bool(
            raw.get("structured_export"), False, field_name="structured_export"
        ),
        figures=_legacy_bool(raw.get("figures"), False, field_name="figures"),
        checksums=_legacy_bool(raw.get("checksums"), False, field_name="checksums"),
    )


def _parse_legacy_experiment(raw: Any) -> SpecExperimentConfig:
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


def _parse_legacy_execution(raw: Any) -> SpecExecutionConfig:
    if not raw:
        return SpecExecutionConfig()
    if not isinstance(raw, dict):
        raise PipelineSpecError("Pipeline 'execution' must be a mapping.")
    error_policy = str(raw.get("error_policy", "fail_fast"))
    if error_policy != "fail_fast":
        raise PipelineSpecError(
            f"Unsupported execution.error_policy {error_policy!r}; Stage 2 "
            "supports only 'fail_fast'."
        )
    seed = raw.get("seed")
    return SpecExecutionConfig(
        error_policy="fail_fast", seed=int(seed) if seed is not None else None
    )


def _parse_legacy_step(
    index: int, item: Any
) -> tuple[str | None, str, dict[str, str], dict[str, Any], dict[str, str]]:
    if not isinstance(item, dict):
        raise PipelineSpecError(f"Step #{index} must be a mapping.")
    uses = item.get("uses")
    if not isinstance(uses, str) or not uses:
        raise PipelineSpecError(f"Step #{index} must define a 'uses' name.")

    return (
        item.get("id"),
        uses,
        _str_mapping(item.get("in", {}), f"step '{uses}' 'in'"),
        dict(item.get("with", {}) or {}),
        _str_mapping(item.get("out", {}), f"step '{uses}' 'out'"),
    )


def _str_mapping(value: Any, label: str) -> dict[str, str]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise PipelineSpecError(f"{label} must be a mapping.")
    return {str(key): str(val) for key, val in value.items()}


# --------------------------------------------------------------------------- #
# Shared: step id generation and uniqueness                                   #
# --------------------------------------------------------------------------- #


def _stable_step_id(position: int, uses: str) -> str:
    return f"step_{position:03d}_{uses.replace('.', '_')}"


def _finalize_step_ids(raw_ids: list[str | None], uses_list: list[str]) -> list[str]:
    resolved = [
        raw_id if raw_id is not None else _stable_step_id(position, uses)
        for position, (raw_id, uses) in enumerate(zip(raw_ids, uses_list))
    ]
    seen: dict[str, int] = {}
    for position, step_id in enumerate(resolved):
        if step_id in seen:
            raise PipelineSpecError(
                f"Duplicate step id {step_id!r} (steps #{seen[step_id]} and "
                f"#{position}). Supply distinct 'id' values or omit them to "
                "use generated ids."
            )
        seen[step_id] = position
    return resolved
