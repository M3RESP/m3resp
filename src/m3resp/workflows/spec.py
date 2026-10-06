"""Load, validate and save YAML or JSON workflow descriptions.

Version 1 specs require recognized keys and correctly typed values.
Unversioned specs use permissive parsing for older files, emitting
FutureWarning when a non-boolean export option is converted with bool().
Both forms produce WorkflowSpec objects.
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

from m3resp.core.exceptions import WorkflowSpecError
from m3resp.core.path_helper import resolve_optional_path

#: Supported integer workflow format versions.
_SUPPORTED_SCHEMA_VERSIONS = (1,)


#: Export modes understood by run_spec. An unversioned spec can omit the
#: mode; a versioned spec states it whenever outputs.dir is supplied.
OutputMode = Literal["automatic", "explicit", "none"]


@dataclass(frozen=True)
class SpecOutputsConfig:
    """Export settings used by run_spec and registered export steps.

    Attributes:
        dir: Output directory, resolved relative to the spec root when parsed.
        mode: automatic for session export after a successful run, explicit
            for declared export steps, or none for skipping runner exports and
            manifests. None lets run_spec infer a mode for an unversioned spec.
        timestamped: Append a local-time YYYYMMDD_HHMMSS subdirectory.
        summary_json: Include summary.json in session export.
        event_csvs: Write CSVs for nonempty event lists.
        parameters_csv: Write legacy grouped parameters when present.
        postprocessing: Include EMG postprocessing in exported legacy parameters.
        structured_export: Write typed result tables and array archives.
        figures: Save available EIT figures after a successful run.
        checksums: Include readable files found in executed-step parameter
            values in the run manifest's SHA-256 checksums.
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
    #: Include checksums of readable files named in executed-step parameters.
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
    """Execution settings recorded for reproducibility.

    error_policy is fail_fast. seed is recorded in execution metadata; steps
    that use randomness receive their own seed through step parameters.
    """

    error_policy: Literal["fail_fast"] = "fail_fast"
    seed: int | None = None


@dataclass(frozen=True)
class StepSpec:
    """One operation and its input, parameter and output bindings.

    Attributes:
        uses: Registered operation name, such as ``"eit.load"``.
        id: Unique step ID. Parsing retains explicit IDs or generates
            ``step_{position:03d}_{operation}``, with dots replaced by underscores.
            Reordering changes generated IDs.
        inputs: Function-argument names mapped to context keys (``in:``).
        params: Static settings (``with:``), which may contain ``@name``
            references to workflow inputs.
        outputs: Returned output names mapped to context keys (``out:``).
    """

    uses: str
    id: str
    #: parameter name -> context key, overriding the step's default ``reads``.
    inputs: dict[str, str] = field(default_factory=dict)
    #: static parameters (``@name`` values reference workflow inputs).
    params: dict[str, Any] = field(default_factory=dict)
    #: natural output name -> context key to store it under.
    outputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSpec:
    """An ordered workflow description with inputs, settings and export options.

    Attributes:
        name: Workflow name.
        schema_version: Format version, or None for an unversioned spec.
        description: Optional description of the analysis.
        inputs: Values referenced from step settings with ``@name``.
        metadata: Additional study, site or session notes.
        execution: Error policy and recorded seed.
        steps: Operations in execution order.
        outputs: Export options used by run_spec and export steps.
        experiment: Study labels used for result-file naming.
        root: Base directory for relative path settings. Loading a file uses
            its parent directory unless a root is supplied.
    """

    name: str
    schema_version: int | None = None
    description: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    #: Additional study, site or session notes.
    metadata: dict[str, Any] = field(default_factory=dict)
    execution: SpecExecutionConfig = field(default_factory=SpecExecutionConfig)
    steps: tuple[StepSpec, ...] = ()
    outputs: SpecOutputsConfig = field(default_factory=SpecOutputsConfig)
    experiment: SpecExperimentConfig = field(default_factory=SpecExperimentConfig)
    #: Base directory used to resolve relative path settings.
    root: Path = field(default_factory=Path.cwd)

    @property
    def is_legacy(self) -> bool:
        """Whether this spec was parsed without a ``schema_version``."""

        return self.schema_version is None


def load_spec(
    spec: str | Path | dict[str, Any] | WorkflowSpec,
    *,
    root: str | Path | None = None,
) -> WorkflowSpec:
    """Load a workflow from an object, dictionary or YAML/JSON file.

    Args:
        spec: A WorkflowSpec returned as supplied, a dictionary to parse, or a
            file path. Files ending in .json use JSON parsing; other files use
            yaml.safe_load. Versioned specs receive strict format validation.
        root: Base directory for relative paths within a parsed spec. Defaults
            to the file's parent directory for a file, or the current working
            directory for a dictionary. An existing WorkflowSpec retains its root.

    Returns:
        WorkflowSpec: Ordered steps, raw parameter references and workflow
            settings. The root is normalized and outputs.dir is resolved against
            it. Operation names and bindings are checked during compilation.

    Raises:
        WorkflowSpecError: If a document is not a mapping, its schema version is
            unsupported, format validation fails, or step IDs are duplicated.
        OSError: If a file cannot be read.
        json.JSONDecodeError: If a JSON file is malformed.
        yaml.YAMLError: If a YAML file is malformed.
    """

    if isinstance(spec, WorkflowSpec):
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
        raise WorkflowSpecError(f"Workflow spec at {path} must be a mapping.")
    return _parse_spec(raw, root=resolved_root or path.parent)


def spec_to_dict(spec: WorkflowSpec) -> dict[str, Any]:
    """Convert a workflow to a dictionary suitable for saving or reloading.

    Args:
        spec: Workflow with serializable input, metadata and parameter values.

    Returns:
        dict[str, Any]: Step bindings with explicit IDs and raw ``@name`` strings.
            Optional blocks at their defaults are omitted. outputs.dir is
            relative when it lies under spec.root and absolute otherwise.
            Reload with the same root to preserve relative-path meanings.
    """

    result: dict[str, Any] = {"name": spec.name}
    if spec.schema_version is not None:
        result["schema_version"] = spec.schema_version
    if spec.description:
        result["description"] = spec.description
    if spec.inputs:
        result["inputs"] = dict(spec.inputs)
    if spec.metadata:
        result["metadata"] = dict(spec.metadata)
    if spec.execution != SpecExecutionConfig():
        result["execution"] = {
            "error_policy": spec.execution.error_policy,
            "seed": spec.execution.seed,
        }
    if spec.outputs != SpecOutputsConfig():
        result["outputs"] = _outputs_to_dict(spec.outputs, spec.root)
    if spec.experiment != SpecExperimentConfig():
        result["experiment"] = {
            "subject_id": spec.experiment.subject_id,
            "mode": spec.experiment.mode,
            "timepoint": spec.experiment.timepoint,
            "run_identifier": spec.experiment.run_identifier,
            "selection": spec.experiment.selection,
        }
    result["steps"] = [_step_to_dict(step_spec) for step_spec in spec.steps]
    return result


def _outputs_to_dict(outputs: SpecOutputsConfig, root: Path) -> dict[str, Any]:
    resolved_dir: str | None = None
    if outputs.dir is not None:
        try:
            resolved_dir = outputs.dir.relative_to(root).as_posix()
        except ValueError:
            resolved_dir = str(outputs.dir)
    return {
        "dir": resolved_dir,
        "mode": outputs.mode,
        "timestamped": outputs.timestamped,
        "summary_json": outputs.summary_json,
        "event_csvs": outputs.event_csvs,
        "parameters_csv": outputs.parameters_csv,
        "postprocessing": outputs.postprocessing,
        "structured_export": outputs.structured_export,
        "figures": outputs.figures,
        "checksums": outputs.checksums,
    }


def _step_to_dict(step_spec: StepSpec) -> dict[str, Any]:
    result: dict[str, Any] = {"id": step_spec.id, "uses": step_spec.uses}
    if step_spec.inputs:
        result["in"] = dict(step_spec.inputs)
    if step_spec.params:
        result["with"] = dict(step_spec.params)
    if step_spec.outputs:
        result["out"] = dict(step_spec.outputs)
    return result


def dump_spec(
    spec: WorkflowSpec,
    path: str | Path,
    *,
    format: Literal["yaml", "json"] | None = None,
) -> Path:
    """Write a workflow description to a YAML or JSON file.

    Args:
        spec: Workflow with serializable input, metadata and parameter values.
        path: Destination path, expanded and resolved. Its parent must exist.
            Existing file contents are replaced, including comments and formatting.
        format: json or yaml. When None, .json selects JSON and other suffixes
            select YAML.

    Returns:
        Path: Resolved path to the written file.

    Raises:
        OSError: If the destination cannot be written.
    """

    target = Path(path).expanduser().resolve()
    resolved_format = format or ("json" if target.suffix.lower() == ".json" else "yaml")
    payload = spec_to_dict(spec)
    if resolved_format == "json":
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        target.write_text(
            yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    return target


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
    """Strict version-1 workflow document used for format validation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str = "workflow"
    description: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution: _ExecutionDocument = Field(default_factory=_ExecutionDocument)
    outputs: _OutputsDocument = Field(default_factory=_OutputsDocument)
    experiment: _ExperimentDocument = Field(default_factory=_ExperimentDocument)
    steps: list[_StepDocument] = Field(min_length=1)


def _parse_spec(raw: dict[str, Any], *, root: Path) -> WorkflowSpec:
    """Select versioned or unversioned parsing using schema_version."""

    schema_version = raw.get("schema_version")
    if schema_version is not None:
        return _parse_versioned_spec(raw, root=root)
    return _parse_legacy_spec(raw, root=root)


def _parse_versioned_spec(raw: dict[str, Any], *, root: Path) -> WorkflowSpec:
    """Validate a versioned document and resolve IDs and output paths.

    Raises WorkflowSpecError for an unsupported version, invalid format or
    duplicate step IDs.
    """

    schema_version = raw.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise WorkflowSpecError(
            f"Unsupported schema_version {schema_version!r}; this version of "
            f"m3resp supports schema_version {_SUPPORTED_SCHEMA_VERSIONS!r}."
        )

    try:
        document = _SpecDocument.model_validate(raw)
    except ValidationError as exc:
        raise WorkflowSpecError(
            f"Workflow spec failed strict validation:\n{exc}"
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
    return WorkflowSpec(
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


def _parse_legacy_spec(raw: dict[str, Any], *, root: Path) -> WorkflowSpec:
    """Parse an unversioned document with permissive value conversion."""

    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise WorkflowSpecError("Workflow spec must define a non-empty 'steps' list.")

    inputs = raw.get("inputs", {})
    if not isinstance(inputs, dict):
        raise WorkflowSpecError("Workflow 'inputs' must be a mapping.")

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise WorkflowSpecError("Workflow 'metadata' must be a mapping.")

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

    return WorkflowSpec(
        name=str(raw.get("name", "workflow")),
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
    """Return a default for None, a bool as supplied, or bool(value).

    Converting another value emits a FutureWarning; e.g. ``"false"`` is True.
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    warnings.warn(
        f"Workflow outputs field {field_name!r} got non-boolean value "
        f"{value!r}; coercing via bool() is deprecated for unversioned specs "
        "and is a hard error once 'schema_version' is set.",
        FutureWarning,
        stacklevel=3,
    )
    return bool(value)


_OUTPUT_MODES: tuple[OutputMode, ...] = ("automatic", "explicit", "none")


def _parse_legacy_output_mode(raw: Any) -> OutputMode | None:
    """Read automatic, explicit or none, preserving None for an omitted mode.

    Raises WorkflowSpecError for an unsupported value.
    """

    if raw is None:
        return None
    if raw not in _OUTPUT_MODES:
        raise WorkflowSpecError(
            f"Workflow 'outputs.mode' must be one of {_OUTPUT_MODES}, got {raw!r}."
        )
    return cast(OutputMode, raw)


def _parse_legacy_outputs(raw: Any, *, root: Path) -> SpecOutputsConfig:
    """Parse unversioned export settings and resolve the output directory."""

    if not raw:
        return SpecOutputsConfig()
    if not isinstance(raw, dict):
        raise WorkflowSpecError("Workflow 'outputs' must be a mapping.")
    resolved_dir = resolve_optional_path(root, raw.get("dir"))
    mode = _parse_legacy_output_mode(raw.get("mode"))
    if mode in ("automatic", "explicit") and resolved_dir is None:
        raise WorkflowSpecError(
            f"Workflow 'outputs.mode' {mode!r} requires 'outputs.dir' to be "
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
    """Read study labels and result-file naming fields from an unversioned spec."""

    if not raw:
        return SpecExperimentConfig()
    if not isinstance(raw, dict):
        raise WorkflowSpecError("Workflow 'experiment' must be a mapping.")
    return SpecExperimentConfig(
        subject_id=raw.get("subject_id"),
        mode=raw.get("mode"),
        timepoint=raw.get("timepoint"),
        run_identifier=raw.get("run_identifier"),
        selection=str(raw.get("selection", "selected")),
    )


def _parse_legacy_execution(raw: Any) -> SpecExecutionConfig:
    """Read fail_fast and an optional integer seed from an unversioned spec."""

    if not raw:
        return SpecExecutionConfig()
    if not isinstance(raw, dict):
        raise WorkflowSpecError("Workflow 'execution' must be a mapping.")
    error_policy = str(raw.get("error_policy", "fail_fast"))
    if error_policy != "fail_fast":
        raise WorkflowSpecError(
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
    """Read an unversioned step's ID, operation, inputs, parameters and outputs."""

    if not isinstance(item, dict):
        raise WorkflowSpecError(f"Step #{index} must be a mapping.")
    uses = item.get("uses")
    if not isinstance(uses, str) or not uses:
        raise WorkflowSpecError(f"Step #{index} must define a 'uses' name.")

    return (
        item.get("id"),
        uses,
        _str_mapping(item.get("in", {}), f"step '{uses}' 'in'"),
        dict(item.get("with", {}) or {}),
        _str_mapping(item.get("out", {}), f"step '{uses}' 'out'"),
    )


def _str_mapping(value: Any, label: str) -> dict[str, str]:
    """Convert mapping keys and values to strings, using an empty dict for empty input.

    Raises WorkflowSpecError for a nonempty value that is not a dictionary.
    """

    if not value:
        return {}
    if not isinstance(value, dict):
        raise WorkflowSpecError(f"{label} must be a mapping.")
    return {str(key): str(val) for key, val in value.items()}


# --------------------------------------------------------------------------- #
# Shared: step id generation and uniqueness                                   #
# --------------------------------------------------------------------------- #


def _stable_step_id(position: int, uses: str) -> str:
    return f"step_{position:03d}_{uses.replace('.', '_')}"


def _finalize_step_ids(raw_ids: list[str | None], uses_list: list[str]) -> list[str]:
    """Fill omitted IDs from step position and operation, rejecting duplicate IDs."""

    resolved = [
        raw_id if raw_id is not None else _stable_step_id(position, uses)
        for position, (raw_id, uses) in enumerate(zip(raw_ids, uses_list))
    ]
    seen: dict[str, int] = {}
    for position, step_id in enumerate(resolved):
        if step_id in seen:
            raise WorkflowSpecError(
                f"Duplicate step id {step_id!r} (steps #{seen[step_id]} and "
                f"#{position}). Supply distinct 'id' values or omit them to "
                "use generated ids."
            )
        seen[step_id] = position
    return resolved
