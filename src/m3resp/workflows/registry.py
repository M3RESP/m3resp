"""Step registry for declarative M3Resp pipelines.

A *step* is a named, reusable operation. Each step declares the context keys it
reads (mapped onto its parameter names) and the context keys it writes. A
pipeline spec lists steps by name and binds them together, so workflows can be
assembled from a YAML/JSON file without writing custom Python.

Steps additionally carry optional, JSON-safe GUI/discovery metadata
(:class:`StepParameter`, :class:`StepArtifact`) describing static parameters
and input/output artifacts in a backend-neutral form. This metadata is being
backfilled module by module (see
``plan/stage2/3_pipeline_structure_implementation_plan.md`` Phase 1); a step
with no declared parameters/artifacts is still a fully valid, executable step.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Literal

from m3resp.core.exceptions import StepMetadataError, UnknownStepError

# A step callable receives bound keyword arguments and returns a mapping of
# {output_name: value}. ``None`` is treated as "no outputs".
StepCallable = Callable[..., Mapping[str, Any] | None]

#: Controlled, JSON-safe static parameter vocabulary (Phase 1.1).
ParameterType = Literal[
    "boolean", "integer", "number", "string", "path", "choice", "mapping", "list"
]

_PARAMETER_TYPES: frozenset[str] = frozenset(
    ("boolean", "integer", "number", "string", "path", "choice", "mapping", "list")
)

#: A step name is ``<prefix>.<operation>``, both lowercase snake_case.
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

#: Context/output names reserved by the engine; a step must not write to them.
_RESERVED_OUTPUT_NAMES = frozenset(("session",))

#: Sentinel ``StepArtifact.artifact_type`` for a genuine passthrough step
#: (e.g. ``eit.slice``, which accepts/returns whatever signal type it was
#: given) - exempt from the compiler's artifact-type compatibility check
#: (Phase 10 of the pipeline-structure plan) on whichever side declares it.
ANY_ARTIFACT_TYPE = "any"


@dataclass(frozen=True)
class StepParameter:
    """Describes one static (``with:``) parameter for GUI/discovery use."""

    name: str
    value_type: ParameterType
    required: bool = False
    default: Any = None
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""
    #: When ``value_type == "path"``: ``"file"``, ``"directory"``, or ``None``.
    path_kind: str | None = None
    #: Hint that a GUI may hide this behind an "advanced" toggle by default.
    advanced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "required": self.required,
            "default": self.default,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": list(self.choices) if self.choices is not None else None,
            "description": self.description,
            "path_kind": self.path_kind,
            "advanced": self.advanced,
        }


@dataclass(frozen=True)
class StepArtifact:
    """Describes one input or output artifact for GUI/discovery use."""

    name: str
    artifact_type: str
    required: bool = True
    default_context_key: str | None = None
    description: str = ""
    unit: str | None = None
    axes: tuple[str, ...] | None = None
    shape_hint: str | None = None
    #: Whether this artifact is part of the native/public result contract.
    public: bool = True
    #: True for a temporary upstream/adapter object excluded from GUI results
    #: by default (see the plan's "Native artifacts are the public contract").
    compatibility_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
            "required": self.required,
            "default_context_key": self.default_context_key,
            "description": self.description,
            "unit": self.unit,
            "axes": list(self.axes) if self.axes is not None else None,
            "shape_hint": self.shape_hint,
            "public": self.public,
            "compatibility_only": self.compatibility_only,
        }


@dataclass(frozen=True)
class StepDefinition:
    """Metadata describing one registered step."""

    name: str
    func: StepCallable
    #: parameter name -> default context key, or None if the binding is required
    #: and must be supplied via ``in:`` in the spec.
    reads: Mapping[str, str | None] = field(default_factory=dict)
    #: natural output names the step returns (default context keys it writes).
    writes: tuple[str, ...] = ()
    #: context keys that must already exist before the step runs, but are not
    #: passed as arguments (e.g. an in-place mutated sequence).
    requires: tuple[str, ...] = ()
    summary: str = ""
    #: Longer, GUI-facing description. Falls back to ``summary`` when unset.
    description: str = ""
    #: Static (``with:``) parameter metadata. Optional during migration; see
    #: the module docstring.
    parameters: tuple[StepParameter, ...] = ()
    #: Whether a user has confirmed ``parameters`` is complete, i.e. that an
    #: empty tuple means "audited, genuinely has no tunable static parameter"
    #: rather than "not looked at yet". Set automatically when ``parameters``
    #: is non-empty; otherwise must be set explicitly via
    #: ``register_step(..., parameters_reviewed=True)`` once audited.
    parameters_reviewed: bool = False
    #: Groups of parameter names that must not be set together in the same
    #: step invocation (e.g. ``emg.ecg_gating``'s ``gate_width_seconds``/
    #: ``gate_width_samples``) - a structural, compile-time constraint on top
    #: of each parameter's own type/range/choice checks (Phase 10 of the
    #: pipeline-structure plan). Every name in every group must be a declared
    #: parameter; checked at registration time.
    mutually_exclusive_parameters: tuple[tuple[str, ...], ...] = ()
    #: Artifact metadata for ``reads``/``requires`` context keys.
    input_artifacts: tuple[StepArtifact, ...] = ()
    #: Artifact metadata for ``writes`` context keys.
    output_artifacts: tuple[StepArtifact, ...] = ()
    #: Dotted resource names this step reads from the shared ``M3Session``
    #: object itself (as opposed to an explicit ``reads``/``in:`` context
    #: key) - e.g. ``"session.processed.emg"`` for a step that calls
    #: ``session.detect_emg_breaths()``, which reads whatever the most
    #: recent preprocessing/ECG-removal step left on the session, not a
    #: declared context key. Optional, additive metadata; see
    #: ``m3resp.workflows.session_deps``.
    session_reads: tuple[str, ...] = ()
    #: Dotted resource names this step writes onto the shared ``M3Session``
    #: object, e.g. ``"session.raw.eit"`` for a loader, or
    #: ``"session.quality"`` for a step that calls ``session.quality.add``.
    #: A resource name is an ancestor of any more specific name sharing its
    #: dotted prefix (``"session.raw"`` covers ``"session.raw.eit"``), so a
    #: step may declare whichever granularity it actually audited.
    session_writes: tuple[str, ...] = ()
    #: Owning modality, e.g. ``"eit"``, ``"emg"``, or ``None`` for a
    #: cross-modality/generic step.
    modality: str | None = None
    #: Free-form grouping for GUI menus, e.g. ``"preprocessing"``, ``"quality"``.
    category: str | None = None
    #: Other step names that produce a compatible result (explicit
    #: alternatives selected in YAML, never auto-selected by the engine).
    alternatives: tuple[str, ...] = ()
    #: Optional third-party packages this step needs at execution time.
    #: Declared for capability discovery; never imported for that purpose.
    optional_packages: tuple[str, ...] = ()
    #: Descriptive-only resource hint, e.g. ``"long_running"``, ``"memory_heavy"``.
    resource_profile: str | None = None
    version: str | None = None
    deprecated_since: str | None = None

    @property
    def operation_id(self) -> str:
        """Stable backend-neutral identifier. Currently the registered name."""

        return self.name


STEP_REGISTRY: dict[str, StepDefinition] = {}

#: Retired step name -> the canonical name it now resolves to. Populated by
#: ``register_step(..., aliases=...)``.
#:
#: Aliases resolve silently and are deliberately kept out of `available_steps`,
#: `describe_steps` and the "available steps" text of `UnknownStepError`: an
#: existing spec keeps running unchanged, while discovery and any GUI built on
#: it only ever offer the current name.
STEP_ALIASES: dict[str, str] = {}


def register_step(
    name: str,
    *,
    reads: Mapping[str, str | None] | None = None,
    writes: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    summary: str = "",
    description: str = "",
    parameters: tuple[StepParameter, ...] = (),
    parameters_reviewed: bool = False,
    mutually_exclusive_parameters: tuple[tuple[str, ...], ...] = (),
    input_artifacts: tuple[StepArtifact, ...] = (),
    output_artifacts: tuple[StepArtifact, ...] = (),
    session_reads: tuple[str, ...] = (),
    session_writes: tuple[str, ...] = (),
    modality: str | None = None,
    category: str | None = None,
    alternatives: tuple[str, ...] = (),
    optional_packages: tuple[str, ...] = (),
    resource_profile: str | None = None,
    version: str | None = None,
    deprecated_since: str | None = None,
    aliases: tuple[str, ...] = (),
) -> Callable[[StepCallable], StepCallable]:
    """Register ``func`` under ``name`` as a pipeline step.

    ``reads`` maps each function parameter to the default context key it is
    bound from; a spec can override the binding per step via ``in:``. ``writes``
    lists the natural output names returned by the function; a spec can rename
    them into other context keys via ``out:``.

    ``parameters``/``input_artifacts``/``output_artifacts`` are optional,
    additive GUI/discovery metadata (Phase 1 of the pipeline-structure plan).
    A step with none of them declared is still fully valid and executable;
    they are being backfilled module by module.

    ``parameters_reviewed`` records whether an empty ``parameters`` tuple has
    been confirmed complete rather than simply never audited; it is set to
    ``True`` automatically whenever ``parameters`` is non-empty, since
    declaring real parameter metadata is itself evidence of review.

    ``session_reads``/``session_writes`` name dotted resources this step
    reads/writes on the shared ``M3Session`` directly, outside its declared
    ``reads``/``writes`` context-key bindings - see the field docstrings on
    :class:`StepDefinition` and ``m3resp.workflows.session_deps``.

    ``aliases`` lists former names this step was registered under, so specs
    written against them keep compiling. They resolve silently and are hidden
    from discovery - see :data:`STEP_ALIASES`.
    """

    def decorator(func: StepCallable) -> StepCallable:
        if name in STEP_REGISTRY:
            raise ValueError(f"Pipeline step '{name}' is already registered.")
        for alias in aliases:
            if alias in STEP_REGISTRY:
                raise ValueError(
                    f"Cannot alias '{alias}' to '{name}': '{alias}' is itself a "
                    "registered step."
                )
            if STEP_ALIASES.get(alias) not in (None, name):
                raise ValueError(
                    f"Pipeline step alias '{alias}' is already mapped to "
                    f"'{STEP_ALIASES[alias]}'."
                )
        definition = StepDefinition(
            name=name,
            func=func,
            reads=dict(reads or {}),
            writes=tuple(writes),
            requires=tuple(requires),
            summary=summary or (func.__doc__ or "").strip().split("\n", 1)[0],
            description=description,
            parameters=tuple(parameters),
            parameters_reviewed=parameters_reviewed or bool(parameters),
            mutually_exclusive_parameters=tuple(
                tuple(group) for group in mutually_exclusive_parameters
            ),
            input_artifacts=tuple(input_artifacts),
            output_artifacts=tuple(output_artifacts),
            session_reads=tuple(session_reads),
            session_writes=tuple(session_writes),
            modality=modality,
            category=category,
            alternatives=tuple(alternatives),
            optional_packages=tuple(optional_packages),
            resource_profile=resource_profile,
            version=version,
            deprecated_since=deprecated_since,
        )
        _validate_step_definition(definition)
        STEP_REGISTRY[name] = definition
        for alias in aliases:
            STEP_ALIASES[alias] = name
        return func

    return decorator


def _validate_step_definition(definition: StepDefinition) -> None:
    """Registration-time checks (Phase 1.2).

    Two tiers: name/output-name checks apply to every step, since every
    currently registered step already satisfies them. Parameter/artifact
    metadata checks only run when that metadata is actually declared, since
    most built-in steps have not been backfilled with it yet — this keeps
    ``register_step`` additive during the module-by-module migration.
    """

    if not _NAME_PATTERN.match(definition.name):
        raise StepMetadataError(
            f"Step name '{definition.name}' must look like "
            "'<prefix>.<operation>' using lowercase snake_case."
        )

    for output_name in (
        *definition.writes,
        *(a.name for a in definition.output_artifacts),
    ):
        if output_name in _RESERVED_OUTPUT_NAMES or output_name.startswith("_"):
            raise StepMetadataError(
                f"Step '{definition.name}' writes to reserved output name "
                f"'{output_name}'."
            )

    _validate_parameters(definition)
    _validate_mutually_exclusive_parameters(definition)
    _validate_artifacts(definition.name, "input_artifacts", definition.input_artifacts)
    _validate_artifacts(
        definition.name, "output_artifacts", definition.output_artifacts
    )
    _validate_json_safe(definition)


def _validate_parameters(definition: StepDefinition) -> None:
    if not definition.parameters:
        return

    seen: set[str] = set()
    for param in definition.parameters:
        if param.name in seen:
            raise StepMetadataError(
                f"Step '{definition.name}' declares duplicate parameter '{param.name}'."
            )
        seen.add(param.name)

        if param.name in definition.reads:
            raise StepMetadataError(
                f"Step '{definition.name}' parameter '{param.name}' collides "
                "with a declared context read of the same name."
            )
        if param.value_type not in _PARAMETER_TYPES:
            raise StepMetadataError(
                f"Step '{definition.name}' parameter '{param.name}' has "
                f"unknown value_type '{param.value_type}'."
            )
        if (
            param.minimum is not None
            and param.maximum is not None
            and param.minimum > param.maximum
        ):
            raise StepMetadataError(
                f"Step '{definition.name}' parameter '{param.name}' has "
                f"minimum {param.minimum} greater than maximum {param.maximum}."
            )
        if (
            param.choices is not None
            and param.default is not None
            and param.default not in param.choices
        ):
            raise StepMetadataError(
                f"Step '{definition.name}' parameter '{param.name}' default "
                f"{param.default!r} is not one of its declared choices."
            )

    _validate_parameters_against_signature(definition)


def _validate_parameters_against_signature(definition: StepDefinition) -> None:
    """When declared, a parameter must name a real keyword of the callable —
    unless the callable accepts ``**kwargs``, in which case we can't tell."""

    try:
        sig = signature(definition.func)
    except (TypeError, ValueError):
        return
    if any(p.kind is Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    accepted = set(sig.parameters)
    for param in definition.parameters:
        if param.name not in accepted:
            raise StepMetadataError(
                f"Step '{definition.name}' declares parameter '{param.name}', "
                "which is not a keyword argument of its function."
            )


def _validate_mutually_exclusive_parameters(definition: StepDefinition) -> None:
    if not definition.mutually_exclusive_parameters:
        return

    declared = {p.name for p in definition.parameters}
    for group in definition.mutually_exclusive_parameters:
        if len(group) < 2:
            raise StepMetadataError(
                f"Step '{definition.name}' declares a mutually-exclusive "
                f"parameter group with fewer than two names: {group!r}."
            )
        for name in group:
            if name not in declared:
                raise StepMetadataError(
                    f"Step '{definition.name}' declares mutually-exclusive "
                    f"group {group!r}, but '{name}' is not a declared parameter."
                )


def _validate_artifacts(
    step_name: str, field_name: str, artifacts: tuple[StepArtifact, ...]
) -> None:
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.name in seen:
            raise StepMetadataError(
                f"Step '{step_name}' declares duplicate {field_name} entry "
                f"'{artifact.name}'."
            )
        seen.add(artifact.name)


def _validate_json_safe(definition: StepDefinition) -> None:
    payload = {
        "parameters": [p.as_dict() for p in definition.parameters],
        "input_artifacts": [a.as_dict() for a in definition.input_artifacts],
        "output_artifacts": [a.as_dict() for a in definition.output_artifacts],
    }
    try:
        json.dumps(payload)
    except TypeError as exc:
        raise StepMetadataError(
            f"Step '{definition.name}' has non-JSON-safe parameter/artifact "
            f"metadata: {exc}."
        ) from exc


def get_step(name: str) -> StepDefinition:
    """Return the registered step definition for ``name``.

    A retired name registered as an alias resolves silently to its current
    step, so specs written against the old name keep working.
    """

    definition = STEP_REGISTRY.get(name)
    if definition is not None:
        return definition

    canonical = STEP_ALIASES.get(name)
    if canonical is not None and canonical in STEP_REGISTRY:
        return STEP_REGISTRY[canonical]

    available = ", ".join(sorted(STEP_REGISTRY)) or "(none registered)"
    raise UnknownStepError(
        f"Unknown pipeline step '{name}'. Available steps: {available}."
    )


def available_steps() -> dict[str, str]:
    """Return a mapping of registered step name -> one-line summary."""

    # Ensure the built-in step package is imported so the registry is populated.
    from m3resp.workflows import steps  # noqa: F401

    return {name: step.summary for name, step in sorted(STEP_REGISTRY.items())}


@dataclass(frozen=True)
class StepDescription:
    """JSON-safe discovery view of a registered step (Phase 1.5).

    Unlike :class:`StepDefinition`, this never carries the callable, so it
    can be handed to the CLI/GUI and serialized directly.
    """

    name: str
    summary: str
    description: str
    modality: str | None
    category: str | None
    parameters: tuple[StepParameter, ...]
    parameters_reviewed: bool
    input_artifacts: tuple[StepArtifact, ...]
    output_artifacts: tuple[StepArtifact, ...]
    alternatives: tuple[str, ...]
    optional_packages: tuple[str, ...]
    resource_profile: str | None
    version: str | None
    deprecated_since: str | None
    capability: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "description": self.description,
            "modality": self.modality,
            "category": self.category,
            "parameters": [p.as_dict() for p in self.parameters],
            "parameters_reviewed": self.parameters_reviewed,
            "input_artifacts": [a.as_dict() for a in self.input_artifacts],
            "output_artifacts": [a.as_dict() for a in self.output_artifacts],
            "alternatives": list(self.alternatives),
            "optional_packages": list(self.optional_packages),
            "resource_profile": self.resource_profile,
            "version": self.version,
            "deprecated_since": self.deprecated_since,
            "capability": self.capability,
        }


def step_capability_state(name: str) -> str:
    """Report step availability without importing its optional packages.

    One of ``"available"``, ``"missing_optional_dependency"``, or
    ``"deprecated"``. ``"unavailable_in_build"`` is reserved for a future
    build-time capability distinction and is not produced by this generic
    implementation.
    """

    definition = get_step(name)
    if definition.deprecated_since is not None:
        return "deprecated"
    for package in definition.optional_packages:
        if importlib.util.find_spec(package) is None:
            return "missing_optional_dependency"
    return "available"


def describe_step(name: str) -> StepDescription:
    """Return the JSON-safe discovery description for one registered step."""

    # Ensure the built-in step package is imported, like available_steps()/
    # describe_steps() already do - otherwise a fresh process that calls
    # this directly (e.g. `m3resp describe <op>`) sees an empty registry.
    from m3resp.workflows import steps  # noqa: F401

    definition = get_step(name)
    return StepDescription(
        name=definition.name,
        summary=definition.summary,
        description=definition.description or definition.summary,
        modality=definition.modality,
        category=definition.category,
        parameters=definition.parameters,
        parameters_reviewed=definition.parameters_reviewed,
        input_artifacts=definition.input_artifacts,
        output_artifacts=definition.output_artifacts,
        alternatives=definition.alternatives,
        optional_packages=definition.optional_packages,
        resource_profile=definition.resource_profile,
        version=definition.version,
        deprecated_since=definition.deprecated_since,
        capability=step_capability_state(name),
    )


def describe_steps(*, prefix: str | None = None) -> list[StepDescription]:
    """Return discovery descriptions for every registered step, sorted by name.

    ``prefix`` restricts the result to names starting with e.g. ``"eit."``.
    """

    from m3resp.workflows import steps  # noqa: F401

    names = sorted(STEP_REGISTRY)
    if prefix is not None:
        names = [name for name in names if name.startswith(prefix)]
    return [describe_step(name) for name in names]
