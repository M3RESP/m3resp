"""Step registry for declarative M3Resp workflows.

A *step* is a named, reusable operation. Each step declares the context keys it
reads (mapped onto its parameter names) and the context keys it writes. A
workflow spec lists steps by name and binds them together, so workflows can be
assembled from a YAML/JSON file without writing custom Python.

Steps can also carry JSON-safe descriptions for step discovery and a GUI
(:class:`StepParameter`, :class:`StepArtifact`): their static parameters with
units and limits, and the values they read and write. A step without these
descriptions runs the same way.

:class:`StepInfo` holds the fields that describe a step. Both
:class:`StepDefinition` (what the registry stores) and
:class:`StepDescription` (what the command line and a GUI get) build on it,
so each of these fields is declared once.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from inspect import Parameter, signature
from typing import Any, Literal

from m3resp.core.exceptions import StepMetadataError, UnknownStepError
from m3resp.workflows.lifecycle import optional_package_version

# A step callable receives bound keyword arguments and returns a mapping of
# {output_name: value}. ``None`` is treated as "no outputs".
StepCallable = Callable[..., Mapping[str, Any] | None]

#: Output key a step registered with ``record=True`` uses for its own
#: provenance details (settings used, source function, ...). The step's
#: wrapper takes this key out of the outputs and adds the details to the
#: step's provenance record. Steps write the key as the text
#: ``"_provenance"``; the registry and the workflow engine use this name
#: to look for it.
PROVENANCE_KEY = "_provenance"

#: Allowed ``StepParameter.value_type`` values.
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

#: ``StepArtifact.artifact_type`` for a step that accepts and returns any
#: signal type (e.g. ``eit.slice_signal``). The compiler skips the artifact
#: type check for a value on whichever side declares it.
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
    #: When ``value_type == "path"``: True if the parameter also takes a list
    #: of paths (several files read as one recording).
    allow_list: bool = False
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
            "allow_list": self.allow_list,
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
    #: True for a temporary upstream/adapter object that a GUI leaves out of
    #: its results by default.
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


@dataclass(frozen=True, kw_only=True)
class StepInfo:
    """The fields that describe a registered step.

    :class:`StepDefinition` and :class:`StepDescription` both build on this
    class. A field added here reaches the registry, ``describe_step`` and
    its JSON output; add the matching keyword to :func:`register_step`.
    """

    name: str
    summary: str = ""
    #: Longer, GUI-facing description. ``describe_step`` uses ``summary``
    #: when this is empty.
    description: str = ""
    #: Owning modality, e.g. ``"eit"``, ``"emg"``, or ``None`` for a
    #: cross-modality/generic step.
    modality: str | None = None
    #: Free-form grouping for GUI menus, e.g. ``"preprocessing"``, ``"quality"``.
    category: str | None = None
    #: Static (``with:``) parameter descriptions.
    parameters: tuple[StepParameter, ...] = ()
    #: Whether someone has checked that ``parameters`` is complete, so that
    #: an empty tuple means "has no static parameter" and not "not looked at
    #: yet". Set automatically when ``parameters`` is non-empty; otherwise
    #: set it with ``register_step(..., parameters_reviewed=True)``.
    parameters_reviewed: bool = False
    #: Descriptions of the ``reads``/``requires`` context keys.
    input_artifacts: tuple[StepArtifact, ...] = ()
    #: Descriptions of the ``writes`` context keys.
    output_artifacts: tuple[StepArtifact, ...] = ()
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


@dataclass(frozen=True, kw_only=True)
class StepDefinition(StepInfo):
    """A registered step: the :class:`StepInfo` fields, the function, and
    the context keys it reads and writes."""

    func: StepCallable
    #: parameter name -> default context key, or None if the binding is required
    #: and must be supplied via ``in:`` in the spec.
    reads: Mapping[str, str | None] = field(default_factory=dict)
    #: natural output names the step returns (default context keys it writes).
    writes: tuple[str, ...] = ()
    #: parameter names in ``reads`` whose context key may be absent. The step
    #: is then called without the argument and uses its own default. For an
    #: input that improves a result when present, e.g. the EMG baseline that
    #: ``emg.detect_breaths`` sets its threshold against.
    optional_reads: tuple[str, ...] = ()
    #: context keys that must already exist before the step runs, but are not
    #: passed as arguments (e.g. an in-place mutated sequence).
    requires: tuple[str, ...] = ()
    #: Groups of parameter names that must not be set together in the same
    #: step invocation (e.g. ``emg.ecg_gating``'s ``gate_width_seconds``/
    #: ``gate_width_samples``), checked when a workflow is compiled. Every
    #: name in every group must be a declared parameter; checked at
    #: registration time.
    mutually_exclusive_parameters: tuple[tuple[str, ...], ...] = ()
    #: Dotted resource names this step reads from the shared ``M3Session``
    #: object itself, outside its ``reads``/``in:`` context keys - e.g.
    #: ``"session.processed.emg"`` for a step that calls
    #: ``session.detect_emg_breaths()``, which reads whatever the most
    #: recent preprocessing/ECG-removal step left on the session. See
    #: ``m3resp.workflows.session_deps``.
    session_reads: tuple[str, ...] = ()
    #: Dotted resource names this step writes onto the shared ``M3Session``
    #: object, e.g. ``"session.raw.eit"`` for a loader, or
    #: ``"session.quality"`` for a step that calls ``session.quality.add``.
    #: A resource name is an ancestor of any more specific name sharing its
    #: dotted prefix (``"session.raw"`` covers ``"session.raw.eit"``), so a
    #: step may declare whichever level of detail was checked.
    session_writes: tuple[str, ...] = ()
    #: Whether each call adds one provenance record to the session; see
    #: ``register_step(..., record=True)``.
    record: bool = False

    @property
    def operation_id(self) -> str:
        """Stable backend-neutral identifier. Currently the registered name."""

        return self.name


STEP_REGISTRY: dict[str, StepDefinition] = {}

#: Retired step name -> the canonical name it now resolves to. Populated by
#: ``register_step(..., aliases=...)``.
#:
#: Aliases resolve silently and are left out of `available_steps`,
#: `describe_steps` and the "available steps" text of `UnknownStepError`: an
#: existing spec keeps running unchanged, while discovery and any GUI built on
#: it only ever offer the current name.
STEP_ALIASES: dict[str, str] = {}


def register_step(
    name: str,
    *,
    reads: Mapping[str, str | None] | None = None,
    optional_reads: tuple[str, ...] = (),
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
    record: bool = False,
    aliases: tuple[str, ...] = (),
) -> Callable[[StepCallable], StepCallable]:
    """Register a function and its workflow inputs, outputs and descriptions.

    Args:
        name: Operation name in lowercase prefix.operation form.
        reads: Function arguments mapped to default context keys. None values
            require explicit input bindings in the spec.
        optional_reads: Names from reads that may be absent at execution time.
        writes: Output names returned by the step, renameable through ``out:``.
        requires: Additional context keys required before execution.
        summary: Short description; defaults to the function docstring's first line.
        description: Longer description for step discovery.
        parameters: Static parameter descriptions, including units and limits.
        parameters_reviewed: Whether an empty parameter list is confirmed complete.
            Set to True automatically when parameter descriptions are supplied.
        mutually_exclusive_parameters: Groups with at most one setting per invocation.
        input_artifacts: Descriptions of values read by the step.
        output_artifacts: Descriptions of values returned by the step.
        session_reads: Dotted names of session resources read directly.
        session_writes: Dotted names of session resources changed directly.
        modality: Device or technique associated with the step.
        category: Processing or measurement category.
        alternatives: Names of related operations.
        optional_packages: Package names used to report availability.
        resource_profile: Optional description of processing resources needed.
        version: Optional operation version.
        deprecated_since: Version from which this operation is deprecated.
        record: Add one provenance record to ``session.provenance`` after
            each call. The record holds the step name, its reads and writes,
            the installed version of its first optional package (None for a
            step without one), and the details the step returns under the
            output key ``"_provenance"``. That key is removed from the
            outputs. The step must read ``session``.
        aliases: Former operation names that resolve to this registration and
            are omitted from step discovery.

    Returns:
        Callable: Decorator that validates metadata, adds the definition and
            aliases to the registries, and returns the supplied function, or
            with ``record=True`` the function wrapped to record each call.

    Raises:
        ValueError: When applying the decorator, if a name or alias conflicts
            with a registration, or optional_reads names an undeclared input.
        StepMetadataError: When applying the decorator, if names, parameters,
            input/output descriptions or their JSON representation are
            invalid, or ``record=True`` is set for a step that does not read
            ``session``.
    """

    def decorator(func: StepCallable) -> StepCallable:
        """Validate and register the step definition and aliases, returning
        func (wrapped when ``record=True``)."""

        if name in STEP_REGISTRY:
            raise ValueError(f"Workflow step '{name}' is already registered.")
        for alias in aliases:
            if alias in STEP_REGISTRY:
                raise ValueError(
                    f"Cannot alias '{alias}' to '{name}': '{alias}' is itself a "
                    "registered step."
                )
            if STEP_ALIASES.get(alias) not in (None, name):
                raise ValueError(
                    f"Workflow step alias '{alias}' is already mapped to "
                    f"'{STEP_ALIASES[alias]}'."
                )
        unknown_optional = set(optional_reads) - set(reads or {})
        if unknown_optional:
            raise ValueError(
                f"Workflow step '{name}' declares optional_reads "
                f"{sorted(unknown_optional)} that are not in 'reads'."
            )
        if record and "session" not in (reads or {}):
            raise StepMetadataError(
                f"Workflow step '{name}' sets record=True, so it must read 'session'."
            )
        step_func = _record_each_call(name, func) if record else func
        definition = StepDefinition(
            name=name,
            func=step_func,
            reads=dict(reads or {}),
            optional_reads=tuple(optional_reads),
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
            record=record,
        )
        _validate_step_definition(definition)
        STEP_REGISTRY[name] = definition
        for alias in aliases:
            STEP_ALIASES[alias] = name
        return step_func

    return decorator


def _record_each_call(name: str, func: StepCallable) -> StepCallable:
    """Wrap a step so that each call adds one provenance record to its session.

    The record is added after the step returns, with the fields described
    under ``record`` in :func:`register_step`.
    """

    func_signature = signature(func)

    @functools.wraps(func)
    def run_and_record(*args: Any, **kwargs: Any) -> Mapping[str, Any] | None:
        result = func(*args, **kwargs)
        if result is None:
            outputs: dict[str, Any] = {}
        else:
            outputs = dict(result)
        details = outputs.pop(PROVENANCE_KEY, {})
        session = func_signature.bind_partial(*args, **kwargs).arguments.get("session")
        if session is None:
            raise TypeError(
                f"Workflow step '{name}' records each call in the session's "
                "provenance, so it needs a session."
            )
        definition = STEP_REGISTRY[name]
        session._record(
            name,
            definition.modality,
            parameters={
                "step": name,
                "reads": sorted(definition.reads),
                "writes": list(definition.writes),
                "upstream_version": (
                    optional_package_version(definition.optional_packages[0])
                    if definition.optional_packages
                    else None
                ),
                **details,
            },
        )
        return None if result is None else outputs

    return run_and_record


def _validate_step_definition(definition: StepDefinition) -> None:
    """Checks run when a step is registered.

    The step name and output names are checked for every step. Parameter
    and artifact descriptions are checked when the step declares them.
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
    """Return a registered step definition, resolving former names through aliases.

    Raises UnknownStepError if the name is unregistered, listing current names.
    """

    definition = STEP_REGISTRY.get(name)
    if definition is not None:
        return definition

    canonical = STEP_ALIASES.get(name)
    if canonical is not None and canonical in STEP_REGISTRY:
        return STEP_REGISTRY[canonical]

    available = ", ".join(sorted(STEP_REGISTRY)) or "(none registered)"
    raise UnknownStepError(
        f"Unknown workflow step '{name}'. Available steps: {available}."
    )


def available_steps() -> dict[str, str]:
    """Return a mapping of registered step name -> one-line summary."""

    # Ensure the built-in step package is imported so the registry is populated.
    from m3resp.workflows import steps  # noqa: F401

    return {name: step.summary for name, step in sorted(STEP_REGISTRY.items())}


@dataclass(frozen=True, kw_only=True)
class StepDescription(StepInfo):
    """What the command line and a GUI get for one registered step: the
    :class:`StepInfo` fields plus ``capability``. All values are JSON-safe,
    and :meth:`as_dict` gives them as a dictionary."""

    #: ``"available"``, ``"missing_optional_dependency"`` or ``"deprecated"``,
    #: from :func:`step_capability_state`.
    capability: str

    def as_dict(self) -> dict[str, Any]:
        """Every field by name; tuples become lists, and parameter and
        artifact descriptions become dictionaries."""

        return {
            item.name: _json_value(getattr(self, item.name)) for item in fields(self)
        }


def _json_value(value: Any) -> Any:
    """A field value as JSON-safe data: tuples become lists, and objects
    with an ``as_dict`` method become dictionaries."""

    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return value


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
    shared = {item.name: getattr(definition, item.name) for item in fields(StepInfo)}
    shared["description"] = definition.description or definition.summary
    return StepDescription(**shared, capability=step_capability_state(name))


def describe_steps(*, prefix: str | None = None) -> list[StepDescription]:
    """Return discovery descriptions for every registered step, sorted by name.

    ``prefix`` restricts the result to names starting with e.g. ``"eit."``.
    """

    from m3resp.workflows import steps  # noqa: F401

    names = sorted(STEP_REGISTRY)
    if prefix is not None:
        names = [name for name in names if name.startswith(prefix)]
    return [describe_step(name) for name in names]
