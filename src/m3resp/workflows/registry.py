"""Step registry for declarative M3Resp pipelines.

A *step* is a named, reusable operation. Each step declares the context keys it
reads (mapped onto its parameter names) and the context keys it writes. A
pipeline spec lists steps by name and binds them together, so workflows can be
assembled from a YAML/JSON file without writing custom Python.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from m3resp.core.exceptions import UnknownStepError

# A step callable receives bound keyword arguments and returns a mapping of
# {output_name: value}. ``None`` is treated as "no outputs".
StepCallable = Callable[..., Mapping[str, Any] | None]


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


STEP_REGISTRY: dict[str, StepDefinition] = {}


def register_step(
    name: str,
    *,
    reads: Mapping[str, str | None] | None = None,
    writes: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    summary: str = "",
) -> Callable[[StepCallable], StepCallable]:
    """Register ``func`` under ``name`` as a pipeline step.

    ``reads`` maps each function parameter to the default context key it is
    bound from; a spec can override the binding per step via ``in:``. ``writes``
    lists the natural output names returned by the function; a spec can rename
    them into other context keys via ``out:``.
    """

    def decorator(func: StepCallable) -> StepCallable:
        if name in STEP_REGISTRY:
            raise ValueError(f"Pipeline step '{name}' is already registered.")
        STEP_REGISTRY[name] = StepDefinition(
            name=name,
            func=func,
            reads=dict(reads or {}),
            writes=tuple(writes),
            requires=tuple(requires),
            summary=summary or (func.__doc__ or "").strip().split("\n", 1)[0],
        )
        return func

    return decorator


def get_step(name: str) -> StepDefinition:
    """Return the registered step definition for ``name``."""

    try:
        return STEP_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(STEP_REGISTRY)) or "(none registered)"
        raise UnknownStepError(
            f"Unknown pipeline step '{name}'. Available steps: {available}."
        ) from exc


def available_steps() -> dict[str, str]:
    """Return a mapping of registered step name -> one-line summary."""

    # Ensure the built-in step package is imported so the registry is populated.
    from m3resp.workflows import steps  # noqa: F401

    return {name: step.summary for name, step in sorted(STEP_REGISTRY.items())}
