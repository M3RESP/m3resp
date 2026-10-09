"""Named workflow values, input-reference resolution and the backing session."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from m3resp.core.exceptions import WorkflowSpecError
from m3resp.core.session import M3Session

#: Context key under which the backing session is always available.
SESSION_KEY = "session"

#: Shared output directory supplied by run_spec, including a timestamp
#: subdirectory when configured. Export steps can read it with
#: ``reads={"output_dir": RESOLVED_OUTPUT_DIR_KEY}``.
RESOLVED_OUTPUT_DIR_KEY = "_resolved_output_dir"

_REF_PREFIX = "@"
_ESCAPE_PREFIX = "@@"


def is_escaped_literal(value: Any) -> bool:
    """Whether ``value`` is a literal string escaped with a leading ``@@``."""

    return isinstance(value, str) and value.startswith(_ESCAPE_PREFIX)


def is_input_reference(value: Any) -> bool:
    """Whether ``value`` is a whole-string ``@name`` input reference."""

    return (
        isinstance(value, str)
        and value.startswith(_REF_PREFIX)
        and not is_escaped_literal(value)
    )


def iter_input_references(value: Any) -> Iterator[str]:
    """Yield input names from ``@name`` strings in lists and dictionary values.

    ``@@text`` represents a literal string. References are yielded in traversal
    order, including repeated names.
    """

    if is_input_reference(value):
        yield value[1:]
    elif isinstance(value, list):
        for item in value:
            yield from iter_input_references(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_input_references(item)


def resolve_value(value: Any, inputs: dict[str, Any]) -> Any:
    """Resolve workflow input references within a parameter value.

    Args:
        value: Parameter value, optionally containing ``@name`` references
            within lists or dictionary values. ``@@text`` becomes ``"@text"``.
        inputs: Declared workflow inputs keyed by name.

    Returns:
        Any: The resolved value. Lists and dictionaries are rebuilt; referenced
            input objects are returned as stored in inputs.

    Raises:
        WorkflowSpecError: If a reference names an undeclared input.
    """

    if is_escaped_literal(value):
        return value[1:]
    if is_input_reference(value):
        ref = value[1:]
        if ref not in inputs:
            available = ", ".join(sorted(inputs)) or "(none)"
            raise WorkflowSpecError(
                f"Workflow references unknown input '@{ref}'. "
                f"Declared inputs: {available}."
            )
        return inputs[ref]
    if isinstance(value, list):
        return [resolve_value(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item, inputs) for key, item in value.items()}
    return value


@dataclass
class WorkflowContext:
    """Named values shared by workflow steps, together with a session.

    Attributes:
        session: Session holding recordings, processing results and provenance.
        inputs: Declared inputs used to resolve ``@name`` parameter references.
        values: Available step inputs and outputs keyed by context name. The
            session is added under ``"session"`` when that key is absent.
        root: Base directory for relative path settings; defaults to the
            current working directory.
    """

    session: M3Session
    inputs: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    #: Base directory for path settings; defaults to the current directory.
    root: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        # Make the session reachable as a normal context value so steps can
        # read it through the same binding mechanism as everything else.
        """Add the session to values when the session key is absent."""

        self.values.setdefault(SESSION_KEY, self.session)

    def has(self, key: str) -> bool:
        """Return whether ``key`` is an available artifact."""

        return key in self.values

    def get(self, key: str) -> Any:
        """Return the value stored under key.

        Raises WorkflowSpecError if the key is missing, listing available keys.
        """

        try:
            return self.values[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.values)) or "(empty)"
            raise WorkflowSpecError(
                f"Workflow step requested missing context key '{key}'. "
                f"Available keys: {available}."
            ) from exc

    def set(self, key: str, value: Any) -> None:
        """Store value under key, replacing any existing value."""

        self.values[key] = value

    def resolve_input(self, value: Any) -> Any:
        """Resolve parameter references using this context's declared inputs.

        Calls `resolve_value` for nested lists and dictionary values, including
        ``@@text`` literals. Raises WorkflowSpecError for an unknown input name.
        """

        return resolve_value(value, self.inputs)
