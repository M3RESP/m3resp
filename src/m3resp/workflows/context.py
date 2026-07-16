"""Shared blackboard for declarative pipeline execution.

``PipelineContext`` holds the named artifacts produced and consumed by steps,
the spec-level ``inputs``, and the backing :class:`~m3resp.core.session.M3Session`
(so loading, event normalization, and export keep flowing through the session).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from m3resp.core.exceptions import PipelineSpecError
from m3resp.core.session import M3Session

#: Context key under which the backing session is always available.
SESSION_KEY = "session"

#: Context key under which ``run_spec()`` seeds the one resolved,
#: possibly-timestamped output directory shared by every export path in a
#: run (Phase 6.1 of the pipeline-structure plan). A custom export step
#: should bind its output-directory parameter to this documented constant
#: (``reads={"output_dir": RESOLVED_OUTPUT_DIR_KEY}``) rather than
#: hand-typing the string, e.g. ``export.rotarc_result``/
#: ``export.session_summary`` already do.
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
    """Recursively yield every ``@name`` reference inside ``value``.

    Walks into lists and mapping values (Phase 2.4); an ``@@``-escaped
    literal is not a reference. Used both by :meth:`PipelineContext.resolve_input`
    and by pre-execution spec validation.
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
    """Resolve a ``with:`` value against a plain ``inputs`` mapping.

    This is the session-free core of :meth:`PipelineContext.resolve_input`,
    reusable at compile/validation time (Phase 3.1) when no
    :class:`PipelineContext` exists yet - resolution only ever depends on the
    spec's declared ``inputs``, never on anything a step produces.
    """

    if is_escaped_literal(value):
        return value[1:]
    if is_input_reference(value):
        ref = value[1:]
        if ref not in inputs:
            available = ", ".join(sorted(inputs)) or "(none)"
            raise PipelineSpecError(
                f"Pipeline references unknown input '@{ref}'. "
                f"Declared inputs: {available}."
            )
        return inputs[ref]
    if isinstance(value, list):
        return [resolve_value(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item, inputs) for key, item in value.items()}
    return value


@dataclass
class PipelineContext:
    """Named-artifact blackboard wrapping an ``M3Session``."""

    session: M3Session
    inputs: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    #: Base directory ``path``-typed static parameters resolve against
    #: (Phase 2.5). Defaults to the current working directory when unset.
    root: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        # Make the session reachable as a normal context value so steps can
        # read it through the same binding mechanism as everything else.
        self.values.setdefault(SESSION_KEY, self.session)

    def has(self, key: str) -> bool:
        """Return whether ``key`` is an available artifact."""

        return key in self.values

    def get(self, key: str) -> Any:
        """Return the artifact stored under ``key``."""

        try:
            return self.values[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.values)) or "(empty)"
            raise PipelineSpecError(
                f"Pipeline step requested missing context key '{key}'. "
                f"Available keys: {available}."
            ) from exc

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under context key ``key``."""

        self.values[key] = value

    def resolve_input(self, value: Any) -> Any:
        """Resolve a ``with:`` value, expanding ``@name`` input references.

        Applies recursively inside lists and mapping values (Phase 2.4); a
        literal string beginning with ``@`` is written ``@@text`` to escape
        it (``@@foo`` resolves to the literal string ``"@foo"``).
        """

        return resolve_value(value, self.inputs)
