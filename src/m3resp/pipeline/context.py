"""Shared blackboard for declarative pipeline execution.

``PipelineContext`` holds the named artifacts produced and consumed by steps,
the spec-level ``inputs``, and the backing :class:`~m3resp.core.session.M3Session`
(so loading, event normalization, and export keep flowing through the session).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from m3resp.core.exceptions import PipelineSpecError
from m3resp.core.session import M3Session

#: Context key under which the backing session is always available.
SESSION_KEY = "session"


@dataclass
class PipelineContext:
    """Named-artifact blackboard wrapping an ``M3Session``."""

    session: M3Session
    inputs: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

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
        """Resolve a ``with:`` value, expanding ``@name`` input references."""

        if isinstance(value, str) and value.startswith("@"):
            ref = value[1:]
            if ref not in self.inputs:
                available = ", ".join(sorted(self.inputs)) or "(none)"
                raise PipelineSpecError(
                    f"Pipeline references unknown input '@{ref}'. "
                    f"Declared inputs: {available}."
                )
            return self.inputs[ref]
        return value
