"""``Pipeline`` contract for named, built-in presets (plan_stage2.md Sec 18,
Milestone 2.4).

This is deliberately a thin, different mechanism from the declarative
step-registry engine in ``m3resp.workflows``: that engine runs a
fully custom YAML/JSON spec of arbitrary steps (Stage 1, still the right tool
for bespoke or batch workflows). A ``Pipeline`` here is a small, named preset
that just calls this session's own already-instrumented methods
(``preprocess_eit``, ``detect_eit_breaths``, ...) in a fixed order, so
``session.run_pipeline("eit")`` is a convenience shortcut, not a second
execution engine. See ``plan/stage2_consolidation.md`` for the reasoning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from m3resp.core.session import M3Session

#: Per-method keyword arguments, keyed by the session method name a concrete
#: `Pipeline` calls (e.g. ``{"preprocess": {...}, "detect_breaths": {...}}``).
PipelineConfig = Mapping[str, Mapping[str, Any]]


class Pipeline(ABC):
    """A named preset that runs a fixed sequence of `M3Session` methods."""

    name: str

    @abstractmethod
    def run(
        self, session: M3Session, *, config: PipelineConfig | None = None
    ) -> M3Session:
        """Run this pipeline against ``session`` and return it."""

    @staticmethod
    def _kwargs_for(config: PipelineConfig | None, step_name: str) -> dict[str, Any]:
        return dict((config or {}).get(step_name, {}))
