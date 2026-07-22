"""``Pipeline`` contract for named, built-in presets.

Why this exists, in one sentence: it's a named shortcut for a sequence of
`M3Session` method calls, not a second way of executing pipeline logic.

There are two ways to run EIT/EMG/multimodal processing in m3resp, and they
solve different problems:

- ``m3resp.workflows`` (``run_pipeline(spec, session=...)``) runs a fully
  custom YAML/JSON spec built from arbitrary, individually composable steps
  (``eit.mdn_filter``, ``eit.global_impedance``, ...). This is for bespoke or
  batch workflows where the exact sequence of operations varies per project -
  see ``docs/pipelines.md``. Those granular steps are pure data transforms;
  they don't populate `session.signals`/`parameter_results`/`quality` or
  record provenance themselves.
- ``Pipeline``/``session.run_pipeline("eit")`` (this module) is for the
  common case: "run the default preprocessing and detection for this
  modality." A concrete `Pipeline.run` just calls `M3Session`'s own
  already-instrumented methods (``preprocess_eit``, ``detect_eit_breaths``,
  ...) in a fixed order - it is *those methods*, not this class, that
  populate the typed collections and record provenance. Every option those
  methods accept is still reachable through ``config``, so this isn't a
  rigid, fixed algorithm - it's a name for "call these methods in this
  order," with the actual behavior fully controlled by whatever `config` is
  passed in.

No new execution machinery is written here: this deliberately avoids
building a second, parallel step-execution engine - that would duplicate
`m3resp.workflows` for no benefit and would need its own copy of the
typed-collection/provenance instrumentation those session methods already
have.
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
