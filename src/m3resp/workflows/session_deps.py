"""Session-resource dependency tracking.

Most pipeline steps communicate through declared context keys (``reads``/
``writes``, bound via ``in:``/``out:``), and ``engine/diagnostics.py``
already tracks that positionally: each read binds to the most recent
preceding writer of the same context key. But a large share of steps also
read or mutate the shared ``M3Session`` object directly - e.g.
``emg.detect_breaths`` calls ``session.detect_emg_breaths()``, which reads
whatever ``session.processed["emg"]`` currently holds, not a declared
context key. That dependency is invisible to ``collect_diagnostics()``
unless a step declares it via ``register_step(..., session_reads=...,
session_writes=...)`` (see ``StepDefinition`` in ``registry.py``).

This module answers two questions from that declared metadata, using the
same "most recent preceding writer, positionally" rule as the context-key
tracking, so a spec's step order can never silently disagree with it:

- :func:`find_session_dependency_conflicts` - did this spec's step order
  put a session-resource read before the step that (later in the same
  spec) writes it?
- :func:`downstream_step_positions` - which steps, transitively, depend on
  a given step, through either an explicit context-key edge or a declared
  session-resource edge? Used to answer "what does re-running/editing this
  step invalidate" (plan/06_gui_readiness_plan.md §5.2's ``rerun-from``).

See ``plan/06_gui_readiness_plan.md`` §4 for the plan this implements.
"""

from __future__ import annotations

from dataclasses import dataclass

from m3resp.core.exceptions import UnknownStepError
from m3resp.workflows.registry import StepDefinition, get_step
from m3resp.workflows.spec import PipelineSpec, StepSpec


def resources_match(write: str, read: str) -> bool:
    """Whether a step's declared ``session_writes`` entry ``write`` covers
    another step's declared ``session_reads`` entry ``read``.

    A resource name is an ancestor of any more specific dotted name sharing
    its prefix, so either side may declare whichever granularity it
    actually audited: a coarse writer (``"session.raw"``) covers a specific
    reader (``"session.raw.eit"``), and a specific writer
    (``"session.raw.eit"``) covers a coarse reader (``"session.raw"``).
    """

    return write == read or write.startswith(read + ".") or read.startswith(write + ".")


def _resolved_steps(
    spec: PipelineSpec,
) -> list[tuple[int, StepSpec, StepDefinition]]:
    """Every step in ``spec`` paired with its registered definition, in
    order. Steps naming an unregistered operation are skipped - that is
    already reported elsewhere (``collect_diagnostics``'s ``unknown_step``
    diagnostic), and this module only reasons about steps it can resolve.
    """

    resolved: list[tuple[int, StepSpec, StepDefinition]] = []
    for position, step_spec in enumerate(spec.steps):
        try:
            definition = get_step(step_spec.uses)
        except UnknownStepError:
            continue
        resolved.append((position, step_spec, definition))
    return resolved


@dataclass(frozen=True)
class SessionDependencyConflict:
    """One step whose declared ``session_reads`` resource is not satisfied
    by anything earlier in the spec, but *is* satisfied by a later step -
    meaning the spec's step order silently reordered a session mutation the
    reader actually depends on."""

    resource: str
    reader_position: int
    reader_step_id: str
    reader_uses: str
    writer_position: int
    writer_step_id: str
    writer_uses: str


def find_session_dependency_conflicts(
    spec: PipelineSpec,
) -> list[SessionDependencyConflict]:
    """Find every session-resource read that a later step in ``spec``
    writes, but no earlier step does.

    A read resource satisfied by nothing anywhere in the spec is not
    flagged: it may come from state the caller pre-populated on the
    session before running the pipeline (e.g. a fixture, or a prior
    pipeline run against the same session), which is legitimate and
    outside the spec's view.
    """

    steps = _resolved_steps(spec)
    conflicts: list[SessionDependencyConflict] = []

    for position, step_spec, definition in steps:
        for read_resource in definition.session_reads:
            if _has_matching_writer_before(steps, position, read_resource):
                continue

            writer = _first_matching_writer_after(steps, position, read_resource)
            if writer is None:
                continue
            writer_position, writer_step_spec, _resource = writer
            conflicts.append(
                SessionDependencyConflict(
                    resource=read_resource,
                    reader_position=position,
                    reader_step_id=step_spec.id,
                    reader_uses=step_spec.uses,
                    writer_position=writer_position,
                    writer_step_id=writer_step_spec.id,
                    writer_uses=writer_step_spec.uses,
                )
            )

    return conflicts


def _has_matching_writer_before(
    steps: list[tuple[int, StepSpec, StepDefinition]],
    position: int,
    read_resource: str,
) -> bool:
    for writer_position, _writer_step_spec, writer_definition in steps:
        if writer_position >= position:
            break
        for write_resource in writer_definition.session_writes:
            if resources_match(write_resource, read_resource):
                return True
    return False


def _first_matching_writer_after(
    steps: list[tuple[int, StepSpec, StepDefinition]],
    position: int,
    read_resource: str,
) -> tuple[int, StepSpec, str] | None:
    for writer_position, writer_step_spec, writer_definition in steps:
        if writer_position <= position:
            continue
        for write_resource in writer_definition.session_writes:
            if resources_match(write_resource, read_resource):
                return writer_position, writer_step_spec, write_resource
    return None


def downstream_step_positions(spec: PipelineSpec, start_position: int) -> set[int]:
    """Positions of every step that depends on the step at ``start_position``,
    transitively, through either an explicit context-key binding or a
    declared session-resource dependency.

    Built from the same "most recent preceding writer" rule used by
    ``collect_diagnostics()`` for context keys, extended with the session
    resources declared via ``session_reads``/``session_writes``, so the two
    never disagree. Used to answer "what would re-running this step
    invalidate downstream" for a results-review rerun-from flow.
    """

    steps = _resolved_steps(spec)
    edges = _build_dependency_edges(steps)

    downstream: set[int] = set()
    frontier = {start_position}
    while frontier:
        next_frontier: set[int] = set()
        for position in frontier:
            for consumer in edges.get(position, ()):
                if consumer not in downstream:
                    downstream.add(consumer)
                    next_frontier.add(consumer)
        frontier = next_frontier
    return downstream


def _build_dependency_edges(
    steps: list[tuple[int, StepSpec, StepDefinition]],
) -> dict[int, set[int]]:
    """producer position -> set of consumer positions, from both explicit
    context-key bindings and declared session resources."""

    edges: dict[int, set[int]] = {}

    def add_edge(producer: int, consumer: int) -> None:
        edges.setdefault(producer, set()).add(consumer)

    # Explicit context keys: each read binds to the most recent preceding
    # writer of the same context key (mirrors engine/diagnostics.py).
    produced_at: dict[str, int] = {}
    for position, step_spec, definition in steps:
        for param, default in definition.reads.items():
            context_key = step_spec.inputs.get(param, default)
            if context_key is not None and context_key in produced_at:
                add_edge(produced_at[context_key], position)
        for context_key in definition.requires:
            if context_key in produced_at:
                add_edge(produced_at[context_key], position)
        for name in definition.writes:
            context_key = step_spec.outputs.get(name, name)
            produced_at[context_key] = position

    # Session resources: each read binds to the most recent preceding
    # writer whose resource matches (see resources_match()).
    for position, _step_spec, definition in steps:
        for read_resource in definition.session_reads:
            writer_position = _most_recent_matching_writer_before(
                steps, position, read_resource
            )
            if writer_position is not None:
                add_edge(writer_position, position)

    return edges


def _most_recent_matching_writer_before(
    steps: list[tuple[int, StepSpec, StepDefinition]],
    position: int,
    read_resource: str,
) -> int | None:
    best: int | None = None
    for writer_position, _writer_step_spec, writer_definition in steps:
        if writer_position >= position:
            break
        for write_resource in writer_definition.session_writes:
            if resources_match(write_resource, read_resource):
                best = writer_position
    return best
