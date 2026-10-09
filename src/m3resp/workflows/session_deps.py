"""Track dependencies through workflow values and shared session resources.

Declared inputs use the most recent preceding output with the same context
key. Declared session reads use the most recent preceding matching resource
writer. These relationships identify possible ordering conflicts and steps
that depend on a changed result.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from m3resp.core.exceptions import UnknownStepError
from m3resp.workflows.registry import StepDefinition, get_step
from m3resp.workflows.spec import StepSpec, WorkflowSpec

#: One resolved context-key producer: the position, ``StepSpec``, and
#: natural output ``name`` (from ``StepDefinition.writes``) of the step
#: that most recently wrote it.
ContextKeyProducer = tuple[int, StepSpec, str]


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


def resolve_step_definitions(
    spec: WorkflowSpec,
) -> list[tuple[int, StepSpec, StepDefinition]]:
    """Return (position, spec step, registered definition) in workflow order.

    Unregistered operations are skipped; structural diagnostics report them.
    """

    resolved: list[tuple[int, StepSpec, StepDefinition]] = []
    for position, step_spec in enumerate(spec.steps):
        try:
            definition = get_step(step_spec.uses)
        except UnknownStepError:
            continue
        resolved.append((position, step_spec, definition))
    return resolved


def iter_context_key_producers(
    steps: list[tuple[int, StepSpec, StepDefinition]],
) -> Iterator[tuple[int, StepSpec, StepDefinition, dict[str, ContextKeyProducer]]]:
    """Yield each resolved step alongside the context-key producer map as it
    stands immediately *before* that step runs - i.e. incorporating every
    earlier step's ``writes``, not this one's.

    This is the "most recent preceding writer" rule ``engine/diagnostics.py``
    already applies for context keys, factored out as a reusable primitive so
    other producer-tracking consumers (this module's own
    :func:`_build_dependency_edges`, and ``m3resp.workflows.graph``) can't
    silently drift from it.
    """

    produced_at: dict[str, ContextKeyProducer] = {}
    for position, step_spec, definition in steps:
        yield position, step_spec, definition, produced_at
        for name in definition.writes:
            context_key = step_spec.outputs.get(name, name)
            produced_at[context_key] = (position, step_spec, name)


def most_recent_matching_session_writer(
    steps: list[tuple[int, StepSpec, StepDefinition]],
    position: int,
    read_resource: str,
) -> tuple[int, StepSpec, str] | None:
    """The latest step before ``position`` whose declared ``session_writes``
    matches ``read_resource`` (see :func:`resources_match`), as
    ``(position, step_spec, matched_write_resource)``, or ``None``."""

    best: tuple[int, StepSpec, str] | None = None
    for writer_position, writer_step_spec, writer_definition in steps:
        if writer_position >= position:
            break
        for write_resource in writer_definition.session_writes:
            if resources_match(write_resource, read_resource):
                best = (writer_position, writer_step_spec, write_resource)
    return best


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
    spec: WorkflowSpec,
) -> list[SessionDependencyConflict]:
    """Find reads satisfied by a later session writer but no earlier writer.

    Args:
        spec: Ordered workflow with declared session reads and writes.

    Returns:
        list[SessionDependencyConflict]: Reader and first later matching writer
            for each potentially reordered dependency. Resources with no writer
            anywhere in the workflow are skipped because they may be supplied
            on the session by the caller.
    """

    steps = resolve_step_definitions(spec)
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


def downstream_step_positions(spec: WorkflowSpec, start_position: int) -> set[int]:
    """Find all steps that depend on a given step's outputs or session changes.

    Args:
        spec: Ordered workflow with registered input/output and session metadata.
        start_position: Zero-based position of the step being changed or rerun.

    Returns:
        set[int]: Zero-based positions of direct and indirect consumers. Reads
            use the most recent preceding output or matching session writer.
            Unregistered operations are skipped.
    """

    steps = resolve_step_definitions(spec)
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
    for position, step_spec, definition, produced_at in iter_context_key_producers(
        steps
    ):
        for param, default in definition.reads.items():
            context_key = step_spec.inputs.get(param, default)
            if context_key is not None and context_key in produced_at:
                add_edge(produced_at[context_key][0], position)
        for context_key in definition.requires:
            if context_key in produced_at:
                add_edge(produced_at[context_key][0], position)

    # Session resources: each read binds to the most recent preceding
    # writer whose resource matches (see resources_match()).
    for position, _step_spec, definition in steps:
        for read_resource in definition.session_reads:
            writer = most_recent_matching_session_writer(steps, position, read_resource)
            if writer is not None:
                add_edge(writer[0], position)

    return edges
