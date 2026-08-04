"""Convert a :class:`PipelineSpec` to and from a node-and-edge graph.

This is the backend piece a node-based workflow editor (see the "Node-based
workflow design panel" section of ``docs/stage3.md``) needs before any
front end can exist: a pipeline spec is already a data-flow graph written
in list form, so what's missing is a pure, JSON-safe conversion layer, not
a new data model.

Two representations, one job each:

- :class:`GraphNode` carries the *authoritative* ``in:``/``with:``/``out:``
  bindings for its step, copied verbatim from :class:`StepSpec`. This is
  what makes the round trip correct: ``graph_to_spec(spec_to_graph(s))``
  reconstructs each step from its node's own bindings, never by trying to
  re-derive them from edges (which would lose information for an output
  that is renamed but never consumed - a real, if rare, case).
- :class:`GraphEdge` is a *derived* connections view for a canvas: which
  node's which output feeds which node's which input, and through which
  context key. It answers "may I draw a line here", not "what should I
  write to the spec" - editing an edge alone does not change a node's
  bindings; a future editor must write to ``GraphNode.inputs``/``outputs``
  when the user rewires a connection, then the edges can be recomputed by
  calling :func:`spec_to_graph` again (or an equivalent incremental update)
  rather than hand-patched, so they can never drift from what
  ``engine/diagnostics.py`` would itself compute.

Edges are built with :func:`m3resp.workflows.session_deps.iter_context_key_producers`
and :func:`m3resp.workflows.session_deps.most_recent_matching_session_writer`
- the same "most recent preceding writer, positionally" rule
``engine/diagnostics.py`` and ``session_deps.py`` already use - so the drawn
graph can never disagree with validation (the trap ``docs/stage3.md``'s
node-based-UI outlook calls out: two steps writing the same context key at
different points in the run must resolve to the correct producer, not
"whichever step shares the name").

``session`` is deliberately never drawn as a node with dozens of edges (the
"hairball" the outlook warns about): a read/write of the literal ``session``
context key is suppressed entirely, and the genuinely meaningful hidden
dependencies show up instead as ``kind="session"`` edges, built from each
step's declared ``session_reads``/``session_writes`` metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from m3resp.workflows.context import SESSION_KEY
from m3resp.workflows.registry import StepDefinition, get_step
from m3resp.workflows.session_deps import (
    iter_context_key_producers,
    most_recent_matching_session_writer,
)
from m3resp.workflows.spec import (
    PipelineSpec,
    SpecExecutionConfig,
    SpecExperimentConfig,
    SpecOutputsConfig,
    StepSpec,
)

#: One connection point's role in a spec-level ``GraphEdge``:
#: ``"artifact"`` - a normal step-to-step context-key binding.
#: ``"session"`` - a hidden dependency through the shared ``M3Session``,
#: declared via a step's ``session_reads``/``session_writes`` metadata.
#: ``"spec_input"`` - the value comes from the spec's own ``inputs:``
#: section, not from another step.
EdgeKind = Literal["artifact", "session", "spec_input"]

#: Prefix for the synthetic ``source_node`` id of a ``"spec_input"`` edge,
#: since a declared pipeline input has no corresponding ``GraphNode``.
_SPEC_INPUT_NODE_PREFIX = "spec_input:"

#: Where per-node canvas UI state (e.g. ``{"x": 120, "y": 40}``) round-trips
#: through a spec's free-form ``metadata`` block. Canvas coordinates belong
#: here, not in a new spec-level field, so that a hand-written spec with no
#: such block still opens (falling back to automatic layout). Never read by
#: the engine itself.
_UI_METADATA_KEY = "ui"
_UI_NODES_KEY = "nodes"


@dataclass(frozen=True)
class GraphNode:
    """One step, as a node. ``inputs``/``parameters``/``outputs`` are the
    step's raw ``in:``/``with:``/``out:`` bindings, copied verbatim from its
    ``StepSpec`` - the source of truth for round-tripping, not derived from
    edges."""

    id: str
    operation_id: str
    position: int
    inputs: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    #: Opaque canvas state (position, collapsed, ...). Round-tripped through
    #: the spec's ``metadata.ui.nodes`` block; never interpreted here.
    ui: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation_id": self.operation_id,
            "position": self.position,
            "inputs": dict(self.inputs),
            "parameters": dict(self.parameters),
            "outputs": dict(self.outputs),
            "ui": dict(self.ui),
        }


@dataclass(frozen=True)
class GraphEdge:
    """One connection: ``source_node``'s ``source_handle`` output feeds
    ``target_node``'s ``target_handle`` input, through ``context_key``.

    For a ``"session"`` edge there is no real context key - ``context_key``
    holds the matched dotted resource name instead (e.g.
    ``"session.processed.emg"``), and ``source_handle``/``target_handle``
    hold the writer's/reader's declared resource strings (which may differ
    in granularity - see
    :func:`m3resp.workflows.session_deps.resources_match`).

    For a ``"spec_input"`` edge, ``source_node`` is a synthetic id
    (``"spec_input:<name>"``), since a declared pipeline input has no
    ``GraphNode`` of its own.
    """

    source_node: str
    source_handle: str
    target_node: str
    target_handle: str
    context_key: str
    kind: EdgeKind

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_node": self.source_node,
            "source_handle": self.source_handle,
            "target_node": self.target_node,
            "target_handle": self.target_handle,
            "context_key": self.context_key,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class PipelineGraph:
    """A :class:`PipelineSpec`'s node-and-edge form. Every field except
    ``nodes``/``edges`` is a direct pass-through of the matching
    ``PipelineSpec`` field - see the module docstring for why nodes, not
    edges, are what ``graph_to_spec`` trusts to reconstruct steps."""

    name: str
    schema_version: int | None
    description: str
    inputs: dict[str, Any]
    metadata: dict[str, Any]
    execution: SpecExecutionConfig
    outputs: SpecOutputsConfig
    experiment: SpecExperimentConfig
    root: Path
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "description": self.description,
            "inputs": dict(self.inputs),
            "metadata": dict(self.metadata),
            "execution": {
                "error_policy": self.execution.error_policy,
                "seed": self.execution.seed,
            },
            "outputs": {
                "dir": str(self.outputs.dir) if self.outputs.dir is not None else None,
                "mode": self.outputs.mode,
                "timestamped": self.outputs.timestamped,
                "summary_json": self.outputs.summary_json,
                "event_csvs": self.outputs.event_csvs,
                "parameters_csv": self.outputs.parameters_csv,
                "postprocessing": self.outputs.postprocessing,
                "structured_export": self.outputs.structured_export,
                "figures": self.outputs.figures,
                "checksums": self.outputs.checksums,
            },
            "experiment": {
                "subject_id": self.experiment.subject_id,
                "mode": self.experiment.mode,
                "timepoint": self.experiment.timepoint,
                "run_identifier": self.experiment.run_identifier,
                "selection": self.experiment.selection,
            },
            "root": str(self.root),
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
        }


def _resolve_steps_or_raise(
    spec: PipelineSpec,
) -> list[tuple[int, StepSpec, StepDefinition]]:
    """Like ``session_deps.resolve_step_definitions``, but raises on an
    unregistered operation instead of skipping it - a graph silently missing
    a node would be worse than an error, unlike diagnostics collection
    (which wants to report every *other* problem too) or the session-
    dependency checks (which are best-effort by nature)."""

    return [
        (position, step_spec, get_step(step_spec.uses))
        for position, step_spec in enumerate(spec.steps)
    ]


def spec_to_graph(spec: PipelineSpec) -> PipelineGraph:
    """Convert ``spec`` into its node-and-edge graph form.

    Raises whatever ``get_step`` raises (``UnknownStepError``) for a step
    naming an unregistered operation - like ``compile_pipeline``, this
    assumes a spec worth graphing is already structurally resolvable;
    call ``collect_diagnostics`` first to report every problem in a spec
    that is not.
    """

    ui_block = spec.metadata.get(_UI_METADATA_KEY)
    nodes_ui = (
        ui_block.get(_UI_NODES_KEY, {})
        if isinstance(ui_block, dict) and isinstance(ui_block.get(_UI_NODES_KEY), dict)
        else {}
    )
    metadata = {
        key: value for key, value in spec.metadata.items() if key != _UI_METADATA_KEY
    }

    steps = _resolve_steps_or_raise(spec)

    nodes = tuple(
        GraphNode(
            id=step_spec.id,
            operation_id=step_spec.uses,
            position=position,
            inputs=dict(step_spec.inputs),
            parameters=dict(step_spec.params),
            outputs=dict(step_spec.outputs),
            ui=dict(nodes_ui.get(step_spec.id, {})),
        )
        for position, step_spec, _definition in steps
    )
    edges = _build_edges(spec, steps)

    return PipelineGraph(
        name=spec.name,
        schema_version=spec.schema_version,
        description=spec.description,
        inputs=dict(spec.inputs),
        metadata=metadata,
        execution=spec.execution,
        outputs=spec.outputs,
        experiment=spec.experiment,
        root=spec.root,
        nodes=nodes,
        edges=edges,
    )


def graph_to_spec(graph: PipelineGraph) -> PipelineSpec:
    """The inverse of :func:`spec_to_graph`.

    Rebuilds each step purely from its node's own ``inputs``/``parameters``/
    ``outputs`` - never from ``graph.edges``, which are a derived rendering
    aid, not load-bearing here. Nodes are ordered by ``position``.
    """

    ordered_nodes = sorted(graph.nodes, key=lambda node: node.position)
    steps = tuple(
        StepSpec(
            uses=node.operation_id,
            id=node.id,
            inputs=dict(node.inputs),
            params=dict(node.parameters),
            outputs=dict(node.outputs),
        )
        for node in ordered_nodes
    )

    nodes_ui = {node.id: dict(node.ui) for node in graph.nodes if node.ui}
    metadata = dict(graph.metadata)
    if nodes_ui:
        metadata[_UI_METADATA_KEY] = {_UI_NODES_KEY: nodes_ui}

    return PipelineSpec(
        name=graph.name,
        schema_version=graph.schema_version,
        description=graph.description,
        inputs=dict(graph.inputs),
        metadata=metadata,
        execution=graph.execution,
        steps=steps,
        outputs=graph.outputs,
        experiment=graph.experiment,
        root=graph.root,
    )


def _build_edges(
    spec: PipelineSpec, steps: list[tuple[int, StepSpec, StepDefinition]]
) -> tuple[GraphEdge, ...]:
    edges: list[GraphEdge] = []

    for position, step_spec, definition, produced_at in iter_context_key_producers(
        steps
    ):
        for param, default in definition.reads.items():
            context_key = step_spec.inputs.get(param, default)
            if context_key is None or context_key == SESSION_KEY:
                continue
            producer = produced_at.get(context_key)
            if producer is not None:
                _producer_position, producer_step_spec, producer_output_name = producer
                edges.append(
                    GraphEdge(
                        source_node=producer_step_spec.id,
                        source_handle=producer_output_name,
                        target_node=step_spec.id,
                        target_handle=param,
                        context_key=context_key,
                        kind="artifact",
                    )
                )
            elif context_key in spec.inputs:
                edges.append(
                    GraphEdge(
                        source_node=f"{_SPEC_INPUT_NODE_PREFIX}{context_key}",
                        source_handle=context_key,
                        target_node=step_spec.id,
                        target_handle=param,
                        context_key=context_key,
                        kind="spec_input",
                    )
                )
            # Otherwise the key comes from engine-seeded/external state
            # (session, run plumbing, or a value supplied outside this
            # spec) - nothing in this spec to draw an edge from.

        for read_resource in definition.session_reads:
            writer = most_recent_matching_session_writer(steps, position, read_resource)
            if writer is None:
                continue
            _writer_position, writer_step_spec, write_resource = writer
            edges.append(
                GraphEdge(
                    source_node=writer_step_spec.id,
                    source_handle=write_resource,
                    target_node=step_spec.id,
                    target_handle=read_resource,
                    context_key=read_resource,
                    kind="session",
                )
            )

    return tuple(edges)
