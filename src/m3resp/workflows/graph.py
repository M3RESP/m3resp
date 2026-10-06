"""Convert workflow specs to and from step-and-connection graphs.

Nodes store the raw step bindings and parameters used to reconstruct a spec.
Edges describe connections from the most recent preceding output or matching
session writer. Graph edits should update node bindings before recomputing
edges. Session dependencies use dotted resource names. Layout settings are
stored in metadata.ui.nodes.
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
    SpecExecutionConfig,
    SpecExperimentConfig,
    SpecOutputsConfig,
    StepSpec,
    WorkflowSpec,
)

#: One connection point's role in a spec-level ``GraphEdge``:
#: ``"artifact"`` - a normal step-to-step context-key binding.
#: ``"session"`` - a hidden dependency through the shared ``M3Session``,
#: declared via a step's ``session_reads``/``session_writes`` metadata.
#: ``"spec_input"`` - the value comes from the spec's own ``inputs:``
#: section, not from another step.
EdgeKind = Literal["artifact", "session", "spec_input"]

#: Prefix for the synthetic ``source_node`` id of a ``"spec_input"`` edge,
#: since a declared workflow input has no corresponding ``GraphNode``.
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
    """A connection between workflow inputs and outputs.

    Attributes:
        source_node: Producing step's ID, or ``"spec_input:<name>"`` for a
            declared workflow input.
        source_handle: Producing output name, input name, or written session
            resource for a session edge.
        target_node: Consuming step's ID.
        target_handle: Consuming parameter or read session resource.
        context_key: Shared value name for artifact/spec-input edges, or the
            read resource's dotted name for a session edge.
        kind: artifact, session or spec_input.
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
class WorkflowGraph:
    """Workflow steps and their connections, with the spec's settings.

    Nodes hold raw inputs, parameters and outputs; graph_to_spec rebuilds
    steps from those fields in position order. Edges describe connections for
    display. Per-node layout settings come from metadata.ui.nodes; remaining
    metadata and workflow settings are retained.
    """

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
        """Return workflow settings, nodes and edges as a dictionary.

        Root and output-directory paths become strings. Input, metadata and
        parameter values are retained as supplied.
        """

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
    spec: WorkflowSpec,
) -> list[tuple[int, StepSpec, StepDefinition]]:
    """Pair each spec step with its position and registered definition.

    Raises UnknownStepError when a step operation is unregistered.
    """

    return [
        (position, step_spec, get_step(step_spec.uses))
        for position, step_spec in enumerate(spec.steps)
    ]


def spec_to_graph(spec: WorkflowSpec) -> WorkflowGraph:
    """Describe workflow steps and their connections as a graph.

    Args:
        spec: Parsed workflow with registered operations and raw step bindings.

    Returns:
        WorkflowGraph: Nodes in spec order with their inputs, parameters,
            outputs and layout settings. Edges link artifact values to the
            latest preceding producer, declared workflow inputs to consumers,
            and session resources to the latest matching writer. The shared
            session context key is omitted from artifact edges.

    Raises:
        UnknownStepError: If a step names an unregistered operation.
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

    return WorkflowGraph(
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


def graph_to_spec(graph: WorkflowGraph) -> WorkflowSpec:
    """Rebuild a workflow spec from its graph nodes and settings.

    Args:
        graph: Graph whose node fields contain the desired step bindings.

    Returns:
        WorkflowSpec: Steps sorted by node position, reconstructed from each
            node's inputs, parameters and outputs. Layout settings are copied
            to metadata.ui.nodes. Changing edges alone does not change the spec.
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

    return WorkflowSpec(
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
    spec: WorkflowSpec, steps: list[tuple[int, StepSpec, StepDefinition]]
) -> tuple[GraphEdge, ...]:
    """Describe preceding output, declared input and session-resource connections."""

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
