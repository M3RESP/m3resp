"""Tests for `m3resp.workflows.graph`: the spec-to-graph, graph-to-spec
conversion layer.

The round-trip property test against every example spec is the honest
go/no-go test for this module: if `graph_to_spec(spec_to_graph(s))` does
not compile to the same result as `s`, the graph model is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.core.exceptions import UnknownStepError
from m3resp.workflows.compiler import compile_pipeline
from m3resp.workflows.graph import GraphNode, graph_to_spec, spec_to_graph
from m3resp.workflows.registry import STEP_REGISTRY, register_step
from m3resp.workflows.spec import load_spec

EXAMPLE_SPECS = sorted(Path("examples").glob("**/*.pipeline.yaml"))


# --------------------------------------------------------------------------- #
# The round-trip property test                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec_path", EXAMPLE_SPECS, ids=lambda p: p.name)
def test_round_trip_compiles_identically_for_every_example_spec(spec_path: Path):
    original = load_spec(spec_path)
    graph = spec_to_graph(original)
    rebuilt = graph_to_spec(graph)

    assert compile_pipeline(rebuilt).as_dict() == compile_pipeline(original).as_dict()


def test_at_least_one_example_spec_was_found():
    """Guards against the parametrized test above silently collecting zero
    cases if the glob pattern or example layout ever changes."""

    assert len(EXAMPLE_SPECS) >= 5


# --------------------------------------------------------------------------- #
# Fixture steps for focused, non-example-file tests                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def _graph_test_steps():
    @register_step(
        "graph_test.produce",
        writes=("value",),
        parameters_reviewed=True,
    )
    def _produce() -> dict[str, Any]:
        return {"value": 1}

    @register_step(
        "graph_test.consume",
        reads={"value": "value"},
        writes=("doubled",),
        parameters_reviewed=True,
    )
    def _consume(value: Any) -> dict[str, Any]:
        return {"doubled": value * 2}

    @register_step(
        "graph_test.session_writer",
        session_writes=("session.thing",),
        parameters_reviewed=True,
    )
    def _session_writer() -> dict[str, Any]:
        return {}

    @register_step(
        "graph_test.session_reader",
        session_reads=("session.thing",),
        parameters_reviewed=True,
    )
    def _session_reader() -> dict[str, Any]:
        return {}

    yield
    for name in (
        "graph_test.produce",
        "graph_test.consume",
        "graph_test.session_writer",
        "graph_test.session_reader",
    ):
        STEP_REGISTRY.pop(name, None)


# --------------------------------------------------------------------------- #
# spec_to_graph: nodes                                                        #
# --------------------------------------------------------------------------- #


def test_spec_to_graph_builds_one_node_per_step_in_order(_graph_test_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "graph_test.produce", "id": "a"},
                {
                    "uses": "graph_test.consume",
                    "id": "b",
                    "with": {"unused": 1},
                },
            ],
        }
    )
    graph = spec_to_graph(spec)
    assert [node.id for node in graph.nodes] == ["a", "b"]
    assert [node.position for node in graph.nodes] == [0, 1]
    assert graph.nodes[0].operation_id == "graph_test.produce"
    assert graph.nodes[1].parameters == {"unused": 1}


def test_spec_to_graph_raises_on_unknown_step():
    spec = load_spec({"name": "p", "steps": [{"uses": "no.such.step"}]})
    with pytest.raises(UnknownStepError):
        spec_to_graph(spec)


def test_spec_to_graph_copies_node_ui_from_spec_metadata(_graph_test_steps):
    spec = load_spec(
        {
            "name": "p",
            "metadata": {"ui": {"nodes": {"a": {"x": 10, "y": 20}}}},
            "steps": [{"uses": "graph_test.produce", "id": "a"}],
        }
    )
    graph = spec_to_graph(spec)
    assert graph.nodes[0].ui == {"x": 10, "y": 20}
    assert "ui" not in graph.metadata


def test_spec_to_graph_node_ui_is_empty_when_no_metadata_ui_block(_graph_test_steps):
    spec = load_spec(
        {"name": "p", "steps": [{"uses": "graph_test.produce", "id": "a"}]}
    )
    graph = spec_to_graph(spec)
    assert graph.nodes[0].ui == {}


# --------------------------------------------------------------------------- #
# spec_to_graph: edges                                                        #
# --------------------------------------------------------------------------- #


def test_spec_to_graph_builds_an_artifact_edge_for_a_context_key_binding(
    _graph_test_steps,
):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "graph_test.produce", "id": "a"},
                {"uses": "graph_test.consume", "id": "b"},
            ],
        }
    )
    graph = spec_to_graph(spec)
    artifact_edges = [e for e in graph.edges if e.kind == "artifact"]
    assert len(artifact_edges) == 1
    edge = artifact_edges[0]
    assert edge.source_node == "a"
    assert edge.source_handle == "value"
    assert edge.target_node == "b"
    assert edge.target_handle == "value"
    assert edge.context_key == "value"


def test_spec_to_graph_suppresses_the_session_context_key(_graph_test_steps):
    @register_step(
        "graph_test.takes_session",
        reads={"session": "session"},
        parameters_reviewed=True,
    )
    def _takes_session(session: Any) -> dict[str, Any]:
        return {}

    try:
        spec = load_spec({"name": "p", "steps": [{"uses": "graph_test.takes_session"}]})
        graph = spec_to_graph(spec)
        assert graph.edges == ()
    finally:
        STEP_REGISTRY.pop("graph_test.takes_session", None)


def test_spec_to_graph_builds_a_spec_input_edge_for_a_declared_pipeline_input(
    _graph_test_steps,
):
    spec = load_spec(
        {
            "name": "p",
            "inputs": {"value": 42},
            "steps": [{"uses": "graph_test.consume", "id": "b"}],
        }
    )
    graph = spec_to_graph(spec)
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.kind == "spec_input"
    assert edge.source_node == "spec_input:value"
    assert edge.target_node == "b"
    assert edge.context_key == "value"


def test_spec_to_graph_builds_a_session_edge_from_session_reads_writes(
    _graph_test_steps,
):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "graph_test.session_writer", "id": "w"},
                {"uses": "graph_test.session_reader", "id": "r"},
            ],
        }
    )
    graph = spec_to_graph(spec)
    session_edges = [e for e in graph.edges if e.kind == "session"]
    assert len(session_edges) == 1
    edge = session_edges[0]
    assert edge.source_node == "w"
    assert edge.target_node == "r"
    assert edge.context_key == "session.thing"


def test_spec_to_graph_omits_an_unresolved_context_key_edge(_graph_test_steps):
    """A read whose context key comes from neither a producing step nor a
    declared pipeline input (e.g. seeded externally at run time) has
    nothing in this spec to draw an edge from."""

    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {
                    "uses": "graph_test.consume",
                    "id": "b",
                    "in": {"value": "externally_seeded"},
                }
            ],
        }
    )
    graph = spec_to_graph(spec)
    assert graph.edges == ()


# --------------------------------------------------------------------------- #
# graph_to_spec                                                               #
# --------------------------------------------------------------------------- #


def test_graph_to_spec_orders_steps_by_node_position(_graph_test_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "graph_test.produce", "id": "a"},
                {"uses": "graph_test.consume", "id": "b"},
            ],
        }
    )
    graph = spec_to_graph(spec)
    # Reversing node order in the graph must not change the spec's order -
    # position is authoritative, not list order.
    shuffled = graph.__class__(
        **{**graph.__dict__, "nodes": tuple(reversed(graph.nodes))}
    )
    rebuilt = graph_to_spec(shuffled)
    assert [s.id for s in rebuilt.steps] == ["a", "b"]


def test_graph_to_spec_writes_back_node_ui_into_metadata(_graph_test_steps):
    spec = load_spec(
        {"name": "p", "steps": [{"uses": "graph_test.produce", "id": "a"}]}
    )
    graph = spec_to_graph(spec)
    moved = graph.__class__(
        **{
            **graph.__dict__,
            "nodes": (
                GraphNode(
                    id="a",
                    operation_id="graph_test.produce",
                    position=0,
                    ui={"x": 5, "y": 7},
                ),
            ),
        }
    )
    rebuilt = graph_to_spec(moved)
    assert rebuilt.metadata["ui"]["nodes"]["a"] == {"x": 5, "y": 7}


def test_graph_to_spec_omits_metadata_ui_when_no_node_has_ui(_graph_test_steps):
    spec = load_spec(
        {"name": "p", "steps": [{"uses": "graph_test.produce", "id": "a"}]}
    )
    rebuilt = graph_to_spec(spec_to_graph(spec))
    assert "ui" not in rebuilt.metadata
