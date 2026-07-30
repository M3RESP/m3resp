"""Tests for `m3resp.workflows.session_deps`: the session-resource
dependency tracking that extends `engine/diagnostics.py`'s positional
context-key producer tracking onto the shared `M3Session` object (see
`plan/06_gui_readiness_plan.md` §4).
"""

from __future__ import annotations

from typing import Any

import pytest

from m3resp.workflows.engine import collect_diagnostics
from m3resp.workflows.registry import STEP_REGISTRY, register_step
from m3resp.workflows.session_deps import (
    downstream_step_positions,
    find_session_dependency_conflicts,
    resources_match,
)
from m3resp.workflows.spec import load_spec

# --------------------------------------------------------------------------- #
# resources_match                                                             #
# --------------------------------------------------------------------------- #


def test_resources_match_exact():
    assert resources_match("session.raw.eit", "session.raw.eit")


def test_resources_match_coarse_writer_covers_specific_reader():
    assert resources_match("session.raw", "session.raw.eit")


def test_resources_match_specific_writer_covers_coarse_reader():
    assert resources_match("session.raw.eit", "session.raw")


def test_resources_match_unrelated_resources_do_not_match():
    assert not resources_match("session.raw.eit", "session.raw.emg")


def test_resources_match_does_not_match_on_shared_prefix_alone():
    assert not resources_match("session.rawish", "session.raw")


# --------------------------------------------------------------------------- #
# find_session_dependency_conflicts                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def _session_dep_steps():
    @register_step(
        "sdeps_test.writer",
        writes=("token",),
        session_writes=("session.thing",),
    )
    def _writer() -> dict[str, Any]:
        return {"token": 1}

    @register_step(
        "sdeps_test.reader",
        session_reads=("session.thing",),
    )
    def _reader() -> dict[str, Any]:
        return {}

    @register_step(
        "sdeps_test.unrelated",
    )
    def _unrelated() -> dict[str, Any]:
        return {}

    yield
    STEP_REGISTRY.pop("sdeps_test.writer", None)
    STEP_REGISTRY.pop("sdeps_test.reader", None)
    STEP_REGISTRY.pop("sdeps_test.unrelated", None)


def test_no_conflict_when_writer_runs_before_reader(_session_dep_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "sdeps_test.writer"},
                {"uses": "sdeps_test.reader"},
            ],
        }
    )
    assert find_session_dependency_conflicts(spec) == []


def test_conflict_when_reader_runs_before_writer(_session_dep_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "sdeps_test.reader"},
                {"uses": "sdeps_test.writer"},
            ],
        }
    )
    conflicts = find_session_dependency_conflicts(spec)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.resource == "session.thing"
    assert conflict.reader_position == 0
    assert conflict.reader_uses == "sdeps_test.reader"
    assert conflict.writer_position == 1
    assert conflict.writer_uses == "sdeps_test.writer"


def test_no_conflict_when_resource_never_written_in_spec(_session_dep_steps):
    """A read resource with no writer anywhere in the spec may come from
    state supplied outside the spec - not a bug this check can see."""

    spec = load_spec({"name": "p", "steps": [{"uses": "sdeps_test.reader"}]})
    assert find_session_dependency_conflicts(spec) == []


def test_no_conflict_for_steps_with_no_session_dependency(_session_dep_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "sdeps_test.unrelated"},
                {"uses": "sdeps_test.unrelated"},
            ],
        }
    )
    assert find_session_dependency_conflicts(spec) == []


def test_collect_diagnostics_surfaces_session_dependency_conflict_as_warning(
    _session_dep_steps,
):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "sdeps_test.reader"},
                {"uses": "sdeps_test.writer"},
            ],
        }
    )
    diagnostics = [
        d for d in collect_diagnostics(spec) if d.code == "session_dependency_reordered"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == "warning"
    assert diagnostics[0].step_position == 0


# --------------------------------------------------------------------------- #
# downstream_step_positions                                                   #
# --------------------------------------------------------------------------- #


def test_downstream_step_positions_follows_explicit_context_key_edges():
    @register_step("sdeps_test.produce", writes=("value",))
    def _produce() -> dict[str, Any]:
        return {"value": 1}

    @register_step("sdeps_test.consume", reads={"value": "value"}, writes=("doubled",))
    def _consume(value: Any) -> dict[str, Any]:
        return {"doubled": value * 2}

    try:
        spec = load_spec(
            {
                "name": "p",
                "steps": [
                    {"uses": "sdeps_test.produce"},
                    {"uses": "sdeps_test.consume"},
                    {"uses": "sdeps_test.produce"},  # unrelated, position 2
                ],
            }
        )
        assert downstream_step_positions(spec, 0) == {1}
        assert downstream_step_positions(spec, 1) == set()
        assert downstream_step_positions(spec, 2) == set()
    finally:
        STEP_REGISTRY.pop("sdeps_test.produce", None)
        STEP_REGISTRY.pop("sdeps_test.consume", None)


def test_downstream_step_positions_follows_session_resource_edges_transitively(
    _session_dep_steps,
):
    @register_step(
        "sdeps_test.relay",
        session_reads=("session.thing",),
        session_writes=("session.other",),
    )
    def _relay() -> dict[str, Any]:
        return {}

    @register_step("sdeps_test.final", session_reads=("session.other",))
    def _final() -> dict[str, Any]:
        return {}

    try:
        spec = load_spec(
            {
                "name": "p",
                "steps": [
                    {"uses": "sdeps_test.writer"},  # position 0
                    {"uses": "sdeps_test.relay"},  # position 1
                    {"uses": "sdeps_test.final"},  # position 2
                ],
            }
        )
        assert downstream_step_positions(spec, 0) == {1, 2}
        assert downstream_step_positions(spec, 1) == {2}
        assert downstream_step_positions(spec, 2) == set()
    finally:
        STEP_REGISTRY.pop("sdeps_test.relay", None)
        STEP_REGISTRY.pop("sdeps_test.final", None)


def test_downstream_step_positions_skips_unknown_steps():
    spec = load_spec({"name": "p", "steps": [{"uses": "no.such.step"}]})
    assert downstream_step_positions(spec, 0) == set()
