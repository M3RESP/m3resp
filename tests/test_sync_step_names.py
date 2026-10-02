"""Every synchronization step is a `sync.*` step, registered from
`m3resp.workflows.steps.sync`. `sync.raw_modalities` keeps its old name
`session.sync_raw` as a silent alias, so existing pipeline specs run
unchanged.
"""

from __future__ import annotations

import pytest

import m3resp.workflows.steps  # noqa: F401 - registers the built-in steps
from m3resp.workflows.registry import available_steps, describe_steps, get_step

RENAMED = {"session.sync_raw": "sync.raw_modalities"}


class TestOldNamesStillWork:
    @pytest.mark.parametrize(("old", "new"), sorted(RENAMED.items()))
    def test_the_old_name_finds_the_new_step(self, old, new):
        assert get_step(old) is get_step(new)
        assert get_step(old).name == new

    def test_a_spec_with_the_old_names_compiles_to_the_new_ones(self):
        from m3resp.workflows.compiler import compile_pipeline
        from m3resp.workflows.spec import load_spec

        spec = {
            "name": "old-sync-names",
            "steps": [{"uses": "session.sync_raw"}, {"uses": "sync.skip"}],
        }

        compiled = compile_pipeline(load_spec(spec))

        assert [step.operation_id for step in compiled.steps] == [
            "sync.raw_modalities",
            "sync.skip",
        ]


class TestOnlyTheNewNamesAreListed:
    def test_available_steps(self):
        names = set(available_steps())
        assert {"sync.raw_modalities", "sync.skip"} <= names
        assert not names & {"session.sync_raw", "session.skip_sync"}

    def test_every_synchronization_step_is_a_sync_step(self):
        names = {d.name for d in describe_steps() if d.category == "synchronization"}
        assert names == {
            "sync.apply_estimated_offset",
            "sync.estimate_offset",
            "sync.raw_modalities",
            "sync.skip",
        }

    def test_no_session_steps_are_left(self):
        assert describe_steps(prefix="session.") == []
