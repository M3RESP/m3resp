"""`export_store` checks the store before writing it (#75).

A store that fails the checks is not written at all: `DataModelValidationError`
lists every problem. `validate=False` writes it anyway; `require_complete=True`
also runs the completeness checks.
"""

from __future__ import annotations

import os

import pytest

from m3resp.adapters import EITProcessingAdapter
from m3resp.core.session import M3Session
from m3resp.datamodel import (
    DataModelRecorder,
    DataModelStore,
    DataModelValidationError,
    QualityAnnotation,
    export_store,
    validate_store,
)


def _broken_store() -> DataModelStore:
    """A store with a quality annotation about a signal it does not hold."""

    store = DataModelStore()
    annotation = QualityAnnotation(
        target_type="signal", target_id="missing-signal", quality_label="invalid"
    )
    # Added directly, the way an imported store could be, bypassing the
    # checks `add_quality_annotation` does.
    store.quality_annotations[annotation.quality_annotation_id] = annotation
    return store


def _recorded_store() -> DataModelStore:
    """A store recorded from a session: its records link up, but it has no
    wall-clock start times or file checksums yet."""

    session = M3Session(
        eit_adapter=EITProcessingAdapter(loader=lambda path, **kwargs: {"path": path})
    )
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)
    session.load_eit("subject.bin")
    return store


class TestAStoreThatFailsIsNotWritten:
    def test_raises_and_writes_nothing(self, tmp_path):
        store = _broken_store()
        out_dir = os.path.join(tmp_path, "datamodel")

        with pytest.raises(DataModelValidationError) as caught:
            export_store(store, out_dir)

        assert not os.path.exists(out_dir)
        assert caught.value.problems == validate_store(store)

    def test_the_message_lists_every_problem_and_the_way_out(self):
        with pytest.raises(DataModelValidationError) as caught:
            export_store(_broken_store(), "unused")

        message = str(caught.value)
        assert message.startswith(
            "The data model store has 1 problem, so nothing was written:"
        )
        assert "  - QualityAnnotation " in message
        assert "no matching signal target 'missing-signal'" in message
        assert "pass validate=False" in message

    def test_validate_false_writes_it_anyway(self, tmp_path):
        written = export_store(_broken_store(), tmp_path, validate=False)

        assert os.path.isfile(written["quality_annotations"])


class TestARecordedStore:
    def test_passes_the_default_checks(self, tmp_path):
        written = export_store(_recorded_store(), tmp_path)

        assert os.path.isfile(written["signal_streams"])

    def test_require_complete_also_runs_the_completeness_checks(self, tmp_path):
        out_dir = os.path.join(tmp_path, "datamodel")

        with pytest.raises(DataModelValidationError) as caught:
            export_store(_recorded_store(), out_dir, require_complete=True)

        assert any("is missing 'start_time'" in p for p in caught.value.problems)
        assert not os.path.exists(out_dir)


def test_require_complete_without_validation_is_refused(tmp_path):
    out_dir = os.path.join(tmp_path, "datamodel")

    with pytest.raises(ValueError, match="require_complete=True has no effect"):
        export_store(DataModelStore(), out_dir, validate=False, require_complete=True)

    assert not os.path.exists(out_dir)
