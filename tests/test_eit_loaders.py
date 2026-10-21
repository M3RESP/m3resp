"""`eit.load` reads vendor files with eitprocessing; `eit.load_custom` uses the session's own loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from m3resp.adapters import EITProcessingAdapter
from m3resp.core.exceptions import UnsupportedWorkflowError, WorkflowSpecError
from m3resp.core.session import M3Session
from m3resp.workflows import load_spec, run_workflow, validate_spec
from m3resp.workflows.registry import get_step
from m3resp.workflows.steps.eit import load, load_custom

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAEGER_PATH = os.path.join(
    REPO_ROOT, "tests", "data", "draeger_synthetic_draeger_20Hz.bin"
)


def _read_draeger(path: str, vendor: Any = None, **kwargs: Any) -> Any:
    from eitprocessing.datahandling.loading import load_eit_data

    return load_eit_data(path, vendor="draeger", **kwargs)


def _failing_loader(*args: Any, **kwargs: Any) -> Any:
    pytest.fail("the custom loader was used")


def test_eit_load_needs_a_vendor():
    spec = {
        "name": "no-vendor",
        "steps": [{"uses": "eit.load", "with": {"file_path": DRAEGER_PATH}}],
    }

    with pytest.raises(WorkflowSpecError, match="vendor"):
        validate_spec(load_spec(spec))


def test_eit_load_reads_with_eitprocessing_when_a_custom_loader_is_set():
    pytest.importorskip("eitprocessing")
    session = M3Session(eit_adapter=EITProcessingAdapter(loader=_failing_loader))

    result = load(session, file_path=DRAEGER_PATH, vendor="draeger", max_frames=50)

    assert result["raw_pixel_impedance_signal"].values.shape[0] == 50


def test_eit_load_custom_without_a_loader_explains_how_to_set_one():
    session = M3Session()

    with pytest.raises(UnsupportedWorkflowError, match="EITProcessingAdapter"):
        load_custom(session, file_path=DRAEGER_PATH)


def test_eit_load_custom_passes_its_options_to_the_loader():
    calls: list[tuple[str, dict[str, Any]]] = []

    def loader(path: str, **kwargs: Any) -> Any:
        calls.append((path, kwargs))
        return _read_draeger(path, max_frames=kwargs["max_frames"])

    session = M3Session(eit_adapter=EITProcessingAdapter(loader=loader))
    pytest.importorskip("eitprocessing")

    result = load_custom(
        session, file_path=DRAEGER_PATH, loader_options={"max_frames": 20}
    )

    assert calls == [(DRAEGER_PATH, {"vendor": None, "max_frames": 20})]
    assert set(get_step("eit.load_custom").writes) <= set(result)
    assert result["raw_pixel_impedance_signal"].values.shape[0] == 20


def test_both_steps_give_the_same_outputs_for_the_same_file():
    pytest.importorskip("eitprocessing")
    vendor_session = M3Session()
    custom_session = M3Session(eit_adapter=EITProcessingAdapter(loader=_read_draeger))

    from_vendor = load(vendor_session, file_path=DRAEGER_PATH, vendor="draeger")
    from_custom = load_custom(custom_session, file_path=DRAEGER_PATH)

    np.testing.assert_array_equal(
        from_vendor["raw_pixel_impedance_signal"].values,
        from_custom["raw_pixel_impedance_signal"].values,
    )
    np.testing.assert_array_equal(
        from_vendor["raw_global_impedance_signal"].values,
        from_custom["raw_global_impedance_signal"].values,
    )


def test_custom_loader_workflow_takes_the_adapter_from_run_workflow():
    pytest.importorskip("eitprocessing")
    spec = {
        "name": "custom-eit",
        "steps": [{"uses": "eit.load_custom", "with": {"file_path": DRAEGER_PATH}}],
    }

    result = run_workflow(spec, eit_adapter=EITProcessingAdapter(loader=_read_draeger))

    assert result.value("raw_pixel_impedance_signal") is not None
    actions = [entry.action for entry in result.session.provenance]
    assert actions.count("eit.load_custom") == 1


def test_session_load_eit_refuses_a_vendor_for_a_custom_loader():
    session = M3Session(eit_adapter=EITProcessingAdapter(loader=_failing_loader))

    with pytest.raises(ValueError, match="takes no vendor"):
        session.load_eit(DRAEGER_PATH, vendor="draeger", loader="custom")


def test_session_load_eit_refuses_an_unknown_loader_choice():
    with pytest.raises(ValueError, match="loader must be"):
        M3Session().load_eit(DRAEGER_PATH, loader="vendor")  # type: ignore[arg-type]
