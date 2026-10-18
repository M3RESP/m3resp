"""Several EIT files loaded as one recording."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from m3resp.adapters import EITProcessingAdapter
from m3resp.core.session import M3Session
from m3resp.datamodel import DataModelRecorder, DataModelStore
from m3resp.workflows import load_spec, run_workflow, validate_workflow
from m3resp.workflows.steps.eit import load

np = pytest.importorskip("numpy")
pytest.importorskip("eitprocessing")

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAEGER_PATH = os.path.join(
    REPO_ROOT, "tests", "data", "draeger_synthetic_draeger_20Hz.bin"
)
# Bytes per frame in this Draeger file format (1200 frames at 20 Hz).
FRAME_SIZE = 4358


def _write_frames(source: str, target: Path, first: int, stop: int) -> str:
    """Write frames [first, stop) of a Draeger file to a new file."""

    with open(source, "rb") as handle:
        handle.seek(first * FRAME_SIZE)
        target.write_bytes(handle.read((stop - first) * FRAME_SIZE))
    return str(target)


@pytest.fixture
def two_halves(tmp_path: Path) -> list[str]:
    return [
        _write_frames(DRAEGER_PATH, tmp_path / "part_1.bin", 0, 600),
        _write_frames(DRAEGER_PATH, tmp_path / "part_2.bin", 600, 1200),
    ]


def test_two_consecutive_files_give_the_same_recording_as_one_file(two_halves):
    whole = M3Session()
    whole.load_eit(DRAEGER_PATH, vendor="draeger")
    joined = M3Session()

    joined.load_eit(two_halves, vendor="draeger")

    np.testing.assert_array_equal(joined.eit.raw.time, whole.eit.raw.time)
    np.testing.assert_array_equal(
        joined.eit.raw.pixel_impedance, whole.eit.raw.pixel_impedance
    )
    assert joined.eit.paths == tuple(Path(p) for p in two_halves)
    assert joined.provenance[-1].parameters["path"] == two_halves


def test_files_out_of_order_are_refused(two_halves):
    with pytest.raises(
        ValueError, match=r"part_1\.bin' starts .* before .*part_2\.bin' ends"
    ):
        M3Session().load_eit(list(reversed(two_halves)), vendor="draeger")


def test_the_same_file_twice_is_refused():
    with pytest.raises(ValueError, match="List the files in recording order"):
        M3Session().load_eit([DRAEGER_PATH, DRAEGER_PATH], vendor="draeger")


def test_a_pause_between_files_is_kept_and_reported(tmp_path):
    files = [
        _write_frames(DRAEGER_PATH, tmp_path / "early.bin", 0, 400),
        _write_frames(DRAEGER_PATH, tmp_path / "late.bin", 800, 1200),
    ]
    session = M3Session()

    with pytest.warns(UserWarning, match=r"gap of 20\.000 s between EIT files"):
        session.load_eit(files, vendor="draeger")

    time = session.eit.raw.time
    assert len(time) == 800
    assert time[400] - time[399] == pytest.approx(20.05)


@pytest.mark.parametrize("option", [{"first_frame": 10}, {"max_frames": 100}])
def test_frame_limits_are_refused_with_several_files(two_halves, option):
    with pytest.raises(ValueError, match="apply to each file"):
        load(M3Session(), file_path=two_halves, vendor="draeger", **option)


def test_one_file_in_a_list_reads_like_a_single_path():
    session = M3Session()

    load(session, file_path=[DRAEGER_PATH], vendor="draeger", max_frames=30)

    assert session.eit.raw.pixel_impedance.shape[0] == 30


def test_custom_loader_receives_the_list_as_given(two_halves):
    received: list[object] = []

    def loader(path, vendor=None, **kwargs):
        received.append(path)
        return EITProcessingAdapter().load_with_eitprocessing(path, vendor="draeger")

    session = M3Session(eit_adapter=EITProcessingAdapter(loader=loader))
    session.load_eit(two_halves, loader="custom")

    assert received == [two_halves]
    assert session.eit.raw.pixel_impedance.shape[0] == 1200


def test_workflow_resolves_a_list_of_relative_paths(two_halves, tmp_path):
    spec_path = tmp_path / "joined.workflow.yaml"
    spec_path.write_text(
        "name: joined\n"
        "steps:\n"
        "  - uses: eit.load\n"
        "    with: { file_path: [part_1.bin, part_2.bin], vendor: draeger }\n",
        encoding="utf-8",
    )

    result = run_workflow(load_spec(spec_path))

    assert result.value("raw_pixel_impedance_signal").values.shape[0] == 1200
    assert result.session.eit.paths == tuple(Path(p).resolve() for p in two_halves)


def test_workflow_check_names_a_missing_file_in_the_list(two_halves, tmp_path):
    spec = load_spec(
        {
            "name": "joined",
            "steps": [
                {
                    "uses": "eit.load",
                    "with": {
                        "file_path": [two_halves[0], str(tmp_path / "missing.bin")],
                        "vendor": "draeger",
                    },
                }
            ],
        }
    )

    report = validate_workflow(spec, readiness=True)

    assert report.is_valid
    messages = [d.message for d in report.readiness if d.code == "missing_file"]
    assert len(messages) == 1
    assert "missing.bin" in messages[0]


def test_workflow_check_refuses_an_empty_list():
    spec = load_spec(
        {
            "name": "joined",
            "steps": [
                {"uses": "eit.load", "with": {"file_path": [], "vendor": "draeger"}}
            ],
        }
    )

    report = validate_workflow(spec)

    assert not report.is_valid


def test_recorder_stores_one_data_file_per_eit_file(two_halves):
    session = M3Session()
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)

    session.load_eit(two_halves, vendor="draeger")

    raw_files = [f.file_path for f in store.data_files.values() if f.file_role == "raw"]
    assert raw_files == two_halves
    load_run = next(
        run for run in store.processing_runs.values() if run.name == "load_eit"
    )
    assert len(load_run.input_file_ids) == 2
