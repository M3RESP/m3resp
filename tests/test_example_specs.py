"""Every shipped example spec must compile/validate cleanly (structurally,
and where its fixture files are present, at readiness level too) without
importing any optional package.

This is deliberately independent of ``test_workflow_spec_baseline.py``'s
frozen-snapshot tests: those guard *exact* parsed structure against
regressions, this file guards the weaker but more directly useful property
that every example is a valid, compilable, GUI-discoverable pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.core.session import M3Session
from m3resp.workflows import run_pipeline
from m3resp.workflows.compiler import compile_pipeline, validate_pipeline
from m3resp.workflows.spec import load_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

#: Every shipped example spec, as of Stage 2 pipeline-structure.
EXAMPLE_SPEC_PATHS: dict[str, Path] = {
    "rotarc": EXAMPLES_DIR / "ROTARC_example" / "breath-duration.pipeline.yaml",
    "multimodal_example": EXAMPLES_DIR
    / "multimodal_example"
    / "multimodal.pipeline.yaml",
    "annemijn_multimodal": EXAMPLES_DIR
    / "annemijn_multimodal"
    / "annemijn.pipeline.yaml",
    "eit_full_preprocessing": EXAMPLES_DIR
    / "eit_full_preprocessing"
    / "eit-full.pipeline.yaml",
    "emg_full_preprocessing": EXAMPLES_DIR
    / "emg_full_preprocessing"
    / "emg-full.pipeline.yaml",
    "multimodal_full": EXAMPLES_DIR
    / "multimodal_full"
    / "multimodal-full.pipeline.yaml",
}

#: schema_version: 1 examples added/upgraded.
_VERSIONED_SPECS = {
    "rotarc",
    "eit_full_preprocessing",
    "emg_full_preprocessing",
    "multimodal_full",
}

#: Examples whose fixture file is a private, site-specific path (see the
#: "USER TEMPLATE" banner in breath-duration.pipeline.yaml) rather than a
#: repo-relative fixture - readiness will flag it as missing on any machine
#: that is not that researcher's, which is the correct behavior, not a bug.
_PRIVATE_PATH_SPECS = {"rotarc"}


@pytest.mark.parametrize("example_name", sorted(EXAMPLE_SPEC_PATHS))
def test_example_spec_file_exists(example_name: str):
    assert EXAMPLE_SPEC_PATHS[example_name].is_file()


@pytest.mark.parametrize("example_name", sorted(EXAMPLE_SPEC_PATHS))
def test_example_spec_compiles_without_optional_packages(example_name: str):
    """``compile_pipeline`` never imports eitprocessing/resurfemg, so this
    must succeed regardless of what is installed in this environment."""

    spec = load_spec(EXAMPLE_SPEC_PATHS[example_name])
    compiled = compile_pipeline(spec)
    assert compiled.steps


@pytest.mark.parametrize("example_name", sorted(EXAMPLE_SPEC_PATHS))
def test_example_spec_has_no_structural_diagnostics(example_name: str):
    spec = load_spec(EXAMPLE_SPEC_PATHS[example_name])
    report = validate_pipeline(spec, readiness=False)
    assert report.structural == ()


@pytest.mark.parametrize("example_name", sorted(_VERSIONED_SPECS))
def test_example_spec_declares_schema_version_1(example_name: str):
    spec = load_spec(EXAMPLE_SPEC_PATHS[example_name])
    assert spec.schema_version == 1
    assert spec.outputs.mode in ("automatic", "explicit", "none")


@pytest.mark.parametrize("example_name", sorted(_VERSIONED_SPECS))
def test_example_spec_steps_have_stable_explicit_ids(example_name: str):
    """Important steps carry an explicit ``id:`` rather than relying
    solely on the generated ``step_NNN_operation`` form, so GUI/provenance
    references survive reordering the spec."""

    spec = load_spec(EXAMPLE_SPEC_PATHS[example_name])
    for step in spec.steps:
        assert not step.id.startswith("step_"), (
            f"{example_name} step {step.uses!r} has no explicit id"
        )


@pytest.mark.parametrize(
    "example_name",
    sorted(set(EXAMPLE_SPEC_PATHS) - _PRIVATE_PATH_SPECS),
)
def test_example_spec_readiness_reports_no_missing_repo_fixtures(example_name: str):
    """Every example whose fixtures ship in the repo must validate cleanly at
    readiness level too (missing optional packages aside) - a broken
    spec-relative path would show up here as a ``missing_file`` diagnostic."""

    spec = load_spec(EXAMPLE_SPEC_PATHS[example_name])
    report = validate_pipeline(spec, readiness=True)
    missing_file_diagnostics = [d for d in report.readiness if d.code == "missing_file"]
    assert missing_file_diagnostics == []


def test_rotarc_readiness_flags_its_private_path_as_missing_when_absent():
    """The ROTARC example's private, site-specific fixture path is expected
    to be reported as missing on any machine other than the researcher's -
    this is the readiness system doing its job, not a spec bug."""

    spec = load_spec(EXAMPLE_SPEC_PATHS["rotarc"])
    report = validate_pipeline(spec, readiness=True)
    codes = {d.code for d in report.readiness}
    assert codes <= {"missing_file", "missing_optional_package"}


def test_multimodal_full_example_runs_end_to_end():
    """Unlike eit-full/emg-full (each covered by their own end-to-end test
    in test_eit_workflow_steps.py/test_emg_workflow_steps.py),
    multimodal-full.pipeline.yaml was previously only compiled/validated,
    never actually executed as a regression test - this closes that gap."""

    pytest.importorskip("eitprocessing")
    pytest.importorskip("resurfemg")

    # run_pipeline (unlike run_spec) does not touch the spec's `outputs:`
    # section, so this exercises the example without writing into the
    # project's real output/ directory.
    result = run_pipeline(EXAMPLE_SPEC_PATHS["multimodal_full"], session=M3Session())

    assert result.status == "succeeded"

    # Raw synchronization ran before any modality-specific preprocessing.
    assert "estimated_offset_seconds" not in result.context.values

    # Full EIT chain, through the ROI lung-space steps.
    assert result.value("size_filtered_roi_result").value.shape == (32, 32)

    # Full EMG/ventilator chain, through the clinical quality steps.
    assert len(result.value("ecg_peak_indices")) > 0
    assert len(result.value("emg_breath_events")) > 0
    assert len(result.value("ventilator_breath_indices")) > 0

    # Native collections were populated exactly once, not duplicated.
    session = result.session
    assert len(session.signals) > 0
    assert len(session.parameter_results) > 0
    assert len(session.quality) > 0
