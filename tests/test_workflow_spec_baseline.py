"""Freezes today's public ``m3resp.workflows`` behavior *before* the schema
migration (typed metadata, versioned specs, compiled plans, ...) begins, so a
regression in the parser/engine shows up as a failing snapshot diff instead of
a silent behavior change. Do not "fix" a failing snapshot by regenerating it
without checking whether the underlying change was intentional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from m3resp.workflows import (
    PipelineContext,
    PipelineResult,
    PipelineSpec,
    StepSpec,
    available_steps,
    load_spec,
    register_step,
    run_pipeline,
    run_spec,
    validate_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots" / "pipeline_specs"

# Every shipped example spec, as of the Stage 2 EIT/EMG gap-migration merge.
EXAMPLE_SPECS: dict[str, Path] = {
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


def _normalize_step(step: StepSpec) -> dict[str, Any]:
    return {
        "id": step.id,
        "uses": step.uses,
        "in": step.inputs,
        "with": step.params,
        "out": step.outputs,
    }


def normalize_spec(spec: PipelineSpec) -> dict[str, Any]:
    """Convert a parsed spec into a JSON-safe, machine-portable structure.

    ``outputs.dir`` is stored relative to the repository root rather than as
    an absolute path, since ``load_spec`` resolves it against the spec's
    directory and an absolute path would not be portable across machines/CI.
    """

    outputs_dir = (
        spec.outputs.dir.relative_to(REPO_ROOT).as_posix()
        if spec.outputs.dir is not None
        else None
    )
    return {
        "name": spec.name,
        "schema_version": spec.schema_version,
        "inputs": spec.inputs,
        "steps": [_normalize_step(step) for step in spec.steps],
        "outputs": {
            "dir": outputs_dir,
            "mode": spec.outputs.mode,
            "timestamped": spec.outputs.timestamped,
            "summary_json": spec.outputs.summary_json,
            "event_csvs": spec.outputs.event_csvs,
            "parameters_csv": spec.outputs.parameters_csv,
            "postprocessing": spec.outputs.postprocessing,
            "structured_export": spec.outputs.structured_export,
            "figures": spec.outputs.figures,
        },
        "experiment": {
            "subject_id": spec.experiment.subject_id,
            "mode": spec.experiment.mode,
            "timepoint": spec.experiment.timepoint,
            "run_identifier": spec.experiment.run_identifier,
            "selection": spec.experiment.selection,
        },
    }


# --------------------------------------------------------------------------- #
# Public API surface                                                          #
# --------------------------------------------------------------------------- #


def test_workflows_public_api_is_stable():
    """Guards the symbols compatibility shim will need to
    re-export from a future ``m3resp.pipeline`` package."""

    assert callable(run_pipeline)
    assert callable(run_spec)
    assert callable(validate_spec)
    assert callable(available_steps)
    assert callable(load_spec)
    assert callable(register_step)
    assert PipelineResult is not None
    assert PipelineContext is not None
    assert PipelineSpec is not None


def test_available_steps_covers_every_modality_prefix():
    steps = set(available_steps())
    prefixes = {name.split(".", 1)[0] for name in steps}
    assert prefixes == {"eit", "emg", "export", "metric", "session", "sync"}


def test_pipeline_result_value_reads_produced_context_key():
    @register_step("baseline.make", writes=("value",))
    def _make(*, x: int) -> dict[str, Any]:
        return {"value": x * 2}

    try:
        result = run_pipeline(
            {
                "name": "baseline-smoke",
                "inputs": {"x": 5},
                "steps": [{"uses": "baseline.make", "with": {"x": "@x"}}],
            }
        )
        assert isinstance(result, PipelineResult)
        assert result.value("value") == 10
    finally:
        from m3resp.workflows.registry import STEP_REGISTRY

        STEP_REGISTRY.pop("baseline.make", None)


# --------------------------------------------------------------------------- #
# Example specs: load + validate                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("example_name", sorted(EXAMPLE_SPECS))
def test_example_spec_loads_and_validates(example_name: str):
    spec = load_spec(EXAMPLE_SPECS[example_name])
    assert spec.steps
    validate_spec(spec)


@pytest.mark.parametrize("example_name", sorted(EXAMPLE_SPECS))
def test_example_spec_dir_resolves_against_spec_root(example_name: str):
    """``outputs.dir`` must resolve relative to the spec file, not the cwd."""

    spec = load_spec(EXAMPLE_SPECS[example_name])
    assert spec.outputs.dir is not None
    assert spec.outputs.dir.is_absolute()
    assert REPO_ROOT in spec.outputs.dir.parents


# --------------------------------------------------------------------------- #
# Example specs: frozen normalized snapshot                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("example_name", sorted(EXAMPLE_SPECS))
def test_example_spec_matches_frozen_snapshot(example_name: str):
    """Fails loudly if a parser/engine change alters step order, resolved
    bindings, or resolved output settings for a shipped example spec.

    If this fails because of an *intended* spec/parser change, regenerate the
    snapshot with the normalized structure from ``normalize_spec`` and review
    the diff before committing it.
    """

    spec = load_spec(EXAMPLE_SPECS[example_name])
    normalized = normalize_spec(spec)
    snapshot_path = SNAPSHOT_DIR / f"{example_name}.json"
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert normalized == expected
