"""Tests for `m3resp.workflows.spec.spec_to_dict`/`dump_spec`: the missing
serializer, the inverse of `load_spec()`'s parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.workflows.compiler import compile_pipeline
from m3resp.workflows.spec import (
    SpecExecutionConfig,
    SpecExperimentConfig,
    SpecOutputsConfig,
    dump_spec,
    load_spec,
    spec_to_dict,
)

EXAMPLE_SPECS = sorted(Path("examples").glob("**/*.pipeline.yaml"))


# --------------------------------------------------------------------------- #
# The round-trip property test                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec_path", EXAMPLE_SPECS, ids=lambda p: p.name)
def test_round_trip_compiles_identically_for_every_example_spec(spec_path: Path):
    original = load_spec(spec_path)
    rebuilt = load_spec(spec_to_dict(original), root=original.root)

    assert compile_pipeline(rebuilt).as_dict() == compile_pipeline(original).as_dict()


def test_at_least_one_example_spec_was_found():
    assert len(EXAMPLE_SPECS) >= 5


@pytest.mark.parametrize("spec_path", EXAMPLE_SPECS, ids=lambda p: p.name)
def test_dump_spec_round_trips_through_an_actual_yaml_file(
    spec_path: Path, tmp_path: Path
):
    """Written to a different directory than the original, with the
    original's root passed explicitly on reload - a relative path (e.g. a
    recording's ``file:``) is only ever meaningful relative to a root, and
    saving a spec to a new location does not, by itself, move the data it
    points at. A caller that wants the saved file to be self-contained in
    its new directory needs to rewrite those paths itself; that is a
    separate concern from whether the serializer preserves the spec it was
    given, which is what this test checks."""

    original = load_spec(spec_path)
    written = dump_spec(original, tmp_path / "saved.pipeline.yaml")
    reloaded = load_spec(written, root=original.root)

    assert compile_pipeline(reloaded).as_dict() == compile_pipeline(original).as_dict()


# --------------------------------------------------------------------------- #
# spec_to_dict: content shape                                                 #
# --------------------------------------------------------------------------- #


def test_spec_to_dict_always_writes_an_explicit_step_id():
    spec = load_spec(
        {"name": "p", "steps": [{"uses": "eit.slice"}]}  # no explicit id
    )
    payload = spec_to_dict(spec)
    assert payload["steps"][0]["id"] == spec.steps[0].id


def test_spec_to_dict_omits_empty_in_with_out():
    spec = load_spec({"name": "p", "steps": [{"uses": "eit.slice"}]})
    payload = spec_to_dict(spec)
    step_payload = payload["steps"][0]
    assert "in" not in step_payload
    assert "with" not in step_payload
    assert "out" not in step_payload


def test_spec_to_dict_keeps_at_ref_values_unresolved():
    spec = load_spec(
        {
            "name": "p",
            "inputs": {"eit_file": "/data/some.bin"},
            "steps": [{"uses": "eit.slice", "with": {"start_s": "@eit_file"}}],
        }
    )
    payload = spec_to_dict(spec)
    assert payload["steps"][0]["with"]["start_s"] == "@eit_file"


def test_spec_to_dict_omits_default_execution_outputs_experiment():
    spec = load_spec({"name": "p", "steps": [{"uses": "eit.slice"}]})
    payload = spec_to_dict(spec)
    assert "execution" not in payload
    assert "outputs" not in payload
    assert "experiment" not in payload
    # Sanity: the defaults really are what a bare spec parses to.
    assert spec.execution == SpecExecutionConfig()
    assert spec.outputs == SpecOutputsConfig()
    assert spec.experiment == SpecExperimentConfig()


def test_spec_to_dict_includes_non_default_execution():
    spec = load_spec(
        {
            "name": "p",
            "execution": {"seed": 7},
            "steps": [{"uses": "eit.slice"}],
        }
    )
    payload = spec_to_dict(spec)
    assert payload["execution"]["seed"] == 7


def test_spec_to_dict_preserves_schema_version():
    legacy = load_spec({"name": "p", "steps": [{"uses": "eit.slice"}]})
    assert "schema_version" not in spec_to_dict(legacy)

    versioned = load_spec(
        {
            "schema_version": 1,
            "name": "p",
            "steps": [{"uses": "eit.slice", "id": "s"}],
        }
    )
    assert spec_to_dict(versioned)["schema_version"] == 1


# --------------------------------------------------------------------------- #
# outputs.dir: relative-to-root handling                                     #
# --------------------------------------------------------------------------- #


def test_spec_to_dict_writes_outputs_dir_relative_to_root_when_possible(tmp_path: Path):
    spec = load_spec(
        {
            "name": "p",
            "outputs": {"dir": "results", "mode": "explicit"},
            "steps": [{"uses": "eit.slice"}],
        },
        root=tmp_path,
    )
    payload = spec_to_dict(spec)
    assert payload["outputs"]["dir"] == "results"


def test_spec_to_dict_writes_outputs_dir_absolute_when_outside_root(tmp_path: Path):
    outside = tmp_path.parent / "elsewhere_outputs"
    spec = load_spec(
        {
            "name": "p",
            "outputs": {"dir": str(outside), "mode": "explicit"},
            "steps": [{"uses": "eit.slice"}],
        },
        root=tmp_path,
    )
    payload = spec_to_dict(spec)
    assert Path(payload["outputs"]["dir"]).is_absolute()
    assert Path(payload["outputs"]["dir"]) == outside.resolve()


# --------------------------------------------------------------------------- #
# dump_spec: file writing                                                     #
# --------------------------------------------------------------------------- #


def test_dump_spec_writes_yaml_by_default(tmp_path: Path):
    spec = load_spec({"name": "p", "steps": [{"uses": "eit.slice"}]})
    written = dump_spec(spec, tmp_path / "out.pipeline.yaml")
    assert written == tmp_path / "out.pipeline.yaml"
    parsed = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert parsed["name"] == "p"


def test_dump_spec_writes_json_for_json_suffix(tmp_path: Path):
    spec = load_spec({"name": "p", "steps": [{"uses": "eit.slice"}]})
    written = dump_spec(spec, tmp_path / "out.json")
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed["name"] == "p"


def test_dump_spec_honors_explicit_format_over_suffix(tmp_path: Path):
    spec = load_spec({"name": "p", "steps": [{"uses": "eit.slice"}]})
    written = dump_spec(spec, tmp_path / "out.pipeline.yaml", format="json")
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed["name"] == "p"
