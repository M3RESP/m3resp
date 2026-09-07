"""Phase 8.4 of the pipeline-structure plan: validate complete registry
coverage. Every built-in step must have a stable operation ID/prefix,
non-empty summary/description/category, complete parameter metadata,
typed public inputs/outputs, optional-dependency information, a JSON-safe
description, valid alternatives where declared, and no eager optional
import at discovery time.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.workflows.registry import STEP_REGISTRY, describe_steps

_KNOWN_PREFIXES = {
    "eit",
    "emg",
    "export",
    "metric",
    "session",
    "sync",
    "ventilator",
}

#: Packages a step module must never import at module scope (Phase 1.4/8.4:
#: "no eager optional import during discovery").
_OPTIONAL_PACKAGES = ("eitprocessing", "resurfemg")

_STEPS_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "m3resp" / "workflows" / "steps"
)


def test_every_step_name_has_a_stable_id_and_known_prefix():
    for name in STEP_REGISTRY:
        prefix, _, operation = name.partition(".")
        assert prefix and operation, name
        assert prefix in _KNOWN_PREFIXES, name


def test_every_step_has_non_empty_summary_description_category():
    for description in describe_steps():
        assert description.summary, description.name
        assert description.description, description.name
        assert description.category, description.name


def test_every_step_description_is_json_serializable():
    for description in describe_steps():
        json.dumps(description.as_dict())


def test_every_read_and_write_has_a_matching_typed_artifact():
    """Phase 8.4's "typed public inputs/outputs": every context key a step
    reads or writes must have a corresponding ``StepArtifact`` entry - it
    may be marked ``public=False``/``compatibility_only=True`` for internal
    plumbing (e.g. the session object, or run_spec's auto-injected
    '_spec_outputs'), but it must still be *typed*, not silently absent."""

    for name, definition in STEP_REGISTRY.items():
        input_names = {a.name for a in definition.input_artifacts}
        output_names = {a.name for a in definition.output_artifacts}
        missing_reads = set(definition.reads) - input_names
        missing_writes = set(definition.writes) - output_names
        assert not missing_reads, (
            f"{name} reads without a typed artifact: {missing_reads}"
        )
        assert not missing_writes, (
            f"{name} writes without a typed artifact: {missing_writes}"
        )


def test_every_declared_alternative_is_a_real_registered_step():
    for description in describe_steps():
        for alternative in description.alternatives:
            assert alternative in STEP_REGISTRY, (
                f"{description.name} declares unknown alternative {alternative!r}"
            )


def test_eit_and_emg_steps_declare_their_optional_package_accurately():
    """Every ``eit.*`` step is genuinely ``eitprocessing``-backed, so all 16
    declare it. ``emg.*`` is a real mix: preprocessing/quality steps wrap
    ``resurfemg``, but several "features"/detection steps
    (`emg.amplitude`, `emg.peak_indices`, ...) call native
    `m3resp.processing.*` primitives and correctly declare none - so this
    only checks that a *declared* package is the real name, not that every
    emg step declares one."""

    for description in describe_steps():
        prefix = description.name.split(".", 1)[0]
        if prefix == "eit":
            assert description.optional_packages == ("eitprocessing",), description.name
        elif prefix == "emg" and description.optional_packages:
            assert description.optional_packages == ("resurfemg",), description.name


def test_every_declared_parameter_matches_a_real_function_keyword():
    """The registration-time check (Phase 1.2) already verifies this one
    direction; re-assert it here so a regression is caught at the registry
    level too, not only at each individual registration call."""

    for name, definition in STEP_REGISTRY.items():
        try:
            signature = inspect.signature(definition.func)
        except (TypeError, ValueError):
            continue
        if any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        ):
            continue  # e.g. emg.preprocess(**kwargs): accepted names aren't enumerable
        accepted = set(signature.parameters)
        for parameter in definition.parameters:
            assert parameter.name in accepted, f"{name}.{parameter.name}"


def test_every_function_keyword_without_a_default_has_declared_metadata():
    """Phase 8.4's "complete parameter metadata": a required static
    parameter (no default in the function signature, not part of ``reads``)
    must have a declared ``StepParameter`` - otherwise a caller gets a raw
    ``TypeError`` instead of a clean pre-execution diagnostic. Skipped for a
    function accepting ``**kwargs`` (its accepted names aren't enumerable)."""

    missing: dict[str, set[str]] = {}
    for name, definition in STEP_REGISTRY.items():
        try:
            signature = inspect.signature(definition.func)
        except (TypeError, ValueError):
            continue
        if any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        ):
            continue
        declared = {p.name for p in definition.parameters}
        for param_name, param in signature.parameters.items():
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            if param_name in definition.reads or param_name in declared:
                continue
            if param.default is inspect.Parameter.empty:
                missing.setdefault(name, set()).add(param_name)

    assert missing == {}, (
        f"Steps with a required keyword missing StepParameter metadata: {missing}"
    )


def _iter_step_module_paths() -> list[Path]:
    return sorted(p for p in _STEPS_DIR.glob("*.py") if p.name != "__init__.py")


def test_no_step_module_eagerly_imports_an_optional_package():
    """Phase 1.4/8.4: an optional package may only be imported inside a
    function body (deferred), never at module scope, so listing/describing
    steps works with neither ``eitprocessing`` nor ``resurfemg`` installed."""

    for path in _iter_step_module_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # only top-level (module-scope) statements
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for package in _OPTIONAL_PACKAGES:
                assert package not in names, (
                    f"{path.name} imports {package!r} at module scope"
                )


def test_describe_steps_works_without_a_real_registry_mutation():
    """Calling discovery twice must be stable/idempotent (no duplicate
    entries, no growth) - a lightweight guard on describe_steps() itself."""

    first = {d.name for d in describe_steps()}
    second = {d.name for d in describe_steps()}
    assert first == second
    assert len(first) == len(STEP_REGISTRY)
