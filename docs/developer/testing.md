# Testing

## Layout

```text
tests/
├── test_*.py           Unit/integration tests for a given module or feature
│                        (session, adapters, data model, exports, workflows, ...)
├── regression/          Equivalence tests against eitprocessing/resurfemg
└── snapshots/
    └── pipeline_specs/  Frozen, normalized snapshots of the shipped example specs
```

There is no separate `tests/unit/`, `tests/adapters/`, `tests/pipelines/`
directory split as originally sketched in early Stage 2 planning - the flat
`tests/test_*.py` layout with one `regression/` subfolder has worked fine at
the current test count and keeps discovery (`pytest tests/`) simple. Split
into subfolders if a directory genuinely becomes hard to navigate, not
preemptively.

## Regression tests (`tests/regression/`)

These are the most important tests in the suite for Stage 2's "don't change
scientific output" guarantee. Each one drives an
adapter's public API on synthetic data and asserts the result is identical -
usually `atol=0, rtol=0`, i.e. exact equality - to calling the underlying
`eitprocessing`/`resurfemg` function directly with the same arguments:

- `test_eit_equivalence.py`, `test_eit_gap_migration_equivalence.py` - EIT adapter vs. `eitprocessing`.
- `test_emg_equivalence.py` - EMG adapter vs. `resurfemg`.
- `test_processing_filter_equivalence.py`, `test_processing_metric_equivalence.py`,
  `test_processing_peak_interval_equivalence.py`,
  `test_processing_quality_equivalence.py`, `test_processing_window_equivalence.py` -
  the shared `m3resp.processing` primitives vs. their `resurfemg` originals.
- `test_ecg_peak_detection_ground_truth.py` - ECG peak detection against a
  hand-labeled ground truth, not just the upstream package.

If one of these starts failing, the wrapper has started transforming data
instead of just passing it through - check the diff against the specific
method involved before assuming the test is wrong. See
[adapters.md](adapters.md).

## Optional-dependency tests

`tests/test_optional_dependency_absence.py` checks that importing `m3resp`
and listing/validating pipeline steps works even when `eitprocessing`/
`resurfemg` are not installed (capability discovery must not import the
optional backend - see [../pipelines.md](../pipelines.md) "Validation and
readiness"). This simulates absence in-process (`sys.modules[name] = None`)
inside one environment that actually has both packages installed.

The `optional-deps` job in `.github/workflows/tests.yml` complements this
with a real install-matrix: four legs (no extras, `.[eit]`, `.[emg]`,
`.[all]`), each installing `m3resp` in its own fresh environment and running
`python -m m3resp steps` plus
`scripts/check_optional_dependency_isolation.py <extra>` - which asserts the
as-installed package set matches the extra, and that every registered
step's `describe_steps()` capability state (`"available"` vs.
`"missing_optional_dependency"`) matches what its declared
`optional_packages` implies for that install.

## Frozen example snapshots

`tests/test_workflow_spec_baseline.py`/`test_example_specs.py` load and
validate every shipped example spec (`examples/*/*.pipeline.yaml`) and
compare a normalized, secret-free structure (step order, `in`/`with`/`out`
bindings, resolved `outputs.dir`) against the frozen JSON files in
`tests/snapshots/pipeline_specs/`. This protects the parser and compiler
against silent behavior changes while their internals evolve. Regenerate a
snapshot only after confirming (by diffing with the changed field stripped)
that nothing else in the normalized structure moved.

## Synthetic fixtures

`m3resp.synthetic` (`src/m3resp/synthetic/`) generates
synthetic EIT/EMG/ventilator recordings so tests and examples never depend on
private clinical data. Examples reference the generated files under
`data/source/synthetic/`.

## Running the suite

```bash
pytest                       # full suite
ruff check .                 # lint
mypy src                     # type check
```

All three must pass before a change is considered done; do not weaken mypy
settings or exclude a package to make it pass.
