# Testing

## Layout

```text
tests/
├── test_*.py           Unit/integration tests for a given module or feature
│                        (session, adapters, data model, exports, workflows, ...)
├── regression/          Equivalence tests against eitprocessing/resurfemg
└── snapshots/
    └── workflow_specs/  Frozen, normalized snapshots of the shipped example specs
```

There is no separate `tests/unit/`, `tests/adapters/`, `tests/workflows/`
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
  hand-labeled ground truth.

If one of these starts failing, the wrapper output now differs from the
upstream output - check the diff against the specific
method involved before assuming the test is wrong. See
[adapters.md](adapters.md).

## Tests on recorded data (`tests/regression/real_data/`)

These tests run m3resp on full recordings: dataset 3 of the 27 July 2026
recordings (TestPS3), with Biopac EMG and airway pressure, FluxMed flow and
Draeger EIT. They use one minute with EIT on (37-97 s on the Biopac clock)
and one minute with EIT off (140-200 s). The FluxMed and EIT recordings are
lined up on the Biopac clock, for each window, by cross-correlating the
airway pressure that all three devices record.

The recordings are not in the repository. The tests read them from the
folder in the environment variable `M3RESP_REAL_DATA` (default
`data/source/Test_PS_27072026`), check the SHA-256 checksum of each file
against `datasets.yaml`, and skip when a file is missing. They have the
marker `real_data`, and the normal test run leaves them out:

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
pytest -m real_data
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv run pytest -m real_data
```
:::
::::

There are three kinds of check:

- Same as upstream: each m3resp step gives exactly what `eitprocessing` or
  `resurfemg` gives when called directly on the same data.
- Same as last time: every collected number equals the stored value in
  `reference_values.json`. Counts and sample indices must match exactly,
  other numbers to a relative difference of 1e-9.
- Physiology: heart rate from EIT against the ECG in the EMG, breathing
  rates and breath onsets against the flow, and the EMG filter chains with a
  notch filter against each other while EIT runs.

After a change that is meant to alter results, rewrite the reference values
with `scripts/update_real_data_references.py` and check the diff of
`reference_values.json`: it shows which numbers moved.

## Optional-dependency tests

`tests/test_optional_dependency_absence.py` checks that importing `m3resp`
and listing/validating workflow steps works even when `eitprocessing`/
`resurfemg` are not installed (capability discovery must not import the
optional backend - see [../workflows.md](../workflows.md) "Validation and
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
validate every shipped example spec (`examples/*/*.workflow.yaml`) and
compare a normalized, secret-free structure (step order, `in`/`with`/`out`
bindings, resolved `outputs.dir`) against the frozen JSON files in
`tests/snapshots/workflow_specs/`. This protects the parser and compiler
against silent behavior changes while their internals evolve. Regenerate a
snapshot only after confirming (by diffing with the changed field stripped)
that nothing else in the normalized structure moved.

## Synthetic fixtures

`m3resp.synthetic` (`src/m3resp/synthetic/`) generates
synthetic EIT/EMG/ventilator recordings so tests and examples never depend on
private clinical data. Examples reference the generated files under
`data/source/synthetic/`.

## Running the suite

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
pytest                       # full suite
ruff check .                 # lint
mypy src                     # type check
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv run pytest                # full suite
uv run ruff check .          # lint
uv run mypy src              # type check
```
:::
::::

All three must pass before a change is considered done; do not weaken mypy
settings or exclude a package to make it pass.
