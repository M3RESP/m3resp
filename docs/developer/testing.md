# Testing

This page shows where the tests are, what each group of tests checks, and how to run them.

## Run the tests

Run the three checks from the source folder. All three must pass before a pull request is ready:

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
pytest           # all tests
ruff check .     # code style
mypy src         # types
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv run pytest           # all tests
uv run ruff check .     # code style
uv run mypy src         # types
```
:::
::::

To make mypy pass, change the code. Keep the mypy settings as they are.

## Test folders

```text
tests/
├── test_*.py              tests per module or feature: session, adapters, data model, export, workflows, ...
├── regression/            comparison tests with eitprocessing and resurfemg
│   └── real_data/         tests on recorded data, not part of the normal test run
├── snapshots/
│   └── workflow_specs/    saved structure of each example workflow file
└── data/                  synthetic recordings for the tests and the examples
```

## Comparison tests

The tests in `tests/regression/` compare M3Resp with the original packages. Each test runs an M3Resp function on synthetic data, and the original function with the same settings. Most tests require exactly equal values (`assert_array_equal`, or `atol=0, rtol=0`).

| Test file | Compares |
|---|---|
| `test_eit_equivalence.py`, `test_eit_gap_migration_equivalence.py` | `EITProcessingAdapter` with `eitprocessing` |
| `test_emg_equivalence.py` | `ReSurfEMGAdapter` with `resurfemg` |
| `test_processing_filter_equivalence.py`, `test_processing_metric_equivalence.py`, `test_processing_peak_interval_equivalence.py`, `test_processing_quality_equivalence.py`, `test_processing_window_equivalence.py` | The copies in `m3resp.processing` with their originals |
| `test_ecg_peak_detection_ground_truth.py` | The ECG peak detection with the R peaks of a simulated ECG signal, found by `neurokit2` |

If one of these tests fails, the M3Resp result differs from the original result. Look at the M3Resp function first, before you change the test. See {doc}`adapters`.

## Tests on recorded data

The tests in `tests/regression/real_data/` run M3Resp on full recordings: dataset 3 of the recordings of 27 July 2026 (TestPS3). This dataset has Biopac EMG and airway pressure, FluxMed flow and Dräger EIT. The tests use one minute with the EIT on (37 to 97 s on the Biopac clock) and one minute with the EIT off (140 to 200 s). For each minute, the tests put the FluxMed and EIT recordings on the Biopac clock. They use the cross-correlation of the airway pressure, which all three devices record.

The recordings are kept outside the repository. The tests read them from the folder in the environment variable `M3RESP_REAL_DATA`, with `data/source/Test_PS_27072026` as the default. They check the SHA-256 checksum of each file against `datasets.yaml`, and skip if a file is missing. The tests have the marker `real_data`, and the normal test run leaves them out. To run them:

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

The tests make three kinds of check:

- Same as the original packages: each M3Resp step gives exactly the result of `eitprocessing` or `resurfemg`, called directly on the same data.
- Same as the last time: each number equals the stored value in `reference_values.json`. Counts and sample numbers must be exactly equal. Other numbers can have a relative difference of at most 1e-9.
- Physiology: the heart rate of the EIT against the ECG in the EMG, the breathing rates and breath starts against the flow, and the EMG filter chains with a notch filter against each other while the EIT is on.

If you change the code to change the results, write the new reference values with `scripts/update_real_data_references.py`. Then look at the changes in `reference_values.json`, which show the numbers that changed.

## Tests without the optional packages

`tests/test_optional_dependency_absence.py` checks that you can import `m3resp`, list the steps and check a workflow file without `eitprocessing` and `resurfemg`. The test hides the two packages in a Python environment that has them.

The job `optional-deps` in `.github/workflows/tests.yml` installs M3Resp four times in new environments: with no extras, and with `.[eit]`, `.[emg]` and `.[all]`. Each time, it runs `python -m m3resp steps` and `scripts/check_optional_dependency_isolation.py`. The script checks that the installed packages match the extra, and that each step shows the right status: `"available"` or `"missing_optional_dependency"`.

## Snapshots of the example workflows

`tests/test_workflow_spec_baseline.py` and `tests/test_example_specs.py` load and check each example workflow file in `examples/`. They compare its structure with the saved JSON files in `tests/snapshots/workflow_specs/`: the order of the steps, the `in`, `with` and `out` of each step, and the results folder. So a change in how M3Resp reads workflow files shows up as a failed test.

Make a new snapshot only after you checked that the change is correct, and that no other part of the structure changed.

## Test data

The recordings in `tests/data` are synthetic. The tests, the tutorials and the example workflows use these files. `m3resp.synthetic` makes synthetic EIT, EMG and ventilator recordings. The `m3resp_multimodal_1_*` files come from the settings in `src/m3resp/synthetic/synthetic_generator_config.yaml`.

:::{warning}
Do not add patient recordings to the repository. Use synthetic recordings for tests and examples.
:::

Continue with {doc}`coding_guidelines`.
