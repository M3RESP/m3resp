# Pipelines

M3Resp workflows are written as YAML specs and executed by the pipeline engine.
A spec is an ordered list of named steps. Each step reads named artifacts from a
shared context, runs one operation, and writes its outputs back. The engine
validates that every step's inputs are produced before it runs, so wiring
mistakes are caught immediately with a clear error.

## Running a pipeline

From the command line:

```bash
m3resp run path/to/pipeline.yaml
```

From Python:

```python
from m3resp import run_spec, run_pipeline

# Run a file end-to-end (handles outputs: section automatically)
result = run_spec("path/to/pipeline.yaml")

# Run a spec dict or file with more control
result = run_pipeline("path/to/pipeline.yaml")
print(result.value("cv"))     # read a produced artifact by context key
print(result.outputs)         # dict of every artifact the spec produced
result.session                # the backing M3Session (events, parameters)
```

List the available built-in steps:

```bash
m3resp steps
```

Or from Python:

```python
from m3resp import available_steps
for name, summary in available_steps().items():
    print(f"{name}: {summary}")
```

## Spec format

A minimal spec looks like this:

```yaml
name: my-pipeline

inputs:
  eit_file: /data/subject.bin   # referenced in steps as "@eit_file"
  vendor: draeger

outputs:
  dir: output/results
  timestamped: true
  summary_json: true
  event_csvs: true

steps:
  - uses: eit.load
    with: { file: "@eit_file", vendor: "@vendor" }

  - uses: eit.detect_rates
    in:  { signal: raw_eit }        # override which context key feeds this param
    with: { subject_type: adult }

  - uses: eit.mdn_filter
    in:  { signal: raw_eit }

  - uses: eit.global_impedance
    in:  { signal: filtered_eit }

  - uses: eit.detect_breaths
    in:  { signal: global_impedance }

  - uses: metric.interval_cv
    in:  { intervals: breath_intervals }
    out: { cv: breath_duration_cv }   # rename output into a different context key
```

The four keys for each step:

- **`uses`** — the registered step name.
- **`in`** — maps a step's parameter name to the context key it reads from.
  Leave it out to use the step's default binding.
- **`with`** — static parameters passed directly to the step function. A string
  value starting with `@` is resolved from `inputs`.
- **`out`** — renames a step's natural output names to different context keys.
  Useful when you want to run the same step twice and keep both outputs.

### Top-level sections

- **`name`** — a label for the pipeline (used in log output).
- **`inputs`** — named values that steps can reference with `@name`.
- **`outputs`** — controls where and what gets exported after the pipeline
  finishes. If `dir` is set and no explicit export step is present, the engine
  calls `export_session_summary` automatically.
- **`experiment`** — study metadata used by `export.rotarc_result` for output
  file naming (`subject_id`, `mode`, `timepoint`, `run_identifier`, `selection`).

### Timestamped output directories

Set `outputs.timestamped: true` to give every run its own subfolder
(`<outputs.dir>/YYYYMMDD_HHMMSS/...`) instead of overwriting the same folder
each time. This works the same way for *any* export path in the run:

- The automatic export (no explicit export step in the spec).
- Built-in export steps that need a directory, e.g. `export.rotarc_result`
  (`<outputs.dir>/<timestamp>/subject_results/<run_identifier>/...`).
- Any custom export step, by reading `_resolved_output_dir` from context.

`run_spec` computes the timestamp exactly once per run and seeds it as
`_resolved_output_dir` (a resolved `Path`, or `None` if `outputs.dir` isn't
set) and `_run_timestamp` (the raw stamp, or `None` if not timestamped) into
context, alongside `_spec_outputs`/`_spec_experiment`. Every export path in
the run reads the *same* resolved directory, so a single run never ends up
split across two different timestamp folders even if it uses more than one
export mechanism. To make your own export step honor it:

```python
@register_step(
    "export.my_thing",
    reads={
        "session": "session",
        "output_dir": "_resolved_output_dir",   # None if outputs.dir isn't set
    },
    writes=("result_path",),
)
def my_thing(session, output_dir, **kwargs) -> dict:
    if output_dir is None:
        raise ValueError("export.my_thing requires 'outputs.dir' to be set.")
    ...
```

## Built-in steps

Run `m3resp steps` to see the full list with descriptions. The main groups are:

| Prefix | What it covers |
|---|---|
| `eit.*` | Load, slice, filter, detect rates/breaths, compute TIV/EELI/pixel TIV |
| `emg.*` | Load (EMG/ventilator), preprocess, detect breaths, compute per-function baseline/event-detection/feature/quality-assessment postprocessing steps |
| `session.*` | Cross-modality operations (raw signal synchronization) |
| `metric.*` | Reduction steps (e.g. coefficient of variation of intervals) |
| `export.*` | Write results to disk (scalar files, JSON, session summary, ROTARC result) |

## Adding a custom step

Register a function with `@register_step`. Declare the context keys it reads
(mapped onto its parameter names) and the output names it returns:

```python
from m3resp.workflows import register_step

@register_step(
    "my.step",
    reads={"signal": "global_impedance"},   # param name → default context key
    writes=("result",),                      # output names returned in the dict
)
def my_step(signal, *, threshold: float = 0.5) -> dict:
    return {"result": signal.values > threshold}
```

The function receives bound keyword arguments and returns a mapping of
`{output_name: value}`. Steps that produce no outputs return `{}` or `None`.
Import the optional modality packages inside the function body so the package
installs cleanly without them.

## Example specs

Three worked examples ship with the repository:

- [`examples/ROTARC_example/breath-duration.pipeline.yaml`](../examples/ROTARC_example/breath-duration.pipeline.yaml) —
  computes breath-duration CV from EIT data and writes a ROTARC-style result
  file.
- [`examples/multimodal_example/multimodal.pipeline.yaml`](../examples/multimodal_example/multimodal.pipeline.yaml) —
  loads EIT, EMG, and ventilator signals, synchronizes them, processes each
  modality, and exports session summaries.
- [`examples/eit_full_preprocessing/eit-full.pipeline.yaml`](../examples/eit_full_preprocessing/eit-full.pipeline.yaml) —
  every EIT operation closed onto `EITProcessingAdapter` by the Stage 2 EIT
  gap migration (`plan/stage2/1_eit_gap_migration_implementation_plan.md`),
  see below.

Run them from the repository root:

```bash
m3resp run examples/ROTARC_example/breath-duration.pipeline.yaml
m3resp run examples/multimodal_example/multimodal.pipeline.yaml
m3resp run examples/eit_full_preprocessing/eit-full.pipeline.yaml
```

Update `inputs.eit_file` in the ROTARC example before running it — it contains
a site-specific path. The other two examples use a relative path to the
committed synthetic Draeger fixture (`data/source/data_from_repo/`), so they
run as-is from the repository root.

### Full EIT preprocessing example

`eit-full.pipeline.yaml` requires the optional `eitprocessing` dependency:

```bash
pip install "m3resp[eit]"
```

It runs the full chain from loading through pixel-resolved parameters and ROI
lung-space masks:

```text
eit.load
  -> eit.detect_rates
  -> eit.mdn_filter
  -> eit.global_impedance
  -> eit.detect_breaths
  -> eit.normalize_breaths
  -> eit.continuous_tiv
  -> eit.eeli
  -> eit.pixel_breaths
  -> eit.pixel_tiv
  -> eit.roi_tiv_lungspace
  -> eit.roi_amplitude_lungspace
  -> eit.roi_watershed
  -> eit.roi_filter_by_size
  -> automatic structured export (outputs.structured_export: true)
```

`eit.roi_filter_by_size.mask` has no default binding — the example binds it
explicitly (`in: { mask: watershed_lungspace_mask }`) rather than relying on
an ambiguous default, since any of the three lung-space masks could otherwise
be the "obvious" choice.

Every step below also emits one or more native `Signal`/`ParameterResult`
objects into `session.signals`/`session.parameter_results` (not shown in the
"writes" column, which lists the raw upstream-shaped context keys used by
later EIT steps) and records per-step provenance via `M3Session._record()`.

| Step | Reads | Writes | Key parameters | Unit | Upstream method |
|---|---|---|---|---|---|
| `eit.load` | `session` | `raw_eit`, `raw_global_impedance`, `eit_sequence` | `file`, `vendor`, `loader_options` | (upstream-defined) | `eitprocessing.datahandling.loading.load_eit_data` |
| `eit.detect_rates` | `signal`, `session` | `respiratory_rate_hz`, `heart_rate_hz`, `rate_detector`, `rate_captures` | `subject_type`, `welch_window_seconds`, `capture` | Hz | `eitprocessing.RateDetection` |
| `eit.mdn_filter` | `signal`, `respiratory_rate_hz`, `heart_rate_hz`, `eit_sequence`, `session` | `filtered_eit`, `filter_captures` | `label` | (upstream-defined) | `eitprocessing.MDNFilter` |
| `eit.global_impedance` | `signal`, `eit_sequence` | `global_impedance` | — | (upstream-defined) | `eitprocessing.EITData.get_summed_impedance` |
| `eit.detect_breaths` | `signal` | `breath_intervals`, `breath_detector` | `min_duration_s` | s | `eitprocessing.BreathDetection` |
| `eit.normalize_breaths` | `breath_intervals`, `session` | — (writes `session.events["eit_breaths"]`) | — | s | n/a |
| `eit.continuous_tiv` | `signal`, `eit_sequence`, `breath_detector` | `continuous_tiv` | — | impedance difference | `eitprocessing.TIV` |
| `eit.eeli` | `signal`, `eit_sequence`, `breath_detector`, `session` | `eeli`, `eeli_result` | `result_label` | impedance | `eitprocessing.EELI` |
| `eit.pixel_breaths` | `eit_data`, `timing_data`, `eit_sequence`, `session` | `pixel_breaths`, `pixel_breath_timing_result` | `phase_correction_mode`, `minimum_duration_seconds`, `result_label` | s | `eitprocessing.PixelBreath` |
| `eit.pixel_tiv` | `filtered_eit`, `signal`, `eit_sequence`, `breath_detector`, `session` | `pixel_tiv`, `pixel_tiv_result` | `tiv_timing`, `result_label` | impedance difference | `eitprocessing.TIV` |
| `eit.roi_tiv_lungspace` | `eit_data`, `timing_data`, `session` | `tiv_lungspace_mask`, `tiv_lungspace_captures` | `threshold` (0 < x < 1) | dimensionless mask | `eitprocessing.TIVLungspace` |
| `eit.roi_amplitude_lungspace` | `eit_data`, `timing_data`, `session` | `amplitude_lungspace_mask`, `amplitude_lungspace_captures` | `threshold` (0 < x < 1) | dimensionless mask | `eitprocessing.AmplitudeLungspace` |
| `eit.roi_watershed` | `eit_data`, `timing_data`, `session` | `watershed_lungspace_mask`, `watershed_captures` | `threshold_fraction` (0 < x < 1) | dimensionless mask | `eitprocessing.WatershedLungspace` |
| `eit.roi_filter_by_size` | `mask` (required, no default binding), `session` | `size_filtered_roi_mask` | `min_region_size` (> 0), `connectivity` (1 or 2) | dimensionless mask | `eitprocessing.FilterROIBySize` |

**`eit.roi_amplitude_lungspace` warning:** upstream does not recommend a
lung-space definition based on amplitude alone, since it can include
reconstruction artifacts; it is computed here primarily to feed
`eit.roi_watershed`. Prefer `eit.roi_tiv_lungspace` or `eit.roi_watershed`'s
output as the functional lung-space mask for downstream analysis.

**NaN in pixel and mask arrays:**

- In a lung-space mask (`tiv_lungspace_mask`, `amplitude_lungspace_mask`,
  `watershed_lungspace_mask`, `size_filtered_roi_mask`), NaN means the pixel
  is excluded from the region of interest; a non-NaN value in `(0, 1]` means
  it is included (optionally weighted).
- In `pixel_breath_timing_result` (shape `(breath, row, column, landmark)`,
  landmark = `[start_time, middle_time, end_time]`), NaN marks a pixel breath
  that could not be determined — including the first and last global breath,
  which `PixelBreath` never resolves by definition (it needs the breaths on
  both sides).
- In `pixel_tiv_result` (shape `(breath, row, column)`), an all-NaN breath is
  kept in place rather than dropped; `metadata["valid_breath_indices"]` and
  `metadata["valid_breath_fraction"]` let you assess coverage without
  scanning the array.

### Array archive and manifest format

Array-valued `ParameterResult`s (EELI, pixel TIV, pixel-breath timing, and
every ROI mask above) are not embedded in `parameter_results.csv` — the
structured export (`outputs.structured_export: true`, or
`session.export_summary(...)`) writes them into one shared,
compressed archive instead:

```text
<output_dir>/parameter_results.csv           # scalar results inline; array
                                              # results reference the archive
<output_dir>/parameter_result_arrays.npz     # one entry per array result
```

Each array row in `parameter_results.csv` has `value` empty and instead
carries:

- **`value_file`** — the archive's filename (`parameter_result_arrays.npz`);
- **`array_key`** — its key in that archive, e.g. `pixel_tivs_0` (deterministic
  and collision-safe: `<result name>_<occurrence index>`);
- **`shape`** / **`dtype`** — the array's shape and dtype;
- **`time_array_key`** — set when the result's `metadata["time"]` was itself
  array-shaped (e.g. per-pixel breath timing); that array moves into the same
  `.npz` under `<array_key>_time`, and `metadata["time"]` in the CSV row
  becomes a short `npz:<key>` reference instead of the raw array.

`method` and the rest of `metadata` (axes, upstream parameters, provenance)
stay in the CSV row as usual. Load the archive back with:

```python
import numpy as np
with np.load("parameter_result_arrays.npz") as archive:
    pixel_tiv = archive["pixel_tivs_0"]
```

When a `DataModelRecorder` is attached and a pipeline run id is available
(`PipelineResult.processing_run_id`), the archive is also recorded as a
`DataFile` (role `parameter`) linked onto that run's
`ProcessingRun.parameter_file_id`. A manual `session.export_summary(...)`
call with no associated run still writes the archive but leaves it unlinked.

## Architecture

| File | Responsibility |
|---|---|
| `workflows/registry.py` | `@register_step` decorator and the global step registry. |
| `workflows/spec.py` | Parse and validate a YAML or JSON spec into typed dataclasses. |
| `workflows/context.py` | `PipelineContext` — the shared artifact store wrapping an `M3Session`. |
| `workflows/engine.py` | `run_pipeline` and `run_spec` — bind arguments, validate, execute. |
| `workflows/steps/` | Built-in steps for EIT, EMG, metrics, session ops, and export. |
| `workflows/utils.py` | Shared utilities: signal slicing, JSON writing, summary logging, `resolve_output_dir` (timestamped output directories). |
| `workflows/summaries.py` | Compact session summaries written to the log after a run. |
