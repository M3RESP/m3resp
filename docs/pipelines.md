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

Worked examples ship with the repository:

- [`examples/ROTARC_example/breath-duration.pipeline.yaml`](../examples/ROTARC_example/breath-duration.pipeline.yaml) —
  computes breath-duration CV from EIT data and writes a ROTARC-style result
  file.
- [`examples/emg_full_preprocessing/emg-full.pipeline.yaml`](../examples/emg_full_preprocessing/emg-full.pipeline.yaml) —
  every EMG/ventilator operation the Stage 2 ReSurfEMG gap migration closed
  onto `ReSurfEMGAdapter`: loading, ECG detection + gating, breath detection,
  moving-baseline, on/offset detection, breath features, ventilator/Pocc
  detection and prerequisites, and all ten clinical quality operations. See
  ["EMG/ventilator pipelines"](#emgventilator-pipelines) below.
- [`examples/multimodal_example/multimodal.pipeline.yaml`](../examples/multimodal_example/multimodal.pipeline.yaml) —
  loads EIT, EMG, and ventilator signals, synchronizes them, processes each
  modality (including ECG removal and Pocc quality), and exports session
  summaries.
- [`examples/annemijn_multimodal/annemijn.pipeline.yaml`](../examples/annemijn_multimodal/annemijn.pipeline.yaml) —
  real EIT + diaphragm sEMG + airway-pressure recording; ECG-contaminated
  surface EMG cleaned with `emg.ecg_gating` before breath detection. No
  ventilator in this recording, so no Pocc steps.

Run them from the repository root:

```bash
m3resp run examples/ROTARC_example/breath-duration.pipeline.yaml
m3resp run examples/emg_full_preprocessing/emg-full.pipeline.yaml
m3resp run examples/multimodal_example/multimodal.pipeline.yaml
m3resp run examples/annemijn_multimodal/annemijn.pipeline.yaml
```

Update `inputs.eit_file` in the ROTARC example before running it — it contains
a site-specific path.

## EMG/ventilator pipelines

Stage 2 of the ReSurfEMG gap migration
(`plan/stage2/2_resurfemg_gap_migration_implementation_plan.md`) keeps
validated ReSurfEMG algorithms behind `ReSurfEMGAdapter` while exposing every
EMG/ventilator operation as a workflow step with native `Signal`/`Event`/
`ParameterResult`/`QualityFlag` outputs. This section documents that step
surface; see the [Stage 2 EIT gap migration
plan](../plan/stage2/1_eit_gap_migration_implementation_plan.md) for the
equivalent EIT-side work.

Using any `emg.*`/ventilator step requires the optional `resurfemg`
dependency:

```bash
pip install "m3resp[emg]"
```

Steps whose upstream call needs it raise `OptionalDependencyError` with that
same install hint if it's missing; `import m3resp` and `m3resp steps` work
without it (upstream imports are lazy, inside the adapter methods that need
them).

### Step table

Every step also writes its raw/compatibility output(s) unchanged (existing
consumers keep working) alongside the native outputs listed here. "Upstream"
means the value is `resurfemg`'s own scientific result, passed through with
provenance recorded (`method`/`metadata.source_function`); "native" means it
comes from an `m3resp.processing` primitive.

| Step | Reads (besides `session`) | Key parameters | Native writes | Implementation |
|---|---|---|---|---|
| `emg.load` | — | `file`, `loader_options` | `raw_emg_signals` (one `Signal`/channel) | upstream loader |
| `emg.preprocess` | — | `channel`, `high_pass_hz`, `low_pass_hz`, `envelope_window_seconds`, `notch_base_frequency`, `notch_quality_factor` | — (raw `processed_emg` dict) | upstream + native notch filter |
| `emg.ecg_detect_peaks` | `processed_emg` | `ecg_channel`, `source` (default `"raw_channel"`), `peak_fraction`, `peak_width_seconds`, `peak_distance_seconds`, `bandpass_filter` | `ecg_peak_events` (one `Event`/peak), `ecg_peak_count_result` | upstream |
| `emg.ecg_gating` | `processed_emg`, `ecg_peak_indices` | `source` (default `"filtered"`), `gate_width_seconds` **xor** `gate_width_samples`, `fill_method` (0-3), `envelope_window_seconds` | `ecg_gated_signal`, `ecg_gate_mask_result` (array) | upstream |
| `emg.ecg_wavelet_denoising` | `processed_emg`, `ecg_peak_indices` | `source`, `hard_thresholding`, `levels`, `wavelet_type`, `fixed_threshold`, `envelope_window_seconds` | `ecg_wavelet_cleaned_signal`, `wavelet_decomposition_result`, `wavelet_thresholds_result`, `wavelet_gate_mask_result` (all arrays) | upstream |
| `emg.detect_breaths` | — | `min_breath_width_seconds`, `half_window_seconds`, `prominence_factor`, `threshold` | `emg_breath_events` | native |
| `emg.moving_baseline` | `processed_emg` | `window_seconds`, `step_seconds`, `percentile` (0-100) | `baseline_signal` | upstream |
| `emg.slopesum_baseline` | `processed_emg` | `window_seconds`, `step_seconds`, `percentile`, `augmented_percentile`, `moving_average_seconds`, `percentile_window_seconds` | `baseline_signal`, `baseline_running_mean_signal`, `baseline_running_std_signal`, `slopesum_baseline_native_detail` (NumPy-only) | upstream |
| `emg.pocc_intervals` | `ventilator_signals`, `pocc_indices` | `peep` (defaults to the pressure median) | `pocc_events` (`BreathEvent`, `modality="pressure"`) | native (`onoff_from_baseline_crossings`) |
| `emg.pocc_time_product` | `ventilator_signals`, `pocc_start_indices`, `pocc_end_indices` | `peep` | `pocc_time_product_result` (unit `<pressure-unit>*s`) | native (`window_integral`) |
| `emg.snr_pseudo` | `processed_emg`, `peak_indices`, `baseline` | `minimum_snr` (optional) | `snr_pseudo_results`; `snr_pseudo_flags` only if `minimum_snr` is set | upstream |
| `emg.percentage_under_baseline` | `processed_emg`, `peak_indices`, `start_indices`, `end_indices`, `baseline` | `aub_window_seconds`, `aub_threshold` | `percentage_under_baseline_results`, `_flags` | upstream |
| `emg.detect_local_high_aub` | `area_under_baseline`, `peak_indices` | `threshold_percentile`, `threshold_factor` | `_flags`, `detect_local_high_aub_threshold_result` | upstream |
| `emg.detect_extreme_time_products` | `time_product`, `peak_indices` | `upper_percentile`, `upper_factor`, `lower_percentile`, `lower_factor` | `_flags`, `detect_extreme_time_products_bounds_result` | upstream |
| `emg.detect_non_consecutive_manoeuvres` | `ventilator_breath_indices`, `pocc_indices` | — | `_flags` (`modality="pressure"`) | upstream |
| `emg.evaluate_bell_curve_error` | `peak_indices`, `start_indices`, `end_indices`, `processed_emg`, `time_product` | — | `_results` (incl. a separate array result per breath for the fitted bell-curve parameters), `_flags` | upstream |
| `emg.evaluate_event_timing` | `peak_indices`, `processed_emg`, `ventilator_breath_indices`, `ventilator_signals` | — | `_results`, `_flags`, `evaluate_event_timing_unmatched_count` | upstream |
| `emg.evaluate_respiratory_rates` | `peak_indices`, `processed_emg`, `ventilator_respiratory_rate` | `minimum_fraction` | `evaluate_respiratory_rates_result`, `_flag` | upstream |
| `emg.pocc_quality` | `ventilator_signals`, `pocc_indices`, `pocc_end_indices`, `pocc_time_products` | `dp_up_10_threshold`, `dp_up_90_threshold`, `dp_up_90_norm_threshold` | `pocc_quality_results` (labeled `dp_up_10`/`dp_up_90`/`dp_up_90_norm`, 3 per manoeuvre), `pocc_quality_flags` (1 per manoeuvre) | upstream |
| `emg.interpeak_dist` | `ecg_peak_indices`, `peak_indices`, `processed_emg` | `threshold` (default `1.1`) | `interpeak_dist_result` (median ECG/EMG interval in seconds + ratio), `interpeak_dist_flag` | upstream |

Run `m3resp steps` for the full list including the pre-existing feature steps
(`emg.time_to_peak`, `emg.pseudo_slope`, `emg.amplitude`, `emg.time_product`,
`emg.area_under_baseline`, `emg.respiratory_rate`) and event-detection steps
(`emg.onoffpeak_baseline_crossing`, `emg.onoffpeak_slope_extrapolation`,
`emg.detect_ventilator_breath`, `emg.find_occluded_breaths`) this migration
routes through the adapter without changing their contracts. The registry
metadata (`reads`/`writes`/`summary`) is enough for a GUI to render controls
and disable an operation whose inputs are missing, without importing
`resurfemg` or inspecting a function's signature.

### Channel selection and time-base rules

- `emg.preprocess`'s `channel` is an index into the raw channel-major array
  from `emg.load` (`session.emg.raw[channel]`), not a label.
- ECG detection's `source` selects a *key* in `processed_emg`
  (`"raw_channel"`/`"filtered"`/`"envelope"`), while `ecg_channel` (when set)
  selects a *raw* channel index instead - use `ecg_channel` when a recording
  has a dedicated reference ECG channel (e.g. the committed
  `emg_data_synth_quiet_breathing.Poly5` fixture's channel 0), and `source`
  when ECG is mixed directly into the EMG channel being processed (no
  separate reference channel).
- No step assumes EMG and ventilator sample rates are equal - Pocc/timing
  steps convert each side to seconds using its own `fs` before comparing
  (`emg.evaluate_event_timing`'s per-pair results record both source sample
  indices and both sampling frequencies for this reason).
- `emg.detect_breaths`'s defaults (`min_breath_width_seconds=1.0`) assume
  breath bursts at least a second wide. Shorter/lower-contrast envelopes
  (e.g. `data_from_repo`'s "quiet breathing" fixtures) need
  `min_breath_width_seconds`/`prominence_factor` tuned down - see the inline
  comments in `emg-full.pipeline.yaml` and `multimodal.pipeline.yaml` for
  values that were verified against each fixture's actual breath timing.

### ECG-removal alternatives

`emg.ecg_gating` and `emg.ecg_wavelet_denoising` both naturally write
`processed_emg_after_ecg`. Pick one; a pipeline uses output renaming to keep
the downstream key `processed_emg`:

```yaml
- uses: emg.preprocess
  out: { processed_emg: processed_emg_before_ecg }

- uses: emg.ecg_detect_peaks
  in: { processed_emg: processed_emg_before_ecg }

- uses: emg.ecg_gating          # or emg.ecg_wavelet_denoising
  in: { processed_emg: processed_emg_before_ecg }
  out: { processed_emg_after_ecg: processed_emg }
```

`gating`'s `fill_method` keeps ReSurfEMG's meanings: `0` zeros the gated
region, `1` interpolates between its edges (the default), `2` fills with the
mean of a neighboring segment, `3` replaces it with a running-RMS estimate.
`wavelet_denoising` additionally zero-pads the signal internally to a
multiple of `2**levels` for its stationary wavelet transform - the cleaned
signal, thresholds, and gate mask are trimmed back to the original length,
but the decomposition array (`wavelet_decomposition_result`) stays at the
padded length; both lengths are recorded in its metadata
(`original_length`/`padded_length`). All four array-valued outputs (gate
masks, decomposition, thresholds) go through the same shared
`parameter_result_arrays.npz` archive as every other array-valued
`ParameterResult` - see "Array export" below.

### Baseline alternatives

`emg.moving_baseline` and `emg.slopesum_baseline` both write `baseline`; the
registry marks them as alternatives for the same reason as the ECG-removal
steps above. `slopesum_baseline`'s compatibility output
(`slopesum_baseline_detail`) keeps ReSurfEMG's `pandas.Series`; its native
sibling (`slopesum_baseline_native_detail`) is NumPy-only and does not cross
that boundary.

### Measurements vs. clinical pass/fail criteria

A quality step's raw upstream output is not automatically a clinical
pass/fail result. `m3resp.processing.quality.quality_flag_from_result` (used
by the shared `to_quality_flags` path) and each granular quality step follow
the same rule: a boolean or boolean array is a real criterion; a numeric
scalar or array (e.g. `snr_pseudo`'s per-peak ratios) is a *measurement*, and
only becomes a `QualityFlag` when a threshold is actually configured
(`emg.snr_pseudo`'s `minimum_snr`, for example). A measurement-only flag uses
`passed=False` with `metadata["measurement_only"]=True` - the same "not
evaluated" convention `metadata["skipped"]=True` uses for a flag skipped for
missing inputs - rather than inventing a pass/fail verdict the upstream
function never defined.

### Pocc prerequisites and thresholds

`emg.pocc_quality` needs Pocc end indices and pressure-time products that
`emg.find_occluded_breaths` alone doesn't produce - `emg.pocc_intervals` and
`emg.pocc_time_product` supply them:

```
emg.find_occluded_breaths  ->  pocc_indices
emg.pocc_intervals          ->  pocc_start_indices, pocc_end_indices
emg.pocc_time_product       ->  pocc_time_products
emg.pocc_quality             (needs all of the above)
```

`emg.pocc_quality`'s three thresholds (`dp_up_10_threshold=0.0`,
`dp_up_90_threshold=2.0`, `dp_up_90_norm_threshold=0.8`) are ReSurfEMG's
defaults for the pressure upslope after occlusion release (Warnaar et al.
2024); its criteria matrix is exposed as three explicitly labeled result
names (`pocc_quality_dp_up_10`/`_dp_up_90`/`_dp_up_90_norm`) rather than a
raw 3-by-N array.

### Missing, skipped, and unmatched events

- A quality step whose prerequisites are entirely missing (no ventilator
  input, no detected breaths) simply has nothing to iterate over - it
  produces no native results for that run rather than raising, so a pipeline
  with a partial dataset still completes.
- `emg.evaluate_event_timing` pairs EMG and ventilator events by position; if
  the two lists have different lengths, it still pairs as many as it can
  (keeping the existing raw-output truncation behavior for backward
  compatibility) but also reports `evaluate_event_timing_unmatched_count` and
  adds an `evaluate_event_timing_unmatched` warning `QualityFlag` - the
  unmatched events are never silently dropped without a trace.
- Per-breath/per-manoeuvre native results use `breath_id=str(position)` (a
  stable event ID is future work) and record the source peak/pressure sample
  index in `metadata["peak_sample_index"]`, so a GUI can always explain which
  detected event a given flag or measurement belongs to.

### Array export

Array-valued `ParameterResult`s (gate masks, wavelet decomposition/
thresholds, bell-curve fit parameters, ...) are not written inline into
`parameter_results.csv` - like EIT's array-valued results, they're added to
`session.parameter_results` and go through the same shared
`parameter_result_arrays.npz` archive (`export_summary`/`export.*`), with a
CSV row (`array_key`, `shape`, `dtype`) pointing into it. There is no
EMG-specific array exporter.

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
