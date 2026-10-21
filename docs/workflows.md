# Workflows

A {term}`workflow` file is a YAML or JSON file with a list of {term}`steps <step>`. M3Resp runs the steps in order. Each step reads {term}`results <result>` by name, does one calculation and writes its results under a name. Before the run, M3Resp checks that each input of a step comes from an earlier step.

This page describes all parts of a workflow file. For the tasks, see {doc}`how-to/write-a-workflow`, {doc}`how-to/check-a-workflow` and {doc}`how-to/add-a-step`. A {term}`preset` (`session.run_preset("eit")`) is a short, built-in alternative to a workflow file (see {doc}`developer/preset-contracts`).

## Run a workflow

From the command line:

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
m3resp run examples/multimodal_example/multimodal.workflow.yaml
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv run m3resp run examples/multimodal_example/multimodal.workflow.yaml
```
:::
::::

From Python:

```python
from m3resp import run_spec, run_workflow

result = run_spec("examples/multimodal_example/multimodal.workflow.yaml")
print(result.status)              # succeeded
print(result.resolved_output_dir) # the folder with the result files
```

| Function | What it runs |
|---|---|
| `run_spec(path)` | The steps, then the `outputs` section: the result files, the figures and the run manifest |
| `run_workflow(spec)` | Only the steps. `spec` is a path, a dictionary or a `WorkflowSpec`. |

Both return a `WorkflowResult`:

| Field or method | Contents |
|---|---|
| `status` | `"succeeded"`, `"failed"` or `"cancelled"` |
| `value(name)` | One result of the run, by name, for example `result.value("breath_intervals")` |
| `outputs` | All results of the run, by name |
| `session` | The {term}`session` with all recordings and results |
| `warnings` | The Python warnings of the steps |
| `step_records` | One record per step, with its status and time |
| `resolved_output_dir`, `manifest_path` | The results folder and the run manifest |
| `run_id`, `processing_run_id`, `started_at`, `finished_at`, `duration_seconds` | The identifiers and the times of the run |

## An example workflow file

```yaml
name: my-workflow

inputs:
  eit_file: ../data/subject.bin   # use it in a step as "@eit_file"
  vendor: draeger

outputs:
  dir: ../output/results
  mode: automatic
  timestamped: true

steps:
  - uses: eit.load
    with: { file_path: "@eit_file", vendor: "@vendor" }

  - uses: eit.detect_rates
    in: { signal: raw_eit }
    with: { subject_type: adult }

  - uses: eit.mdn_filter
    in: { signal: raw_eit }

  - uses: eit.global_impedance
    in: { signal: filtered_eit }

  - uses: eit.detect_breaths
    in: { signal: global_impedance }

  - uses: metric.interval_cv
    in: { intervals: breath_intervals }
    out: { cv: breath_duration_cv }
```

## The keys of a step

| Key | Contents |
|---|---|
| `uses` | The name of the step, for example `eit.detect_breaths` |
| `id` | An optional fixed name of this step in the file (see {ref}`workflow-step-ids`) |
| `in` | Which result each input of the step reads: `{input name: result name}`. Without `in`, the step reads its default result. |
| `with` | The {term}`settings <setting>` of the step |
| `out` | New names for the results of the step: `{result name: new name}`. Use it to run the same step twice and keep both results. |

In `with`, a text that starts with `@` is a value from `inputs`, also inside lists and dictionaries: `with: {paths: ["@a", "@b"]}`. Write `@@text` for the text `@text`.

M3Resp reads a relative file path in a step (for example `file_path` of `eit.load`) and in `outputs.dir` from the folder of the workflow file.

## The sections of a workflow file

| Section | Contents |
|---|---|
| `name` | The name of the workflow, for the log |
| `schema_version` | `1` for strict checks (see {ref}`workflow-strict`) |
| `description` | A short text about the workflow |
| `inputs` | Values that the steps use with `@name`, for example file paths |
| `outputs` | The results folder and the result files (see {ref}`workflow-outputs`) |
| `experiment` | Study details for the file names of `export.rotarc_result`: `subject_id`, `mode`, `timepoint`, `run_identifier`, `selection` (default `"selected"`) |
| `metadata` | Free notes about the study, the site or the session, as a dictionary |
| `execution` | `error_policy` (only `fail_fast`) and `seed`. M3Resp records the seed, but no step uses it. |
| `steps` | The list of steps |

(workflow-outputs)=
## The outputs section

| Setting | Default | Meaning |
|---|---|---|
| `dir` | None | The results folder. Without `dir`, M3Resp writes no files. |
| `mode` | Set by M3Resp | What happens after the steps (see the next table) |
| `timestamped` | `true` | Each run writes to its own subfolder `<dir>/YYYYMMDD_HHMMSS` |
| `summary_json` | `true` | Write `summary.json` |
| `event_csvs` | `true` | Write one CSV file per list of events |
| `parameters_csv` | `false` | Write `parameters.csv` |
| `postprocessing` | `false` | Add the EMG results of `postprocess_emg` |
| `structured_export` | `false` | Write the other result files (see {doc}`how-to/export-results`) |
| `figures` | `false` | Save the figures |
| `checksums` | `false` | Add a SHA-256 checksum of each input file to the run manifest |

| `mode` | What happens after the steps |
|---|---|
| `automatic` | M3Resp writes the result files after a successful run. Use it for a workflow without an `export.*` step. |
| `explicit` | The `export.*` steps write the files. M3Resp writes the run manifest and the figures. |
| `none` | M3Resp writes no result files, figures or run manifest. The `export.*` steps still run. |

- With `schema_version: 1`, set `mode` when you set `dir`. Without `schema_version`, M3Resp selects `explicit` if the file has an `export.*` step, else `automatic`, and gives a `FutureWarning`.
- M3Resp saves the figures only after a successful run. After a failed or cancelled run, the files of the finished steps stay.
- All export paths of one run use the same results folder, also with `timestamped: true`. A step of your own can read this folder from `_resolved_output_dir` (see {doc}`how-to/add-a-step`).

(workflow-strict)=
## Strict and lenient files

| | Without `schema_version` | `schema_version: 1` |
|---|---|---|
| Unknown key | Ignored | Error (`WorkflowSpecError`) |
| A setting that must be true or false, given as text (`"yes"`) | Converted, with a `FutureWarning` | Error |
| `outputs.mode` with `outputs.dir` | Optional | Required |

With `schema_version: 1`, true and false are the YAML words `true` and `false` (also `yes`, `no`, `on` and `off` without quotes), or `true` and `false` in JSON. Both kinds of files give the same steps and the same results. Use `schema_version: 1` for new files.

(workflow-step-ids)=
## Step identifiers

Each step has an `id`. If you do not give one, M3Resp makes `step_<position>_<name>`, for example `step_003_eit.detect_breaths`. This identifier changes when you add or move steps. Give an `id` to a step that other tools must find again, for example in `run_manifest.json`:

```yaml
steps:
  - id: detect_breaths
    uses: eit.detect_breaths
    in: { signal: global_impedance }
```

The `id` of each step must be different from the others.

(workflow-command-line)=
## The command line

| Command | What it does |
|---|---|
| `m3resp run file.yaml` | Runs the workflow. `--dry-run` shows the steps and runs nothing. `--debug` shows the full Python error. Ctrl-C stops the run after the current step. |
| `m3resp validate file.yaml` | Checks the file (see {doc}`how-to/check-a-workflow`). `--readiness` also checks the input files and the packages. `--json` gives the report as JSON. |
| `m3resp steps` | Lists all steps. A step that needs a package that is not installed is marked `[missing_optional_dependency]`. `--details` adds the settings and the results of each step. `--json` gives JSON. |
| `m3resp describe <step>` | Shows all details of one step, for example `m3resp describe eit.detect_rates` |

If you installed M3Resp with uv, put `uv run` in front of each command, for example `uv run m3resp steps`.

| Exit code | Meaning |
|---:|---|
| 0 | Success |
| 1 | A wrong command or option |
| 2 | The workflow file has an error, or M3Resp cannot read it |
| 3 | Readiness error, for example a missing input file |
| 4 | A step failed during the run |
| 5 | The run was cancelled |

## Built-in steps

Run `m3resp steps` for the list of all steps. The groups are:

| Group | Steps for |
|---|---|
| `eit.*` | Load and cut EIT data, filter, find the rates and the breaths, calculate TIV, EELI, pixel TIV and lung regions |
| `emg.*` | Load and cut EMG data, filter, remove the ECG, find the baseline and the breaths, calculate the parameters and the quality checks |
| `ventilator.*` | Load and cut ventilator data, select the channels, find the breaths and the occlusions (Pocc) |
| `sync.*` | Put the recordings on the shared clock: `sync.raw_modalities`, `sync.skip`, `sync.estimate_offset`, `sync.apply_estimated_offset` (see {doc}`concepts/synchronization-overview`) |
| `metric.*` | Calculations on lists, for example the coefficient of variation of the breath durations |
| `export.*` | Write results to files: values, JSON, the session summary and the ROTARC result |

The steps that cut data (see {doc}`concepts/slicing`):

| Step | What it cuts |
|---|---|
| `emg.slice_recording`, `eit.slice_recording` | The loaded recording, with all data in the same file. The start time of the recording moves, so it stays in line with the other recordings. Times in seconds from the first sample. |
| `ventilator.slice_recording` | Each separate ventilator recording (loaded with `source: ventilator`) |
| `eit.slice_signal` | One signal in the workflow. The loaded recording does not change. |

The older names `emg.slice` and `eit.slice` still work.

### Load EIT data

Two steps load EIT data:

| Step | Reads |
|---|---|
| `eit.load` | A Dräger, Sentec or Timpel file, with `eitprocessing`. `vendor` is required. `sample_frequency`, `first_frame` and `max_frames` are optional. |
| `eit.load_custom` | Any other file, with a loader that you write in Python. M3Resp calls it as `loader(file_path, vendor=None, **loader_options)`, and it returns an `eitprocessing` `Sequence`. |

Both steps give the same results (`raw_eit`, `raw_global_impedance_signal`, `raw_pixel_impedance_signal`, ...), so the next steps work with both. Give the loader to the EIT adapter of the session before the run:

```python
from m3resp.adapters import EITProcessingAdapter
from m3resp.workflows import run_workflow

result = run_workflow(
    "my-workflow.yaml", eit_adapter=EITProcessingAdapter(loader=my_loader)
)
```

Without a loader, `eit.load_custom` stops with an error.

A device can split one recording into several files. The `file_path` of both steps also takes a list of these files, in the order of the recording:

```yaml
- uses: eit.load
  with: { file_path: [part_1.bin, part_2.bin], vendor: draeger }
```

- `eit.load` reads each file and joins them into one recording, in the order of the list.
- Each file must start after the end of the file before it. If the files are in the wrong order or overlap, the workflow stops with an error that names the two files.
- A pause between two files stays in the time axis and gives a warning with its length, because the filters and the breath detection run across the pause.
- The files must come from the same device, with the same image size and sampling rate.
- `first_frame` and `max_frames` work only with one file, because `eitprocessing` would apply them to each file. To cut the joined recording, use `eit.slice_recording`.
- `eit.load_custom` gives the list to the loader as it is.

### Ventilator channels

`ventilator.channels` selects `airway_pressure`, `flow` and `volume` by default. For a recording with only pressure, set `channels: [airway_pressure]`.

- M3Resp selects the channels by their names. If no name is known, it uses the positions 0, 1 and 2. An unknown name gives a warning.
- A channel number (for example `airway_pressure_channel: 0`, counted from 0) replaces the selection by name.
- The channels keep their recorded units. The Pocc pressure limits use the pressure unit, and the pressure time products use the pressure unit × s.
- With more than one pressure channel, the steps use the main airway pressure channel, also with names such as `airway_pressure__pod`.

## EMG and ventilator steps

The EMG and ventilator steps need the package `resurfemg` (`python -m pip install -e ".[emg]"`, or `uv sync --extra emg`). Without it, these steps raise `OptionalDependencyError`. `import m3resp` and `m3resp steps` work also without it.

Each step also writes its result in the form that `resurfemg` gives, beside the M3Resp results in the table. In the last column, "resurfemg" means that the calculation comes from `resurfemg`, and "M3Resp" that it is part of M3Resp.

| Step | Reads (besides the session) | Main settings | M3Resp results | Calculation |
|---|---|---|---|---|
| `emg.load` | | `file_path`, `loader_options` | `raw_emg_signals` (one signal per channel) | resurfemg |
| `emg.slice_recording` | | `start_seconds`, `end_seconds` (default: to the end) | `emg_slice` | M3Resp. Also cuts the ventilator channels from the same file. |
| `ventilator.slice_recording` | | `start_seconds`, `end_seconds` (default: to the end) | `ventilator_slice` | M3Resp. Refuses ventilator data from the EIT or EMG file. |
| `emg.preprocess` | | `channel`, `high_pass_hz` (20), `low_pass_hz` (500, at most 95% of the Nyquist frequency), `notch_base_frequency`, `notch_max_frequency`, `notch_quality_factor`, `notch_before_bandpass` (`false`) | `processed_emg`, without an envelope | `emg.select_channel`, `emg.notch_filter` and `emg.bandpass_filter` in one step |
| `emg.select_channel` | | `channel` | `processed_emg` with the selected channel | M3Resp |
| `emg.notch_filter` | `processed_emg` | `base_frequency` (Hz, required), `max_frequency` (the Nyquist frequency), `quality_factor` (30) | `notch_filtered_signal` | M3Resp |
| `emg.bandpass_filter` | `processed_emg` | `high_pass_hz` (20), `low_pass_hz` (500, at most 95% of the Nyquist frequency) | `bandpass_filtered_signal` | resurfemg |
| `emg.envelope` | `processed_emg` | `window_seconds` (0.25), `method` (`"rms"`, `"arv"` or `"median"`), `source` (the signal after the ECG removal, else the filtered signal) | `envelope_signal` | M3Resp |
| `emg.ecg_detect_peaks` | `processed_emg` | `ecg_channel`, `source` (`"raw_channel"`), `peak_fraction`, `peak_width_seconds`, `peak_distance_seconds`, `bandpass_filter` | `ecg_peak_events` (one event per R peak), `ecg_peak_count_result` | resurfemg |
| `emg.ecg_gating` | `processed_emg`, `ecg_peak_indices` | `source` (`"filtered"`), `gate_width_seconds` or `gate_width_samples`, `fill_method` (0 to 3) | `ecg_gated_signal`, `ecg_gate_mask_result` | resurfemg |
| `emg.ecg_wavelet_denoising` | `processed_emg`, `ecg_peak_indices` | `source`, `hard_thresholding`, `levels`, `wavelet_type`, `fixed_threshold` | `ecg_wavelet_cleaned_signal`, `wavelet_decomposition_result`, `wavelet_thresholds_result`, `wavelet_gate_mask_result` | resurfemg |
| `emg.detect_breaths` | `baseline` (optional) | `min_breath_width_seconds`, `half_window_seconds`, `prominence_factor`, `threshold`, `merge_close_peaks_within_width` (`false`) | `emg_breath_events` | M3Resp |
| `emg.moving_baseline` | `processed_emg` | `window_seconds`, `step_seconds`, `percentile` (0 to 100) | `baseline_signal` | resurfemg |
| `emg.slopesum_baseline` | `processed_emg` | `window_seconds`, `step_seconds`, `percentile`, `augmented_percentile`, `moving_average_seconds`, `percentile_window_seconds` | `baseline_signal`, `baseline_running_mean_signal`, `baseline_running_std_signal`, `slopesum_baseline_native_detail` | resurfemg |
| `ventilator.detect_pressure_breaths` | `ventilator_signals` (with `airway_pressure`) | `smoothing_seconds` (0.2), `min_depth` (0.15, pressure unit), `min_interval_seconds` (2.0) | `ventilator_breath_indices` (the lowest point of each pressure dip, for spontaneous breathing only) | M3Resp |
| `ventilator.pocc_intervals` | `ventilator_signals`, `pocc_indices` | `baseline_window_seconds` (7.5), `baseline_step_seconds` (0.2), `baseline_percentile` (33): a moving baseline of the airway pressure | `pocc_events`, `pressure_baseline` | M3Resp |
| `ventilator.pocc_time_product` | `ventilator_signals`, `pocc_start_indices`, `pocc_end_indices`, `pressure_baseline`, `pocc_indices` | `include_aub` (`true`: adds the area under the baseline, as the PTPocc of resurfemg), `aub_window_seconds` (5) | `pocc_time_product_result` (pressure unit × s) | M3Resp |
| `emg.snr_pseudo` | `processed_emg`, `peak_indices`, `baseline` | `minimum_snr` (optional) | `snr_pseudo_results`, and `snr_pseudo_flags` if `minimum_snr` is set | resurfemg |
| `emg.percentage_under_baseline` | `processed_emg`, `peak_indices`, `start_indices`, `end_indices`, `baseline` | `aub_window_seconds`, `aub_threshold` | `percentage_under_baseline_results`, `percentage_under_baseline_flags` | resurfemg |
| `emg.detect_local_high_aub` | `area_under_baseline`, `peak_indices` | `threshold_percentile`, `threshold_factor` | `detect_local_high_aub_flags`, `detect_local_high_aub_threshold_result` | resurfemg |
| `emg.detect_extreme_time_products` | `time_product`, `peak_indices` | `upper_percentile`, `upper_factor`, `lower_percentile`, `lower_factor` | `detect_extreme_time_products_flags`, `detect_extreme_time_products_bounds_result` | resurfemg |
| `ventilator.detect_non_consecutive_manoeuvres` | `ventilator_breath_indices`, `pocc_indices` | | `detect_non_consecutive_manoeuvres_flags` | resurfemg |
| `emg.evaluate_bell_curve_error` | `peak_indices`, `start_indices`, `end_indices`, `processed_emg`, `time_product` | | `evaluate_bell_curve_error_results` (with the fitted bell curve per breath), `evaluate_bell_curve_error_flags` | resurfemg |
| `emg.evaluate_event_timing` | `peak_indices`, `processed_emg`, `ventilator_breath_indices`, `ventilator_signals` | | `evaluate_event_timing_results`, `evaluate_event_timing_flags`, `evaluate_event_timing_unmatched_count` | resurfemg |
| `emg.evaluate_respiratory_rates` | `peak_indices`, `processed_emg`, `ventilator_respiratory_rate` | `minimum_fraction` | `evaluate_respiratory_rates_result`, `evaluate_respiratory_rates_flag` | resurfemg |
| `ventilator.pocc_quality` | `ventilator_signals`, `pocc_indices`, `pocc_end_indices`, `pocc_time_products` | `dp_up_10_threshold`, `dp_up_90_threshold`, `dp_up_90_norm_threshold` | `pocc_quality_results` (`dp_up_10`, `dp_up_90` and `dp_up_90_norm` per occlusion), `pocc_quality_flags` (one per occlusion) | resurfemg |
| `emg.interpeak_dist` | `ecg_peak_indices`, `peak_indices`, `processed_emg` | `threshold` (1.1) | `interpeak_dist_result` (the median ECG and EMG peak distance in seconds, and their ratio), `interpeak_dist_flag` | resurfemg |

`m3resp steps` also lists the parameter steps (`emg.time_to_peak`, `emg.pseudo_slope`, `emg.amplitude`, `emg.time_product`, `emg.area_under_baseline`, `emg.respiratory_rate`) and the detection steps (`emg.onoffpeak_baseline_crossing`, `emg.onoffpeak_slope_extrapolation`, `ventilator.detect_breaths`, `ventilator.find_occluded_breaths`).

### Channels and sampling rates

- The `channel` of `emg.preprocess` and `emg.select_channel` is a channel number of the loaded file, counted from 0. It is not a channel name.
- In `emg.ecg_detect_peaks`, `source` selects a signal of `processed_emg`: `"raw_channel"`, `"filtered"` or `"envelope"`. `ecg_channel` selects a raw channel instead. Use `ecg_channel` for a separate ECG channel, and `source` when the ECG is inside the EMG channel.
- The EMG and the ventilator data can have different sampling rates. The Pocc and timing steps change each side to seconds with its own sampling rate before they compare.
- The default `min_breath_width_seconds` of `emg.detect_breaths` is 0.5 s. It also finds the breaths of fast breathing, for example 22 breaths per minute. For noisy recordings, change `min_breath_width_seconds` or `prominence_factor`.

### The order of the EMG steps and the envelope

The EMG steps run in this order:

```text
emg.select_channel -> emg.notch_filter (optional) -> emg.bandpass_filter
  -> ECG removal (emg.ecg_gating or emg.ecg_wavelet_denoising)
  -> emg.envelope -> baseline -> emg.detect_breaths
```

- The notch filter can also come after the band-pass filter.
- `emg.preprocess` runs the first three steps as one step, with all their settings. To change the order, or to add a step of your own between them, use the separate steps.
- Each step gives the processed EMG to the next step as a dictionary (`processed_emg`). It holds the selected channel (`"raw_channel"`), the filtered signal (`"filtered"`), the signal after the ECG removal (`"ecg_cleaned"`), the envelope (`"envelope"`), the sampling rate in Hz (`"fs"`) and the settings that the steps used (`"filter"`).

Only `emg.envelope` calculates the envelope. It uses the signal after the ECG removal, if there is one. The filter and ECG removal steps give the processed EMG without an envelope, because an envelope of an earlier signal does not match the new signal. The steps that need the envelope (baselines, breath detection, parameters per breath and quality checks) stop with an error that names `emg.envelope` if there is no envelope.

Each step writes a result name of its own, so connect the steps with `in` and `out`. The last step before the baseline writes to `processed_emg`, which the next steps read:

```yaml
- uses: emg.select_channel
  with: { channel: 1 }
  out: { processed_emg: selected_emg }

- uses: emg.notch_filter
  in: { processed_emg: selected_emg }
  with: { base_frequency: 50.0 }

- uses: emg.bandpass_filter
  in: { processed_emg: processed_emg_after_notch }
  with: { high_pass_hz: 20.0, low_pass_hz: 500.0 }

- uses: emg.ecg_detect_peaks
  in: { processed_emg: processed_emg_after_bandpass }
  with: { source: filtered }

- uses: emg.ecg_wavelet_denoising
  in: { processed_emg: processed_emg_after_bandpass }

- uses: emg.envelope
  in: { processed_emg: processed_emg_after_ecg }
  with: { window_seconds: 0.25, method: rms }
  out: { processed_emg_with_envelope: processed_emg }
```

(workflow-ecg-removal)=
### Remove the ECG: gating or wavelet denoising

`emg.ecg_gating` and `emg.ecg_wavelet_denoising` both write `processed_emg_after_ecg`. Use one of them, then calculate the envelope from its result:

```yaml
- uses: emg.preprocess
  out: { processed_emg: processed_emg_before_ecg }

- uses: emg.ecg_detect_peaks
  in: { processed_emg: processed_emg_before_ecg }

- uses: emg.ecg_gating  # or emg.ecg_wavelet_denoising
  in: { processed_emg: processed_emg_before_ecg }

- uses: emg.envelope
  in: { processed_emg: processed_emg_after_ecg }
  out: { processed_emg_with_envelope: processed_emg }
```

- The `fill_method` of gating: 0 fills the gap with zeros, 1 interpolates between its edges (default), 2 uses the mean of a segment next to the gap, 3 uses a running RMS.
- Wavelet denoising makes the signal longer with zeros, to a multiple of `2**levels`, for its calculation. It cuts the cleaned signal, the thresholds and the gate mask back to the original length. `wavelet_decomposition_result` keeps the longer length. Its metadata has both lengths (`original_length`, `padded_length`).

### Baseline: moving or slope sum

`emg.moving_baseline` and `emg.slopesum_baseline` both write `baseline`. Use one of them. `slopesum_baseline` also writes the `pandas.Series` of `resurfemg` (`slopesum_baseline_detail`), and the same values as NumPy arrays (`slopesum_baseline_native_detail`).

### Measurements and pass or fail

A quality step gives a pass or a fail only for a real criterion:

- A result that is true or false is a criterion. It gives a quality flag that passed or failed.
- A number (for example the SNR of each peak of `emg.snr_pseudo`) is a measurement. It gives a pass or a fail only when you set a limit, for example `minimum_snr`. Without a limit, the flag has `passed=False` and `metadata["measurement_only"] == True`.
- A check that could not run because an input is missing has `passed=False` and `metadata["skipped"] == True`.

### Pocc steps

`ventilator.pocc_quality` needs the end of each occlusion and the pressure time products. Run these steps first:

```text
ventilator.find_occluded_breaths  ->  pocc_indices
ventilator.pocc_intervals         ->  pocc_start_indices, pocc_end_indices
ventilator.pocc_time_product      ->  pocc_time_products
ventilator.pocc_quality              (needs all of the above)
```

The default limits of `ventilator.pocc_quality` (`dp_up_10_threshold=0.0`, `dp_up_90_threshold=2.0`, `dp_up_90_norm_threshold=0.8`) are the limits of `resurfemg` for the pressure rise after the release of the occlusion (Warnaar et al. 2024).

### Missing and unmatched events

- A quality step with no inputs (no ventilator data, no breaths) gives no results and does not stop the workflow.
- `emg.evaluate_event_timing` pairs the EMG and the ventilator events by their position. If the two lists have different lengths, it pairs as many as it can, gives `evaluate_event_timing_unmatched_count`, and adds a warning flag `evaluate_event_timing_unmatched`.
- The results per breath or per occlusion have `breath_id` = the position in the list, counted from 0. Their metadata has the sample of the extremum in `extremum_sample_index`, and its time in seconds in `extremum_time` when the sampling rate is known. `emg.remove_invalid_breaths` finds the breaths of the quality flags by this sample. It also reads the older key `peak_sample_index`, with a `UserWarning`.

### Array results

Results with an array as value (gate masks, wavelet results, bell curve fits) go to `parameter_result_arrays.npz`. Their row in `parameter_results.csv` gives the key, the shape and the data type of the array (see {doc}`how-to/export-results`).

## Example workflow files

The `examples` folder has these workflow files. Run them from the M3Resp source folder.

| File | What it does |
|---|---|
| `examples/multimodal_example/multimodal.workflow.yaml` | A short EIT, EMG and ventilator workflow, easy to read from start to end |
| `examples/eit_full_preprocessing/eit-full.workflow.yaml` | All EIT steps: loading, rates, MDN filter, breaths, TIV, EELI, pixel breaths, pixel TIV and the four lung region steps |
| `examples/emg_full_preprocessing/emg-full.workflow.yaml` | All EMG and ventilator steps: loading, ECG removal, baseline, breaths, start and end of breaths, parameters, Pocc and the ten quality checks |
| `examples/multimodal_full/multimodal-full.workflow.yaml` | All steps of the three modalities, with `sync.raw_modalities` before the processing |
| `examples/ROTARC_example/breath-duration.workflow.yaml` | The coefficient of variation of the EIT breath durations, written as a ROTARC result file |
| `examples/multidomain_recording1/recording1_a2.workflow.yaml` | The EMG analysis of one time window of a real diaphragm recording: notch filter, band-pass filter, wavelet ECG removal, median envelope and breaths. It needs `tests/data/TestPS3.txt`, which is not in the repository. |

All examples except the last one use the synthetic recordings in `tests/data`. The examples `eit-full`, `emg-full`, `multimodal-full` and `breath-duration` use `schema_version: 1`, a fixed `id` for each step and an explicit `outputs.mode`.

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
m3resp run examples/eit_full_preprocessing/eit-full.workflow.yaml
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv run m3resp run examples/eit_full_preprocessing/eit-full.workflow.yaml
```
:::
::::

## Run events, cancellation and errors

`run_spec` and `run_workflow` accept two optional arguments, `event_sink` and `cancellation_token`:

- `event_sink` is a function that M3Resp calls at each event of the run: `workflow_started`, `step_started`, `step_completed`, `step_failed`, `step_warning`, `workflow_completed`, `workflow_failed` and `workflow_cancelled`. Use it, for example, for a progress bar.
- A `CancellationToken` stops the run: after `token.cancel()`, the current step finishes and the next step does not start. The status is then `"cancelled"`, and the finished results and the manifest stay. Ctrl-C in `m3resp run` does the same. A step that hangs does not stop with Ctrl-C. Stop it with Ctrl-\ or `kill`.
- M3Resp collects the Python warnings of each step in its step record and in `WorkflowResult.warnings`.
- If a step fails, M3Resp raises `WorkflowExecutionError`. It has the step (`step_id`, `position`, `operation_id`), the run (`run_id`, `started_at`), the step records up to the error, and the original error as its cause.

## The run manifest

With `outputs.dir` and a `mode` other than `none`, `run_spec` writes `run_manifest.json` to the results folder. It writes the file with `status: "running"` before the first step, and replaces it with the final status at the end, also after a failure or a cancellation. So a stopped run always leaves a manifest with its status.

The manifest has the run identifier, the status, the name of the workflow, the times, the `inputs`, each step record, the warnings, and an error block after a failure. M3Resp hides the values of `inputs` with `password`, `secret`, `token`, `credential` or `api_key` in their name. With `outputs.checksums: true`, it also has a SHA-256 checksum of each input file.

## Results in two forms

Some steps write the same result in two forms: an M3Resp object (`Signal`, `Event`, `ParameterResult` or `QualityFlag`) with its method and source, and the form of the outside package, for example a `pandas.Series`. Use the M3Resp object in new code. `m3resp steps --details` and `m3resp describe <step>` show the results of each step.

## For applications: WorkflowService

`m3resp.workflows.WorkflowService` is for an application, for example a graphical interface, that uses M3Resp. Each method takes a workflow (a path, a dictionary or a `WorkflowSpec`) and returns only JSON data:

```python
from m3resp.workflows import WorkflowService

service = WorkflowService()
service.list_capabilities(prefix="eit.")         # all eit.* steps
service.describe_capability("eit.detect_rates")  # one step
service.validate_workflow(spec, readiness=True)  # the check report
service.compile_workflow(spec)                   # the steps that will run
service.run_workflow(spec, event_sink=..., cancellation_token=...)
```

For each step, `describe_capability` gives `"available"`, `"missing_optional_dependency"` or `"deprecated"`, without an import of the outside package. So an application can show the steps and their settings, and disable a step whose package is not installed.

## Where the code is

| File | Contents |
|---|---|
| `workflows/registry.py` | `@register_step` and the list of steps |
| `workflows/spec.py` | Reads and checks a YAML or JSON workflow file |
| `workflows/context.py` | `WorkflowContext`: the results of a run, with the session |
| `workflows/engine/` | `run_workflow` (`execution.py`), `run_spec` (`spec_runner.py`) and the checks (`diagnostics.py`) |
| `workflows/steps/` | The built-in steps |
| `workflows/utils.py` | Shared functions, for example `resolve_output_dir` for the results folder |
| `workflows/summaries.py` | The short summary of a session in the log after a run |

Continue with {doc}`how-to/write-a-workflow`.
