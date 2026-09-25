# Pipelines

M3Resp workflows are written as YAML specs and executed by the pipeline engine. A spec is an ordered list of named steps. Each step reads named artifacts from a shared context, runs one operation, and writes its outputs back. The engine validates that every step's inputs are produced before it runs, so wiring mistakes are caught immediately with a clear error.

This is the declarative engine (`m3resp.workflows`). For the smaller, named `Pipeline`/preset mechanism (`session.run_pipeline("eit" | "emg" | "multimodal")`), see [developer/pipeline-contracts.md](developer/pipeline-contracts.md).

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
    with: { file_path: "@eit_file", vendor: "@vendor" }

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
- **`in`** — maps a step's parameter name to the context key it reads from. Leave it out to use the step's default binding.
- **`with`** — static parameters passed directly to the step function. A string value starting with `@` is resolved from `inputs`; this applies recursively inside lists and mappings, so `with: {paths: ["@a", "@b"]}` resolves both entries. Write `@@text` to get the literal string `@text` instead of a reference. A parameter a step declares as a `path` (e.g. `eit.load`'s `file_path`) resolves relative to the spec file's own directory, not the process working directory — same as `outputs.dir` below — after any `@` reference is substituted.
- **`out`** — renames a step's natural output names to different context keys. Useful when you want to run the same step twice and keep both outputs.

### Top-level sections

- **`name`** — a label for the pipeline (used in log output).
- **`inputs`** — named values that steps can reference with `@name`.
- **`outputs`** — controls where and what gets exported after the pipeline finishes. If `dir` is set and no explicit export step is present, the engine calls `export_session_summary` automatically.
- **`experiment`** — study metadata used by `export.rotarc_result` for output file naming (`subject_id`, `mode`, `timepoint`, `run_identifier`, `selection`).
- **`metadata`** — a free, JSON-serializable mapping for study/site/session notes that are not part of file naming (unlike `experiment`).
- **`execution`** — reproducibility controls: `error_policy` (only `fail_fast` is supported) and an optional `seed`, recorded but not yet threaded into any step.
- **`schema_version`** — omit it for the permissive legacy parser (unknown keys ignored, booleans loosely coerced with a `FutureWarning`). Set it to `1` for strict parsing: unknown top-level/step keys and non-boolean booleans become hard errors instead.

### Strict (`schema_version: 1`) versus legacy specs

A spec with no top-level `schema_version` key is parsed by the permissive legacy parser: unknown top-level/step keys are silently ignored, and a non-boolean `outputs.*` flag (e.g. `"yes"`) is coerced with `bool()` behind a `FutureWarning` rather than rejected. This exists for backward compatibility with specs written before Stage 2; new specs should set `schema_version: 1`.

A versioned spec (`schema_version: 1`) is parsed strictly by a pydantic model:

- An unknown key anywhere (top level, a step, `outputs`, `experiment`, `execution`) is a hard `PipelineSpecError`.
- Every boolean must be a real YAML/JSON boolean. In YAML that means unquoted `true`/`false` (`yes`/`no` and `on`/`off` also work); quoted strings such as `"yes"`, and `y`/`n`, are rejected. In JSON, only `true`/`false`.
- `outputs.mode` must be set whenever `outputs.dir` is set. Legacy specs may still omit it and let the engine infer it (see "Output modes" below).

Both parsers build the same `PipelineSpec`/`StepSpec` dataclasses, so the engine, `compile_pipeline`, and every step behave identically either way; only what gets accepted at parse time differs. Four of the six example specs under `examples/` use `schema_version: 1`; the other two are the smaller introductory ones (`multimodal_example`, `multidomain_recording1_a2`) which are kept legacy on purpose, as a lower-ceremony starting point.

### Step ids

Every step gets a stable `id`: an explicit `id:` in the spec is used as-is (and must be unique across the spec), otherwise the engine generates `step_{position:03d}_{operation}` (e.g. `step_003_eit.detect_breaths`). A generated id changes if you reorder or insert steps, so anything that needs to reference *this specific step* across runs or across spec edits — a GUI's per-step provenance link, a downstream tool reading `run_manifest.json`'s `step_records` — should use an explicit `id:` instead:

```yaml
steps:
  - id: detect_breaths
    uses: eit.detect_breaths
    in: { signal: global_impedance }
```

Every step in `eit-full.pipeline.yaml`, `emg-full.pipeline.yaml`, `multimodal-full.pipeline.yaml`, and `breath-duration.pipeline.yaml` declares an explicit `id:` for this reason.

### Timestamped output directories

Set `outputs.timestamped: true` to give every run its own subfolder (`<outputs.dir>/YYYYMMDD_HHMMSS/...`) instead of overwriting the same folder each time. This works the same way for *any* export path in the run:

- The automatic export (no explicit export step in the spec).
- Built-in export steps that need a directory, e.g. `export.rotarc_result` (`<outputs.dir>/<timestamp>/subject_results/<run_identifier>/...`).
- Any custom export step, by reading `_resolved_output_dir` from context.

`run_spec` computes the timestamp exactly once per run and seeds it as `_resolved_output_dir` (a resolved `Path`, or `None` if `outputs.dir` isn't set) and `_run_timestamp` (the raw stamp, or `None` if not timestamped) into context, alongside `_spec_outputs`/`_spec_experiment`. Every export path in the run reads the *same* resolved directory, so a single run never ends up split across two different timestamp folders even if it uses more than one export mechanism. To make your own export step honor it:

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

### Output modes

`outputs.mode` (Phase 6.2) states, explicitly, what happens to `outputs.dir` after the pipeline runs:

| Mode | Behavior |
|---|---|
| `automatic` | The engine calls `export_session_summary` itself once the run succeeds — the same behavior as omitting an explicit export step. Use this when the spec has no `export.*` step, e.g. `eit-full.pipeline.yaml`/`emg-full.pipeline.yaml`/`multimodal-full.pipeline.yaml`. |
| `explicit` | The engine writes nothing on its own; the spec's own `export.*` step(s) (already run during execution, e.g. `export.rotarc_result`) are entirely responsible for what lands in `outputs.dir`. Use this whenever the spec has its own export step, so output is never written twice. |
| `none` | Nothing is written to `outputs.dir` at all (still useful for `outputs.figures` alone, or for a spec run purely for its in-memory result). |

A versioned (`schema_version: 1`) spec must set `mode` explicitly whenever `dir` is set — this is enforced at parse time — precisely so a reader never has to guess which of the three behaviors above will happen. A legacy spec may omit it; the engine then infers `explicit` if any `export.*` step is present, `automatic` otherwise, and warns with a `FutureWarning` that this inference is deprecated. Regardless of mode, `outputs.figures` (when true) is written for a successful run before the mode branches, and *nothing* in `outputs:` is written for a failed or cancelled run except the run manifest below (Phase 6.4: no success summary after a failure).

### Validation and readiness

Two distinct checks are available before running anything, both without importing `eitprocessing`/`resurfemg` or touching the filesystem for anything but the readiness check itself:

- **Structural** validation (`m3resp validate spec.yaml`, or `validate_pipeline(spec)` from `m3resp.workflows.compiler`) checks that the spec parses, every `uses` name is a registered step, every step's inputs are bound to something produced earlier (or a declared default), every static parameter has the right value type, and no unknown/duplicate names exist. This is always safe to run and always cheap — it never imports an optional package or reads a data file.
- **Readiness** validation (`m3resp validate --readiness spec.yaml`, or `validate_pipeline(spec, readiness=True)`) additionally checks things that depend on *this* machine/environment: whether a step's declared optional package (`eitprocessing`/`resurfemg`) is actually importable, and whether a `path`-typed parameter's file actually exists on disk. A readiness diagnostic is expected and correct, not a bug, when it fires for the right reason — e.g. `breath-duration.pipeline.yaml`'s `eit_file` is a private, site-specific path (see its "USER TEMPLATE" banner) and reports `missing_file` on any machine other than that researcher's.

Both return every independent problem found in one pass (a `ValidationReport` with separate `structural`/`readiness` diagnostic tuples, each JSON-safe via `.as_dict()`), rather than raising on the first one, so a GUI or CI job can show a complete list instead of a fix-one-rerun loop. `compile_pipeline(spec)` raises on the first structural error instead of collecting them — it is meant for "give me the resolved plan or fail," not for validation reporting.

### Execution lifecycle: progress, warnings, cancellation, failure

A run is more than success/failure. `run_pipeline`/`run_spec` accept an optional `event_sink` callback and `cancellation_token`, and every `PipelineResult` carries a full accounting of what happened:

- **Progress events** — if `event_sink` is supplied, it is called with a framework-neutral event for `pipeline_started`, each step's `step_started`/`step_completed`/`step_failed`/`step_warning`, and the run's own `pipeline_completed`/`pipeline_failed`/`pipeline_cancelled`. A GUI wires this straight to a progress bar without importing anything m3resp-internal beyond the event shape itself.
- **Warnings** — every Python warning raised inside a step (including one raised immediately before that step's own exception — this ordering is deliberately preserved, not dropped) is captured and attached to that step's `StepExecutionRecord`, and also collected onto `PipelineResult.warnings`, rather than only printing to stderr.
- **Cancellation** — a `CancellationToken` is checked before and after each step; calling `.cancel()` on it (the CLI does this from a `SIGINT` handler, Ctrl-C) lets the current step finish, then stops before the next one starts. The result's `status` becomes `"cancelled"`, not `"failed"` — completed work and its manifest are preserved, not discarded. This is cooperative, not preemptive: while a step is genuinely stuck (not just slow), the cancellation check between steps never runs, so repeated Ctrl-C is a no-op for the remainder of that step. The CLI installs no handler for `SIGQUIT`/`SIGTERM`, so Ctrl-\ or `kill`/`kill -9` from another terminal still forcibly kill the process at its default OS disposition, bypassing cooperative cancellation entirely - use one of those to recover from a hung step.
- **Failure** — a step's own exception is wrapped in `PipelineExecutionError` (carrying `step_id`/`position`/`operation_id`, the run's `run_id`/`started_at`, and every `step_record` up to the failure, with the original exception as `__cause__`), so a caller always gets structured context, not just a bare traceback.

### Run manifests

Whenever `outputs.dir` is set and the resolved mode is not `"none"`, `run_spec` writes `<output_dir>/run_manifest.json` — once with `status: "running"` before any step executes, then atomically replaced (temp file + `os.replace()`, so a reader never observes a half-written file) with the terminal state once the run finishes, *including on failure or cancellation*. The manifest is JSON: `run_id`, `status`, `pipeline_name`, `schema_version`, timestamps, the spec's resolved `root`/`description`, `inputs` (with any key containing `password`/`secret`/`token`/`credential`/`api_key` redacted), the execution context, every step record, every diagnostic/warning, an `error` block on failure, and (only when `outputs.checksums: true`) a sha256 of every existing file referenced by a `path`-typed step parameter. A crashed run therefore always leaves an honestly-marked-`"failed"` manifest behind instead of nothing.

### Native versus compatibility artifacts

Every context key a step reads or writes is a typed `StepArtifact` (`m3resp steps --details` / `m3resp describe <op>` show these). Several steps write two forms of the same result: a **native** `Signal`/`Event`/`ParameterResult`/`QualityFlag` object with full provenance (`method`/`metadata.source_function`), and a **compatibility** output that keeps the exact raw shape (e.g. a `pandas.Series`) older consumers already expect. Both are typed and both are exported; prefer the native form in new code; compatibility outputs exist so upgrading an EMG/EIT step's implementation never breaks an existing consumer of its raw shape. See the EMG step table below for the specific native/compatibility pairs on the EMG/ventilator side.

### GUI discovery and the `PipelineService` boundary

`m3resp.workflows.service.PipelineService` is the intended integration surface for a GUI or any other application embedding m3resp — every method takes a spec (path/dict/`PipelineSpec`) and returns only JSON-safe dictionaries, never a live `M3Session`, adapter instance, upstream package object, NumPy array, or callable:

```python
from m3resp.workflows.service import PipelineService

service = PipelineService()
service.list_capabilities(prefix="eit.")       # every eit.* step's discovery description
service.describe_capability("eit.detect_rates") # one step, incl. capability state
service.validate_pipeline(spec, readiness=True) # ValidationReport.as_dict()
service.compile_pipeline(spec)                  # CompiledPipeline.as_dict()
service.run_pipeline(spec, event_sink=..., cancellation_token=...)  # run summary
```

`list_capabilities`/`describe_capability` (backed by `describe_steps`/`describe_step` in `workflows/registry.py`) let a GUI render step pickers and per-step parameter forms, and disable/grey out an operation whose optional package (`eitprocessing`/`resurfemg`) is not installed, all without importing that package or inspecting a function's signature — `step_capability_state` reports `"available"`, `"missing_optional_dependency"`, or `"deprecated"` per step. `event_sink`/`cancellation_token` are the one exception to "JSON-safe only": they are the caller's own objects, passed through unchanged to the engine (see "Execution lifecycle" above) — GUI threading/process management is the caller's responsibility, not this service's.

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

Register a function with `@register_step`. Declare the context keys it reads (mapped onto its parameter names) and the output names it returns:

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

The function receives bound keyword arguments and returns a mapping of `{output_name: value}`. Steps that produce no outputs return `{}` or `None`. Import the optional modality packages inside the function body so the package installs cleanly without them.

## Example specs

Worked examples ship with the repository. All except the two smaller introductory ones use `schema_version: 1` and declare an explicit `outputs.mode` (see above); every step in the `schema_version: 1` examples has a stable, explicit `id:`.

- [`examples/ROTARC_example/breath-duration.pipeline.yaml`](https://github.com/M3RESP/m3resp/blob/main/examples/ROTARC_example/breath-duration.pipeline.yaml) — computes breath-duration CV from EIT data and writes a ROTARC-style result file (`outputs.mode: explicit`, since `export.rotarc_result` is its own export step). Its `inputs.eit_file` is a private, site-specific path — point it at your own recording before running it (see the file's "USER TEMPLATE" banner); `m3resp validate --readiness` correctly reports it missing on any other machine.
- [`examples/eit_full_preprocessing/eit-full.pipeline.yaml`](https://github.com/M3RESP/m3resp/blob/main/examples/eit_full_preprocessing/eit-full.pipeline.yaml) — every EIT operation the Stage 2 EIT gap migration closed onto `EITProcessingAdapter`: loading, rate detection, MDN filtering, global breath detection, continuous TIV/EELI, pixel-breath timing, pixel TIV, and the four ROI lung-space steps (`outputs.mode: automatic`).
- [`examples/emg_full_preprocessing/emg-full.pipeline.yaml`](https://github.com/M3RESP/m3resp/blob/main/examples/emg_full_preprocessing/emg-full.pipeline.yaml) — every EMG/ventilator operation the Stage 2 ReSurfEMG gap migration closed onto `ReSurfEMGAdapter`: loading, ECG detection + gating, breath detection, moving-baseline, on/offset detection, breath features, ventilator/Pocc detection and prerequisites, and all ten clinical quality operations. See ["EMG/ventilator pipelines"](#emgventilator-pipelines) below.
- [`examples/multimodal_full/multimodal-full.pipeline.yaml`](https://github.com/M3RESP/m3resp/blob/main/examples/multimodal_full/multimodal-full.pipeline.yaml) — the canonical, most complete multimodal example: loads EIT + EMG + ventilator, performs raw synchronization via `session.sync_raw` *before* any modality-specific preprocessing, then runs the full EIT and EMG/ventilator chains above, then exports native results. Does not restore any post-detection alignment step — raw sync is the only cross-modality timing adjustment.
- [`examples/multimodal_example/multimodal.pipeline.yaml`](https://github.com/M3RESP/m3resp/blob/main/examples/multimodal_example/multimodal.pipeline.yaml) — a smaller, introductory multimodal pipeline (legacy spec, no `schema_version`); shorter than `multimodal-full` and easier to read end-to-end.
- [`examples/multidomain_recording1/recording1_a2.pipeline.yaml`](https://github.com/M3RESP/m3resp/blob/main/examples/multidomain_recording1/recording1_a2.pipeline.yaml) — the multidomain results chain on real diaphragm sEMG (Recording1, window A2, EIT off): cut to the window, 50 Hz notch before the 20-500 Hz band-pass, wavelet ECG removal, median envelope, baseline subtracted, breath detection with close peaks merged. Gives the same 9 breaths and 9.72 breaths/min as the multidomain notebook, value for value.

Run them from the repository root:

```bash
m3resp run examples/ROTARC_example/breath-duration.pipeline.yaml
m3resp run examples/eit_full_preprocessing/eit-full.pipeline.yaml
m3resp run examples/emg_full_preprocessing/emg-full.pipeline.yaml
m3resp run examples/multimodal_full/multimodal-full.pipeline.yaml
m3resp run examples/multimodal_example/multimodal.pipeline.yaml
m3resp run examples/multidomain_recording1/recording1_a2.pipeline.yaml
```

Update `inputs.eit_file` in the ROTARC example before running it — it contains a site-specific path.

### CLI reference

`m3resp` has four subcommands, plus stable, documented exit codes so a calling script can distinguish failure categories without parsing output:

| Command | Purpose |
|---|---|
| `m3resp run spec.yaml [--dry-run] [--debug]` | Execute a spec end-to-end. `--dry-run` compiles and prints the resolved plan (`compile_pipeline(...).as_dict()`) without running any step; `--debug` prints a full traceback instead of a short message on error. Ctrl-C cooperatively cancels (finishes the current step, keeps completed work and its manifest, exits `EXIT_CANCELLED`). |
| `m3resp validate spec.yaml [--readiness] [--json]` | Structural (and, with `--readiness`, capability/file-existence) validation without running anything — see "Validation and readiness" above. |
| `m3resp steps [--details] [--json]` | List every registered step, tagging any whose optional package isn't installed (e.g. `[missing_optional_dependency]`) without importing it; `--details` adds parameters/artifacts/full capability state per step. |
| `m3resp describe <operation>` | One step's full discovery description (e.g. `m3resp describe eit.detect_rates`). |

| Exit code | Meaning |
|---:|---|
| `0` | Success. |
| `1` | Usage error (bad CLI arguments). |
| `2` | Invalid spec — structural validation failed, or the spec path could not be read. |
| `3` | Readiness failure — `validate --readiness` found a missing optional package or file. |
| `4` | Execution failure — a step raised during `run`. |
| `5` | Cancelled — Ctrl-C during `run`. |

## EMG/ventilator pipelines

The ReSurfEMG gap migration keeps validated ReSurfEMG algorithms behind `ReSurfEMGAdapter` while exposing every EMG/ventilator operation as a workflow step with native `Signal`/`Event`/`ParameterResult`/`QualityFlag` outputs (see [developer/adapters.md](developer/adapters.md) for the adapter conversion boundary). This section documents that step surface; the equivalent EIT-side steps follow the same pattern via `EITProcessingAdapter`.

Using any `emg.*`/ventilator step requires the optional `resurfemg` dependency:

```bash
pip install "m3resp[emg]"
```

Steps whose upstream call needs it raise `OptionalDependencyError` with that same install hint if it's missing; `import m3resp` and `m3resp steps` work without it (upstream imports are lazy, inside the adapter methods that need them).

### Step table

Every step also writes its raw/compatibility output(s) unchanged (existing consumers keep working) alongside the native outputs listed here. "Upstream" means the value is `resurfemg`'s own scientific result, passed through with provenance recorded (`method`/`metadata.source_function`); "native" means it comes from an `m3resp.processing` primitive.

| Step | Reads (besides `session`) | Key parameters | Native writes | Implementation |
|---|---|---|---|---|
| `emg.load` | — | `file_path`, `loader_options` | `raw_emg_signals` (one `Signal`/channel) | upstream loader |
| `emg.slice` | — | `start_seconds`, `end_seconds` (optional; default: to the end) | `emg_slice` (kept window, samples removed at each end) | native (cuts `session.raw`; ventilator channels from the same file are cut too) |
| `emg.preprocess` | — | `channel`, `high_pass_hz` (default 20), `low_pass_hz` (default 500, Nyquist-capped), `envelope_window_seconds`, `envelope_method` (`"rms"` default / `"arv"` / `"median"`), `notch_base_frequency`, `notch_quality_factor`, `notch_before_bandpass` (default `false`) | — (raw `processed_emg` dict) | upstream + native notch filter |
| `emg.ecg_detect_peaks` | `processed_emg` | `ecg_channel`, `source` (default `"raw_channel"`), `peak_fraction`, `peak_width_seconds`, `peak_distance_seconds`, `bandpass_filter` | `ecg_peak_events` (one `Event`/peak), `ecg_peak_count_result` | upstream |
| `emg.ecg_gating` | `processed_emg`, `ecg_peak_indices` | `source` (default `"filtered"`), `gate_width_seconds` **xor** `gate_width_samples`, `fill_method` (0-3), `envelope_window_seconds`, `envelope_method` (defaults to preprocessing's) | `ecg_gated_signal`, `ecg_gate_mask_result` (array) | upstream |
| `emg.ecg_wavelet_denoising` | `processed_emg`, `ecg_peak_indices` | `source`, `hard_thresholding`, `levels`, `wavelet_type`, `fixed_threshold`, `envelope_window_seconds`, `envelope_method` (defaults to preprocessing's) | `ecg_wavelet_cleaned_signal`, `wavelet_decomposition_result`, `wavelet_thresholds_result`, `wavelet_gate_mask_result` (all arrays) | upstream |
| `emg.detect_breaths` | `baseline` (optional; from `emg.moving_baseline`/`emg.slopesum_baseline`) | `min_breath_width_seconds`, `half_window_seconds`, `prominence_factor`, `threshold`, `merge_close_peaks_within_width` (default `false`) | `emg_breath_events` | native |
| `emg.moving_baseline` | `processed_emg` | `window_seconds`, `step_seconds`, `percentile` (0-100) | `baseline_signal` | upstream |
| `emg.slopesum_baseline` | `processed_emg` | `window_seconds`, `step_seconds`, `percentile`, `augmented_percentile`, `moving_average_seconds`, `percentile_window_seconds` | `baseline_signal`, `baseline_running_mean_signal`, `baseline_running_std_signal`, `slopesum_baseline_native_detail` (NumPy-only) | upstream |
| `ventilator.detect_pressure_breaths` | `ventilator_signals` (needs `pressure`) | `smoothing_seconds` (default 0.2), `min_depth` (default 0.15, pressure unit), `min_interval_seconds` (default 2.0) | `ventilator_breath_indices` (lowest point of each airway pressure dip; spontaneous breathing only) | native (`detect_pressure_dip_breaths`) |
| `ventilator.pocc_intervals` | `ventilator_signals`, `pocc_indices` | `baseline_window_seconds` (default 7.5), `baseline_step_seconds` (default 0.2), `baseline_percentile` (default 33) - a moving baseline of the airway pressure | `pocc_events` (`BreathEvent`, `modality="pressure"`), `pressure_baseline` | native (`onoff_from_baseline_crossings`) |
| `ventilator.pocc_time_product` | `ventilator_signals`, `pocc_start_indices`, `pocc_end_indices`, `pressure_baseline`, `pocc_indices` | `include_aub` (default `true`, adds the area under the baseline as ReSurfEMG's PTPocc does), `aub_window_seconds` (default 5) | `pocc_time_product_result` (unit `<pressure-unit>*s`) | native (`window_integral`) |
| `emg.snr_pseudo` | `processed_emg`, `peak_indices`, `baseline` | `minimum_snr` (optional) | `snr_pseudo_results`; `snr_pseudo_flags` only if `minimum_snr` is set | upstream |
| `emg.percentage_under_baseline` | `processed_emg`, `peak_indices`, `start_indices`, `end_indices`, `baseline` | `aub_window_seconds`, `aub_threshold` | `percentage_under_baseline_results`, `_flags` | upstream |
| `emg.detect_local_high_aub` | `area_under_baseline`, `peak_indices` | `threshold_percentile`, `threshold_factor` | `_flags`, `detect_local_high_aub_threshold_result` | upstream |
| `emg.detect_extreme_time_products` | `time_product`, `peak_indices` | `upper_percentile`, `upper_factor`, `lower_percentile`, `lower_factor` | `_flags`, `detect_extreme_time_products_bounds_result` | upstream |
| `ventilator.detect_non_consecutive_manoeuvres` | `ventilator_breath_indices`, `pocc_indices` | — | `_flags` (`modality="pressure"`) | upstream |
| `emg.evaluate_bell_curve_error` | `peak_indices`, `start_indices`, `end_indices`, `processed_emg`, `time_product` | — | `_results` (incl. a separate array result per breath for the fitted bell-curve parameters), `_flags` | upstream |
| `emg.evaluate_event_timing` | `peak_indices`, `processed_emg`, `ventilator_breath_indices`, `ventilator_signals` | — | `_results`, `_flags`, `evaluate_event_timing_unmatched_count` | upstream |
| `emg.evaluate_respiratory_rates` | `peak_indices`, `processed_emg`, `ventilator_respiratory_rate` | `minimum_fraction` | `evaluate_respiratory_rates_result`, `_flag` | upstream |
| `ventilator.pocc_quality` | `ventilator_signals`, `pocc_indices`, `pocc_end_indices`, `pocc_time_products` | `dp_up_10_threshold`, `dp_up_90_threshold`, `dp_up_90_norm_threshold` | `pocc_quality_results` (labeled `dp_up_10`/`dp_up_90`/`dp_up_90_norm`, 3 per manoeuvre), `pocc_quality_flags` (1 per manoeuvre) | upstream |
| `emg.interpeak_dist` | `ecg_peak_indices`, `peak_indices`, `processed_emg` | `threshold` (default `1.1`) | `interpeak_dist_result` (median ECG/EMG interval in seconds + ratio), `interpeak_dist_flag` | upstream |

Run `m3resp steps` for the full list including the pre-existing feature steps (`emg.time_to_peak`, `emg.pseudo_slope`, `emg.amplitude`, `emg.time_product`, `emg.area_under_baseline`, `emg.respiratory_rate`) and event-detection steps (`emg.onoffpeak_baseline_crossing`, `emg.onoffpeak_slope_extrapolation`, `ventilator.detect_breaths`, `ventilator.find_occluded_breaths`) this migration routes through the adapter without changing their contracts. The registry metadata (`reads`/`writes`/`summary`) is enough for a GUI to render controls and disable an operation whose inputs are missing, without importing `resurfemg` or inspecting a function's signature.

### Channel selection and time-base rules

- `emg.preprocess`'s `channel` is an index into the raw channel-major array from `emg.load` (`session.emg.raw[channel]`), not a label.
- ECG detection's `source` selects a *key* in `processed_emg` (`"raw_channel"`/`"filtered"`/`"envelope"`), while `ecg_channel` (when set) selects a *raw* channel index instead - use `ecg_channel` when a recording has a dedicated reference ECG channel (e.g. the committed `emg_data_synth_quiet_breathing.Poly5` fixture's channel 0), and `source` when ECG is mixed directly into the EMG channel being processed (no separate reference channel).
- No step assumes EMG and ventilator sample rates are equal - Pocc/timing steps convert each side to seconds using its own `fs` before comparing (`emg.evaluate_event_timing`'s per-pair results record both source sample indices and both sampling frequencies for this reason).
- `emg.detect_breaths`'s default `min_breath_width_seconds=0.5` keeps breath bursts at least half a second wide, which also covers faster breathing (the `data_from_repo` "quiet breathing" fixture breathes at about 22 breaths/min; 1.0 s found none of its breaths). Noisy recordings can still need `min_breath_width_seconds`/`prominence_factor` tuned - see the inline comments in `emg-full.pipeline.yaml` and `multimodal.pipeline.yaml` for values that were verified against each fixture's actual breath timing.

### ECG-removal alternatives

`emg.ecg_gating` and `emg.ecg_wavelet_denoising` both naturally write `processed_emg_after_ecg`. Pick one; a pipeline uses output renaming to keep the downstream key `processed_emg`:

```yaml
- uses: emg.preprocess
  out: { processed_emg: processed_emg_before_ecg }

- uses: emg.ecg_detect_peaks
  in: { processed_emg: processed_emg_before_ecg }

- uses: emg.ecg_gating  # or emg.ecg_wavelet_denoising
  in: { processed_emg: processed_emg_before_ecg }
  out: { processed_emg_after_ecg: processed_emg }
```

`gating`'s `fill_method` keeps ReSurfEMG's meanings: `0` zeros the gated region, `1` interpolates between its edges (the default), `2` fills with the mean of a neighboring segment, `3` replaces it with a running-RMS estimate. `wavelet_denoising` additionally zero-pads the signal internally to a multiple of `2**levels` for its stationary wavelet transform - the cleaned signal, thresholds, and gate mask are trimmed back to the original length, but the decomposition array (`wavelet_decomposition_result`) stays at the padded length; both lengths are recorded in its metadata (`original_length`/`padded_length`). All four array-valued outputs (gate masks, decomposition, thresholds) go through the same shared `parameter_result_arrays.npz` archive as every other array-valued `ParameterResult` - see "Array export" below.

### Baseline alternatives

`emg.moving_baseline` and `emg.slopesum_baseline` both write `baseline`; the registry marks them as alternatives for the same reason as the ECG-removal steps above. `slopesum_baseline`'s compatibility output (`slopesum_baseline_detail`) keeps ReSurfEMG's `pandas.Series`; its native sibling (`slopesum_baseline_native_detail`) is NumPy-only and does not cross that boundary.

### Measurements vs. clinical pass/fail criteria

A quality step's raw upstream output is not automatically a clinical pass/fail result. `m3resp.processing.quality.quality_flag_from_result` (used by the shared `to_quality_flags` path) and each granular quality step follow the same rule: a boolean or boolean array is a real criterion; a numeric scalar or array (e.g. `snr_pseudo`'s per-peak ratios) is a *measurement*, and only becomes a `QualityFlag` when a threshold is actually configured (`emg.snr_pseudo`'s `minimum_snr`, for example). A measurement-only flag uses `passed=False` with `metadata["measurement_only"]=True` - the same "not evaluated" convention `metadata["skipped"]=True` uses for a flag skipped for missing inputs - rather than inventing a pass/fail verdict the upstream function never defined.

### Pocc prerequisites and thresholds

`ventilator.pocc_quality` needs Pocc end indices and pressure-time products that `ventilator.find_occluded_breaths` alone doesn't produce - `ventilator.pocc_intervals` and `ventilator.pocc_time_product` supply them:

```
ventilator.find_occluded_breaths  ->  pocc_indices
ventilator.pocc_intervals          ->  pocc_start_indices, pocc_end_indices
ventilator.pocc_time_product       ->  pocc_time_products
ventilator.pocc_quality             (needs all of the above)
```

`ventilator.pocc_quality`'s three thresholds (`dp_up_10_threshold=0.0`, `dp_up_90_threshold=2.0`, `dp_up_90_norm_threshold=0.8`) are ReSurfEMG's defaults for the pressure upslope after occlusion release (Warnaar et al. 2024); its criteria matrix is exposed as three explicitly labeled result names (`pocc_quality_dp_up_10`/`_dp_up_90`/`_dp_up_90_norm`) rather than a raw 3-by-N array.

### Missing, skipped, and unmatched events

- A quality step whose prerequisites are entirely missing (no ventilator input, no detected breaths) simply has nothing to iterate over - it produces no native results for that run rather than raising, so a pipeline with a partial dataset still completes.
- `emg.evaluate_event_timing` pairs EMG and ventilator events by position; if the two lists have different lengths, it still pairs as many as it can (keeping the existing raw-output truncation behavior for backward compatibility) but also reports `evaluate_event_timing_unmatched_count` and adds an `evaluate_event_timing_unmatched` warning `QualityFlag` - the unmatched events are never silently dropped without a trace.
- Per-breath/per-manoeuvre native results use `breath_id=str(position)` (a stable event ID is future work) and record the source peak/pressure sample index in `metadata["peak_sample_index"]`, so a GUI can always explain which detected event a given flag or measurement belongs to.

### Array export

Array-valued `ParameterResult`s (gate masks, wavelet decomposition/thresholds, bell-curve fit parameters, ...) are not written inline into `parameter_results.csv` - like EIT's array-valued results, they're added to `session.parameter_results` and go through the same shared `parameter_result_arrays.npz` archive (`export_summary`/`export.*`), with a CSV row (`array_key`, `shape`, `dtype`) pointing into it. There is no EMG-specific array exporter.

## Architecture

| File | Responsibility |
|---|---|
| `workflows/registry.py` | `@register_step` decorator and the global step registry. |
| `workflows/spec.py` | Parse and validate a YAML or JSON spec into typed dataclasses. |
| `workflows/context.py` | `PipelineContext` — the shared artifact store wrapping an `M3Session`. |
| `workflows/engine/` | `run_pipeline` (`execution.py`), `run_spec` (`spec_runner.py`), and spec validation (`diagnostics.py`) — bind arguments, validate, execute. |
| `workflows/steps/` | Built-in steps for EIT, EMG, metrics, session ops, and export. |
| `workflows/utils.py` | Shared utilities: signal slicing, JSON writing, summary logging, `resolve_output_dir` (timestamped output directories). |
| `workflows/summaries.py` | Compact session summaries written to the log after a run. |
