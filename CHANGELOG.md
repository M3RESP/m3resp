# Changelog

## Unreleased

### `FilterType` renamed to `ButterworthFilterType` (#90)

`m3resp.processing.filters` holds more than Butterworth filters, so the
list of allowed `filter_type` values (`"lowpass"`, `"highpass"`,
`"bandpass"`, `"bandstop"`) now says which filter it belongs to. The values
themselves do not change. **The old name is removed.**

| Before | Now |
|---|---|
| `from m3resp.processing.filters import FilterType` | `from m3resp.processing.filters import ButterworthFilterType` |

The `mode` choices of the `eit.butterworth_filter` step now come from the
same list, so the two cannot drift apart.

Found while doing the rename, all in `butterworth_filter`:

- **Fixes** infinite samples getting through the check for bad samples. One
  `inf` in the signal turned every filtered sample into NaN, with no error.
  Now NaN and infinite samples are both refused.
- An unknown `filter_type` (such as `"low"` or `"notch"`) is now refused with
  a clear message. It used to give a wrong message about `cutoff_frequency`,
  or an error from inside SciPy.
- numpy numbers (`np.float32`, `np.int64`, ...) are now accepted for
  `cutoff_frequency`, `sample_frequency` and `order`, so values read from
  data files work as they are.
- `True` is no longer taken as a filter `order` or a `sample_frequency`. It
  used to run quietly as order 1 or as 1 Hz.

### `compute_multimodal_parameters` renamed to `compute_breath_timing_parameters` (#116)

The old name could be read as "all results that use more than one
modality". What it computes is breath timing: the delay between two
modalities' breaths, the difference in breath duration, and how often the
modalities agree that a breath happened. **The old names are removed.**

| Before | Now |
|---|---|
| `session.compute_multimodal_parameters()` | `session.compute_breath_timing_parameters()` |
| `m3resp.synchronization.compute_multimodal_parameters` | `m3resp.synchronization.compute_breath_timing_parameters` |
| Module `m3resp.synchronization.multimodal_parameters` | `m3resp.synchronization.breath_timing_parameters` |
| Provenance action `"compute_multimodal_parameters"` | `"compute_breath_timing_parameters"` |

The results keep their names, units and `modality="multimodal"`.

Found while doing the rename:

- **Fixes** the event-agreement score counting linked breaths that hold
  neither modality of the pair. With a ventilator breath linked on its own,
  `eit_emg_event_agreement` dropped although EIT and EMG agreed on every
  breath they saw. Now only linked breaths holding a breath from at least
  one of the two count, so the score can be higher than before when a third
  modality was linked.
- **Fixes** a second call to `session.compute_breath_timing_parameters()`
  adding a second copy of every result, which then counted twice in
  `parameter_results.csv`. A second call now replaces the earlier results.
- With no linked breath holding either modality of a pair (for example
  when `link_breaths()` was not called), the event-agreement result is left
  out instead of being stored as 0.0.
- An unknown `anchor` (such as the old `"peak"`) is now always an error. It
  used to pass when no linked breath held both modalities.
- The provenance record now keeps `delay_pairs` and `duration_pairs`, so the
  call can be repeated from it.

### A labelled ventilator file no longer hands on an unnamed column

- **Fixes** ventilator channels being taken from columns 0, 1 and 2 when the
  file has labels but none for that channel. A file with `Paw, EMGdi, EMGsc`
  (airway pressure recorded next to the sEMG) used to give EMGdi as flow and
  EMGsc as volume, with no warning, and those then went into PEEP estimation
  and ventilator breath detection. Now, once one label names a known
  channel, a channel not named among the labels is reported as missing.
  Read such a file with `channels=("airway_pressure",)`.
- A file whose labels name no known channel at all (such as ReSurfEMG's
  synthetic `P, F, V`) is still read as columns 0, 1 and 2, but now with a
  warning. A file without labels is read the same way, without a warning.

### The ventilator's `pressure` channel is now `airway_pressure` (#117)

The channel only ever held the airway pressure, never an esophageal,
transpulmonary or gastric pressure, but its name did not say so, while the
other pressures next to it did (`esophageal_pressure`, ...). It is now
`airway_pressure`, the same as its category. **The old names are removed.**

| Before | Now |
|---|---|
| `session.ventilator.pressure` (`VentilatorRecording.pressure`) | `session.ventilator.airway_pressure` |
| Channel key `"pressure"`: `bundle["pressure"]`, `channels=("pressure", "flow", "volume")`, `Signal.channel == "pressure"` | `"airway_pressure"` |
| A second airway pressure `"pressure__pod"`, `"pressure__<recording>"` | `"airway_pressure__pod"`, `"airway_pressure__<recording>"` |
| `pressure_channel=` on `split_channels`, `VentilatorAdapter.preprocess` and the `ventilator.channels` step (`with: {pressure_channel: 0}` in a workflow file) | `airway_pressure_channel=` |
| `ventilator_pressure_channel=` on EMG postprocessing | `ventilator_airway_pressure_channel=` |
| `medibus.pressure_channel` in the synthetic data generator config | `medibus.airway_pressure_channel` |

Files are read as before: a column labelled `Pressure`, `Paw`, `Pvent` or
`airway pressure` still becomes the airway pressure channel. Exported signals
and the `channel` column carry the new name. `register_channel_alias` (and so
`load_channel_aliases`) still reads an alias pointing to `"pressure"`: it is
registered for `"airway_pressure"`, with a warning naming the new channel.

Found while doing the rename:

- **Fixes** EMG postprocessing reading the ventilator columns by position.
  It always asked for columns 0, 1 and 2 as airway pressure, flow and volume,
  which overrode the column labels. Labels now decide, as they already did in
  `ventilator.channels`; a recording without recognised labels still uses
  columns 0, 1 and 2.
- **Fixes** the unit on `pocc_time_product` and `pocc_quality` results. It was
  always cmH2O; it is now the unit the recording reports for its airway
  pressure. The `pocc_quality` thresholds are in that same unit, and their
  defaults are meant for cmH2O.
- **Fixes** `ventilator.normalize_breaths` letting the last breath run past
  the end of the recording when the airway pressure was not loaded.
- A column index given for a channel that is not being read (for example
  `channel_indices={"pressure": 2}`, or `flow_channel=` when flow is not
  asked for) is now an error. It used to be ignored.
- The Pocc steps now say they need an `airway_pressure` channel when it is
  missing, instead of failing with a bare `KeyError`. They also find the
  airway pressure of a second ventilator recording.

### "Pipeline" renamed to "workflow" everywhere; built-in pipelines are now presets (#112)

The engine module was already `m3resp.workflows`, but most names, files and
docs still said "pipeline", so one thing had two names. Everything now says
**workflow**. The small built-in shortcuts (`"eit"`, `"emg"`, `"multimodal"`)
are a different thing from a YAML workflow, so they are now called
**presets**, like the package they live in. **The old names are removed.**

| Before | Now |
|---|---|
| `run_pipeline(spec, session=...)`, `compile_pipeline`, `validate_pipeline` | `run_workflow`, `compile_workflow`, `validate_workflow` |
| `PipelineSpec`, `PipelineResult`, `PipelineContext`, `PipelineService`, `PipelineGraph`, `CompiledPipeline`, `PipelineStatus` | `WorkflowSpec`, `WorkflowResult`, `WorkflowContext`, `WorkflowService`, `WorkflowGraph`, `CompiledWorkflow`, `WorkflowStatus` |
| `PipelineError`, `PipelineSpecError`, `PipelineExecutionError` | `WorkflowError`, `WorkflowSpecError`, `WorkflowExecutionError` |
| `summarize_pipeline_result`, `DataModelRecorder.record_pipeline_result` | `summarize_workflow_result`, `record_workflow_result` |
| Events `pipeline_started`, `pipeline_completed`, `pipeline_failed`, `pipeline_cancelled` | `workflow_started`, `workflow_completed`, `workflow_failed`, `workflow_cancelled` |
| `ProcessingRun.pipeline_name`, `ProcessingRun.pipeline_version` | `ProcessingRun.name`, `ProcessingRun.version`, plus a new `ProcessingRun.kind`: `"workflow"`, `"step"` or `"session_action"` |
| Key `pipeline_name` in `run_manifest.json` | `workflow_name` (a manifest with `pipeline_name` was written by an older version) |
| `session.run_pipeline("eit")` | `session.run_preset("eit")` |
| `Pipeline`, `EITPipeline`, `EMGPipeline`, `MultimodalPipeline`, `PipelineConfig` | `Preset`, `EITPreset`, `EMGPreset`, `MultimodalPreset`, `PresetConfig` |
| `available_pipelines`, `get_pipeline`, `register_pipeline`, `PIPELINE_REGISTRY`, `UnknownPipelineError` | `available_presets`, `get_preset`, `register_preset`, `PRESET_REGISTRY`, `UnknownPresetError` |
| `examples/*/*.pipeline.yaml` | `examples/*/*.workflow.yaml` |
| `docs/pipelines.md`, `docs/developer/pipeline-contracts.md` | `docs/workflows.md`, `docs/developer/preset-contracts.md` (the old pages forward to the new ones) |

A `ProcessingRun` records a whole workflow, one step, or one session method
call such as `postprocess_emg`, so its name field is now plain `name`, and
`kind` says which of the three it is.

The old import path `m3resp.pipeline`, kept with a warning since v0.2.0, is
removed; import from `m3resp.workflows`. Its warning promised it would stay
until at least 0.3.0, so **the next release must be 0.3.0 or later**. A workflow file without a `name`
is now called `"workflow"` instead of `"pipeline"`. Error messages say
"Workflow" too.

### `peak_time` and `peak_index` renamed to `extremum_time` and `extremum_index` (#93)

"Peak" says the signal turns at a maximum. That holds for impedance, volume
and the EMG envelope, but not for esophageal pressure or the deepest pressure
of an occlusion, which turn at a minimum. "Extremum" covers both. **The old
names are removed:**

| Before | Now |
|---|---|
| `BreathEvent.peak_time`, `BreathEvent.peak_index` | `BreathEvent.extremum_time`, `BreathEvent.extremum_index` |
| Breath and linked-breath export columns `peak_time`, `peak_index`, `eit_peak_time`, `emg_peak_time`, `ventilator_peak_time` | `extremum_time`, `extremum_index`, `eit_extremum_time`, `emg_extremum_time`, `ventilator_extremum_time`; linked breaths also gain `eit_extremum_index`, `emg_extremum_index`, `ventilator_extremum_index` |
| `compute_breath_timing_parameters(anchor="peak")`, `compute_timing_delay(anchor="peak")` | `anchor="extremum"` (the first is called `compute_multimodal_parameters` before #116, see above) |
| Metadata `peak_sample_index`, `peak_time` on EMG and ventilator results and quality flags | `extremum_sample_index`, `extremum_time` |

`coerce_breath_event` still reads `peak_time`/`peak_index` from
dictionaries and objects made by other detectors, with a warning naming the
new keys. It tries `extremum_time`/`extremum_index` first, then
`peak_time`/`peak_index`, then eitprocessing's `middle_time`, and takes the
time and its position from the same pair. Likewise
`emg.remove_invalid_breaths` still matches a quality flag carrying the old
`peak_sample_index`, with a warning.

### EIT results per breath, pixel maps and masks get their own types; global and regional impedance (#120, #107)

`ParameterResult` held real results (respiratory rate) next to in-between
EIT data: TIV and EELI as bare arrays with no breaths, per-pixel breath
timing, and lung-space masks. It was hard to tell what a `ParameterResult`
was. TIV, EELI, pixel TIV and pixel breath timing now keep each value next to
its breath, and masks have their own type.

| Step | Before | Now |
|---|---|---|
| `eit.continuous_tiv` | upstream `SparseData` only | also `continuous_tiv_result`: `IntervalData`, one TIV per breath |
| `eit.eeli` | `eeli_result`: array `ParameterResult` | `IntervalData`, one EELI per breath |
| `eit.pixel_tiv` | `pixel_tiv_result`: (breath, row, column) `ParameterResult` | `IntervalData`, one `PixelMap` per breath |
| `eit.pixel_breaths` | `pixel_breath_timing_result`: (breath, row, column, 3) `ParameterResult` | `IntervalData`, one (row, column, 3) grid per breath |
| `eit.roi_tiv_lungspace`, `eit.roi_amplitude_lungspace`, `eit.roi_watershed`, `eit.roi_filter_by_size` | `*_result`: 2D `ParameterResult` | `PixelMask` |

New:

| Name | What it is |
|---|---|
| `PixelMap` | One number per EIT pixel, a 2D (row, column) grid |
| `PixelMask` | Which pixels belong to a region: NaN (left out), 1, or a weight between 0 and 1. A 0, or a number outside 0 to 1, raises `ValueError`. |
| `session.interval_data` | `IntervalDataCollection` holding the `IntervalData` results |
| `session.pixel_masks` | `PixelMaskCollection` holding the masks |
| `EITProcessingAdapter.to_interval_data` | Converts the TIV, EELI and pixel TIV of `preprocess_eit()` into `IntervalData` |

What changes for existing code:

- These results are no longer in `session.parameter_results` or in
  `parameter_results.csv`/`parameter_result_arrays.npz`. The export writes
  them to `interval_data.csv`, `interval_data_metadata.json` and
  `interval_data_arrays.npz` (values per breath) and `pixel_masks.csv` and
  `pixel_masks.npz` (masks). With a `DataModelRecorder` attached, numbers
  per breath are stored as one `DerivedFeature` per breath, with the breath's
  start and end as its time window. A missing value (NaN) is stored as no
  value, so the saved store stays valid JSON; a value that is not a number is
  stored as no value with a warning.
- Steps that use the same breath detector and signal share the same breath
  objects, and reuse the breath in `session.events["eit_breaths"]` with the
  same start and end time. The same holds for `preprocess_eit()` and
  `detect_eit_breaths()`, in either order: each TIV or EELI value points to
  the stored breath object. Stored items that are not a `BreathEvent` are
  ignored. `eit.pixel_breaths` stores its breaths with their
  turning point (`extremum_time`), like TIV and EELI.
- `preprocess_eit()` used to store TIV and EELI as one `ParameterResult` per
  breath and **dropped breaths whose value was NaN**. They are now
  `IntervalData` in `session.interval_data`, with every breath kept.
- Pixel TIV no longer keeps the per-pixel breath times in its metadata. With
  `tiv_timing="pixel"` these times are the result of `eit.pixel_breaths`.
- `EITProcessingAdapter.to_parameters` returns only the respiratory and heart
  rate. `_sparse_data_to_parameters` is removed; use
  `sparse_data_to_interval_data` from `m3resp.adapters.eitprocessing_adapter`.
- `eit.roi_filter_by_size` accepts a `PixelMask` or the eitprocessing mask,
  no longer a `ParameterResult`.
- The workflow input/output type `eit_global_impedance` is renamed
  `eit_impedance_waveform`, because these steps accept a global *or* a
  regional impedance waveform. The words follow the chest EIT consensus
  (Physiol. Meas. 2026, doi:10.1088/1361-6579/ae8b55): **global** is the
  whole image plane (all pixels), **regional** is a region of interest such
  as a lung mask. The step `eit.global_impedance` and the `global_impedance`
  output keep their names, so workflow files do not change.

`ProcessingRun.parameter_file_id` (one file) is replaced by
`ProcessingRun.parameter_file_ids` (a list), because one run can now write
three array files: `parameter_result_arrays.npz`, `interval_data_arrays.npz`
and `pixel_masks.npz`. The export links all three to the run. Exporting again
to the same path replaces the earlier link, and `validate_store` reports a run
that names a file missing from the store. **The old field is removed**, so
code reading `run.parameter_file_id` must read `run.parameter_file_ids`.

Also fixed: two array results whose names clean up to the same archive key
(for example `"a-b"` and `"a b"`) no longer overwrite each other in
`parameter_result_arrays.npz`.

See [Pixel maps and masks](docs/concepts/pixel-maps.md).

### New `Interval`, `IntervalData` and `EventData` types; `BreathEvent` is now an `Interval` (#103)

Only breaths could be stored with a start and an end. Other things that last,
such as an occlusion or a period of noise, had no type, and results with one
value per breath were kept as bare arrays that lost which breath each value
belonged to.

| New | What it is |
|---|---|
| `Interval` | Something that lasts: `modality`, `start_time`, `end_time`, and a required `name` such as `"occlusion"` |
| `IntervalData` | One value (a number or an array, such as a pixel map) per interval, in the same order |
| `EventData` | One value per `Event`, in the same order |
| `coerce_interval`, `coerce_intervals` | Turn a dictionary, a `(start, end)` pair or eitprocessing's `Interval` into an m3resp `Interval` |

`BreathEvent` is now an `Interval` whose `name` is always `"breath"`, plus
`extremum_time` and `extremum_index` (named `peak_time` and `peak_index` before #93). What changes for existing code:

- Only `modality`, `start_time` and `end_time` can be given by position.
  Every other field must be given by name, e.g.
  `BreathEvent("eit", 1.0, 2.0, extremum_time=1.5)`.
- Exported breath tables have two new columns, `name` (always `"breath"`)
  and `label`, and `extremum_time`/`extremum_index` are now the last two columns.
- `align_events_by_modality_offset` and `align_events_manual_offset` shift
  intervals as well as events and breaths.
- `coerce_breath_event` refuses an interval named anything other than
  `"breath"` (for example an occlusion), and now keeps `label` when it reads
  a dictionary or an object. A dictionary without `start_time` or `end_time`
  raises a `ValueError` that names the missing entry.

All new names can be imported from `m3resp` and `m3resp.data`. See
[Events, intervals and breaths](docs/concepts/events-and-breaths.md).

### `Event` and `BreathEvent` moved to `m3resp.data` (#96)

The event types sat in `m3resp.core` only because they were written before
the `m3resp.data` package existed. They now live with the other data types.
**The old path is removed**, so imports from it need updating:

| Before | Now |
|---|---|
| `from m3resp.core.events import BreathEvent` | `from m3resp.data.events import BreathEvent` |
| `from m3resp.core import BreathEvent` (also `Event`, `coerce_*`, `event_to_dict`) | `from m3resp.data import ...` |
| `from m3resp.data import Breath` | `from m3resp.data import BreathEvent` |

`from m3resp import BreathEvent` still works. `m3resp.data` now also offers
`coerce_event`, `coerce_breath_event`, `coerce_breath_events` and
`event_to_dict`. `m3resp.core` now only holds `M3Session`. The second name
`Breath` is gone, because `m3resp.datamodel.Breath` (the saved breath record)
and eitprocessing's `Breath` are different classes with that same name.

### `export_store` checks the data model store before writing it (#75)

`validate_store` had to be called separately, and it only returned a list of
problems, so a store whose records did not link up could be saved without
anyone noticing. `export_store` now runs the same checks first. If one fails,
**nothing is written** (not even the output folder) and a
`DataModelValidationError` lists every problem (also in its `problems`
attribute).

| Call | What it does |
|---|---|
| `export_store(store, path)` | Checks that every record points at records that exist and that no time window ends before it starts, then writes |
| `export_store(store, path, require_complete=True)` | Also checks that every record has the details a finished dataset needs (units, sampling rate, start time, file checksums) |
| `export_store(store, path, validate=False)` | Writes without checking, as before |

A store recorded from a session passes the default checks, so existing
exports keep working; all the example workflows were checked. Passing
`require_complete=True` together with `validate=False` raises an error
instead of being ignored.

### Synchronization code and steps tidied up; `Timebase` removed

Every synchronization step is now a `sync.*` step, in one file
(`m3resp.workflows.steps.sync`):

| Before | Now |
|---|---|
| `session.sync_raw` | `sync.raw_modalities` |
| - | `sync.skip` (new, see #115 below) |

The old name still works in specs, so existing YAML files run unchanged; only
the new name is listed by `m3resp steps` and the step catalogue. The example
specs use the new name. The Python module `m3resp.workflows.steps.session` is
gone; the step functions are now `raw_modalities` and `skip` in
`m3resp.workflows.steps.sync`.

`m3resp.synchronization.ventilator` was not about synchronization: it turns
ventilator breath detections into `BreathEvent`s. It moved to the ventilator
adapter. For code that imported from it:

| Was in `m3resp.synchronization.ventilator` | Now in |
|---|---|
| `iter_ventilator_detections`, `normalize_ventilator_breath` | `m3resp.adapters.ventilator_adapter` (still importable from `m3resp.synchronization`) |
| `infer_ventilator_fs`, `infer_ventilator_duration` | `m3resp.adapters.ventilator_adapter` |

Two smaller moves, with no change in behavior: the rules for which recording
is the reference when `reference_modality` is not given moved out of
`M3Session` into `m3resp.synchronization.alignment`
(`raw_reference_modality`, `breath_reference_modality`), and
`ventilator_recording` moved from `m3resp.synchronization.start_times` to
`m3resp.modalities.ventilator`, beside the other ventilator lookups.

`m3resp.synchronization.Timebase` was removed. Nothing used it: a signal's
own time axis, `TimeWindow` and the per-recording start times now do its job.

### Warn when recordings that were never synchronized are compared (#115)

A session is meant to hold recordings of the same stretch of real time, but
nothing checked that the recordings had been synchronized. Now:

- `session.sync_methods` records, per recording (`"eit"`, `"emg"`,
  `"ventilator:<name>"` for each standalone ventilator recording), how it was
  placed on the shared clock, using the data model's vocabulary: `"manual"`
  (`synchronize_raw_modalities`, `synchronize_multimodal_breaths`) or `"none"`.
- `session.skip_synchronization()` (workflow step `sync.skip`) is for
  recordings that really did start together: it records `"none"`.
- `link_breaths` and `emg.evaluate_event_timing` warn with
  `UnsynchronizedDataWarning` when they compare recordings on different
  clocks and one of them has no record. They do not stop. Data from one clock
  (EMG with airway pressure from the same file) never warns.
- **Fix:** `emg.evaluate_event_timing` now compares EMG and ventilator times
  on the shared clock. Since start times replaced cutting (#118), a
  standalone ventilator recording with a start time was compared on the
  wrong clock. Ventilator data from the EMG file gives the same results as
  before.
- Loading a file again clears that recording's start time and record: it is a
  new recording, back at full length, so an earlier start time (for example
  one moved by slicing) no longer applies.
- `summary.json` has a new `"synchronization"` section with the start times
  and the sync methods.
- With a data model recorder attached, `SignalStream.sync_method` and
  `SignalStream.time_offset_ms` (the start time in milliseconds) are now
  filled from the session, and updated after every session action. Ventilator
  signals from `preprocess_ventilator` carry the name of their recording in
  `metadata["recording"]`, so each stream is matched to the right recording.

### Cutting code moved to where it belongs; `m3resp.synchronization.cropping` removed

Synchronization no longer cuts recordings (see "Raw synchronization sets a
start time" below), so the module named after cutting was split up. For code
that imported from it:

| Was in `m3resp.synchronization.cropping` | Now in |
|---|---|
| `VENTILATOR`, `normalize_modality` | `m3resp.modalities.names` |
| `DEFAULT_VENTILATOR_NAME`, `ventilator_raw`, `ventilator_payload`, `ventilator_clock`, `ventilator_recordings` | `m3resp.modalities.ventilator` |
| `resolve_alignment_offsets`, `offsets_relative_to_reference`, `normalize_offset_key`, `ventilator_start_key` | `m3resp.synchronization.alignment` |
| `raw_synchronization_traces` | `m3resp.synchronization.raw_traces` |
| `crop_loaded_modality` and the `_crop_*` helpers | removed: cut a recording with `slice_emg` / `slice_eit` / `slice_ventilator` |

Cutting one recording now lives beside loading it, in
`m3resp.modalities.{emg,eit,ventilator}` (`sample_window` / `frame_window`
and `keep_samples` / `keep_frames`); the session methods only decide what is
cut together and update the start times. The signal-cutting helpers
`slice_by_index`, `slice_by_time` and `slice_signal_by_mode` moved from
`m3resp.workflows.utils` to `m3resp.processing.slicing` (still importable from
the old place), and every cutting step is registered from a `slicing.py` in
its step folder. See docs/concepts/slicing.md.

### Cutting steps renamed: `*.slice_recording` cuts the recording, `*.slice_signal` one signal

`emg.slice` cut the whole loaded EMG recording, while `eit.slice` cut only one
signal passed along in a workflow. The names now say which:

| Before | Now | What it cuts |
|---|---|---|
| `emg.slice` | `emg.slice_recording` | The loaded EMG recording |
| - | `eit.slice_recording` (new, below) | The loaded EIT recording |
| `eit.slice` | `eit.slice_signal` | One EIT signal in a workflow |
| - | `ventilator.slice_recording` (new, below) | The standalone ventilator recordings |

The old names still work in specs, so existing YAML files run unchanged; only
the new names are listed by `m3resp steps` and the step catalogue. The
example specs use the new names.

`eit.slice_signal` now also cuts m3resp's own `Signal` (values and time axis
together, with the kept sample range noted in its metadata), not only
eitprocessing data. On a `Signal` a window that keeps no samples raises an
error instead of returning an empty result. Its time mode is unchanged: times
are the signal's own time values, which for data read from an EIT file is the
time of day.

### Cut standalone ventilator recordings to a time window: `session.slice_ventilator` / `ventilator.slice_recording`

Ventilator data could only be cut when it came inside the EMG or EIT file (by
`slice_emg` / `slice_eit`); a standalone ventilator or monitor export could not
be cut at all. `slice_ventilator(start_seconds, end_seconds=None)` now does
this, with times in seconds from the start of the recording. It cuts the
recording named with `name=`, or every standalone ventilator recording, and
moves each cut recording's start time later by `start_seconds` (see below). It
refuses ventilator data from the EIT or EMG file, which must be cut with its
host recording, and must run before `preprocess_ventilator`.

The `ventilator.load` step has a new `source` option (`ventilator`, `emg` or
`eit`), so a workflow can mark a file as a standalone export. Without it, a
`.bin` file is taken as EIT and anything else as the EMG device's export, as
before.

### Standalone ventilator recordings can each have their own start time

All standalone ventilator recordings shared the one `"ventilator"` start time,
so a ventilator export and a separate monitor export that were started at
different moments could not both be lined up. Each can now get its own start
time under `"ventilator:<name>"` (the name it was loaded under), e.g.
`synchronize_raw_modalities(offset_seconds={"ventilator": -30.0,
"ventilator:monitor": 12.5})`. A recording without its own entry uses
`"ventilator"` as before. A name that is not loaded, or one for ventilator data
inside the EIT or EMG file, raises an error rather than being ignored.

- `slice_ventilator(..., name=...)` cuts one named recording; without `name`
  it still cuts them all. Each cut recording's new start time is stored under
  its own `"ventilator:<name>"` entry, so no other recording moves.
- The `ventilator.load` step has a new `name` option, so a workflow can load
  more than one ventilator recording; `ventilator.slice_recording` has one
  too.
- A `"ventilator:<name>"` recording can be the `reference_modality`.

### Cut the loaded EIT recording to a time window: `session.slice_eit` / `eit.slice_recording`

The EIT counterpart of `slice_emg` / `emg.slice_recording`. It keeps only the part of
the loaded EIT recording between two times (seconds from the first frame, not
the time of day in the file). The cutting is done by `eitprocessing`'s own
`Sequence.select_by_index` (through `EITProcessingAdapter.slice_sequence`), so
pixel data, global impedance, pressure/flow channels and markers are cut to
the same frames, chosen by their time stamps. A ventilator recording loaded
from the same EIT file is cut at exactly those frames, and the EIT start time
moves later by the part cut off the front. It must run before
`preprocess_eit`.

### Raw synchronization sets a start time per recording instead of cutting samples (#118)

`synchronize_raw_modalities` (and the `session.sync_raw` /
`sync.apply_estimated_offset` workflow steps) used to line recordings up by
removing samples: a negative offset cut the start of a recording, a positive
one cut its end. That lost data - an EIT recording starting 90 minutes into a
three-hour pressure recording cost the first 90 minutes of pressure - and a
positive offset did not move the recording in time at all, it only shortened
it.

**This is a behavior change.**

- No samples are removed. Each recording gets a start time on a shared clock,
  stored in `session.start_times` (and in
  `session.parameters["raw_alignment"]["start_time_seconds"]`).
- The start times are added when modalities are compared:
  `synchronize_multimodal_breaths` adds them on top of its own offset (its
  `parameters["alignment"]["start_time_shift_seconds"]` records how much),
  `link_breaths` adds them to any list `synchronize_multimodal_breaths` did not
  already shift, and `plot_synchronization_comparison` draws the "after"
  traces moved rather than cut.
- `session.events` and each modality's signals stay on their own recording's
  clock.
- EIT breath times are now counted from the EIT recording's first sample when
  modalities are compared. A Draeger file's time axis is the time of day, so
  EIT breaths used to sit hours away from EMG and ventilator breaths.
- `slice_emg` moves the EMG start time later by `start_seconds`, so a sliced
  EMG stays lined up.
- `synchronize_raw_modalities` returns `{modality: {"start_time_seconds": ...}}`
  for every loaded modality. `cropped_samples` is gone from its return value
  and from `parameters["raw_alignment"]`.
- Breath detection now sees the whole recording, so it can find breaths near
  the start that used to be cut off. The multimodal example finds all 14 EMG
  breaths instead of 13.

## 0.2.0 (2026-09-25)

### The standard EMG pipeline now removes ECG, band-passes 20-500 Hz, and computes an RMS envelope

The `"emg"` preset was `preprocess_emg` -> `detect_emg_breaths` ->
`postprocess_emg`, with no ECG removal anywhere, so the pipeline advertised as
the standard EMG chain produced an ECG-contaminated envelope - and every breath
detection and amplitude-derived parameter computed from it. The standard chain
is band-pass -> ECG peak detection -> gating -> envelope -> baseline, and the
preset now runs it.

**These are behavior changes: EMG amplitudes, envelopes, and breath detections
will differ from previous runs.**

- `EMGPipeline` (`session.run_pipeline("emg")`) runs `emg.ecg_detect_peaks` +
  `emg.ecg_gating` between preprocessing and breath detection. Gating
  recomputes the envelope from the gated signal, so nothing downstream sees the
  pre-gating envelope.
  - `config={"ecg_detect_peaks": {...}}` / `config={"ecg_gating": {...}}` pass
    keyword arguments to either step; `{"ecg_detect_peaks": {"ecg_channel": n}}`
    points detection at a dedicated reference ECG channel.
  - `config={"ecg_removal": {"ecg_peak_indices": [...]}}` gates already-known
    peaks and skips detection.
  - `config={"ecg_removal": {"enabled": False}}` restores the old behavior. This
    is a data-check/exploratory path only.
- The default EMG band-pass is **20-500 Hz** (`high_pass_hz` was `10.0`).
  `low_pass_hz` still caps at 0.95 x Nyquist below 1053 Hz sampling. The
  high-pass no longer doubles as weak ECG suppression, since gating now owns
  that.
- The default EMG envelope is **RMS**, not ARV. `preprocess_emg` and
  `emg.preprocess` gained `envelope_method` (`"rms"` default, `"arv"`);
  `emg.ecg_gating` gained the same parameter, defaulting to whatever
  preprocessing used so the recomputed envelope cannot silently switch method.
  Pass `envelope_method="arv"` to reproduce `resurfemg`'s `full_rolling_arv`
  exactly.
- New shared primitive `m3resp.processing.rolling_envelope(values,
  window_length=..., method=...)` over the existing `rolling_rms`/`rolling_arv`,
  plus `ENVELOPE_METHODS`.
- `_preprocess_default` itself is unchanged in scope: it still only filters and
  envelopes. ECG removal lives in the preset, not in the adapter primitive, so
  composing the blocks by hand stays possible.

### The Annemijn example is replaced by the multidomain Recording1 example

`examples/annemijn_multimodal/annemijn.pipeline.yaml` is now
`examples/multidomain_recording1/recording1_a2.pipeline.yaml`. It runs the multidomain
results chain (`tools/visualization_tools/paper_results_v2.py`) on
Recording1 (`TestPS3.txt`), window A2 (140-200 s, EIT off), without the
adaptive harmonic stage, which A2 does not need. It gives the notebook's
numbers value for value: 77 ECG peaks, 9 breaths, 9.72 breaths/min, and the
same on/offsets, amplitudes and time products. EIT and FluxMed files are not
used (there is no EIT data in A2).

New options this needed, all off by default so existing pipelines do not
change:

- `emg.preprocess`: `notch_before_bandpass` (notch the raw signal first, as
  the multidomain results chain does) and `envelope_method: median` (median of the absolute
  signal, which ignores short spikes such as heartbeat leftovers).
- `emg.detect_breaths`: `merge_close_peaks_within_width` merges peaks closer
  than the minimum breath width, keeping the higher one.

### Biopac `.txt` files with a leading time column were read one column off

Some Biopac exports start every row with a time column (`min` before `CH1`).
The reader labelled that time column as the first channel, so every channel
was shifted by one and the last channel was dropped. The time column is now
skipped and named in `metadata["skipped_time_column"]`.

### `emg.ecg_wavelet_denoising` keeps the preprocessing envelope method

It always rebuilt the envelope as ARV, whatever preprocessing used. It now
uses the same method as preprocessing (like `emg.ecg_gating`), or the one
given in its new `envelope_method` setting. **Pipelines that combine this step
with the default RMS preprocessing now get an RMS envelope instead of ARV.**

### New steps: `emg.slice` and `ventilator.detect_pressure_breaths`

- `emg.slice` / `session.slice_emg(start_seconds, end_seconds=None)` keeps only
  a time window of the loaded EMG recording, for example to leave out a stretch
  where another device disturbed the EMG. Ventilator channels read from the same
  file (such as airway pressure on a Biopac export) are cut the same way, so
  they stay lined up. A window outside the recording raises `ValueError`.
- `ventilator.detect_pressure_breaths` finds spontaneous breaths as dips in the
  airway pressure, for recordings without a ventilator volume channel. It
  writes `ventilator_breath_indices`, so `ventilator.respiratory_rate` and
  `ventilator.normalize_breaths` work on its output. Missing pressure samples
  warn and never hold a breath.

### Respiratory rate with fewer than two breaths warns instead of crashing

`respiratory_rate_from_indices` (used by the `emg.respiratory_rate` and
`ventilator.respiratory_rate` steps) crashed with `index -1 is out of bounds`
when fewer than two breaths were found. It now warns and returns NaN and an
empty breath-to-breath array.

### EMG breath detection accepts breaths from 0.5 s wide (was 1.0 s)

The default `min_breath_width_seconds` of `emg.detect_breaths`,
`session.detect_emg_breaths()` and `run_pipeline("emg")` is now **0.5 s**. At
1.0 s the `emg_data_synth_quiet_breathing` file (about 22 breaths/min) gave
**0** breaths; at 0.5 s it gives 154 (22.0 breaths/min, against 154
ventilator breaths).

**This is a behavior change: EMG breath counts, and everything computed from
them, can differ from previous runs.** Pipelines that set
`min_breath_width_seconds` themselves are not affected. Pass
`min_breath_width_seconds=1.0` to get the old behavior.

### EMG preprocessing no longer analyses channel 0 by default

`preprocess_emg()` (and so `run_pipeline("emg")` and the `emg.preprocess` step)
used channel 0 whenever no `channel` was given. In the
`emg_data_synth_quiet_breathing` file, channel 0 is the ECG, so the breathing
analysis ran on the heart signal.

**This is a behavior change: code that relied on the silent channel 0 either
gets a different channel or an error.**

- When `channel` is not given, it is picked from the channel names: a
  recording with one channel uses it, and a channel named ECG/EKG is never
  picked. If exactly one other channel is left, that one is used.
- If the names do not show which channel is the breathing muscle (for example
  `emg_0` and `emg_1` from the synthetic generator), an `UnresolvedChannelError`
  lists the channels and asks for `channel=`.
- A `channel` you pass is always used as given.

### The EMG preset no longer crashes when saving the respiratory rate

`session.run_pipeline("emg")` and `session.postprocess_emg()` failed with
`ValueError: ... inhomogeneous shape` on every recording with at least two
detected breaths. The respiratory rate is a pair (median rate, rate of each
breath) with two different shapes, and it was saved as one value.

- The pair is now saved as two `ParameterResult`s: `respiratory_rate` (the
  median, a single number) and `respiratory_rate_breath_to_breath` (one value
  per breath, NaN for outlier breaths). Both are in breaths/min.
- A new test runs the EMG preset from start to end on a made-up recording.

### Signals carry a data *category* alongside their modality

`Signal`, `ParameterResult`, and `QualityFlag` gained a `category` field.
`modality` now means only "which device/technique produced this" (`"eit"`,
`"emg"`, `"ventilator"`); `category` means "what physical quantity is this"
(`"impedance"`, `"airway_pressure"`, `"airflow"`, `"volume"`). They are
independent axes: one device emits several quantities, and one quantity can
come from several devices.

- Physical quantities were removed from the `modality` vocabulary. Pocc results
  that were tagged `modality="pressure"` are now
  `modality="ventilator", category="airway_pressure"`. This value was never
  released on `main`, so no published API changes.
- Query either axis: `for_category(...)` joins `for_modality(...)` on
  `session.signals`, `session.parameter_results`, and `session.quality`.
  `for_modality` is unchanged.
- **Fixes** persisted signal types being recorded incorrectly. Resolving a
  Layer 2 `SignalStream.signal_type` needs both axes, so with only `modality`
  every ventilator signal was stored as `ventilator_pressure` regardless of
  what it measured, and `ventilator_volume` was unreachable. An uncategorized
  ventilator signal is now skipped with a warning instead of guessed.
- **Fixes** derived features being attributed to the wrong signal stream: the
  recorder cached streams by modality alone, so a device's channels overwrote
  one another.
- `parameter_results.csv` gains a `category` column.
- The category vocabulary is open and extensible at runtime via
  `register_category_alias` / `load_category_aliases`, so an externally
  maintained taxonomy can be adopted without vendoring it.

### Ventilator becomes a first-class modality

- Ventilator pipeline steps moved out of the `emg.*` namespace into
  `ventilator.*`, and now declare `modality="ventilator"`, so they are
  discoverable under their own prefix instead of buried among the EMG steps:

  | Was | Now |
  |---|---|
  | `emg.load_ventilator` | `ventilator.load` |
  | `emg.ventilator_channels` | `ventilator.channels` |
  | `emg.detect_ventilator_breath` | `ventilator.detect_breaths` |
  | `emg.normalize_ventilator_breaths` | `ventilator.normalize_breaths` |
  | `emg.ventilator_respiratory_rate` | `ventilator.respiratory_rate` |
  | `emg.find_occluded_breaths` | `ventilator.find_occluded_breaths` |
  | `emg.pocc_intervals` | `ventilator.pocc_intervals` |
  | `emg.pocc_time_product` | `ventilator.pocc_time_product` |
  | `emg.pocc_quality` | `ventilator.pocc_quality` |
  | `emg.detect_non_consecutive_manoeuvres` | `ventilator.detect_non_consecutive_manoeuvres` |

  Every former id keeps working as a **silent alias**, so existing pipeline
  specs compile and run unchanged with no warning. Aliases resolve to the
  current step, so a spec written against an old id still records the new
  `operation_id` in provenance. They are deliberately hidden from
  `available_steps()`/`describe_steps()` and from the "available steps" list in
  `UnknownStepError`, so discovery and any GUI built on it only ever offer
  current names. `register_step` gained an `aliases=` argument and the registry
  exports `STEP_ALIASES`.

  The step *functions* moved too, into a new `m3resp.workflows.steps.ventilator`
  package (`loading.py`/`detection.py`/`normalization.py`/`features.py`/
  `quality.py`, plus its own `_shared.py` mirroring the EIT/EMG packages'
  pattern of a per-modality provenance helper rather than a cross-package
  import). `from m3resp.workflows.steps.emg import pocc_quality` etc. must
  become `from m3resp.workflows.steps.ventilator import pocc_quality`.

  Moving these off the EMG package's shared `_record_step` also **fixes** a
  latent bug: `ventilator.pocc_intervals`/`.pocc_time_product`/`.pocc_quality`/
  `.detect_non_consecutive_manoeuvres` were recording their step-level
  provenance under `modality="emg"` (the EMG helper's hardcoded value) even
  though nothing about them is EMG-specific. They now correctly record
  `modality="ventilator"`.
- New `VentilatorAdapter` (`m3resp.adapters.ventilator_adapter`), completing the
  set alongside `EITProcessingAdapter` and `ReSurfEMGAdapter`. It is the first
  adapter that wraps **no upstream library**: neither `eitprocessing` nor
  `resurfemg` implements ventilator preprocessing, which is why ventilator
  channels were previously used unfiltered. Its defaults are native, built on
  `m3resp.processing.filters` and `m3resp.processing.peaks`.
  - `preprocess()` splits a recording into pressure/flow/volume and returns the
    channels as the ventilator recorded them. Low-passing ventilator waveforms
    is not standard practice, so no filter is applied unless `lowpass_hz` is
    given (clamped below Nyquist); `SUGGESTED_LOWPASS_HZ` (20 Hz) is offered as
    a starting point. When a cutoff is used the unfiltered arrays stay
    available under `"raw"`.
  - `to_signals()` emits `modality="ventilator"` with a per-channel `category`
    (`airway_pressure`/`airflow`/`volume`), so a ventilator's three quantities
    are finally distinguishable. Ventilator data has never reached
    `session.signals` before.
  - `detect_breaths()` returns `BreathEvent`s from the volume channel.
  - Loading delegates to `ReSurfEMGAdapter` unless a loader is injected, since
    ventilator channels usually share the sEMG's file.
- New `M3Session.preprocess_ventilator(variant=..., overwrite=..., **kwargs)`
  and `M3Session.detect_ventilator_breaths(variant=...)`, with the same
  `variant`/`overwrite`/`allow_overwrite` semantics as their EIT/EMG
  counterparts and a `processed_variants["ventilator"]` slot to match.
  - Ventilator signals now reach `session.signals` for the first time.
  - Ventilator breath detection used to be reachable only as a side effect of
    `postprocess_emg`; it is now a method of its own. `postprocess_emg` still
    populates `session.events["ventilator_breaths"]` unchanged.
- New `M3Session.load_ventilator(path, **kwargs)`, mirroring
  `load_eit`/`load_emg`. It stores a new `VentilatorRecording`
  (`m3resp.modalities.ventilator`) on `session.ventilator` and under
  `session.raw["ventilator"]`, and records `load_ventilator` provenance.
- `M3Session(ventilator_adapter=...)` allows a dedicated loader. It defaults to
  the EMG adapter, since ventilator channels usually share the sEMG's file, so
  an injected EMG loader keeps covering both.
- **Breaking:** `session.raw["ventilator"]`/`["vent"]` now hold a
  `VentilatorRecording` rather than the bare `{"array", "metadata"}` payload,
  matching what `raw["eit"]`/`raw["emg"]` have always held. Code reading
  `session.raw["vent"]["array"]` should use `session.ventilator.data["array"]`,
  or `m3resp.synchronization.cropping.ventilator_payload(...)`, which unwraps
  either shape. Assigning a bare dict to `raw["vent"]` still works.
- The `emg.load_ventilator` pipeline step now delegates to the session method,
  so `session.raw` bookkeeping and provenance happen in one place. Its emitted
  `ventilator_raw` artifact is unchanged.

### The ventilator modality canonicalizes to `"ventilator"`

Stage 1 used `"vent"` internally while the docs, `M3Session.link_breaths`, and
`Signal.modality` used `"ventilator"` - so a breath could be tagged
`modality="vent"` while the `LinkedBreath` holding it was keyed `"ventilator"`.
Everything now normalizes to `"ventilator"`.

`"vent"` keeps working everywhere it was previously accepted:

- `session.raw` stores the ventilator recording under **both** keys, pointing
  at the same object, so `session.raw["vent"]` still resolves. Cropping mutates
  that object in place, so the two views cannot drift apart.
- Alignment canonicalizes both the event's modality and the offset keys before
  matching, so events tagged `"vent"` still shift under a `"ventilator"` offset
  and vice versa.
- `sync.estimate_offset` accepts `vent` and `ventilator` as source values, so
  existing pipeline specs keep validating.

Detected ventilator breaths now carry `modality="ventilator"`, and
`session.parameters["alignment"]["offset_seconds"]` is keyed `"ventilator"`.
Code comparing those values against the literal `"vent"` needs updating.

## 0.1.0

- Create the initial Stage 1 M3Resp package skeleton.
- Add `M3Session`, event models, modality adapters, manual synchronization, and
  export helpers.
