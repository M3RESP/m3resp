# Changelog

## Unreleased

### m3resp reads the EIT files

`eit.load` reads Draeger `.bin`, Sentec `.zri` and Timpel `.txt` files with the readers in `m3resp.eit.io` (`read_eit_file`, `read_draeger`, `read_sentec`, `read_timpel`). Each reader returns an `EITFileData`: the images and the waveforms as `Signal`s, the device marks (breath minima and maxima, event markers, QRS complexes) as `EventData`, and the Timpel device breaths as `BreathEvent`s. The readers are copied from eitprocessing 1.8.7 and give the same values. The tests compare them with the saved eitprocessing results in `tests/regression/golden/eit/` and with eitprocessing itself.

The steps after `eit.load` still read an eitprocessing `Sequence`. `m3resp.adapters.eitprocessing_adapter.file_data_to_sequence` builds it from the reader result, field by field, so the `eit` extra still installs eitprocessing. The provenance record of `eit.load` names `m3resp.eit.io.read_eit_file`.

A recording loaded with `M3Session().load_eit` holds the m3resp data next to the `Sequence`: `EITRecording.pixel_impedance`, `signals`, `events`, `intervals` and `sample_frequency`. `m3resp.eit.io.join_eit_files` joins files recorded one after the other. A custom loader may return an `EITFileData`.

### The EIT steps pass Signals

The EIT images and impedance waveforms that the EIT steps give and read are `Signal`s: `raw_eit`, `raw_global_impedance`, `filtered_eit`, `global_impedance`, `regional_impedance` and `functional_impedance`. Before, these were eitprocessing `EITData` and `ContinuousData` objects. The artifact types keep their names: `eit_pixel_signal` is a `Signal` with EIT images (frames, rows, columns) and `eit_impedance_waveform` a `Signal` with one impedance value per frame. They are no longer marked as compatibility types. The outputs with the same Signal under a second name (`raw_pixel_impedance_signal`, `filtered_eit_signal`, ...) stay and have the same artifact type. A step input of type `signal` accepts both EIT types; an input of type `eit_pixel_signal` refuses a waveform before the workflow runs. A step that calls eitprocessing turns its `Signal` inputs into eitprocessing objects first; `m3resp.adapters.eitprocessing_adapter` has the conversions (`signal_to_eit_data`, `signal_to_continuous_data` and back).

`eit.global_impedance` sums the pixels with `m3resp.eit.processing.global_impedance`, a copy of eitprocessing's `EITData.get_summed_impedance` that gives the same values.

`eit.regional_impedance` and `eit.functional_impedance` sum masked images with `m3resp.eit.processing.regional_impedance`. `PixelMask.apply` multiplies the last two axes of an array, a `Signal` or a `PixelMap` by the mask. Masked pixels and regional waveforms have the same values as in eitprocessing.

### Native EIT slicing and device data

`session.slice_eit` and `eit.slice_recording` cut the native images, waveforms, device marks and breaths, then rebuild the eitprocessing Sequence. `eit.slice_signal` cuts native Signals and accepts `start_inclusive` and `end_inclusive` in time mode. The time-selection rules match eitprocessing 1.8.7, including its zero-second interval-boundary behaviour.

Ventilator waveforms can be read from `EITRecording.signals` or an `EITFileData`. Device marks and Timpel breaths remain available in `session.eit.events` and `session.eit.intervals`. Raw EIT Signals record the vendor in `source`. The raw global waveform is available as `session.eit.raw_global_impedance`; when the file has none, m3resp sums the native images.

### m3resp finds the EIT rates and breaths

`eit.detect_rates`, `EITProcessingAdapter.detect_rates` and `session.preprocess_eit()` find the breathing rate and the heart rate with `m3resp.eit.processing.detect_rates`. `eit.detect_breaths`, `EITProcessingAdapter.find_breaths` and `session.preprocess_eit()` find the breaths with `m3resp.eit.processing.find_breaths`. The two functions are copied from `RateDetection` and `BreathDetection` of eitprocessing 1.8.7 and give the same numbers. Their settings are in `RateDetectionSettings` and `BreathDetectionSettings`, with the same names and defaults as in eitprocessing. The `method` of the rate results is `m3resp.eit.processing.detect_rates`.

Some step outputs have a new form:

- `breath_intervals` of `eit.detect_breaths` is an m3resp `IntervalData` whose `intervals` are `BreathEvent`s, with the artifact type `interval_data`. Before, it was an eitprocessing `IntervalData` of `Breath`s (type `interval_collection`). The peak of a breath is in `extremum_time`; eitprocessing calls it `middle_time`.
- `rate_detector` of `eit.detect_rates` and `breath_detector` of `eit.detect_breaths` are the settings records. The artifact types `eit_rate_detector` and `eit_breath_detector` keep their names and now stand for the m3resp settings records. The TIV, EELI and pixel steps make an eitprocessing `BreathDetection` from the settings record.
- `metric.interval_cv` reads `BreathEvent`s and `Interval`s, and its input has the type `interval_data`.
- `eit.detect_rates`, `eit.detect_breaths` and `eit.normalize_breaths` run without eitprocessing.

`session.preprocess_eit()` also stores the breaths in the eitprocessing `Sequence` under `"eit_breaths"`, in the form of eitprocessing. `m3resp.visualization.plot_rate_detection` draws the figure of the rate detection (`rate_detection.png`), with the same look as the eitprocessing figure.

Three cases give a different result from eitprocessing 1.8.7:

- A rate limit of 0 Hz stays 0 Hz. eitprocessing replaces it by the default limit of the subject type.
- When no frequency of the spectrum lies inside a rate range, `detect_rates` gives a ValueError that names the range in breaths per minute. eitprocessing stops with a numpy error.
- When the first and the last valley are removed and no valley is left, `find_breaths` gives no breaths. eitprocessing stops with an IndexError.

### m3resp filters the EIT data

`eit.mdn_filter`, `EITProcessingAdapter.apply_mdn` and `session.preprocess_eit()` (filter `"mdn"`) remove the heart signal with `m3resp.eit.processing.mdn_filter`. The function is copied from `MDNFilter` of eitprocessing 1.8.7 and gives the same numbers, with the same settings and defaults. `mdn_frequency_bands` gives the bands that the filter removes. `eit.butterworth_filter` and `session.preprocess_eit()` (filters `"lowpass"` and `"bandpass"`) run `m3resp.processing.filters.butterworth_filter`, which gives the same numbers as eitprocessing's `ButterworthFilter`.

Changes that a user can see:

- `eit.mdn_filter` has three new settings: `noise_frequency_limit_hz` (default 220 per minute), `notch_distance_hz` (default 10 per minute) and `order` (default 10, from 1 to 10). The defaults give the same result as before.
- `apply_mdn` returns the filtered images as a `Signal`. Before, it returned an eitprocessing `EITData`.
- In `filter_captures` of both steps and of `session.preprocess_eit()`, `unfiltered_data` and `filtered_data` are the images before and after the filter as `Signal`s. Before, they were eitprocessing `EITData` objects (MDN) or arrays (Butterworth).
- The filtered images have the device maker in `source` and the filter function in `method`. The provenance of both steps names m3resp.
- The Butterworth order of EIT images is 1 to 10 in `eit.butterworth_filter` and in `session.preprocess_eit()`. Before, `session.preprocess_eit()` accepted a higher order.
- A pixel with NaN at some samples becomes NaN at every sample of the MDN result, as in eitprocessing. M3Resp also gives a warning that names these pixels.

### Fixed: EIT reading options that stopped with an error

- Sentec: `first_frame` above 0 together with `max_frames` gives `max_frames` frames from `first_frame` on. eitprocessing 1.8.7 stops with an IndexError.
- Timpel: with no frame left after `first_frame`, the reader gives a ValueError about `first_frame`. With one frame left, it reads that frame. eitprocessing 1.8.7 stops with an IndexError in both cases.
- Timpel: when the device minima give fewer than two breath starts, the list of device breaths is empty. eitprocessing 1.8.7 stops with a ValueError.
- `vendor="simulated"` gives the "No loading method" error of an unknown vendor. eitprocessing 1.8.7 stops with a KeyError.

### Fixed: equal signals with NaN values

`TimeSeries ==`, `Signal ==` and `ParameterResult ==` count a NaN at the same place in both values as equal. Before, two copies of the same Sentec or Timpel images were "not equal", because the pixels outside the chest are NaN. A NaN at a place where the other signal has a number still makes the signals different. Text values compare as before.

## 0.2.1 (2026-10-26)

### Interactive plotly figures

`plot_session_overview`, `plot_synchronization_comparison` and
`plot_eit_processing_summary` take `backend="plotly"` and then return an
interactive plotly figure: zoom, pan and read the values with the mouse. The
default `backend="matplotlib"` gives the same static figure as before. For a
plotly figure, `max_points` (default 5000) sets the largest number of points
drawn for one signal; a longer signal keeps the lowest and the highest value
of each short time bin. The tutorials use the plotly figures. The `eit`,
`emg` and `all` extras install plotly, and the `docs` extra also installs
kaleido, which needs Chrome to save the PNG thumbnails of the tutorials.

### Tests on recorded data (part of #92)

New tests run m3resp on dataset 3 of the 27 July 2026 recordings (TestPS3):
one minute with EIT on (37-97 s) and one with EIT off (140-200 s). The
FluxMed and EIT recordings are lined up on the Biopac clock by
cross-correlating the airway pressure. The tests check that each step gives
exactly what `eitprocessing` and `resurfemg` give, that the results equal the
stored values in `tests/regression/real_data/reference_values.json`, and that
heart rate, breathing rate and breath onsets agree with the ECG and the flow.
They run with `pytest -m real_data` and read the recordings from the folder
in `M3RESP_REAL_DATA`; the normal test run leaves them out.
`scripts/update_real_data_references.py` rewrites the stored values.

### Steps after a cut read the cut recording

`eit.slice_recording` cut the recording in the session, but `raw_eit`,
`eit_sequence` and `raw_global_impedance` kept the whole recording, so the
steps after it (rates, MDN filter, global impedance, breaths) used the whole
file. On one minute of TestPS3 this gave a heart rate of 40 per minute from
all 106 s; the cut minute gives 75.5 per minute. `eit.slice_recording` now
gives the cut recording under the names `eit.load` uses, and
`emg.slice_recording` does the same for `emg_recording` and
`raw_emg_signals`. The raw signals of the cut recording replace the uncut
ones in `session.signals`. Step registrations have a new field, `replaces`:
the outputs a step may write again under a name an earlier step wrote.

### uv commands next to the pip commands (#88)

The install, test, docs and run commands in the README, `CONTRIBUTING.md`,
the documentation site and the synthetic data README are shown both for pip
and for [uv](https://docs.astral.sh/uv/). On the
documentation site each command block has a "pip" and a "uv" tab, and the
choice carries over to every page. The `uv synx` typo in the README is fixed.

### Origin of the EIT code noted in each module (#87)

`processing/peaks.py` (`remove_duplicate_extrema`,
`remove_low_amplitude_peaks`) and `data/pixel_maps.py` (the checks of
`PixelMask`) have a Provenance block naming the eitprocessing code they
follow, and are listed in `NOTICE.md`. The eitprocessing blocks in
`filters.py`, `windows.py` and `metrics.py` give the commit m3resp installs
(`m3resp-integration`, `6d6da6a`, version 1.8.7). `CONTRIBUTING.md` describes
the block, and `tests/test_notice_provenance.py` checks that the blocks and
`NOTICE.md` list the same files.

### EMG polishing: session baseline and adapter docs (#72)

`M3Session.emg_moving_baseline(window_seconds=7.5, step_seconds=0.2,
percentile=33.0, variant=None)` computes the moving baseline of the
processed EMG envelope, with the same calculation as the
`emg.moving_baseline` step. The baseline is added to `session.signals` as
`<channel>_moving_baseline` (`<channel>_moving_baseline:<variant>` for a
preprocessing variant), replacing the one from an earlier call, and the
call is recorded in `session.provenance` as `emg_moving_baseline` with its
settings and variant. The EMG tutorial now runs preprocess, baseline, then
breath detection:

```python
session.preprocess_emg(channel=1)
baseline = session.emg_moving_baseline()
session.detect_emg_breaths(baseline=baseline)
```

The `"emg"` preset uses it, with the same numbers as before; its baseline
now also appears in `session.signals` and `session.provenance`.
`config={"baseline": {...}}` takes `window_seconds`, `step_seconds` and
`percentile` (converted to numbers, so `"7.5"` works); other keys,
including `variant`, raise `TypeError`, so the baseline always comes from
the processed EMG that breath detection uses. A zero window or step now
raises `ValueError`; before, it became one sample.

`M3Session.emg_slopesum_baseline(...)` computes the slope-sum baseline the
same way, with the settings and calculation of the `emg.slopesum_baseline`
step and the same `variant` argument. It adds the baseline, and the running
mean and running standard deviation of the moving baseline, to
`session.signals` as
`<channel>_slopesum_baseline`, `<channel>_slopesum_baseline_running_mean`
and `<channel>_slopesum_baseline_running_std` (`:<variant>` at the end for a
variant), replacing the ones from an earlier call, and records the call in
`session.provenance` as `emg_slopesum_baseline`.

`ReSurfEMGAdapter` and its method files have plain-language docstrings
(inputs, outputs, units, and which values come from ReSurfEMG and which are
computed in m3resp), and the adapter page in the developer docs lists every
method that way. References to planning documents and project stages are
removed from the EMG and ventilator code.

### Several EIT files loaded as one recording (#121)

`eit.load`, `eit.load_custom` and `session.load_eit()` take a list of files
recorded one after the other and read them as one recording:

```yaml
- uses: eit.load
  with: { file_path: [part_1.bin, part_2.bin], vendor: draeger }
```

The files are joined in the given order. A file that starts before the one
before it ends is refused with an error naming both files. A pause between
two files stays in the time axis and gives a warning with its length.
`first_frame` and `max_frames` are refused with several files, because
eitprocessing applies them to each file; cut the joined recording with
`eit.slice_recording`. A custom loader receives the list as given.

`EITRecording.path` is now `EITRecording.paths`, the files read, in
recording order (one file gives a list of one); the provenance record of `load_eit` lists them all, and the
data-model recorder stores one raw `DataFile` per file. Step parameters
have a new `allow_list` field, true for a `path` parameter that also takes
a list of paths; workflow checks resolve and check each path in the list.

### `eit.load_custom` for your own EIT loader; `eit.load` needs a vendor (#106)

`eit.load` reads Draeger, Sentec and Timpel files with eitprocessing, and
its `vendor` parameter is required. It uses eitprocessing even when the
session's EIT adapter has a loader of its own.

The new step `eit.load_custom` reads a file with the loader given as
`EITProcessingAdapter(loader=...)`, either on the session or through
`run_workflow(..., eit_adapter=...)`. It takes `file_path` and
`loader_options`, gives the same outputs as `eit.load`, and stops with an
error when no loader was given.

| Before | Now |
|---|---|
| `eit.load` with a custom loader set and no `vendor` | `eit.load_custom` |

`session.load_eit()` has a new `loader` argument: `"eitprocessing"`,
`"custom"`, or `"auto"` (default, the custom loader when there is one,
else eitprocessing, as before). `EITProcessingAdapter` has the matching
methods `load_with_eitprocessing` and `load_with_custom_loader`.

### EMG preprocessing as separate steps; the envelope has its own step (#124)

Four new workflow steps split EMG preprocessing into its parts:

| Step | Does |
|---|---|
| `emg.select_channel` | takes one channel of the loaded EMG recording |
| `emg.notch_filter` | removes a frequency and its harmonics (mains hum, EIT frame-rate comb) |
| `emg.bandpass_filter` | band-pass filter, 20-500 Hz by default |
| `emg.envelope` | RMS, ARV or median envelope over a moving window (0.25 s, RMS by default) |

`emg.preprocess` runs channel selection, the notch filter and the band-pass
filter in one step, and its settings are now listed as step parameters.
**The envelope is computed only by `emg.envelope`**, normally after ECG
removal:

- `emg.preprocess` returns the filtered EMG with `"envelope"` None. Its
  former `envelope_window_seconds`, `envelope_method` and
  `compute_envelope` settings are refused with an error that names
  `emg.envelope`.
- `emg.ecg_gating` and `emg.ecg_wavelet_denoising` drop their
  `envelope_window_seconds` and `envelope_method` parameters, and the
  processed EMG they return has `"envelope"` None.
- Steps that need the envelope stop with an error naming `emg.envelope`
  when it is missing.

| Before | Now |
|---|---|
| `emg.preprocess` with `envelope_window_seconds: 0.5`, then `emg.ecg_gating` with `out: { processed_emg_after_ecg: processed_emg }` | `emg.preprocess`, then `emg.ecg_gating`, then `emg.envelope` with `in: { processed_emg: processed_emg_after_ecg }`, `with: { window_seconds: 0.5 }` and `out: { processed_emg_with_envelope: processed_emg }` |

The four example workflows are changed this way and give the same numbers
as before. `session.preprocess_emg()` still filters and computes the
envelope in one call. The `"emg"` preset runs `emg.envelope` after ECG
gating with the window and method preprocessing used, or with
`config={"envelope": {...}}`; its results are the same as before.

The parts are also `ReSurfEMGAdapter` methods: `select_emg_channel`,
`notch_filter_emg`, `bandpass_filter_emg` and `emg_envelope`.

A notch `max_frequency` above the Nyquist frequency (half the sampling
rate) gives a warning and is lowered to the Nyquist frequency, in
`emg.notch_filter` and `emg.preprocess`. The recorded
`notch_max_frequency` is the frequency the filter used.

### `register_step(..., record=True)` logs each call (#104)

A step registered with `record=True` adds one record to
`session.provenance` after each call: step name and modality, its reads
and writes, the version of its upstream package, and the details the step
returns under the output key `"_provenance"`. The key is removed from the
outputs. The 33 built-in steps that logged their own call now use it, and
the three `_record_step` helpers in the EIT, EMG and ventilator step
modules are removed. See "Adding a custom step" in the workflow docs.

The records are the same as before, with one change: `upstream_version` is
the installed version of the first package in the step's
`optional_packages`, and None for a step without one.
`ventilator.pocc_time_product` and `emg.remove_invalid_breaths` compute
their values in m3resp, so their records now have None in place of the
ReSurfEMG version.

A workflow run warns when a step registered without `record=True` returns
`"_provenance"` details, because those details are not saved.

A workflow run also writes one `session.processing_history` entry per step,
as before. Steps with `record=True` therefore appear in both
`session.provenance` and `session.processing_history`.

### Step descriptions share one field list (#111)

The fields that describe a step (name, summary, description, modality,
category, parameters, inputs and outputs, alternatives, optional packages,
version, ...) are declared once, in the new `StepInfo` class in
`m3resp.workflows.registry`. `StepDefinition` and `StepDescription` both
build on it, and `describe_step` and `StepDescription.as_dict()` copy every
field from that list. A new step field is added in `StepInfo` and as a
`register_step` keyword; a test fails when one of the two is missing.

The output of `describe_step`, `describe_steps` and `m3resp describe` is
the same as before. `StepDefinition` and `StepDescription` take keyword
arguments only.

### `resample_signal` replaced by `interpolate_to_time` (#110)

`resample_signal(signal, target_frequency_hz)` built a new, evenly spaced
time axis from a rate in Hz. Device time axes are not perfectly even, so
that axis drifted away from the other device's real timestamps.
`interpolate_to_time(signal, time)` takes the time points themselves, so a
signal can be put on exactly the time points of another signal. It lives
with the other signal operations in `m3resp.processing`. **The old name is
removed.**

| Before | Now |
|---|---|
| `m3resp.resample_signal(emg, eit.sample_frequency)` | `m3resp.interpolate_to_time(emg, eit.time)` |
| `from m3resp.synchronization import resample_signal` | `from m3resp.processing import interpolate_to_time` |

Other differences:

- Time points before the first or after the last sample get NaN, with a
  warning. `resample_signal` copied the first or last value outward without
  a warning.
- Values with more than one number per time point (channels, EIT pixel
  frames) work. `resample_signal` handled one number per time point only.
- A signal with fewer than two samples, or a time axis that is not strictly
  increasing, raises `ValueError`.
- The result has `metadata["interpolation"] = "linear"`, and
  `sample_frequency` is 1 / the median time step of the new time points,
  rounded to 9 significant digits (a 20 Hz time axis gives exactly 20.0).
- Both time axes must be on the same clock. Synchronize the recordings
  first; the function cannot check this.

As before, the interpolation is linear with no anti-aliasing filter:
low-pass filter the signal first when the new time points are further
apart than the original samples.

### `capture_value` moved to `m3resp.processing.captures` (#91)

`capture_value` stores an extra result of a processing function in the
optional `captures` dictionary. It is used by the filters and the peak
detection, so it has its own module.

| Before | Now |
|---|---|
| `from m3resp.processing.filters import capture_value` | `from m3resp.processing.captures import capture_value` |

### One `QualityFlag` can cover several breaths (#94)

`QualityFlag` has a new field, `breath_ids`, for a check that judges a
set of breaths together, such as ten breaths marked as a stable stretch.
It works like `ParameterResult.breath_ids`. The list reaches
`quality_flags.csv` and `summary.json`, and `QualityFlag.to_dict()` gives
the same fields as a dictionary.

A `QualityFlag` or `ParameterResult` sets `breath_id` for one breath or
`breath_ids` for several breaths. Setting both, or giving an empty
`breath_ids` list, now raises `ValueError`.

- **Fixes** `DataModelRecorder.record_quality_flag` dropping the flag's
  breath and time window. The stored `QualityAnnotation` now keeps
  `start_time`/`end_time` as its time window, and its breaths in the new
  `QualityAnnotation.breath_ids` field.
- `validate_store` reports a `QualityAnnotation` whose time window ends
  before it starts, as it already did for derived features and clinical
  events.

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
`peak_time`/`peak_index`, then eitprocessing's `middle_time`. It selects
the first pair with either value set and reads both from that pair,
even if one is `None`. Likewise
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
