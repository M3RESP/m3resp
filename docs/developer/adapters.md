# Adapters

`m3resp` does not reimplement EIT/EMG algorithms - it calls
`eitprocessing`/`resurfemg` and converts their output into the Layer 1
objects described in the [concept guides](../concepts/index.md). The adapter is the
conversion boundary: everything on the near side of it (`M3Session`,
`m3resp.data`, exports) only ever sees `Signal`/`ParameterResult`/
`QualityFlag`/`BreathEvent`, never a raw `eitprocessing`/`resurfemg` object.

```text
eitprocessing output
    |
    v
EITProcessingAdapter
    |
    v
m3resp Signal / BreathEvent / ParameterResult / QualityFlag
```

```text
resurfemg output
    |
    v
ReSurfEMGAdapter
    |
    v
m3resp Signal / BreathEvent / ParameterResult / QualityFlag
```

## `EITProcessingAdapter` (`src/m3resp/adapters/eitprocessing_adapter/`)

| Method | Responsibility |
| --- | --- |
| `load(path, vendor, **kwargs)` | Load a vendor EIT file into an `eitprocessing` sequence. |
| `get_raw_eit` / `get_global_impedance` | Extract pixel/global impedance from a loaded sequence. |
| `detect_rates`, `apply_mdn`, `find_pixel_breaths`, `compute_eeli`, `compute_pixel_tiv`, `compute_tiv_lungspace`, `compute_amplitude_lungspace`, `compute_watershed_lungspace`, `filter_roi_by_size` | Individual `eitprocessing` operations, exposed one-to-one so a caller can compose a custom sequence. |
| `preprocess(...)` | Chains the operations above into one call - what `session.preprocess_eit()` uses by default. |
| `detect_breaths(data, **kwargs)` | Returns `list[BreathEvent]` directly - already converted. |
| `compute_tiv(sequence, **kwargs)` | Runs TIV computation on a loaded sequence. |
| `to_signals(preprocessed)` | **Conversion boundary.** Turns a `preprocess()` result into `list[Signal]`. |
| `to_parameters(preprocessed)` | **Conversion boundary.** Turns a `preprocess()` result into `list[ParameterResult]` (respiratory and heart rate). |
| `to_interval_data(preprocessed)` | **Conversion boundary.** Turns the per-breath TIV, EELI and pixel TIV of a `preprocess()` result into `list[IntervalData]`, each value next to the breath it was computed over. |
| `to_quality_flags(preprocessed)` | **Conversion boundary.** Turns a `preprocess()` result into `list[QualityFlag]`. |

## `ReSurfEMGAdapter` (`src/m3resp/adapters/resurfemg_adapter/`)

The session holds one as `session.emg_adapter`. Its methods take plain
arrays or the recording dictionary from `load` (samples under `"array"`,
one row per channel; sampling rate in Hz under `metadata["fs"]`). Values
keep the units of the recording; times are in seconds and positions in
samples. The last column says whether a value comes from ReSurfEMG (called
with ReSurfEMG's own settings and defaults) or is computed in m3resp.

| Method | What it does | Computed by |
| --- | --- | --- |
| `load(path, **kwargs)` | Read an EMG file into a recording dictionary (also used by `VentilatorAdapter` for ventilator channels stored in the same file). Biopac `.txt` exports are read by m3resp. | ReSurfEMG `load_file` |
| `select_emg_channel(recording, channel=)` | Take one channel as the trace to analyse; with no channel, choose it from the labels, leaving out ECG/EKG channels. | m3resp |
| `notch_filter_emg(values, ...)` | Remove a frequency and its harmonics (mains hum, EIT frame-rate comb). | m3resp (`processing.filters`) |
| `bandpass_filter_emg(values, ...)` | Band-pass filter, 20-500 Hz by default. | ReSurfEMG `emg_bandpass_butter` |
| `emg_envelope(values, ...)` | RMS, ARV or median envelope over a moving window. | m3resp (`processing.windows`) |
| `preprocess(recording, **kwargs)` | The four methods above in one call; what `session.preprocess_emg()` uses. | as above |
| `detect_ecg_peaks`, `gate_ecg`, `wavelet_denoise_ecg` | Find heartbeats in the EMG (or an ECG channel) and remove them. | ReSurfEMG |
| `moving_baseline`, `slopesum_baseline` | The quiet level of the envelope. | ReSurfEMG |
| `detect_breaths(processed_emg, **kwargs)` | Find breaths as envelope peaks above the baseline; returns `list[BreathEvent]`. | m3resp (`processing.peaks`) |
| `postprocess(processed_emg, events, **kwargs)` | Baselines, breath on/offsets, per-breath measures (amplitude, time product, ...) and quality checks in one call; what `session.postprocess_emg()` uses. | m3resp measures, ReSurfEMG baselines and quality checks |
| `snr_pseudo`, `pocc_quality`, `interpeak_distance`, `percentage_under_baseline`, `detect_local_high_aub`, `detect_extreme_time_products`, `detect_non_consecutive_manoeuvres`, `evaluate_bell_curve_error`, `evaluate_event_timing`, `evaluate_respiratory_rates` | Quality checks on breaths and Pocc manoeuvres. | ReSurfEMG |
| `available_postprocessing()` / `run_postprocessing_function(category, function_name, ...)` | List, and call by name, ReSurfEMG postprocessing functions that have no method of their own. | ReSurfEMG |
| `compute_features(signal, events, compute=...)` | Run a feature function you pass as `compute=`. | your function |
| `to_signals(processed_emg)` | **Conversion boundary.** Raw channel, filtered trace and envelope as `list[Signal]`. | m3resp |
| `to_parameters(postprocessed)` | **Conversion boundary.** Per-breath measures of a `postprocess()` result as `list[ParameterResult]`. | m3resp |
| `to_quality_flags(postprocessed)` | **Conversion boundary.** Quality checks of a `postprocess()` result as `list[QualityFlag]`. | m3resp |

The `emg.*` workflow steps (`workflows/steps/emg/`) and the ventilator steps
(`workflows/steps/ventilator/`) call these methods one by one.

## `VentilatorAdapter` (`src/m3resp/adapters/ventilator_adapter/`)

Ventilator pressure/flow/volume data has its own adapter, available on the
session as `session.ventilator_adapter`. It does not read files itself: it
hands each file to the adapter that already knows the format.

| Method | Responsibility |
| --- | --- |
| `load(path, **kwargs)` | Load a ventilator recording from one of three sources: a file shared with the EMG (read by `ReSurfEMGAdapter`), an EIT `.bin` file that stores ventilator waveforms next to the impedance frames (read by `EITProcessingAdapter`), or any other format with a reader registered for its file extension. `source="eit"`/`"emg"`/`"ventilator"` picks the source explicitly; otherwise `.bin` is read as EIT and everything else as EMG. |
| `preprocess(recording, **kwargs)` | Split the recording into pressure/flow/volume channels and filter them. |
| `detect_breaths(processed, **kwargs)` | Returns `list[BreathEvent]` directly - already converted. |
| `to_signals(processed_ventilator)` | **Conversion boundary.** Turns a `preprocess()` result into `list[Signal]`. |
| `to_parameters` / `to_quality_flags` | Return empty lists: ventilator preprocessing computes no parameters or quality checks. Pocc and ventilator quality results come from the `ventilator.*` steps instead. |

## Regression guarantee

`tests/regression/` pins down that the adapters are still thin wrappers:
each test drives the adapter's public API on synthetic data and asserts the
result is identical to calling the underlying `eitprocessing`/`resurfemg`
function directly with the same arguments. If one of these starts failing,
the adapter output now differs from the upstream output - check the diff against the adapter method involved before assuming
the test is wrong. See [testing.md](testing.md).

## Adding a new upstream algorithm

1. Add a thin wrapper method on the relevant adapter that calls the upstream
   function with the same argument names/defaults it has.
2. If the result should be visible on `M3Session`'s typed collections, extend
   `to_signals`/`to_parameters`/`to_quality_flags` (or add a new `to_*`
   method if the shape doesn't fit those three).
3. Add a regression test asserting your wrapper's output matches calling the
   upstream function directly.
4. Optionally register a step under `workflows/steps/` so the operation is also
   reachable from a declarative YAML/JSON workflow (see
   [preset-contracts.md](preset-contracts.md)).
