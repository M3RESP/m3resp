# Adapters

An adapter connects M3Resp to an outside package. It calls `eitprocessing` or `resurfemg`, and changes the results into the Layer 1 objects of the {doc}`concept guides <../concepts/index>`. The session, the workflow steps and the export read only these M3Resp objects.

```text
eitprocessing or resurfemg result
    |
    v
EITProcessingAdapter, ReSurfEMGAdapter or VentilatorAdapter
    |   to_signals, to_parameters, to_interval_data, to_quality_flags
    v
Signal, BreathEvent, ParameterResult, IntervalData, QualityFlag
```

The session has one adapter of each type: `session.eit_adapter`, `session.emg_adapter` and `session.ventilator_adapter`.

## The conversion methods

Each adapter has the same conversion methods. They read the result of `preprocess` or `postprocess` and give lists of M3Resp objects:

| Method | Gives |
|---|---|
| `to_signals(result)` | `list[Signal]` |
| `to_parameters(result)` | `list[ParameterResult]` |
| `to_interval_data(result)` (EIT only) | `list[IntervalData]`: TIV, EELI and pixel TIV, each value next to its breath |
| `to_quality_flags(result)` | `list[QualityFlag]` |
| `detect_breaths(data)` | `list[BreathEvent]` |

## EITProcessingAdapter

The code is in `src/m3resp/adapters/eitprocessing_adapter/`.

| Methods | What they do |
|---|---|
| `load(path, vendor)` | Load a vendor EIT file into an `eitprocessing` sequence |
| `get_raw_eit`, `get_global_impedance`, `get_regional_impedance` | Get the pixel, global or regional impedance from a sequence |
| `slice_sequence` | Cut native images, waveforms, marks and breaths, then rebuild a sequence |
| `detect_rates`, `apply_mdn`, `find_breaths`, `find_pixel_breaths`, `compute_eeli`, `compute_pixel_tiv` | Run one `eitprocessing` operation each |
| `compute_tiv_lungspace`, `compute_amplitude_lungspace`, `compute_watershed_lungspace`, `filter_roi_by_size`, `as_pixel_mask` | Make and filter regions of interest |
| `preprocess(sequence, ...)` | Run rate detection, filter, breath detection, TIV, EELI and pixel TIV in one call. `session.preprocess_eit()` uses it. |
| `compute_tiv(sequence, compute=...)` | Run a TIV function that you give |

## ReSurfEMGAdapter

The code is in `src/m3resp/adapters/resurfemg_adapter/`, with one file per task: `_core.py`, `_ecg.py`, `_baseline.py`, `_quality.py` and `_defaults.py`.

The methods take plain arrays or the recording dictionary of `load`. This dictionary has the samples under `"array"`, one row per channel, and the sampling rate in Hz under `metadata["fs"]`. The values keep the units of the recording. Times are in seconds, and positions are sample numbers. The last column says which package calculates the result. A `resurfemg` function runs with the settings and defaults of `resurfemg`.

| Methods | What they do | Calculated by |
|---|---|---|
| `load(path)` | Load an EMG file into a recording dictionary. The `VentilatorAdapter` uses it for ventilator channels in the same file. | `resurfemg` (`load_file`). M3Resp reads Biopac `.txt` files. |
| `select_emg_channel(recording, channel=)` | Take one channel to analyse. Without `channel`, choose it from the channel labels, and leave out ECG/EKG channels. | M3Resp |
| `notch_filter_emg(values, ...)` | Remove one frequency and its harmonics, for example the mains hum or the frame rate of the EIT | M3Resp (`processing.filters`) |
| `bandpass_filter_emg(values, ...)` | Band-pass filter, 20 to 500 Hz by default. If 95% of the Nyquist frequency is below 500 Hz, the upper limit is 95% of the Nyquist frequency. | `resurfemg` (`emg_bandpass_butter`) |
| `emg_envelope(values, ...)` | RMS, ARV or median envelope over a moving window | M3Resp (`processing.windows`) |
| `preprocess(recording, ...)` | The four methods above in one call. `session.preprocess_emg()` uses it. | As above |
| `detect_ecg_peaks`, `gate_ecg`, `wavelet_denoise_ecg` | Find and remove the ECG | `resurfemg` |
| `moving_baseline`, `slopesum_baseline` | Calculate the baseline of the envelope | `resurfemg` |
| `detect_breaths(processed, ...)` | Find the breaths as envelope peaks above the baseline | M3Resp (`processing.peaks`) |
| `postprocess(processed, events, ...)` | Calculate the baselines, the start and end of each breath, the parameters of each breath and the quality checks in one call. `session.postprocess_emg()` uses it. | M3Resp for the parameters, `resurfemg` for the baselines and the quality checks |
| `snr_pseudo`, `pocc_quality`, `interpeak_distance`, `percentage_under_baseline`, `detect_local_high_aub`, `detect_extreme_time_products`, `detect_non_consecutive_manoeuvres`, `evaluate_bell_curve_error`, `evaluate_event_timing`, `evaluate_respiratory_rates` | Run one quality check each | `resurfemg` |
| `available_postprocessing()`, `run_postprocessing_function(category, name, ...)` | List and call 23 functions of `resurfemg.postprocessing` by name | `resurfemg` |
| `compute_features(signal, events, compute=...)` | Run a feature function that you give | Your function |

The workflow steps in `workflows/steps/emg/` and `workflows/steps/ventilator/` call these methods one by one.

## VentilatorAdapter

The code is in `src/m3resp/adapters/ventilator_adapter/`. The adapter reads ventilator data from three sources:

| Source | Read by |
|---|---|
| An EMG file with ventilator channels | `ReSurfEMGAdapter` |
| An EIT `.bin` file with ventilator waveforms | `EITProcessingAdapter` |
| A file type with a reader of its own | The reader registered for its file extension |

`source="eit"`, `"emg"` or `"ventilator"` selects the source. Without `source`, the adapter uses the reader registered for the file extension. Without a registered reader, it reads `.bin` files as EIT and all other files as EMG.

`preprocess` splits the recording into pressure, flow and volume channels. With `lowpass_hz`, it also filters them with a low-pass filter. `to_parameters` and `to_quality_flags` give empty lists. The `ventilator.*` workflow steps calculate the ventilator parameters and quality checks.

## Comparison tests

The tests in `tests/regression/` compare the result of each adapter method with the direct call of `eitprocessing` or `resurfemg`, with the same settings. Most tests require exactly equal values. If one of these tests fails, look at the adapter method first. See {doc}`testing`.

## Add a function of eitprocessing or resurfemg

1. Add a method to the adapter. Give it the same argument names and defaults as the original function.
2. To store the result in the session, change it into M3Resp objects in a `to_*` method.
3. Add a test in `tests/regression/` that compares the method with the original function.
4. To use the method in workflow files, add a step. See {doc}`../how-to/add-a-step`.

Continue with {doc}`preset-contracts`.
