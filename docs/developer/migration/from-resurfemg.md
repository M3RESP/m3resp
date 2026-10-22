# Move from resurfemg

This page lists the `resurfemg` calls and the M3Resp calls that do the same work. See the {doc}`../migration` for the tasks that are the same for all modalities.

To run all EMG steps in one call, use `session.run_preset("emg")`. See {doc}`../../how-to/emg-processing-settings`.

## Load a recording

`session.load_emg` calls `load_file` of `resurfemg`:

```python
# resurfemg
from resurfemg.data_connector.converter_functions import load_file

array, dataframe, metadata = load_file("recording.Poly5")

# M3Resp
session.load_emg("recording.Poly5")
```

M3Resp reads Biopac `.txt` exports itself, with the sampling rate and the channel names from the file header.

## Filter and make the envelope

`session.preprocess_emg` runs `emg_bandpass_butter` and then calculates the {term}`envelope`:

```python
session.preprocess_emg(channel=1, envelope_method="arv", envelope_window_seconds=0.25)
```

:::{note}
The default envelope of M3Resp is the RMS envelope. For the envelope of `full_rolling_arv`, set `envelope_method="arv"`.
:::

The default band-pass filter is 20 Hz to 500 Hz. Give the window of the envelope in seconds.

## Remove the ECG

The adapter has one method for each ECG function of `resurfemg.preprocessing.ecg_removal`:

| `resurfemg` | M3Resp |
|---|---|
| `detect_ecg_peaks` | `session.emg_adapter.detect_ecg_peaks(signal, sample_frequency=...)` |
| `gating` | `session.emg_adapter.gate_ecg(signal, peak_indices)` |
| `wavelet_denoising` | `session.emg_adapter.wavelet_denoise_ecg(signal, peak_indices, sample_frequency=...)` |

The `"emg"` preset runs the peak detection and the gating. In a workflow file, use the steps of {ref}`workflow-ecg-removal`.

## Detect breaths

`session.detect_emg_breaths()` finds the breaths in the envelope and gives `BreathEvent` objects. The detection uses the same rule as `detect_emg_breaths` of `resurfemg`. See {doc}`../../concepts/events-and-breaths`.

## Calculate parameters and quality checks

`session.postprocess_emg()` calculates the parameters of `resurfemg.postprocessing.features` with the M3Resp copies in `m3resp.processing.metrics`. It also runs the checks of `resurfemg.postprocessing.quality_assessment`. It stores the results as `ParameterResult` objects in `session.parameter_results`, and the checks as `QualityFlag` objects in `session.quality`. See {doc}`../../concepts/parameters` and {doc}`../../concepts/quality`.

## Call another resurfemg function

The adapter can call 23 functions of `resurfemg.postprocessing` by name. `available_postprocessing()` lists them per module:

```python
print(session.emg_adapter.available_postprocessing()["features"])
session.emg_adapter.run_postprocessing_function("features", "time_product", *args, **kwargs)
```

Continue with {doc}`../../tutorials/20_emg_only` for a full example.
