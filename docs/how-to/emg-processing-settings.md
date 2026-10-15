# How to change the EMG processing settings

The `"emg"` {term}`preset` runs the EMG processing in this order: band-pass filter, {term}`ECG` peak detection, ECG gating, {term}`envelope`, {term}`baseline`, breath detection and parameters. Each step has a group in the preset configuration:

| Group | Step | Settings go to |
|---|---|---|
| `"preprocess"` | Band-pass filter and envelope | `ReSurfEMGAdapter.preprocess` |
| `"ecg_detect_peaks"` | Detection of the R peaks | The step `emg.ecg_detect_peaks` |
| `"ecg_gating"` | Removal of the ECG at each R peak | The step `emg.ecg_gating` |
| `"ecg_removal"` | Turn ECG removal off, or give the R peaks | The preset |
| `"baseline"` | Baseline of the envelope | `ReSurfEMGAdapter.moving_baseline` |
| `"detect_breaths"` | Breath detection | `ReSurfEMGAdapter.detect_breaths` |
| `"postprocess"` | Parameters and quality checks | `ReSurfEMGAdapter.postprocess` |

For example:

```python
session.run_preset(
    "emg",
    config={
        "preprocess": {"channel": 1, "high_pass_hz": 80.0},
        "baseline": {"window_seconds": 7.5, "step_seconds": 0.2},
    },
)
```

See [Workflows](../workflows.md) for the same steps in a {term}`workflow` file.

## Select the channel

To select the EMG {term}`channel`, set `"preprocess": {"channel": n}`. The channel numbers start at 0.

If you do not set `channel`, M3Resp uses the channel names. It never uses a channel with the name ECG or EKG as EMG. If the names do not show the breathing muscle channel, M3Resp raises `UnresolvedChannelError` and asks for `channel`.

## Change the band-pass filter

The default band-pass filter is 20 Hz to 500 Hz. If the {term}`sampling rate` is below 1053 Hz, the upper limit is 95% of the Nyquist frequency (half the sampling rate).

- To change the lower limit, set `"preprocess": {"high_pass_hz": ...}`.
- To change the upper limit, set `"preprocess": {"low_pass_hz": ...}`.

## Remove mains hum or EIT interference

A notch filter removes one frequency and its multiples, for example 50 Hz mains hum. An EIT device that records at the same time can also add a frequency and its multiples. To use the notch filter, set `"preprocess": {"notch_base_frequency": 50.0}`.

| Setting | Default | Meaning |
|---|---|---|
| `notch_base_frequency` | None (no notch filter) | The lowest frequency to remove, in Hz |
| `notch_max_frequency` | None (the Nyquist frequency) | The highest multiple to remove, in Hz |
| `notch_quality_factor` | 30.0 | Width of each notch: a higher value gives a narrower notch |
| `notch_before_bandpass` | False | If True, the notch filter runs before the band-pass filter |

## Change the envelope

| Setting | Default | Meaning |
|---|---|---|
| `envelope_method` | `"rms"` | `"rms"`: root mean square. `"arv"`: average rectified value, the method of `resurfemg`'s `full_rolling_arv`. `"median"`: median of the absolute signal, which ignores short spikes such as parts of the ECG that stay after gating. |
| `envelope_window_seconds` | 0.25 | Length of the envelope window, in seconds |

Put both settings in the `"preprocess"` group. ECG gating calculates the envelope again with the same settings.

## Change the ECG removal

The preset finds the R peaks in the EMG channel. Then it removes a part of the signal around each R peak and fills the gap.

- If you recorded a separate ECG channel, set `"ecg_detect_peaks": {"ecg_channel": n}`.
- If you already have the R peaks (as sample numbers), set `"ecg_removal": {"ecg_peak_indices": [...]}`. The preset then skips the peak detection. Do not also set `"ecg_detect_peaks"`.
- To change the width of the removed part, set `"ecg_gating": {"gate_width_seconds": ...}`. The default is 205 samples.
- To change how M3Resp fills the gap, set `"ecg_gating": {"fill_method": n}`: 0 fills with zeros, 1 interpolates (default), 2 uses the signal before the gap, 3 uses the RMS of the signal.

:::{warning}
`"ecg_removal": {"enabled": False}` turns ECG removal off. Then the ECG stays in the envelope, in the breath detection and in all amplitude parameters. Use it only to look at the data.
:::

To remove the ECG with wavelet denoising, use the step `emg.ecg_wavelet_denoising` in a workflow file. See {ref}`workflow-ecg-removal`.

## Change the baseline

The preset calculates the baseline as the 33rd percentile of the envelope in a moving window. Set the window in the `"baseline"` group:

| Setting | Default | Meaning |
|---|---|---|
| `window_seconds` | 7.5 | Length of the window, in seconds |
| `step_seconds` | 0.2 | Step of the window, in seconds |
| `percentile` | 33.0 | Percentile of the envelope in each window |

## Change the breath detection

Set the breath detection in the `"detect_breaths"` group:

| Setting | Default | Meaning |
|---|---|---|
| `min_breath_width_seconds` | 0.5 | The shortest breath, in seconds |
| `merge_close_peaks_within_width` | False | If True, M3Resp keeps only the higher peak of two peaks that are closer than `min_breath_width_seconds` |

## Call another resurfemg function

The adapter can call 23 functions of `resurfemg.postprocessing` by name. `available_postprocessing()` lists them per module:

```python
print(session.emg_adapter.available_postprocessing()["quality_assessment"])
session.emg_adapter.run_postprocessing_function("quality_assessment", "snr_pseudo", *args, **kwargs)
```
