# How to change the EIT processing settings

`session.preprocess_eit()` filters the {term}`EIT` data, finds the breaths and calculates the results per breath. Give the settings as keyword arguments. The settings go to `EITProcessingAdapter.preprocess`.

## Change the filter

To select the filter, set `filter_mode`:

| `filter_mode` | What it does |
|---|---|
| `"mdn"` (default) | Removes the heart signal with a multiple digital notch filter. The filter needs the respiratory rate and the heart rate, so M3Resp always calculates them for this mode. |
| `"lowpass"` | A Butterworth low-pass filter with the cutoff `lowpass_hz` (default 1.0 Hz). |
| `"bandpass"` | A Butterworth band-pass filter from `highpass_hz` (default 0.05 Hz) to `lowpass_hz` (default 1.0 Hz). |
| `"none"` | No filter. |

`filter_order` sets the order of the low-pass and band-pass filters (default 4, from 1 to 10).

```python
session.preprocess_eit(filter_mode="bandpass", highpass_hz=0.05, lowpass_hz=1.0)
```

## Change the rate detection

M3Resp finds the respiratory rate and the heart rate in the raw EIT data. If the patient is a neonate, set `subject_type="neonate"` (default `"adult"`). To change the length of the analysis window, set `welch_window_seconds` (default 30.0 s).

## Change the breath detection

M3Resp finds the breaths during `preprocess_eit`. To change the shortest breath that M3Resp accepts, set `breath_min_duration_seconds` (default 2/3 s).

`detect_eit_breaths()` then stores these breaths as `BreathEvent` objects in `session.events["eit_breaths"]`.

## Calculate fewer results

Set a result to `False` to skip it:

| Setting | Result |
|---|---|
| `compute_rates` | Respiratory rate and heart rate, in Hz |
| `compute_breath_intervals` | The start and end of each breath |
| `compute_continuous_tiv` | The tidal impedance variation (TIV) of each breath |
| `compute_eeli` | The end-expiratory lung impedance (EELI) of each breath |
| `compute_pixel_tiv` | A TIV map per breath, with one value per pixel |
| `compute_pixel_eeli` | An EELI map per breath, with one value per pixel. Off by default: set it to `True`. |

TIV, EELI, pixel TIV and pixel EELI need the breaths. If you set `compute_breath_intervals=False`, also set the TIV and EELI settings to `False`.

## Keep more than one result

To compare two settings, store each result under a name with `variant`:

```python
session.preprocess_eit(variant="lowpass", filter_mode="lowpass")
session.detect_eit_breaths(variant="lowpass")
```

M3Resp stores the breaths of this result in `session.events["eit_breaths:lowpass"]`.

## Use the same settings with the preset

Put the settings under `"preprocess"` in the preset configuration:

```python
session.run_preset("eit", config={"preprocess": {"filter_mode": "lowpass"}})
```
