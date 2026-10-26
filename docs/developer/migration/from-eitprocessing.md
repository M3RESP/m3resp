# Move from eitprocessing

This page lists the `eitprocessing` calls and the M3Resp calls that do the same work. See the {doc}`../migration` for the tasks that are the same for all modalities.

## Load a recording

`session.load_eit` calls `load_eit_data` of `eitprocessing`, with the same `vendor`:

```python
# eitprocessing
from eitprocessing.datahandling.loading import load_eit_data

sequence = load_eit_data("recording.bin", vendor="draeger")

# M3Resp
session.load_eit("recording.bin", vendor="draeger")
```

`m3resp.io.load_eit(path, vendor=...)` loads a recording without a session.

## Filter and detect breaths

`session.preprocess_eit()` runs the `eitprocessing` steps in one call. Each step has settings in `preprocess_eit`:

| `eitprocessing` | Settings of `preprocess_eit` |
|---|---|
| `RateDetection` | `subject_type`, `welch_window_seconds` |
| `MDNFilter` | `filter_mode="mdn"` (default) |
| Butterworth low-pass or band-pass filter | `filter_mode="lowpass"` or `"bandpass"`, `lowpass_hz`, `highpass_hz`, `filter_order` |
| `BreathDetection` | `breath_min_duration_seconds` |
| `TIV` | `compute_continuous_tiv`, `compute_pixel_tiv` |
| `EELI` | `compute_eeli` |

See {doc}`../../how-to/eit-processing-settings` for the default of each setting.

## Read the results

`session.detect_eit_breaths()` gives the breaths of `BreathDetection` as `BreathEvent` objects. The values per breath are in `session.interval_data`:

```python
session.preprocess_eit()
breaths = session.detect_eit_breaths()

tiv = session.interval_data.for_name("continuous_tivs")[0]
print(len(breaths), len(tiv.values))  # one TIV value per breath
```

| `eitprocessing` result | Place in M3Resp |
|---|---|
| Respiratory rate and heart rate | `session.parameter_results`. See {doc}`../../concepts/parameters`. |
| Breaths of `BreathDetection` | `session.events["eit_breaths"]` |
| TIV, EELI and pixel TIV per breath | `session.interval_data`, with the names `continuous_tivs`, `continuous_eelis` and `pixel_tivs` |
| `IntervalData` (values per interval) | `m3resp.IntervalData`. See [Values per interval or event](../../concepts/events-and-breaths.md#values-per-interval-or-event). |
| `SparseData` (values at single times) | `m3resp.EventData` |

`coerce_intervals` changes `eitprocessing` intervals into M3Resp `Interval` objects. `coerce_events` does the same for single time points.

Continue with {doc}`../../tutorials/10_eit_only` for a full example, or with {doc}`from-resurfemg`.
