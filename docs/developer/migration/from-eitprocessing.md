# Move from eitprocessing

This page lists the `eitprocessing` calls and the M3Resp calls that do the same work. See the {doc}`../migration` for the tasks that are the same for all modalities.

## Load a recording

`session.load_eit` reads Draeger, Sentec and Timpel files with M3Resp's readers, using the same `vendor` names as `load_eit_data`:

```python
# eitprocessing
from eitprocessing.datahandling.loading import load_eit_data

sequence = load_eit_data("recording.bin", vendor="draeger")

# M3Resp
session.load_eit("recording.bin", vendor="draeger")
```

`m3resp.io.load_eit(path, vendor=...)` loads a recording without a session.

The recording in `session.eit` holds native data:

| Field | Data |
|---|---|
| `pixel_impedance` | EIT images as a `Signal`, with shape (frames, rows, columns) |
| `signals` | Recorded global impedance and device waveforms, keyed by vendor label |
| `events` | Device marks as `EventData`, including marker values and text |
| `intervals` | Device breaths as `IntervalData`; Timpel breaths are `BreathEvent`s with a turning point |
| `raw_global_impedance` | Recorded global waveform, or the sum of the native images |
| `data` | The `eitprocessing` Sequence used by the remaining processing calls |

The raw EIT Signals identify the vendor in `source`. `eit.load` and `eit.slice_recording` expose the native images as `raw_eit` and the global waveform as `raw_global_impedance`. Their older `*_signal` output names refer to the same objects. Device marks and breaths are available through the recording fields above.

`session.slice_eit(start, end)` cuts native images, waveforms, marks and breaths using seconds from the first image. See {doc}`../../concepts/slicing` for boundary rules. Ventilator waveforms can be read from the native recording or an `EITFileData` with `ventilator_payload_from_sequence` in `m3resp.adapters.ventilator_adapter`.

## Find the rates and the breaths

`m3resp.eit.processing` has the rate detection and the breath detection of `eitprocessing`. They give the same numbers as `RateDetection` and `BreathDetection` of eitprocessing 1.8.7. The settings have the same names and defaults; they are in a settings record, and the calculation is a function:

```python
# eitprocessing
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.features.rate_detection import RateDetection

respiratory_rate, heart_rate = RateDetection("adult").apply(eit_data)
breaths = BreathDetection(minimum_duration=2 / 3).find_breaths(global_impedance)

# M3Resp
from m3resp.eit.processing import (
    BreathDetectionSettings,
    RateDetectionSettings,
    detect_rates,
    find_breaths,
)

respiratory_rate, heart_rate = detect_rates(raw_eit, RateDetectionSettings("adult"))
breaths = find_breaths(global_impedance, BreathDetectionSettings(minimum_duration=2 / 3))
```

`raw_eit` is a `Signal` with the EIT images (frames, rows, columns), and `global_impedance` is a `Signal` with one value per frame. Both rates are in Hz. `find_breaths` returns a list of `BreathEvent`s. The `middle_time` of an eitprocessing `Breath` is the `extremum_time` of a `BreathEvent`.

`detect_rates(..., captures={})` fills the dictionary with the spectra under the keys of eitprocessing (`frequencies`, `normalized_total_power`, `average_normalized_pixel_power`, `diff_total_averaged_power`, `estimated_respiratory_rate`, `estimated_heart_rate`). `m3resp.visualization.plot_rate_detection(settings, **captures)` draws the figure of `RateDetection.plotting.plot`.

The steps `eit.detect_rates` and `eit.detect_breaths` run these functions. Their outputs `rate_detector` and `breath_detector` are the settings records. `breath_intervals` is an m3resp `IntervalData` whose `intervals` are the `BreathEvent`s. The TIV, EELI and pixel steps still run eitprocessing, and they make an eitprocessing `BreathDetection` from the settings record.

## Filter the EIT data

`m3resp.eit.processing.mdn_filter` is the MDN filter of eitprocessing. It gives the same numbers as `MDNFilter` of eitprocessing 1.8.7, with the same settings and defaults. The settings are keyword arguments of the function, in Hz:

```python
# eitprocessing
from eitprocessing.filters.mdn import MDNFilter

filtered = MDNFilter(respiratory_rate=0.25, heart_rate=1.2).apply(eit_data)

# M3Resp
from m3resp.eit.processing import mdn_filter

filtered = mdn_filter(raw_eit, respiratory_rate=0.25, heart_rate=1.2)
```

`raw_eit` is a `Signal` with the EIT images (frames, rows, columns) or one waveform; the result is a new `Signal` with the same shape, unit and time. `noise_frequency_limit` (default 220 per minute), `notch_distance` (default 10 per minute) and `order` (default 10) have the names of eitprocessing. `mdn_frequency_bands(respiratory_rate=..., heart_rate=...)` gives the removed bands without filtering.

`mdn_filter(..., captures={})` fills the dictionary with the keys of eitprocessing: `low_pass_frequency`, `n_harmonics`, `frequency_bands`, and the images before and after the filter as `Signal`s in `unfiltered_data` and `filtered_data`. A pixel with NaN at some samples becomes NaN at every sample, as in eitprocessing; M3Resp also gives a warning that names these pixels.

The Butterworth filters of eitprocessing (`ButterworthFilter`) are `m3resp.processing.filters.butterworth_filter`, a function with the filter type, the cutoff, the sample frequency and the order as arguments. It gives the same numbers. For EIT images, the order is 1 to 10, as in eitprocessing.

The steps `eit.mdn_filter` and `eit.butterworth_filter` run these functions. `eit.mdn_filter` has the settings `noise_frequency_limit_hz`, `notch_distance_hz` and `order`. Both steps give the filtered images as a `Signal` and keep the captures under `filter_captures`. They also add the filtered images to the eitprocessing `Sequence`, for the steps that still run eitprocessing.

## Filter and detect breaths

`session.preprocess_eit()` runs rate detection, filter, breath detection, TIV, EELI and pixel TIV in one call. The rate detection, the filters and the breath detection use M3Resp code; TIV, EELI and pixel TIV use eitprocessing. Each part has settings in `preprocess_eit`:

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
