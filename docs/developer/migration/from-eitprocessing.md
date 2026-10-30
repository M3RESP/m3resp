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

The steps `eit.detect_rates` and `eit.detect_breaths` run these functions. Their outputs `rate_detector` and `breath_detector` are the settings records. `breath_intervals` is an m3resp `IntervalData` whose `intervals` are the `BreathEvent`s. The TIV, EELI and pixel steps find the breaths with the settings record, or take them from the input `breaths`.

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

## TIV and EELI

`m3resp.eit.processing.tidal_impedance_variation` and `end_expiratory_lung_impedance` are the waveform part of `TIV` and `EELI` of eitprocessing. They take the breaths as a list of `BreathEvent`s, so the breaths can come from `find_breaths`, from the EIT device or from another step:

```python
# eitprocessing
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.parameters.eeli import EELI
from eitprocessing.parameters.tidal_impedance_variation import TIV

detector = BreathDetection()
tiv = TIV(breath_detection=detector).compute_parameter(global_impedance)
eeli = EELI(breath_detection=detector).compute_parameter(global_impedance)

# M3Resp
from m3resp.eit.processing import (
    end_expiratory_lung_impedance,
    find_breaths,
    tidal_impedance_variation,
)

breaths = find_breaths(global_impedance)
tiv = tidal_impedance_variation(global_impedance, breaths)
eeli = end_expiratory_lung_impedance(global_impedance, breaths)
```

With the breaths of `find_breaths`, both give the numbers of eitprocessing 1.8.7. `method` is the `tiv_method` of eitprocessing: `"inspiratory"` (default), `"expiratory"` or `"mean"`. The result is an `IntervalData` with the given breaths and one value per breath, in the unit of the waveform. eitprocessing gives a `SparseData` with no unit; its `time` is the `extremum_time` of each breath for TIV and the `end_time` for EELI.

Each breath time is read at the nearest sample. A breath that lies partly outside the data or in a gap of the time axis, and a TIV breath with no `extremum_time`, gets NaN, with a warning. Data whose category is not an impedance gives a `ValueError`, as `EELI` of eitprocessing.

The steps `eit.continuous_tiv` and `eit.eeli` run these functions and give `continuous_tiv_result` and `eeli_result`. `eit.continuous_tiv` has the setting `tiv_method`. Without the input `breaths`, the steps find the breaths on their waveform with the settings of `eit.detect_breaths`, as eitprocessing does. With `breaths`, they use those breaths; see [Breaths of another modality](../../concepts/events-and-breaths.md#breaths-of-another-modality).

## Pixel breaths, pixel TIV and pixel EELI

`m3resp.eit.processing.find_pixel_breaths` is `PixelBreath.find_pixel_breaths` of eitprocessing. `pixel_tidal_impedance_variation` is the pixel part of `TIV` (`compute_pixel_parameter`). `pixel_end_expiratory_lung_impedance` is new: eitprocessing has no pixel EELI. The functions take the global breaths as a list of `BreathEvent`s, and the breaths of each pixel as a `PixelBreaths`:

```python
# eitprocessing
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.features.pixel_breath import PixelBreath
from eitprocessing.parameters.tidal_impedance_variation import TIV

detector = BreathDetection()
pixel_breaths = PixelBreath(breath_detection=detector).find_pixel_breaths(
    eit_data, global_impedance
)
tiv = TIV(breath_detection=detector).compute_pixel_parameter(
    eit_data, global_impedance, sequence, tiv_timing="pixel"
)

# M3Resp
from m3resp.eit.processing import (
    find_breaths,
    find_pixel_breaths,
    pixel_end_expiratory_lung_impedance,
    pixel_tidal_impedance_variation,
)

breaths = find_breaths(global_impedance)
pixel_breaths = find_pixel_breaths(pixels, global_impedance, breaths)
tiv = pixel_tidal_impedance_variation(pixels, pixel_breaths)
eeli = pixel_end_expiratory_lung_impedance(pixels, pixel_breaths)
```

With the breaths of `find_breaths`, the pixel breaths and the pixel TIV are the numbers of eitprocessing 1.8.7, with the three phase correction modes, both timings and the three TIV methods. The differences:

| eitprocessing | M3Resp |
|---|---|
| `PixelBreath` finds the global breaths with its own `BreathDetection` | `find_pixel_breaths` takes the global breaths as input |
| An `IntervalData` with a (row, column) grid of `Breath` or `None` per global breath | A `PixelBreaths` with three (breath, row, column) arrays of times, NaN for no breath; `pixel(row, column)` gives the breaths of one pixel as `BreathEvent`s |
| A pixel TIV with `tiv_timing="pixel"` finds the pixel breaths again, always in the mode "negative amplitude" | `pixel_tidal_impedance_variation` takes the `PixelBreaths` of any mode |
| A `SparseData` of pixel TIV with no unit; with missing pixel breaths, its `time` values are out of step with the values | An `IntervalData` of one `PixelMap` per global breath, in the unit of the images |
| One warning for each pixel with no cross-correlation peak or with too many breaths left out; the text says "more than half", the limit is 25 % | One warning for each of the two reasons, with the number of pixels and their positions; `metadata["skipped_pixels"]` lists them |
| Fewer than 2 global breaths: `ValueError` | Fewer than 2 global breaths: no pixel breaths |

eitprocessing takes the sign of each pixel from a second breath detection with the default settings, whatever the settings of `PixelBreath`. `find_pixel_breaths` takes the sign from `sign_breaths`, or from `breaths` when `sign_breaths` is None. To get the numbers of eitprocessing with other detection settings, pass the breaths found with the default settings as `sign_breaths`. The steps do this.

The step `eit.pixel_breaths` runs `find_pixel_breaths` and gives `pixel_breaths` (the `PixelBreaths`) and `pixel_breath_timing_result` (the same times as grids). It finds the global breaths with `minimum_duration_seconds`, or takes the EIT breaths of the input `breaths`. `eit.pixel_tiv` and `eit.pixel_eeli` have the settings `tiv_timing` and `eeli_timing` (`continuous` or `pixel`); `eit.pixel_tiv` also has `tiv_method`. With the `pixel` timing, they use the `PixelBreaths` of the input `pixel_breaths` (by default the output of an earlier `eit.pixel_breaths` step), or find the pixel breaths themselves in the mode "negative amplitude", as eitprocessing does. See {doc}`../../concepts/pixel-maps`.

## Pixel maps, masks and lung space

`m3resp.PixelMap` and `m3resp.PixelMask` have the methods of `PixelMap` and `PixelMask` of eitprocessing. M3Resp replaces the 8 subclasses of `PixelMap` (`TIVMap`, `AmplitudeMap`, `ODCLMap`, `DifferenceMap`, `PerfusionMap`, `PendelluftMap`, `SignedPendelluftMap`, `IntegerMap`) with one field, `kind`:

| eitprocessing | M3Resp |
|---|---|
| `TIVMap(values)`, `AmplitudeMap(values)`, ... | `PixelMap(name=..., values=..., kind="tiv")`, `kind="amplitude"`, ... |
| `PixelMap.from_aggregate`, `normalize`, `create_mask_from_threshold`, `convert_to`, `to_boolean_array`, `to_integer_array`, `to_non_nan_array`, `+ - * /` | The same names on `m3resp.PixelMap` |
| `PixelMask * + -`, `is_weighted` | The same on `m3resp.PixelMask`; the result is named `"a and b"`, `"a or b"` or `"a without b"` |
| `PixelMaskCollection.combine`, `apply` | `m3resp.PixelMaskCollection.combine(method, name=...)`, `apply(data)` |
| `get_geometric_mask(region, shape)` | `m3resp.eit.roi.geometric_mask(region, shape)` |
| `TIVLungspace(threshold).apply(eit_data, timing_data=...)` | `m3resp.eit.roi.tiv_lungspace(pixels, breaths, threshold=...)` |
| `AmplitudeLungspace(threshold).apply(...)` | `amplitude_lungspace(pixels, global_impedance, breaths, threshold=...)` |
| `WatershedLungspace(threshold_fraction).apply(...)` | `watershed_lungspace(pixels, global_impedance, breaths, threshold_fraction=...)` |
| `FilterROIBySize(min_region_size, connectivity).apply(mask)` | `filter_mask_by_size(mask, min_region_size=..., connectivity=...)` |

```python
# eitprocessing
from eitprocessing.roi.watershed import WatershedLungspace

captures = {}
mask = WatershedLungspace(threshold_fraction=0.15).apply(
    eit_data, timing_data=global_impedance, captures=captures
)

# M3Resp
from m3resp.eit.processing import BreathDetectionSettings, find_breaths
from m3resp.eit.roi import watershed_lungspace

breaths = find_breaths(global_impedance, BreathDetectionSettings())
captures = {}
mask = watershed_lungspace(pixels, global_impedance, breaths, captures=captures)
```

With the breaths of `find_breaths` and the default settings, the masks of the three test files are the masks of eitprocessing 1.8.7. The differences:

| eitprocessing | M3Resp |
|---|---|
| The lung-space classes find the breaths with their own `BreathDetection` (default settings) | The functions take the global breaths as input |
| Without `timing_data`, the classes use the sum of all pixels | The adapter methods do the same; the steps read `timing_data` (by default `global_impedance`) |
| `WatershedLungspace` runs `TIVLungspace` and `AmplitudeLungspace` in full and uses only their captures | The pixel TIV and the pixel amplitude are computed once; the masks are the same |
| The watershed maxima are found with NaN read as 0, and the watershed regions on an image with NaN pixels, so the region borders near NaN pixels depend on how scikit-image treats NaN | Pixels with a NaN mean amplitude are in no region (number 0) and cannot be a maximum. At the default threshold the final mask of the test files is the same; at a threshold of 0.05, the Timpel file has 2 (raw) or 7 (filtered) pixels fewer. |
| The capture `"included watershed regions"` holds every region, because `np.where` reads the mask object as one true value | It holds the included regions, NaN elsewhere |
| Watershed captures `"mean tiv"`, `"mean amplitude"` | `"mean TIV of shared breaths"`, `"mean amplitude of shared breaths"` |
| Captures of `SparseData`, `TIVMap` and NumPy arrays | `IntervalData` with one `PixelMap` per breath, `PixelMap`, `PixelMask`; peaks as (row, column) arrays |
| The `get_geometric_mask` error for a height that does not divide shows `{height_divisor}` | The message gives the number |
| `PixelMap / 0` gives NaN with no warning; every division gives a NumPy warning | `PixelMap / 0` gives NaN with a warning; no NumPy warning |

The amplitude uses the pixel breaths in the mode "phase shift". M3Resp leaves out a pixel breath whose end of inspiration the lag moves outside the data (see the section above), so on a short recording the amplitude mask can differ by some pixels at a low threshold. `FilterROIBySize` and `filter_mask_by_size` both lose the weights of a weighted mask and raise `RuntimeError` when no group is large enough.

The steps `eit.roi_tiv_lungspace`, `eit.roi_amplitude_lungspace`, `eit.roi_watershed` and `eit.roi_filter_by_size` run these functions. They give `tiv_lungspace_result`, `amplitude_lungspace_result`, `watershed_lungspace_result` and `size_filtered_roi_result` (M3Resp `PixelMask`s), and the captures. They have the optional input `breaths`: breaths of any modality for the TIV mask, EIT breaths for the amplitude and watershed masks. The step `eit.geometric_mask` gives `geometric_mask_result`. In a spec, bind the mask of `eit.roi_filter_by_size` to a `*_result` output:

```yaml
- uses: eit.roi_watershed
- uses: eit.roi_filter_by_size
  in: { mask: watershed_lungspace_result }
```

See {doc}`../../concepts/pixel-maps`.

## Filter and detect breaths

`session.preprocess_eit()` runs rate detection, filter, breath detection, TIV, EELI and pixel TIV in one call, and pixel EELI with `compute_pixel_eeli=True`. All parts use M3Resp code. Each part has settings in `preprocess_eit`:

| `eitprocessing` | Settings of `preprocess_eit` |
|---|---|
| `RateDetection` | `subject_type`, `welch_window_seconds` |
| `MDNFilter` | `filter_mode="mdn"` (default) |
| Butterworth low-pass or band-pass filter | `filter_mode="lowpass"` or `"bandpass"`, `lowpass_hz`, `highpass_hz`, `filter_order` |
| `BreathDetection` | `breath_min_duration_seconds` |
| `TIV` | `compute_continuous_tiv` (method `inspiratory`), `compute_pixel_tiv` (method `inspiratory`, global timing) |
| `EELI` | `compute_eeli` |
| (no pixel EELI) | `compute_pixel_eeli` (off by default) |

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
| TIV, EELI and pixel TIV per breath | `session.interval_data`, with the names `continuous_tivs`, `continuous_eelis` and `pixel_tivs` (and `pixel_eelis`) |
| `IntervalData` (values per interval) | `m3resp.IntervalData`. See [Values per interval or event](../../concepts/events-and-breaths.md#values-per-interval-or-event). |
| `SparseData` (values at single times) | `m3resp.EventData` |

`coerce_intervals` changes `eitprocessing` intervals into M3Resp `Interval` objects. `coerce_events` does the same for single time points.

Continue with {doc}`../../tutorials/10_eit_only` for a full example, or with {doc}`from-resurfemg`.
