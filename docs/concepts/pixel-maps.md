# Pixel maps and masks

An EIT image is a grid of pixels, in rows and columns. M3Resp has two objects with one number per pixel: a `PixelMap` holds a value per pixel, and a `PixelMask` selects the pixels of a region. This page also gives the names of the EIT impedance waveforms.

## Global, regional and pixel impedance

M3Resp uses the names of the consensus on chest EIT terms (*Updated consensus terminology and definitions in chest electrical impedance tomography*, Physiol. Meas. 47(8), 2026, [doi:10.1088/1361-6579/ae8b55](https://doi.org/10.1088/1361-6579/ae8b55)):

| Name | Meaning |
|---|---|
| Global impedance waveform | The whole EIT image: the sum or the mean of all pixels over time. A filtered global waveform is still global. |
| Regional impedance waveform | The impedance change in a region of interest (ROI) over time. An ROI is an area of the image, for example a layer, a quadrant or the ventilated lung space. |
| Pixel impedance waveform | The value of one pixel over time |

A waveform inside a mask is regional. The regional waveform of the ventilated (functional) lung space is the functional impedance.

| Step | Result | Waveform |
|---|---|---|
| `eit.global_impedance` | `global_impedance` | Global: the sum of all pixels |
| `eit.regional_impedance` | `regional_impedance` | Regional: each pixel times its mask value, then the sum |
| `eit.functional_impedance` | `functional_impedance` | Regional, for the functional lung space mask (by default `tiv_lungspace_result` from `eit.roi_tiv_lungspace`) |

`eit.regional_impedance` and `eit.functional_impedance` do the same calculation. Both store the waveform as a {term}`signal` in `session.signals`, with the result name as its `channel`.

Each of the three waveforms is a {term}`signal` with the artifact type `eit_impedance_waveform`. The steps for breath detection, TIV, EELI, pixel TIV, pixel breaths and the lung space masks accept each of them as input. The EIT images have the artifact type `eit_pixel_signal`.

## Pixel maps

A `PixelMap` holds one number for each pixel, for example the TIV of each pixel in one breath:

```python
import numpy as np

from m3resp import PixelMap

tiv_map = PixelMap(name="pixel_tivs", values=np.ones((32, 32)), unit="a.u.")
print(np.nanmean(tiv_map))  # 1.0
```

| Field | Contents |
|---|---|
| `name` | What the numbers are |
| `values` | A grid of rows and columns, as decimal numbers. NaN means that the pixel has no value. |
| `modality` | The device: `"eit"` by default |
| `category`, `unit` | The physical quantity and the unit |
| `method`, `metadata` | The calculation and other details |

M3Resp raises `ValueError` for values that are not a grid of rows and columns, and `TypeError` for values that are not numbers. NumPy functions use the grid of a `PixelMap`, as in the example.

## Pixel masks

A `PixelMask` selects the pixels of a region, for example the functional lung space. Each pixel is NaN (outside the region), 1 (inside the region) or a weight from 0 to 1:

```python
from m3resp import PixelMask

lung = PixelMask(name="tiv_lungspace_mask", values=grid)
print(lung.included_pixel_count)  # the pixels that are not NaN
```

M3Resp checks the values as `eitprocessing` does:

| Input | Result |
|---|---|
| 0 in a grid of numbers | Becomes NaN, with a warning. With `keep_zeros=True`, it stays a weight of 0: inside the region, but it adds nothing to a sum. `suppress_zero_conversion_warning=True` stops the warning. |
| A grid of `True` and `False` | `True` becomes 1. `False` becomes NaN, or 0 with `keep_zeros=True`. |
| A number below 0 or above 1 | `ValueError`, unless `suppress_value_range_error=True` |
| A grid with only NaN | A warning, unless `suppress_all_nan_warning=True` |

```python
weighted = PixelMask(name="ventral_weights", values=grid, keep_zeros=True)
```

`lung.apply(data)` multiplies the last two axes of a NumPy array, a `Signal` or a `PixelMap` by the mask, returning the same kind of object. A masked `Signal` or `PixelMap` keeps its other fields. The image size must match the mask size.

`m3resp.eit.processing.regional_impedance(pixel_impedance, lung)` sums the masked images into one value per frame. `functional_impedance` performs the same calculation for a functional lung-space mask. Both return a `Signal` with the time, sampling frequency and device source of the images, in arbitrary units (`a.u.`). NaN pixels are left out of the sum; an image with only NaNs gives 0.

Masks from `eitprocessing`, and masks that M3Resp gives to `eitprocessing`, keep their values, also the zeros.

The lung region steps (`eit.roi_tiv_lungspace`, `eit.roi_amplitude_lungspace`, `eit.roi_watershed`, `eit.roi_filter_by_size`) store their mask in `session.pixel_masks`. `eit.roi_filter_by_size` accepts a `PixelMask` or an `eitprocessing` mask.

## Pixel results per breath

Pixel results with one map per breath are [values per breath](events-and-breaths.md#values-per-interval-or-event) in `session.interval_data`. Each map stays next to its breath:

| Step | Name | One value per breath |
|---|---|---|
| `eit.pixel_tiv` | `pixel_tivs` | A `PixelMap` of TIV |
| `eit.pixel_eeli` | `pixel_eelis` | A `PixelMap` of EELI: the impedance of each pixel at the end of the breath |
| `eit.pixel_breaths` | `pixel_breaths` | A grid of (row, column, 3): the start, end of inspiration and end time of the breath in each pixel, in seconds. NaN for a pixel without a breath. `metadata["landmarks"]` names the three times (`start_time`, `extremum_time`, `end_time`) and `metadata["grid_shape"]` gives the rows and columns. |

```python
pixel_tiv = session.interval_data.for_name("pixel_tivs")[0]
breath = pixel_tiv.intervals[3]   # the fourth breath
tiv_map = pixel_tiv.values[3]     # its TIV map
```

## Pixel breaths

The breath of a pixel can start and end a little earlier or later than the global breath. `m3resp.eit.processing.find_pixel_breaths` finds the breath of each pixel between the ends of inspiration of two neighbouring global breaths, as `PixelBreath` of eitprocessing does. The first and the last global breath never have pixel breaths. The setting `phase_correction_mode` sets what happens to a pixel that is out of phase with the global waveform: `"negative amplitude"` (default) turns a pixel with a negative mean TIV upside down, `"phase shift"` moves the search windows of the pixel by its lag to the global waveform, and `"none"` makes no correction.

The result is a `PixelBreaths`. It holds the global breaths, and three arrays of shape (breath, row, column) in seconds: `start_time`, `extremum_time` (the end of inspiration) and `end_time`. A pixel with no breath has NaN for all three times. `metadata["skipped_pixels"]` lists the pixels that have no breaths, for four reasons: `no_tiv` (the TIV is NaN in every breath), `flat` (the impedance does not change), `no_correlation_peak` and `too_many_breaths_skipped` (more than 25 % of the breaths of the pixel are left out). A pixel breath is left out when one of its two search windows lies partly outside the data, for example when global breaths of a whole recording go with a slice of its EIT data. The search gives a warning for the pixels with no cross-correlation peak, for the pixels with too many breaths left out, and when no pixel has a breath. The waveform must be on the time axis of the images.

```python
from m3resp.eit.processing import find_breaths, find_pixel_breaths

global_breaths = find_breaths(global_impedance)
pixel_breaths = find_pixel_breaths(pixels, global_impedance, global_breaths)
print(pixel_breaths.shape)                # (32, 32): rows and columns
print(pixel_breaths.has_breath.sum())     # number of pixel breaths
breaths = pixel_breaths.pixel(16, 12)     # the breaths of one pixel
```

`pixel(row, column)` gives the breaths of one pixel as `BreathEvent` objects. Each breath has the start and end time of the pixel, and the end of inspiration as `extremum_time`. Its `metadata` has `row`, `column` and `global_breath_id`, the `id` of the global breath. A global breath without a breath in the pixel gives no entry.

The step `eit.pixel_breaths` gives the `PixelBreaths` as `pixel_breaths`, and the same times as the grids of `pixel_breath_timing_result` in `session.interval_data`. `PixelBreaths.from_interval_data` and `to_interval_data` convert between the two. `pixel_breath_events` gives the breaths of one pixel of the grids:

```python
from m3resp import pixel_breath_events

pixel_breaths = session.interval_data.for_name("pixel_breaths")[0]
breaths = pixel_breath_events(pixel_breaths, row=16, column=12)
durations = [breath.duration for breath in breaths]
```

## Pixel TIV and pixel EELI

`m3resp.eit.processing.pixel_tidal_impedance_variation` and `pixel_end_expiratory_lung_impedance` give one `PixelMap` per global breath. They take the breaths in one of two forms:

| Breaths | Timing | The value of each pixel |
|---|---|---|
| A list of `BreathEvent`s | Global (`"continuous"`) | At the start, end of inspiration and end of the global breath, the same times for every pixel |
| A `PixelBreaths` | Pixel (`"pixel"`) | At the times of the breath of the pixel. NaN for a pixel with no breath. |

```python
from m3resp.eit.processing import (
    pixel_end_expiratory_lung_impedance,
    pixel_tidal_impedance_variation,
)

tiv_global_timing = pixel_tidal_impedance_variation(pixels, global_breaths)
tiv_pixel_timing = pixel_tidal_impedance_variation(pixels, pixel_breaths)
eeli = pixel_end_expiratory_lung_impedance(pixels, global_breaths)
```

With the global timing, the sum of the pixel values of a breath (NaN pixels left out) is the value of the global impedance waveform. The steps `eit.pixel_tiv` and `eit.pixel_eeli` choose the timing with `tiv_timing` and `eeli_timing`. The global timing does not use the input `pixel_breaths`. Without an `eit.pixel_breaths` step, the pixel timing finds the pixel breaths itself in the mode `"negative amplitude"`. With such a step, the input `pixel_breaths` reads its output `pixel_breaths`:

```yaml
- uses: eit.pixel_breaths
  with: { phase_correction_mode: "phase shift" }
- uses: eit.pixel_tiv
  with: { tiv_timing: pixel }
```

The input `pixel_breaths` takes only a `PixelBreaths`, so the workflow check refuses `pixel_breath_timing_result` there.

## Export

`session.export_summary` writes the masks to `pixel_masks.csv` (one row per mask) and their grids to `pixel_masks.npz`. It writes the pixel maps per breath to `interval_data.csv`, `interval_data_metadata.json` and `interval_data_arrays.npz` (see {doc}`../how-to/export-results`).

Continue with {doc}`slicing`.
