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

The steps for breath detection, TIV, EELI, pixel TIV, pixel breaths and the lung space masks accept each of the three waveforms as input. Their input is named `eit_impedance_waveform`.

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

Masks from `eitprocessing`, and masks that M3Resp gives to `eitprocessing`, keep their values, also the zeros.

The lung region steps (`eit.roi_tiv_lungspace`, `eit.roi_amplitude_lungspace`, `eit.roi_watershed`, `eit.roi_filter_by_size`) store their mask in `session.pixel_masks`. `eit.roi_filter_by_size` accepts a `PixelMask` or an `eitprocessing` mask.

## Pixel results per breath

Pixel results with one map per breath are [values per breath](events-and-breaths.md#values-per-interval-or-event) in `session.interval_data`. Each map stays next to its breath:

| Step | Name | One value per breath |
|---|---|---|
| `eit.pixel_tiv` | `pixel_tivs` | A `PixelMap` of TIV |
| `eit.pixel_breaths` | `pixel_breaths` | A grid of (row, column, 3): the start, middle and end time of the breath in each pixel, in seconds. NaN for a pixel without a breath. |

```python
pixel_tiv = session.interval_data.for_name("pixel_tivs")[0]
breath = pixel_tiv.intervals[3]   # the fourth breath
tiv_map = pixel_tiv.values[3]     # its TIV map
```

`pixel_breath_events` gives the breaths of one pixel as `BreathEvent` objects, the same objects as the global breaths:

```python
from m3resp import pixel_breath_events

pixel_breaths = session.interval_data.for_name("pixel_breaths")[0]
breaths = pixel_breath_events(pixel_breaths, row=16, column=12)
durations = [breath.duration for breath in breaths]
```

Each breath has the start and end time of the pixel, and the middle time (the end of inspiration) as `extremum_time`. Its `metadata` has `row`, `column` and `global_breath_id`, the `id` of the global breath. A global breath without a breath in the pixel gives no entry.

## Export

`session.export_summary` writes the masks to `pixel_masks.csv` (one row per mask) and their grids to `pixel_masks.npz`. It writes the pixel maps per breath to `interval_data.csv`, `interval_data_metadata.json` and `interval_data_arrays.npz` (see {doc}`../how-to/export-results`).

Continue with {doc}`slicing`.
