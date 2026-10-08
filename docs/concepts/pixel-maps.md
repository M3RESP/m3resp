# Pixel maps and masks

An EIT image is a grid of pixels (rows by columns). This page covers the two
types that hold one number per pixel, and the words used for impedance
waveforms.

## Words: global, regional and pixel impedance waveforms

m3resp uses the words of the consensus on chest EIT terminology
(*Updated consensus terminology and definitions in chest electrical
impedance tomography*, Physiol. Meas. 47(8), 2026,
[doi:10.1088/1361-6579/ae8b55](https://doi.org/10.1088/1361-6579/ae8b55)).

| Word | Meaning (from the consensus) |
|---|---|
| **Global impedance waveform** | Represents the *entire* EIT image plane: the sum or the average of all pixel values over time. Filtered or not, it is still global. |
| **Regional impedance waveform** | The impedance change in a region of interest (ROI) over time. An ROI is any predefined area of the image: a layer, a quadrant, or the ventilated (functional) lung space. |
| **Pixel impedance waveform** | The value of one pixel over time. |

Anything computed inside a mask is **regional**. The regional waveform of the
functional (ventilated) lung space is commonly called the **functional
impedance**, and m3resp uses that name for it.

| Step | Output | Waveform |
|---|---|---|
| `eit.global_impedance` | `global_impedance` | Global: the sum of all pixels |
| `eit.regional_impedance` | `regional_impedance` | Regional: each pixel multiplied by its mask value, then summed |
| `eit.functional_impedance` | `functional_impedance` | Regional, for the functional lung-space mask (by default `tiv_lungspace_result` from `eit.roi_tiv_lungspace`) |

`eit.regional_impedance` and `eit.functional_impedance` do the same
calculation. Both store the waveform as a `Signal` in `session.signals`, with
the output name as its `channel`.

Steps that work on any one-channel impedance waveform, global or regional
(breath detection, TIV, EELI, pixel TIV, pixel breaths and the lung-space
masks), declare their input as `eit_impedance_waveform`, so any of the three
waveforms above can feed them.

## `PixelMap`

One number for each pixel, such as the tidal impedance variation of each pixel
in one breath.

```python
from m3resp import PixelMap

tiv_map = PixelMap(name="pixel_tivs", values=grid, unit="AU")
```

| Field | Meaning |
|---|---|
| `name` | What the numbers are |
| `values` | A 2D grid (row, column), stored as floats. NaN means the pixel has no value. |
| `modality` | The device, `"eit"` by default |
| `category`, `unit` | Physical quantity and unit |
| `method`, `metadata` | Which method made the map, and any extra information |

Anything that is not a 2D grid raises `ValueError`; values that are not
numbers raise `TypeError`. numpy reads a `PixelMap` as its grid, so
`np.nanmean(tiv_map)` works.

## `PixelMask`

Which pixels belong to a region, such as the functional lung space. Each pixel
is NaN (not part of the region), 1 (part of it), or a weight from 0 to 1.

```python
from m3resp import PixelMask

lung = PixelMask(name="tiv_lungspace_mask", values=grid)
lung.included_pixel_count  # pixels that are not NaN, weight 0 included
```

The checks follow eitprocessing's `PixelMask`:

| Input | Result |
|---|---|
| 0 in a grid of numbers | Becomes NaN, with a warning. `keep_zeros=True` keeps it as weight 0 (inside the region, adds nothing to a sum); `suppress_zero_conversion_warning=True` skips the warning. |
| A grid of true/false values | True becomes 1. False becomes NaN without a warning, or 0 with `keep_zeros=True`. |
| A number below 0 or above 1 | `ValueError`, unless `suppress_value_range_error=True`. |
| A grid that is NaN everywhere | A warning, unless `suppress_all_nan_warning=True`. |

```python
weighted = PixelMask(name="ventral_weights", values=grid, keep_zeros=True)
```

Masks read from eitprocessing, and masks passed back to it, keep their values
unchanged, zeros included.

The ROI steps (`eit.roi_tiv_lungspace`, `eit.roi_amplitude_lungspace`,
`eit.roi_watershed`, `eit.roi_filter_by_size`) store their mask in
`session.pixel_masks`. `eit.roi_filter_by_size` accepts a `PixelMask` or the
eitprocessing mask.

## Pixel results per breath

Pixel results that have one map per breath are stored as
[`IntervalData`](events-and-breaths.md#values-per-interval-or-event) in
`session.interval_data`, each map next to its breath:

| Step | Name | One value per breath |
|---|---|---|
| `eit.pixel_tiv` | `pixel_tivs` | a `PixelMap` of TIV |
| `eit.pixel_breaths` | `pixel_breaths` | a (row, column, 3) grid of each pixel's breath start, middle and end time, in seconds; NaN where a pixel has no breath |

```python
pixel_tiv = session.interval_data.for_name("pixel_tivs")[0]
breath = pixel_tiv.intervals[3]   # the fourth breath
tiv_map = pixel_tiv.values[3]     # its TIV map
```

## Export

`session.export_summary(...)` writes the masks to `pixel_masks.csv` (one row
per mask, with its metadata) and their grids to `pixel_masks.npz`. Pixel maps
per breath go to `interval_data.csv` (one row per breath),
`interval_data_metadata.json` and `interval_data_arrays.npz`, see
[../tutorials/export-results.md](../tutorials/export-results.md).
