# Pixel maps and masks

An EIT image is a grid of pixels, in rows and columns. M3Resp has two objects with one number per pixel: a `PixelMap` holds a value per pixel, and a `PixelMask` selects the pixels of a region. This page also gives the names of the EIT impedance waveforms, and shows how to draw the maps and masks.

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
| `kind` | What the numbers are: `"tiv"`, `"amplitude"`, `"odcl"`, `"difference"`, `"perfusion"`, `"pendelluft"`, `"signed_pendelluft"` or `"integer"`, or None for a plain map |

M3Resp raises `ValueError` for values that are not a grid of rows and columns, and `TypeError` for values that are not numbers. NumPy functions use the grid of a `PixelMap`, as in the example.

The `kind` follows the types of pixel map in `eitprocessing`. A map of kind `"amplitude"`, `"perfusion"` or `"pendelluft"` with a negative value gives a warning, unless `suppress_negative_warning=True`. A map of kind `"integer"`, for example the region numbers of the watershed method, holds whole numbers only; NaN gives `TypeError`.

A `PixelMap` has these methods:

| Method | Result |
|---|---|
| `PixelMap.from_aggregate(maps, np.nanmean)` | One map from several maps of the same shape, for example the mean TIV of each pixel over the breaths. A pixel that is NaN in every map stays NaN. |
| `normalize(mode=...)` | A copy divided by a reference. `"zero-based"` (default): the values go from 0 to 1. `"symmetric"`: divided by the largest absolute value. `"maximum"`: divided by the largest value. `"reference"`: divided by `reference`. |
| `create_mask_from_threshold(threshold)` | A `PixelMask` with the pixels whose value is at least `threshold`. With `fraction_of_max=True`, the threshold is that fraction of the largest value. `comparator` (for example `np.less`) and `use_magnitude` change the comparison. |
| `+`, `-`, `*`, `/` | A new map, pixel by pixel, with a number, a grid or another map. A NaN pixel stays NaN. A division by 0 gives NaN and a warning. A map minus a map has kind `"difference"`. |
| `convert_to(kind)` | A copy with another kind |
| `to_non_nan_array(nan=0.0)`, `to_integer_array()`, `to_boolean_array()` | A NumPy grid with a number for NaN pixels; whole numbers with 0 for NaN; `True` for the pixels with a value other than 0 and other than NaN |

```python
mean_tiv = PixelMap.from_aggregate(pixel_tiv.values, np.nanmean, name="mean TIV")
large = mean_tiv.create_mask_from_threshold(0.15, fraction_of_max=True)
```

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

Two masks of the same shape combine pixel by pixel. The name of the result comes from the two names:

| Rule | Pixels of the result | Weights | Name |
|---|---|---|---|
| `a * b` | In both masks | Multiplied | `"a and b"` |
| `a + b` | In one mask or in both | Added, up to 1 | `"a or b"` |
| `a - b` | In `a` and not in `b` | The weight of `b` subtracted, down to 0; a pixel at 0 becomes NaN | `"a without b"` |

`is_weighted` is `True` when a pixel holds a number other than NaN or 1. `PixelMaskCollection.combine("sum", name=...)` adds all masks of a collection and `combine("product", name=...)` multiplies them. `PixelMaskCollection.apply(data)` applies each mask to the same data and gives a dictionary with the mask names as keys.

`lung.apply(data)` multiplies the last two axes of a NumPy array, a `Signal` or a `PixelMap` by the mask and returns the same kind of object. A masked `Signal` or `PixelMap` keeps its other fields. The image size must match the mask size.

`m3resp.eit.processing.regional_impedance(pixel_impedance, lung)` sums the masked images into one value per frame. `functional_impedance` performs the same calculation for a functional lung-space mask. Both return a `Signal` with the time, sampling frequency and device source of the images, in arbitrary units (`a.u.`). The sum leaves out NaN pixels; an image with only NaNs gives 0.

Masks from `eitprocessing`, and masks that M3Resp gives to `eitprocessing`, keep their values, including the zeros.

## Geometric masks

`m3resp.eit.roi.geometric_mask(region, shape=(32, 32))` gives the mask of a fixed part of the image, with 1 inside and NaN outside. The rows go from ventral (top of the image) to dorsal (bottom). The anatomical right of the patient is the left side of the image.

| Region | Short name | Pixels |
|---|---|---|
| `"ventral"`, `"dorsal"` | `V`, `D` | The top or bottom half of the rows |
| `"anatomical right"`, `"anatomical left"` | `R`, `L` | The left or right half of the columns |
| `"layer 1"` to `"layer 4"` | `L1` to `L4` | A quarter of the rows, from ventral to dorsal |
| `"quadrant 1"` to `"quadrant 4"` | `Q1` to `Q4` | Ventral right, ventral left, dorsal right, dorsal left |

The number of rows must divide by 2 (halves) or 4 (layers), and the number of columns by 2; other shapes give `ValueError`. The masks follow the set rules above: `ventral * anatomical right` is quadrant 1, and `layer 1 + layer 2` is the ventral half.

```python
from m3resp.eit.processing import regional_impedance
from m3resp.eit.roi import geometric_mask

ventral = geometric_mask("ventral")
ventral_impedance = regional_impedance(pixels, ventral)
```

The step `eit.geometric_mask` makes one mask (parameter `region`) with the rows and columns of its input `eit_data`. Four steps give the four layers or quadrants.

## Functional lung space

The functional lung space is the part of the image that takes part in ventilation. Three functions of `m3resp.eit.roi` find it from the EIT images and the global breaths, as the lung-space methods of `eitprocessing` 1.8.7 do. Each gives a `PixelMask` with 1 for the pixels of the lung space and NaN elsewhere.

| Function | Step | A pixel is in the lung space when |
|---|---|---|
| `tiv_lungspace(pixels, breaths, threshold=0.15)` | `eit.roi_tiv_lungspace` | Its mean TIV over the breaths is at least 15 % of the largest mean TIV. The TIV uses the times of the global breaths for every pixel. |
| `amplitude_lungspace(pixels, signal, breaths, threshold=0.15)` | `eit.roi_amplitude_lungspace` | Its mean amplitude is at least 15 % of the largest mean amplitude. The amplitude is the TIV between the pixel's own breath start and end of inspiration, found in the mode `"phase shift"`, so it is positive for a pixel that is out of phase. The first and the last breath have no amplitude. |
| `watershed_lungspace(pixels, signal, breaths, threshold_fraction=0.15)` | `eit.roi_watershed` | It lies in a watershed region whose largest amplitude lies inside the TIV mask, and its mean amplitude reaches the TIV threshold. See below. |

Warning: the amplitude lung space can include reconstruction artefacts, so do not use it alone. The watershed method uses it together with the TIV.

The watershed method works in four parts:

1. It computes the TIV and the amplitude of each pixel in each breath, and keeps the breaths that have both. In practice, the first and the last breath drop out.
2. Over these breaths, it computes the mean TIV and the mean amplitude of each pixel. The TIV mask holds the pixels with a mean TIV of at least `threshold_fraction` times the largest mean TIV. The amplitude mask holds the pixels with a mean amplitude of at least that same TIV value.
3. The local maxima of the mean amplitude are the centres of the watershed regions. Each pixel belongs to the region of the maximum that it reaches by moving to higher values. Pixels with a NaN mean amplitude belong to no region.
4. The lung space is the union of the regions whose maximum lies inside the TIV mask, multiplied by the amplitude mask.

A pixel with a low TIV because of pendelluft (air that moves between lung regions during the breath) can still be in the watershed lung space, because its amplitude is large and its region has a maximum in the TIV mask.

There are two TIV masks. `tiv_lungspace` and `eit.roi_tiv_lungspace` use all breaths. The TIV mask inside the watershed method (the capture `"functional tiv mask"`) uses the breaths with both TIV and amplitude, so without the first and the last breath. For the same data and threshold, the two masks can differ by some pixels.

The breaths are an input of the functions. The steps take them from the optional input `breaths`; without it, they find the breaths on `timing_data` with the default `BreathDetectionSettings`, as `eitprocessing` does. `eit.roi_tiv_lungspace` takes breaths of any modality, moved to the EIT clock with the start times of the session. `eit.roi_amplitude_lungspace` and `eit.roi_watershed` take EIT breaths only, because they find the breaths of each pixel from them.

```python
from m3resp.eit.processing import BreathDetectionSettings, find_breaths
from m3resp.eit.roi import watershed_lungspace

breaths = find_breaths(global_impedance, BreathDetectionSettings())
captures = {}
lung = watershed_lungspace(pixels, global_impedance, breaths, captures=captures)
print(lung.included_pixel_count, lung.metadata["breath_count"])
```

With a `captures` dictionary, the functions store their intermediate results. The steps give these dictionaries as `tiv_lungspace_captures`, `amplitude_lungspace_captures` and `watershed_captures`:

| Function | Captures |
|---|---|
| `tiv_lungspace` | `"TIV per breath"` (an `IntervalData` with one `PixelMap` per breath), `"mean TIV"` (`PixelMap`) |
| `amplitude_lungspace` | `"amplitude per breath"`, `"mean amplitude"` |
| `watershed_lungspace` | `"mean TIV of shared breaths"` and `"mean amplitude of shared breaths"` (`PixelMap`); `"functional tiv mask"`, `"functional amplitude mask"` and `"included region"` (`PixelMask`); `"local peaks"`, `"included peaks"` and `"excluded peaks"` (row and column of each local maximum); `"included marker indices"` (the numbers of the maxima inside the TIV mask); `"watershed regions"` (a `PixelMap` of kind `"integer"` with the region number of each pixel, 0 for a pixel with a NaN mean amplitude); `"included watershed regions"` (the numbers of the included regions, NaN elsewhere) |

`filter_mask_by_size(mask, min_region_size=10, connectivity=1)` keeps the groups of connected pixels of a mask with at least `min_region_size` pixels. With `connectivity=1`, pixels that share an edge are connected; with `2`, also pixels that share a corner. The result holds 1 for the kept pixels, so the weights of a weighted mask are lost. When no group is large enough, it raises `RuntimeError`. The step `eit.roi_filter_by_size` takes a `PixelMask` (for example `watershed_lungspace_result`) or a 2D grid as `mask`; the workflow check refuses other artifact types there.

All lung-space steps store their mask in `session.pixel_masks`. The `metadata` of each mask holds the settings, `breath_count`, `included_pixel_count` and `included_pixel_fraction`.

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

## The minute image

`m3resp.eit.processing.minute_image(pixels, breaths, start=..., end=...)` gives the mean pixel TIV over the breaths that lie fully inside a time window, in seconds. A breath is inside when it starts at or after `start` and ends at or before `end`. The TIV uses the times of the global breaths for every pixel. The result is a `PixelMap` of kind `"tiv"`; its `metadata` holds `start`, `end`, `breath_count` and `tiv_method`. Without `start` and `end`, the window is the whole recording. When no breath lies inside the window, the function raises `ValueError`.

```python
from m3resp.eit.processing import minute_image

image = minute_image(pixels, global_breaths, start=60.0, end=120.0)
print(image.metadata["breath_count"])
```

## Draw a pixel map

`m3resp.visualization.pixel_maps.plot_pixel_map(item, ax)` draws a `PixelMap` or a `PixelMask` with Matplotlib, with a colour bar, and returns the Matplotlib image. It gives the picture of `PixelMap.plotting.imshow` of `eitprocessing` 1.8.7. Each map kind has its own style: the colours, the two ends of the colour scale and the label of the colour bar.

| Kind | Colours | Ends of the colour scale |
|---|---|---|
| None (plain map) | `viridis` | The smallest and the largest value |
| `"tiv"`, `"amplitude"` | Dark blue (0) to white (largest value); values below 0 are purple | 0 and the largest value |
| `"odcl"` | White, black, dark orange | −1 and 1; the colour bar shows the absolute value in % |
| `"difference"` | `vanimo` | Centred on 0, out to the value farthest from 0 |
| `"perfusion"` | Black to red | 0 and the largest value |
| `"pendelluft"` | Black to forest green | 0 and the largest value |
| `"signed_pendelluft"` | Deep pink, black, forest green | Centred on 0; the colour bar shows the absolute value |
| `"integer"` | `tab20`: one colour for each whole number from 0 to 19 | −0.5 and 19.5 |
| A `PixelMask` | One colour (`tab:blue`), no colour bar | 0 and 2 |

A NaN pixel shows the dark grey background. For `"tiv"` and `"amplitude"`, the colour bar ends in a point at the low end, for the purple values below 0.

```python
import matplotlib.pyplot as plt

from m3resp.visualization.pixel_maps import add_region_labels, plot_pixel_map

fig, ax = plt.subplots()
plot_pixel_map(image, ax)  # the minute image, kind "tiv"
```

The arguments `colorbar`, `normalize`, `percentage`, `absolute`, `hide_axes` and `background` change one setting of the style for this figure. With `normalize=True`, the values are divided by the largest value, or by 1 when the largest value is below 1, as in `eitprocessing`. `style=pixel_map_style("tiv", colorbar_label="TIV (mL)")` gives a changed style. Other arguments, such as `vmin`, `vmax` and `cmap`, go to Matplotlib `imshow`.

`add_region_labels(regions, ax)` writes the number of each region at a point inside that region, in a white box. `regions` is a map of kind `"integer"`, for example the capture `"watershed regions"`. `labels` gives other texts for some region numbers.

`pixel_map_style(kind)` in `m3resp.visualization.pixel_map_style` gives the style of a kind. `scale_limits(style, values)` gives the two ends of the colour scale, and `pixel_colors(style, values)` the colour of each pixel, as red, green, blue and alpha from 0 to 1. These three functions work without Matplotlib. The colours are stored in the package, so the figures and the GUI use the same colours.

## Plot data for the GUI

The GUI draws the EIT results in the browser. The functions of `m3resp.visualization.plot_data` give these results as dictionaries of numbers, text, lists, True/False and None, so `json.dumps` writes them as they are. A grid is a list of rows, with None for a NaN pixel. Each picture comes with its style and the two ends of its colour scale.

| Function | Plot data |
|---|---|
| `pixel_map_plot_data(pixel_map)` | A `PixelMap`: name, kind, unit, shape, values and style |
| `pixel_mask_plot_data(mask)` | A `PixelMask`: values (1 or a weight, None outside), number and fraction of included pixels |
| `eit_image_plot_data(pixels, time)` | The EIT image at the sample nearest to `time`, with the time and the number of that sample |
| `eit_images_plot_data(pixels, start, end, step)` | One image per `step` seconds from `start` to `end`, with one colour scale for all images |
| `lungspace_plot_data(captures, mask=...)` | The captures of a lung-space method: mean maps, masks, watershed regions with the point of each region label, and peaks |
| `rate_spectrum_plot_data(captures, settings)` | The spectra of `detect_rates`, in Hz, and the breathing rate and heart rate |

A time outside the time axis of the images gives `ValueError`. An infinite value gives `ValueError`, because JSON has no infinite numbers.

```python
import json

from m3resp.visualization.plot_data import eit_image_plot_data, pixel_map_plot_data

frame = eit_image_plot_data(pixels, time=12.3)
print(frame["time"], frame["style"]["limits"])
text = json.dumps(pixel_map_plot_data(image), allow_nan=False)
```

## Export

`session.export_summary` writes the masks to `pixel_masks.csv` (one row per mask) and their grids to `pixel_masks.npz`. It writes the pixel maps per breath to `interval_data.csv`, `interval_data_metadata.json` and `interval_data_arrays.npz` (see {doc}`../how-to/export-results`).

Continue with {doc}`slicing`.
