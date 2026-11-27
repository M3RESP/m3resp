# Coding guidelines

This page has the rules for choices that come back often in the code: the return value of a function, named arguments, settings objects, the choice of a field, names, two arguments for one value, checks of argument values, the registration of workflow steps, module docstrings, and function names in the docs. Add a rule here when the team agrees on a new one.

See `AGENTS.md` in the source folder for the rules for docstrings and comments.

## Extra outputs go into `captures`

A function always returns the same type. To give extra outputs, for example values for a plot or a check, add an optional `captures` argument. The function fills this dictionary with `capture_value` from `m3resp.processing.captures`:

```python
def detect_peaks(..., captures: dict[str, Any] | None = None) -> np.ndarray:
    ...
    capture_value(captures, "properties", properties)
    return peak_indices
```

A caller who wants the extra outputs gives an empty dictionary and reads it after the call. `m3resp.processing.filters` and `m3resp.processing.peaks` use this rule.

Do not add a True/False argument that changes the return type, for example from a value to a tuple. A caller who forgets to unpack the tuple gets an error much later in the code.

## Arguments after the data go by name

The first argument of a function is the data it works on, for example the pixel impedance or a waveform. Put `*` after it, so the caller gives every other argument by name:

```python
def amplitude_lungspace(
    pixels: Signal,
    *,
    breaths: Sequence[BreathEvent],
    signal: Signal | None = None,
    threshold: float = 0.15,
) -> PixelMask: ...


mask = amplitude_lungspace(pixels, breaths=breaths, signal=global_impedance)
```

A reader then sees what each value is without the order of the arguments. Two arguments can stay positional when their order is clear from the function name: a start and an end time (`slice_intervals(data, start_time, end_time)`), a row and a column, and `read_eit_file(path, vendor)`.

## Named arguments or a settings object

The settings of a function are named arguments, for example `mdn_filter(pixels, respiratory_rate=0.25, heart_rate=1.2, order=10)`. Values measured on the recording, such as the breathing rate and the heart rate that the MDN filter needs, are always arguments.

A settings object holds the settings when they depend on each other, or when many settings go together to several functions. In `RateDetectionSettings`, `subject_type` (`"adult"` or `"neonate"`) sets the default limits of the breathing rate and the heart rate. `BreathDetectionSettings` holds the seven settings of the breath detection, and the same object goes to `find_breaths(...)`, `preprocess_eit_recording(...)` and the EIT steps that find breaths, the pixel breaths and the lung space.

To use the same settings in several calls, put the named arguments in a dictionary:

```python
mdn_options = {"notch_distance": 0.2, "order": 8}  # notch_distance in Hz
filtered = mdn_filter(pixels, respiratory_rate=rate, heart_rate=heart, **mdn_options)
```

In a workflow file, each step takes its settings as parameters, also when the function behind the step has a settings object.

## Choose a field by what the method calculates

Some objects have two fields for similar things, for example `ParameterResult.breath_id` and `breath_ids`. Choose the field when you write the code, from what the method calculates:

- A method that calculates one value per breath writes `breath_id`.
- A method that calculates one value from a group of breaths writes `breath_ids`, also when the group has only one breath.

Do not choose the field from the data while the code runs, for example from the length of a list.

## Names: what the function calculates, and the term of the literature

A shared function in `m3resp.processing` gets a name for what it calculates. A value of one modality keeps the name that the literature uses.

For example, `window_integral(...)` in `m3resp.processing.metrics` calculates the area between a signal and its baseline in each window. The EMG code uses it for the time product (ETP), with the result name `time_product` and the step `emg.time_product`. The ventilator code uses it for the pressure time product of an occlusion, `pocc_time_product`. So the shared function has a general name.

## Default names of results

A function that makes a new `Signal` has the argument `name`. With `name=None`, the function makes the name from its input. For example, `global_impedance(pixels)` gives `"Global impedance (mdn_filtered)"` for pixel impedance with `metadata["label"]` `"mdn_filtered"`, and `regional_impedance(pixel impedance, mask=ventral)` gives `"ventral_impedance"`. Give `name` only when the result needs another name.

## Two arguments for one value

Some functions accept one value in two forms, for example `gate_width_seconds` and `gate_width_samples`, or `min_peak_width_samples` and `min_peak_width_s`:

- Both arguments have the default `None`.
- If both are `None`, the function uses its default value.
- If both are set, the function raises `ValueError`.

With a default other than `None`, the function cannot tell if the caller set the value. See the step `emg.ecg_gating` and `detect_emg_breath_peaks(...)` in `m3resp.processing.peaks` for examples.

## Checks of argument values

The workflow checks the type and the limits of each step setting before the step runs, so the GUI gives a clear message for a wrong value. A function trusts its type hints. It checks only the values that would otherwise give a wrong result or an unclear error, for example a negative `first_frame` or a `max_frames` below 1 in `read_eit_file(...)`.

## Settings of a workflow step

Write the inputs, outputs and parameters of a step in its `@register_step(...)` decorator, so a reader finds them in one place. Use a constant only for a block that two or more steps share, and name these steps in a comment above it. For example, `_PIXEL_INPUTS` in `m3resp.workflows.steps.eit.pixel` holds the inputs of both `eit.pixel_tiv` and `eit.pixel_eeli`, so the two steps read the same inputs.

## Module docstrings

The API pages show the docstrings of functions and classes, so a user reads what a function does there. A module docstring has one line on what the module holds, and the provenance block when the code comes from another package. Put the steps of a method, the settings and the reference into the docstring of the function. Put longer background, for example the orientation of the EIT image, on a concepts page.

## Function names in the docs

In the docs pages, write a function with `(...)`, for example `global_impedance(...)`, so that a reader sees that it is a function. Classes, attributes and constants have no brackets: `PixelMap`, `PixelMap.values`, `TIV_METHODS`.

Continue with {doc}`offset-estimation`.
