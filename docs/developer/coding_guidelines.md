# Coding guidelines

This page has the rules for choices that come back often in the code: the return value of a function, the choice of a field, names, and two arguments for one value. Add a rule here when the team agrees on a new one.

See `AGENTS.md` in the source folder for the rules for docstrings and comments.

## Extra outputs go into `captures`

A function always returns the same type. To give extra outputs, for example values for a plot or a check, add an optional `captures` argument. The function fills this dictionary with `capture_value` from `m3resp.processing.filters`:

```python
def detect_peaks(..., captures: dict[str, Any] | None = None) -> np.ndarray:
    ...
    capture_value(captures, "properties", properties)
    return peak_indices
```

A caller who wants the extra outputs gives an empty dictionary and reads it after the call. `m3resp.processing.filters` and `m3resp.processing.peaks` use this rule.

Do not add a True/False argument that changes the return type, for example from a value to a tuple. A caller who forgets to unpack the tuple gets an error much later in the code.

## Choose a field by what the method calculates

Some objects have two fields for similar things, for example `ParameterResult.breath_id` and `breath_ids`. Choose the field when you write the code, from what the method calculates:

- A method that calculates one value per breath writes `breath_id`.
- A method that calculates one value from a group of breaths writes `breath_ids`, also when the group has only one breath.

Do not choose the field from the data while the code runs, for example from the length of a list.

## Names: what the function calculates, and the term of the literature

A shared function in `m3resp.processing` gets a name for what it calculates. A value of one modality keeps the name that the literature uses.

For example, `window_integral` in `m3resp.processing.metrics` calculates the area between a signal and its baseline in each window. The EMG code uses it for the time product (ETP), with the result name `time_product` and the step `emg.time_product`. The ventilator code uses it for the pressure time product of an occlusion, `pocc_time_product`. So the shared function has a general name.

## Two arguments for one value

Some functions accept one value in two forms, for example `gate_width_seconds` and `gate_width_samples`, or `min_peak_width_samples` and `min_peak_width_s`:

- Both arguments have the default `None`.
- If both are `None`, the function uses its default value.
- If both are set, the function raises `ValueError`.

With a default other than `None`, the function cannot tell if the caller set the value. See the step `emg.ecg_gating` and `detect_emg_breath_peaks` in `m3resp.processing.peaks` for examples.

Continue with {doc}`offset-estimation`.
