# Parameters

A {term}`parameter` is a number that M3Resp calculates from the data, for example the respiratory rate, the EMG amplitude or the delay between the EIT and the EMG breath. M3Resp stores each parameter as a `ParameterResult` object in `session.parameter_results`. The value can be one number or an array.

## Look at a parameter

```python
rate = session.parameter_results.for_name("respiratory_rate")[0]
print(rate.value)     # 0.2313914079219103
print(rate.unit)      # Hz
print(rate.modality)  # eit
print(rate.method)    # eitprocessing.RateDetection
```

To find parameters, use `for_name`, `for_modality` or `for_category`:

```python
session.parameter_results.for_modality("emg")
```

## What a parameter holds

| Field | Contents |
|---|---|
| `name` | The name of the parameter, for example `"respiratory_rate"` |
| `value` | One number, or an array |
| `modality` | The device that the parameter comes from |
| `category` | The physical quantity that the parameter is calculated from (see {doc}`signals`) |
| `unit` | The unit, for example `"Hz"` or `"s"` |
| `metric_type` | The standard name of the parameter. M3Resp sets it from `name` for a known parameter, and leaves it `None` for a name of your own. |
| `breath_id`, `breath_ids`, `event_id`, `start_time`, `end_time` | What the value belongs to (see below) |
| `region`, `channel` | The lung region or the channel of the value |
| `method` | The calculation that made the value |
| `metadata` | Other details, as a dictionary |

For example, a parameter with `name="RR"` and one with `name="breath_rate"` both get `metric_type="respiratory_rate"`, so you can group them. A parameter with `name="mean_amplitude"` gets `metric_type=None`.

`is_scalar` is `True` when the value is one number. `to_dict()` gives the content for a JSON file, with arrays as lists.

## What a value belongs to

The value of a parameter can belong to the whole recording or to a part of it. Set the fields that apply. You can combine them, for example one value per breath in a time window. The one exception is `breath_id` together with `breath_ids`: a parameter with both gives a `ValueError`.

| Fields | The value belongs to |
|---|---|
| None of the fields | The whole recording |
| `breath_id` | One breath |
| `breath_ids` | Several breaths, for example a mean over five breaths |
| `event_id` | One {doc}`event <events-and-breaths>`, for example a blood gas sample |
| `start_time` | One time point, in seconds |
| `start_time` and `end_time` | One time window, in seconds. For several windows, make one parameter per window. |

```python
from m3resp import ParameterResult

mean_amplitude = ParameterResult(
    name="mean_amplitude",
    value=3.8,
    modality="emg",
    start_time=30.0,
    end_time=60.0,
)
```

Values with one value per breath, such as the TIV and EELI of the EIT, are stored as [values per breath](events-and-breaths.md#values-per-interval-or-event) in `session.interval_data`. Each value then stays next to its breath.

## Where parameters come from

| Call | Parameters |
|---|---|
| `preprocess_eit` | The respiratory rate and the heart rate of the EIT, in Hz |
| `postprocess_emg` (or the `"emg"` preset) | The EMG parameters, for example the amplitude, the time product and the respiratory rate |
| `compute_breath_timing_parameters` | The delay and the difference in duration between modalities, and the fraction of breaths that both modalities found (see {doc}`synchronization`) |
| Workflow steps | Many steps add parameters, for example the ECG gate mask of `emg.ecg_gating` |

The timing parameters between modalities use the start, extremum and end times of the breaths. Each modality marks these times at its own moment of the breath, so use the delays to check the detection and the synchronization.

`session.export_summary` writes the parameters to `parameter_results.csv` (see {doc}`../how-to/export-results`).

Continue with {doc}`quality`.
