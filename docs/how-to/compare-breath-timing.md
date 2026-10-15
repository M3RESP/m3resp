# How to compare the breath timing of two modalities

See {ref}`breath-timing-parameters` for the definition of each timing parameter.

Before you start, synchronize the recordings (see {doc}`synchronize-recordings`) and detect the breaths in each {term}`modality`.

## Link the breaths

Call `link_breaths`. Set `time_tolerance` to the largest time difference, in seconds, between the {term}`extremum` times of two breaths that belong together. The default is 0.5 s.

```python
linked_breaths = session.link_breaths(time_tolerance=0.5)
```

The closest breath links. A breath with no partner within `time_tolerance` gets a {term}`linked breath` of its own. M3Resp stores the result in `session.linked_breaths`.

## Calculate the timing parameters

```python
timing_parameters = session.compute_breath_timing_parameters()
```

M3Resp adds the results to `session.parameter_results`. A new call replaces all timing results of the previous call.

For each pair of modalities, M3Resp calculates:

| Parameter | Value | Unit |
|---|---|---|
| `<a>_to_<b>_delay` | Time of the breath in modality b minus time of the breath in modality a, for each linked breath. A positive value means that b comes later. | s |
| `<a>_<b>_duration_difference` | Duration of the breath in modality a minus duration of the breath in modality b, for each linked breath. | s |
| `<a>_<b>_event_agreement` | Of the linked breaths that have a breath of a or b, the fraction that has both. | fraction (0 to 1) |

## Compare another moment of the breath

The delay uses the start time of each breath. To use another moment, set `anchor`:

| `anchor` | Moment |
|---|---|
| `"start"` (default) | The start of the breath |
| `"extremum"` | The extremum: the highest impedance (EIT), the highest muscle activity (EMG) |
| `"end"` | The end of the breath |

```python
session.compute_breath_timing_parameters(anchor="extremum")
```

:::{note}
The default EMG breath detection gives only the time of the highest point of the envelope. The start time and the end time of each EMG breath are equal to that time. So with `anchor="start"`, the EIT to EMG delay compares the start of the EIT breath with the highest EMG activity.
:::

## Select the modality pairs

By default, M3Resp calculates the parameters for each pair of modalities that has breaths, in alphabetical order. To select the pairs, set `delay_pairs` and `duration_pairs`. A pair `(a, b)` gives the delay from a to b. An empty list skips that parameter.

```python
session.compute_breath_timing_parameters(delay_pairs=[("emg", "eit")], duration_pairs=[])
```

This call calculates only the EMG to EIT delay and the EMG and EIT event agreement.

## Print the delays

```python
for parameter in timing_parameters:
    if parameter.name == "eit_to_emg_delay":
        print(parameter.breath_id, parameter.value, parameter.unit)
```
