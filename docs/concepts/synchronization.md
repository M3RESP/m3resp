# Synchronization and breath timing

EIT, EMG and ventilator recordings come from different devices, and each device has its own clock. {term}`Synchronization` puts the recordings on one shared clock, so you can compare their breaths. For a summary in tables, see {doc}`synchronization-overview`. For the steps, see {doc}`../how-to/synchronize-recordings`.

## Start times

`synchronize_raw_modalities` gives each {term}`recording` a {term}`start time`: the time of its first sample on the shared clock, in seconds. The reference recording starts at 0 s. A negative start time means that the recording started earlier than the reference.

```python
session.synchronize_raw_modalities(
    method="manual_offset",
    offset_seconds={"eit": 0.0, "emg": -2.0},
    reference_modality="eit",
)
print(session.start_times)  # {'eit': 0.0, 'emg': -2.0, 'ventilator': 0.0}
```

You give the {term}`offset`. M3Resp does not measure it from the signals. For example, you can find it when you compare filtered signals of both recordings.

M3Resp changes and removes no samples. A pressure recording of three hours stays three hours long, also beside an EIT recording of thirty minutes that started 90 minutes later. A second call replaces the start times.

## Each modality keeps its own clock

The results of each modality stay on the clock of their recording. For example, the times in `session.events["emg_breaths"]` count from the start of the EMG file. M3Resp adds the start times only when it compares modalities: in `synchronize_multimodal_breaths`, in `link_breaths` and in `plot_synchronization_comparison`.

$$t_{shared} = t_{modality} - t_{modality}(0) + t_{start}$$

The first time $t_{modality}(0)$ is 0 s for EMG and ventilator data. An EIT file has the time of day instead, for example 36528.6 s in a Dräger file. So M3Resp subtracts the first EIT time.

## Ventilator recordings

Ventilator data in the EIT or EMG file uses the start time of that file. A separate ventilator file, loaded with `source="ventilator"`, uses the start time `"ventilator"`. If two separate ventilator recordings started at different times, give one of them a name and its own start time:

```python
session.load_ventilator("ventilator.txt", source="ventilator")
session.load_ventilator("monitor.csv", source="ventilator", name="monitor")
session.synchronize_raw_modalities(
    offset_seconds={"eit": 0.0, "ventilator": -30.0, "ventilator:monitor": 12.5},
    reference_modality="eit",
)
```

M3Resp refuses a name that is not loaded, and a name of ventilator data in the EIT or EMG file.

## Cut recordings

When you cut a recording with `slice_eit`, `slice_emg` or `slice_ventilator`, M3Resp moves its start time later by the time that it cut from the start. So the recording stays in line with the other recordings (see {doc}`slicing`).

## Was each recording synchronized?

`session.sync_methods` records how each recording was put on the shared clock:

```python
print(session.sync_methods)  # {'eit': 'manual', 'emg': 'manual'}
```

| Key | Recording |
|---|---|
| `"eit"`, `"emg"` | The EIT and the EMG recording |
| `"ventilator:<name>"` | Each separate ventilator recording, loaded with `source="ventilator"` |

| Value | Set by | Meaning |
|---|---|---|
| `"manual"` | `synchronize_raw_modalities`, `synchronize_multimodal_breaths` | An offset that you gave |
| `"none"` | `skip_synchronization()` | No offset: the recording started together with the others |

`link_breaths` (and so the breath timing parameters) and `emg.evaluate_event_timing` compare recordings. If one of the recordings has no record in `session.sync_methods`, they give an `UnsynchronizedDataWarning` and compare the recordings as if all started at the same time. Data from one clock gives no warning, for example EMG with the airway pressure in the same file.

If one trigger started all devices, call `session.skip_synchronization()` (or the step `sync.skip`). The warnings then stop. The provenance log and the `"synchronization"` part of `summary.json` show this choice, next to the start times.

If you load a file again, M3Resp removes its start time and its record in `session.sync_methods`.

With a data model recorder (`session.datamodel`), each `SignalStream` has the same information: `sync_method` (`"manual"`, `"none"`, or empty) and `time_offset_ms` (the start time, in milliseconds). M3Resp updates them after each session action.

## Shift the breaths after detection

`synchronize_multimodal_breaths` shifts the detected breaths by an extra offset, on top of the start times. Use it when you find a better offset after the breath detection:

```python
session.synchronize_multimodal_breaths(method="manual_offset", offset_seconds={"emg": 0.2})
```

`offset_seconds` is one number (for the EMG) or an offset per modality. M3Resp stores the shifted breaths in `session.processed["synchronized"]`, and `link_breaths` uses them.

## Put two signals on the same time points

Breath linking and the breath timing parameters use times in seconds, so each modality keeps its own sampling rate. To compare two signals sample by sample, use `interpolate_to_time`. It gives a copy of a signal with its values at the time points of another signal:

```python
from m3resp import interpolate_to_time

envelope_on_eit_time = interpolate_to_time(envelope, global_impedance.time)
```

- The two time axes must be on the same clock. The function cannot check this, so with two different clocks the result is wrong and no warning is given. Each signal keeps the clock of its recording (see "Each modality keeps its own clock" above), so put the times of two recordings on the shared clock first.
- Time points before the first sample or after the last sample of the signal get NaN, with a warning.
- `interpolate_to_time` uses linear interpolation and no anti-aliasing filter. So apply a low-pass filter before you lower the sampling rate.

## Linked breaths

A {term}`linked breath` (`LinkedBreath`) is one breath as each modality sees it. `link_breaths` makes the linked breaths:

```python
linked_breaths = session.link_breaths(time_tolerance=0.5)
first = linked_breaths[0]
print(first.modalities)   # for example ['eit', 'emg']
print(first.breaths["eit"].extremum_time)
print(first.confidence)
```

| Field | Contents |
|---|---|
| `breaths` | The breath of each modality, by modality name |
| `time_tolerance` | The largest time difference for a link, in seconds |
| `confidence` | 1 - (largest time difference / `time_tolerance`). `None` for a breath with no partner. |
| `metadata` | Other details, as a dictionary |
| `modalities` | The names of the modalities that have a breath in this link |

- M3Resp compares the {term}`extremum` times of the breaths. Without an extremum, it uses the middle of the breath.
- Each breath links once, and the closest pair links first.
- A breath with no partner within `time_tolerance` gets a linked breath of its own. So M3Resp drops no breath.
- M3Resp does not correct clocks that drift apart.

`m3resp.link_breaths_by_time` does the same for lists of breaths that you give yourself.

(breath-timing-parameters)=
## Breath timing parameters

`compute_breath_timing_parameters` calculates {doc}`parameters <parameters>` from the linked breaths and adds them to `session.parameter_results`:

```python
results = session.compute_breath_timing_parameters()
```

| Parameter | Value | Unit |
|---|---|---|
| `<a>_to_<b>_delay` | For each linked breath: the time of the breath in modality b minus the time in modality a. A positive value means that b comes later. | s |
| `<a>_<b>_duration_difference` | For each linked breath: the duration in modality a minus the duration in modality b | s |
| `<a>_<b>_event_agreement` | Of the linked breaths with a breath of a or b, the fraction with both | fraction (0 to 1) |

- The results have `modality="multimodal"`. The `breath_id` is the position of the linked breath in the list, as a text.
- `anchor` selects the moment of the breath for the delay: `"start"` (default), `"extremum"` or `"end"`.
- By default, M3Resp uses each pair of modalities that has breaths, in alphabetical order. Select pairs with `delay_pairs` and `duration_pairs`. An empty list skips that parameter.
- A delay or a duration difference needs a breath of both modalities in the linked breath.
- Each call replaces all earlier timing results. Other parameters stay. An unknown `anchor` raises `ValueError`, and the earlier results stay.

The three calculations are also available as functions in `m3resp.synchronization`:

| Function | Result |
|---|---|
| `compute_timing_delay(linked, from_modality, to_modality, anchor="start")` | The delay in seconds, or `None` if a breath or its anchor time is missing |
| `compute_breath_duration_difference(linked, modality_a, modality_b)` | The duration of a minus the duration of b in seconds, or `None` if a breath is missing |
| `compute_event_agreement(linked_breaths, modalities)` | The fraction of linked breaths with a breath of each modality, of the linked breaths with at least one of them. 0.0 if there are no such linked breaths. |

Each modality marks its breath times at its own moment of the breath. For example, the start of the default EMG breath is the time of the highest EMG activity. So use the delays to check the detection and the synchronization.

Continue with {doc}`slicing`, or follow the {doc}`multimodal tutorial <../tutorials/30_multimodal>`.
