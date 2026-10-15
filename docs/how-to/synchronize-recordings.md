# How to synchronize recordings

See {doc}`../concepts/synchronization` for how M3Resp uses {term}`start times <start time>`.

## Give each recording a start time

Call `synchronize_raw_modalities` before the processing. Give the start time of each {term}`recording` in seconds from the start of the reference recording. A negative value means that the recording started earlier than the reference.

```python
session.synchronize_raw_modalities(
    method="manual_offset",
    offset_seconds={"eit": 0.0, "emg": -2.0, "ventilator": 0.0},
    reference_modality="eit",
)
```

- If you do not set `reference_modality`, the reference is the ventilator. Without ventilator data, the reference is the EIT, and without EIT, the EMG.
- A single number for `offset_seconds` is the start time of the EMG.
- A second call replaces the start times. It does not add to them.

M3Resp stores the start times in `session.start_times`. Each recording keeps all its samples and its own clock.

## Synchronize ventilator data

The key `"ventilator"` sets the start time of a ventilator file that you loaded with `source="ventilator"`. Ventilator data inside an EIT or EMG file always has the start time of that file.

If two separate ventilator recordings started at different times, give each one a name when you load it. Then use the key `"ventilator:<name>"`:

```python
session.load_ventilator("monitor.txt", name="monitor", source="ventilator")
session.synchronize_raw_modalities(
    offset_seconds={"eit": 0.0, "ventilator": 0.0, "ventilator:monitor": 12.5},
)
```

## Use recordings that started together

If one trigger started all devices, the recordings need no start times. Call `skip_synchronization` to record this choice:

```python
session.skip_synchronization()
```

M3Resp then stores `"none"` for each recording in `session.sync_methods`. `link_breaths` gives a warning for a recording with no synchronization record, so this call also stops that warning.

## Shift the breaths after detection

If you find a better offset after the breath detection, shift the breaths with `synchronize_multimodal_breaths`. This offset adds to the start times:

```python
session.synchronize_multimodal_breaths(method="manual_offset", offset_seconds={"emg": 0.2})
```

M3Resp stores the shifted breaths in `session.processed["synchronized"]`. `link_breaths` uses these breaths.

## Overview of the three calls

| Call | What it changes | When |
|---|---|---|
| `synchronize_raw_modalities` | It stores the start time of each recording in `session.start_times` | Before the processing |
| `synchronize_multimodal_breaths` | It shifts the detected breaths by an extra offset | After the breath detection, only if the offset changed |
| `link_breaths` | Nothing. It puts the breaths that belong together into a {term}`linked breath` | Last |

The `"multimodal"` {term}`preset` runs `synchronize_raw_modalities` and then `synchronize_multimodal_breaths`. Run it after the `"eit"` and `"emg"` presets.
