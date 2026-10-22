# Synchronization at a glance

This page lists in tables how M3Resp puts recordings on one shared clock. For the explanation with examples, see {doc}`synchronization`. To slice recordings, see {doc}`slicing`.

All methods use an {term}`offset` that you give. M3Resp does not measure the offset from the data.

## Synchronization methods

| Method | Workflow step | What it does | Recorded as |
|---|---|---|---|
| `synchronize_raw_modalities(offset_seconds=...)` | `sync.raw_modalities` | Gives each loaded recording a {term}`start time` on the shared clock. It removes no samples. Use it before or after the processing. | `"manual"` |
| `synchronize_multimodal_breaths(offset_seconds=...)` | Part of the `"multimodal"` preset | Shifts the detected breaths (`eit_breaths`, `emg_breaths`, `ventilator_breaths`) by an extra offset, on top of the start times | `"manual"` |
| `sync.estimate_offset`, then `sync.apply_estimated_offset` | The same two names | The first step gives the offset that you entered. The second step gives it to `synchronize_raw_modalities`. | `"manual"` |
| `skip_synchronization()` | `sync.skip` | Records that the recordings started together. It shifts nothing. | `"none"` |

M3Resp stores the "Recorded as" value of each recording in `session.sync_methods`.

You can also call these functions yourself. They record nothing in the session:

| Function | What it does |
|---|---|
| `compute_offsets_from_timestamps` | Calculates offsets from the start times of the devices |
| `m3resp.synchronization.align_events_by_modality_offset` | Shifts a list of events by an offset per modality |
| `interpolate_to_time` | Puts a signal on the time points of another signal |

## How synchronization works

| Step | What happens |
|---|---|
| 1. Load | Each recording keeps its own clock and all its samples. |
| 2. Synchronize | Each recording gets a start time in `session.start_times` and a record in `session.sync_methods`. With a data model recorder, its `SignalStream` objects get `sync_method` and `time_offset_ms`. |
| 3. Process | The breaths and signals of each modality stay on the clock of their recording. |
| 4. Compare | M3Resp adds the start times only when it compares modalities: in `synchronize_multimodal_breaths`, `link_breaths`, `emg.evaluate_event_timing` and the synchronization plot. |
| 5. Check | If you compare recordings and one of them has no synchronization record, M3Resp gives an `UnsynchronizedDataWarning`. The step still runs. |

```text
time on the shared clock = own time - own first time + start time
```

## The clock of each type of data

| Data | Its own time starts at | Its clock | How M3Resp decides |
|---|---|---|---|
| EIT (`.bin`) | The time of day in the file, for example 36528.6 s. M3Resp subtracts the time of the first image. | Its own (`"eit"`) | |
| EMG | 0 s at the first sample (sample number / sampling rate) | Its own (`"emg"`) | |
| Ventilator data in the EMG file, for example the airway pressure in a Biopac export | 0 s | The clock of the EMG. M3Resp slices it together with the EMG. | Each file that does not end in `.bin` |
| Ventilator data in the EIT `.bin` file | 0 s, one sample per EIT image | The clock of the EIT. M3Resp slices it together with the EIT. | Each `.bin` file |
| A separate ventilator or monitor file | 0 s | Its own: `"ventilator"`, or `"ventilator:<name>"` | Only if you load it with `source="ventilator"` |

## Assumptions

| Assumption | What it means for you |
|---|---|
| Without synchronization, each start time is 0. | M3Resp takes the recordings as started together. The steps that compare recordings give a warning, unless you call `skip_synchronization()`. |
| Each recording has one fixed offset. | M3Resp does not correct clocks that drift apart during a long recording. |
| The sampling rate is exact. | The time of a sample is its sample number / sampling rate. The EIT uses the times in the file. |
| The reference recording starts at 0 s. | If you give no reference, it is the ventilator, else the EIT, else the EMG. For the breaths, it is the ventilator if it has breaths, else the EIT. |
| One number means the EMG. | `offset_seconds=2.0` sets only the start time of the EMG. |
| A new call replaces all start times. | It does not add to the earlier start times. If you load a file again, its start time and its record are removed. |
| The file name gives the type of a ventilator file. | M3Resp takes a separate `.txt` file as a part of the EMG file, unless you give `source="ventilator"`. It then ignores the offset of that file without a message. |
| Breath linking uses the nearest breath. | Two breaths link if their times differ by 0.5 s or less (the default). Each breath links once, the closest pair first. |

## Where the code is

| What | File |
|---|---|
| The session methods | `src/m3resp/core/session.py` |
| Start times and the shift to the shared clock | `src/m3resp/synchronization/start_times.py` |
| The synchronization record and the warning | `src/m3resp/synchronization/sync_methods.py` |
| Offsets, the shift of events and the reference recording | `src/m3resp/synchronization/alignment.py` |
| The clock of a ventilator recording | `src/m3resp/modalities/ventilator.py` (`ventilator_clock`, `ventilator_recording`) |
| Synchronization in the data model | `src/m3resp/datamodel/recorder.py` (`record_synchronization`) |
| The workflow steps `sync.*` | `src/m3resp/workflows/steps/sync.py` |

Continue with {doc}`synchronization`.
