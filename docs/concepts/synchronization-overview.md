# Synchronization at a glance

A short reference: which synchronization methods exist, how synchronization
works, and what the package assumes about each recording's clock. For the
full explanation see [Synchronization](synchronization.md); for cutting
recordings see [Cutting data to a time window](slicing.md).

Every method below takes an offset **you enter**. The package does not
measure an offset from the data itself.

## Sync methods

| Method | Workflow step | What it does | Works on | Recorded as |
|---|---|---|---|---|
| `synchronize_raw_modalities(offset_seconds=...)` | `sync.raw_modalities` | Gives each **recording** a start time on the shared clock. No samples are removed. | Loaded recordings (EIT, EMG, standalone ventilators), before or after processing | `"manual"` |
| `synchronize_multimodal_breaths(offset_seconds=...)` | — (part of the `"multimodal"` preset) | Moves detected **breath times** by an extra offset, on top of the start times | Breath lists: `eit_breaths`, `emg_breaths`, `ventilator_breaths` | `"manual"` |
| `sync.estimate_offset` then `sync.apply_estimated_offset` | the same two names | Two workflow steps: the first returns the offset you entered, the second passes it to `synchronize_raw_modalities` | Same as `synchronize_raw_modalities` | `"manual"` |
| `skip_synchronization()` | `sync.skip` | No shift: uses the recordings as they are and treats them as having started together | Loaded recordings not yet synchronized | `"none"` |

The "Recorded as" value is stored per recording in `session.sync_methods`.

**Helpers you can call yourself** (these record nothing):

| Helper | What it does |
|---|---|
| `compute_offsets_from_timestamps` | Turns device start timestamps into offsets |
| `align_events_by_modality_offset` | Shifts a list of events by a per-modality offset |
| `resample_signal` | Changes a signal's sampling rate |
| Offset estimators in `tools/visualization_tools/` | Tuned for one dataset (Annemijn's); **not part of the package** |

## How it works

| Step | What happens |
|---|---|
| 1. Load | Each recording keeps its own clock and all of its samples |
| 2. Synchronize | Each recording gets a start time in `session.start_times` and a record in `session.sync_methods`; with a data model recorder attached, its `SignalStream`s get `sync_method` and `time_offset_ms` |
| 3. Process | Each modality's breaths and signals stay **on their own clock** |
| 4. Compare | Start times are added only where modalities meet: `synchronize_multimodal_breaths`, `link_breaths`, `emg.evaluate_event_timing`, and the synchronization plot |
| 5. Check | Comparing recordings where one was never synchronized gives an `UnsynchronizedDataWarning`; the step still runs |

```text
time on the shared clock = own time - own first time + start time
```

## Clocks by data type

| Data | Own time counts from | Which clock it is on | How that is decided |
|---|---|---|---|
| **EIT** (`.bin`) | The file's time of day (e.g. 36528.6 s); the first frame's time is taken off when comparing | Its own (`"eit"`) | — |
| **EMG** | 0 s at the first sample (sample number ÷ sampling rate) | Its own (`"emg"`) | — |
| **Ventilator inside the EMG file** (e.g. airway pressure on a Biopac export) | 0 s | **The EMG's clock**: always in step with it, and cut with it | Any file not ending in `.bin` is assumed to be this |
| **Ventilator inside the EIT `.bin`** | 0 s, one sample per EIT frame | **The EIT's clock**: always in step with it, and cut with it | A `.bin` file is assumed to be this |
| **Standalone ventilator or monitor export** | 0 s | Its own: shares `"ventilator"`, or has its own `"ventilator:<name>"` | **Only** if loaded with `source="ventilator"` |

## Built-in assumptions

| Assumption | What it means for you |
|---|---|
| **Without synchronizing, every start time is 0** | All recordings are taken to have started together. Steps that compare recordings warn, unless you called `skip_synchronization()` |
| **One fixed offset per recording** | No correction for clocks that slowly drift apart over a long recording |
| **The sampling rate is exact** | Times are sample number ÷ sampling rate, except EIT, which uses the time stamps in the file |
| **The reference recording starts at 0** | When none is given, the reference is the ventilator if loaded, otherwise EIT, otherwise EMG. For breath alignment it is the ventilator if there are ventilator breaths, otherwise EIT |
| **A single number means EMG** | `offset_seconds=2.0` sets only the EMG start time |
| **Calling it again replaces everything** | Start times are not added to earlier ones. Reloading a file clears that recording's start time and record |
| **The ventilator file type is guessed from the file name** | A standalone `.txt` export is treated as part of the EMG file unless you pass `source="ventilator"`. Any offset you give it is then ignored without a message (a known open gap) |
| **Breath matching** | Nearest breath within a tolerance (0.5 s by default), one-to-one, closest pair first |

## Where the code lives

| What | File |
|---|---|
| Session methods (`synchronize_raw_modalities`, `synchronize_multimodal_breaths`, `skip_synchronization`, `link_breaths`) | `src/m3resp/core/session.py` |
| Start times and the shared-clock shift | `src/m3resp/synchronization/start_times.py` |
| Sync record and the warning | `src/m3resp/synchronization/sync_methods.py` |
| Offsets, event shifting, and which recording is the reference | `src/m3resp/synchronization/alignment.py` |
| Which clock a ventilator recording is on, and finding a ventilator recording by name | `src/m3resp/modalities/ventilator.py` (`ventilator_clock`, `ventilator_recording`) |
| Copying synchronization onto data model streams | `src/m3resp/datamodel/recorder.py` (`record_synchronization`) |
| Workflow steps (all `sync.*`) | `src/m3resp/workflows/steps/sync.py` |
