# Slicing data to a time window

M3Resp can slice your data to a {term}`time window`, for example to analyse one ventilator setting of a protocol, or to leave out a period with interference from another device. This page lists all ways to slice and says which one to use.

## Slice a recording or a signal

First decide what you want to slice:

- A {term}`recording` is all data from one file: for an EIT `.bin` file, the pixel data, the global impedance, the other channels in the file (airway pressure, flow, CO2, pod pressures) and the markers. A recording has its own clock and a {term}`start time` in `session.start_times` (see {doc}`synchronization`). The slice cannot be undone: the samples outside the time window are removed from the loaded recording, and M3Resp moves the start time, so the recording stays in line with the other recordings. To get the removed samples back, load the file again.
- A {term}`signal` is one quantity over time, for example the global impedance or an EMG envelope. One recording can give many signals. A slice of a signal is a copy of the selected segment, available to the next workflow steps. The signal and the recording do not change, and no samples are removed.

The step names show this: `*.slice_recording` slices a recording, and `*.slice_signal` slices one signal.

## All ways to slice

| # | Call | What it slices | Time window given as | Alignment with other recordings |
|---|---|---|---|---|
| 1 | `session.slice_emg(start, end)`, step `emg.slice_recording` | The EMG recording, and the ventilator channels in the same file (for example the airway pressure in a Biopac export) | Seconds from the first sample | Yes: the EMG start time moves later |
| 2 | `session.slice_eit(start, end)`, step `eit.slice_recording` | The EIT recording (pixel data, global impedance, other channels, markers), and the ventilator data in the same `.bin` file | Seconds from the first image | Yes: the EIT start time moves later |
| 3 | `session.slice_ventilator(start, end, name=None)`, step `ventilator.slice_recording` | The separate ventilator recordings (loaded with `source="ventilator"`): the one with this name, or all | Seconds from the first sample | Yes: the start time of each sliced recording moves later |
| 4 | Step `eit.slice_signal` | One signal in a workflow: an M3Resp `Signal`, or `eitprocessing` data (raw EIT, global impedance, a `Sequence`). The loaded recording does not change. | Sample numbers (`mode: index`), or the time values of the signal (`mode: time`) | No |
| 5 | `session.load_eit(..., first_frame=, max_frames=)`, step `eit.load` | Reads only a part of the EIT file | Image numbers | Not needed: it happens before the synchronization |

- Ways 1 to 4 keep the samples from `start` up to the sample before `end`.
- Ways 1 to 3 keep all samples from `start` to the end of the recording if you leave out `end`. `eit.slice_signal` always needs `end`.
- Way 5 takes the number of the first image and the number of images.
- `slice_eit` cuts the native images, waveforms, device marks and breaths with M3Resp, then rebuilds the `eitprocessing` Sequence used by the remaining processing steps. `eit.slice_signal` cuts a native `Signal` with M3Resp; it also accepts `eitprocessing` objects for existing workflows.

For a recording slice, device marks at the first kept image are included and marks at the final kept image are excluded. Complete device breaths can end at the final kept image. Interval bounds follow eitprocessing 1.8.7: a boundary of exactly 0 s uses the corresponding outer interval boundary. A single-frame slice at time 0 s therefore keeps all device breaths.

In time mode, `eit.slice_signal` also accepts `start_inclusive` and `end_inclusive`, both False by default. An exact start is included with either start flag. When the start falls between samples, `start_inclusive=True` keeps the preceding sample. `end_inclusive=True` keeps the first sample at or after the end time. The helper `m3resp.processing.slicing.slice_by_time` exposes the same flags and permits empty selections with `allow_empty=True`.

## Slice ventilator data

The file of the ventilator data gives its clock, so it also gives the way to slice it:

| Ventilator data from | Slice it with |
|---|---|
| The EMG file, for example the airway pressure in a Biopac export | 1, `emg.slice_recording`, together with the EMG |
| The EIT `.bin` file | 2, `eit.slice_recording`, together with the EIT |
| Its own file, loaded with `source="ventilator"` | 3, `ventilator.slice_recording` |

`slice_ventilator` refuses ventilator data from the EIT or EMG file, because a separate slice would move it out of line with its recording. Each separate ventilator recording can have its own start time (`"ventilator:<name>"`), so `slice_ventilator` can slice one of them and leave the others.

## Rules

- Slice the recordings before the preprocessing. `slice_eit` and `slice_ventilator` refuse to run after the preprocessing of that modality, because the preprocessed results cover the whole recording. `slice_emg` runs, but it slices only the loaded EMG: the filtered EMG and the envelope in `session.signals` keep their full length.
- A slice removes the samples outside the time window from the loaded recording. Load the file again to get them back.
- In a workflow, the steps after a slice use the sliced recording. `eit.slice_recording` gives it under the names of `eit.load` (`raw_eit`, `raw_global_impedance`, `eit_sequence` and the two raw signals). `emg.slice_recording` gives it under the names of `emg.load` (`emg_recording`, `raw_emg_signals`). The raw signals of the sliced recording replace the full ones in `session.signals`.
- `eit.slice_signal` with `mode: time` uses the time values of the signal. An EIT file can have the time of day as its times: a Dräger file can start at 36528.6 s. The `slice_recording` steps always use seconds from the start.
- `slice_eit` selects the images by their times. The times of a device can be a little uneven, so 100 s can hold a slightly different number of images than the image rate × 100 s.
- The older step names still work: `emg.slice` runs `emg.slice_recording`, and `eit.slice` runs `eit.slice_signal`.

## Example: slice two recordings to the same time

Put an EIT recording and a separate ventilator file on the shared clock, then slice both to the same period:

```python
session.load_eit("recording.bin", vendor="draeger")
session.load_ventilator("ventilator.txt", source="ventilator")

# The ventilator started 40 s before the EIT.
session.synchronize_raw_modalities(offset_seconds={"eit": 0.0, "ventilator": -40.0}, reference_modality="eit")

session.slice_eit(600.0, 900.0)         # 10 to 15 minutes into the EIT file
session.slice_ventilator(640.0, 940.0)  # the same period on the clock of the ventilator

# Both sliced recordings now start at 600 s on the shared clock.
```

The same in a {term}`workflow` file:

```yaml
- uses: eit.load
  with: { file_path: "@eit_file", vendor: draeger }
- uses: eit.slice_recording
  with: { start_seconds: 600.0, end_seconds: 900.0 }
- uses: ventilator.load
  with: { file_path: "@ventilator_file", source: ventilator }
- uses: ventilator.slice_recording
  with: { start_seconds: 640.0, end_seconds: 940.0 }
```

## Example: slice one signal

Slice the global impedance for a breath detection window, and keep the recording whole:

```yaml
- id: detection_window
  uses: eit.slice_signal
  in: { signal: global_impedance }
  with: { start: 0, end: 1550, mode: index }
  out: { result: detection_signal }
```

## Where the code is

| What | File |
|---|---|
| The session methods `slice_emg`, `slice_eit` and `slice_ventilator` | `src/m3resp/core/session.py` |
| The slice of one recording | `src/m3resp/modalities/emg.py`, `eit.py`, `ventilator.py` (`sample_window`, `frame_window`, `keep_samples`, `keep_frames`), and `TimeWindow` in `modalities/time_window.py` |
| Native EIT images, waveforms, device marks and breaths | `src/m3resp/eit/processing/slicing.py` (`slice_file_data`) |
| Rebuild the Sequence from sliced native data | `src/m3resp/adapters/eitprocessing_adapter/_slicing.py` |
| The slice of one signal | `src/m3resp/processing/slicing.py` (`slice_by_index`, `slice_by_time`) |
| The workflow steps | `src/m3resp/workflows/steps/emg/slicing.py`, `eit/slicing.py`, `ventilator/slicing.py` |
| Start times | `src/m3resp/synchronization/start_times.py` |

Continue with {doc}`provenance`.
