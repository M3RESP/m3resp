# Cutting data to a time window

M3Resp has several ways to keep only part of your data, for example to
analyse one ventilator setting, or to leave out a stretch where another
device disturbed the signal. This page lists all of them and says which one
to use.

## Recording or signal?

The first question is what you want to cut.

- A **recording** is everything loaded from one file, as it came from the
  device: for an EIT `.bin` file the pixel data, global impedance and every
  channel stored beside it (airway pressure, flow, CO2, pod pressures), plus
  the device's markers. It lives in the session (`session.eit`,
  `session.emg`, `session.ventilators`), and it has its own clock, so each
  recording has a start time in `session.start_times` (see
  [Synchronization](synchronization.md)). Cutting a recording changes what
  every later step reads, so anything recorded in the same file is cut with
  it, and its start time is moved so it stays lined up with the other
  recordings.
- A **signal** is one measured or calculated quantity over time: the global
  impedance, a filtered EIT signal, an EMG envelope. Many signals can come
  from one recording. Cutting a signal makes a shorter copy for the next
  workflow step, for example a detection window, and leaves the recording and
  everything else untouched.

The step names follow this: `*.slice_recording` cuts a recording, and
`*.slice_signal` cuts one signal.

## All the ways to cut

| # | How you call it | What it cuts | Times given as | Keeps recordings lined up? | Done by |
|---|---|---|---|---|---|
| 1 | `session.slice_emg(start, end)`<br>step `emg.slice_recording` | The whole loaded EMG recording, plus ventilator channels from the same file (e.g. airway pressure on a Biopac export) | Seconds from the first sample | Yes: the EMG start time moves later | m3resp |
| 2 | `session.slice_eit(start, end)`<br>step `eit.slice_recording` | The whole loaded EIT recording (pixel data, global impedance, pressure/flow channels, markers), plus ventilator data loaded from the same `.bin` file | Seconds from the first frame | Yes: the EIT start time moves later | eitprocessing (`Sequence.select_by_index`) |
| 3 | `session.slice_ventilator(start, end, name=None)`<br>step `ventilator.slice_recording` | Standalone ventilator recordings (loaded with `source="ventilator"`): the one named, or all of them | Seconds from the first sample | Yes: each cut recording's start time moves later | m3resp |
| 4 | step `eit.slice_signal` | One signal in a workflow: an m3resp `Signal`, or eitprocessing data (raw EIT, global impedance, a `Sequence`). The loaded recording is not changed. | Sample numbers (`mode: index`), or the signal's own time values (`mode: time`) | No | eitprocessing for its own data (`[a:b]`, `.t[a:b]`); m3resp for a `Signal` |
| 5 | `session.load_eit(..., first_frame=, max_frames=)`<br>step `eit.load` | Reads only part of the EIT file in the first place | Frame numbers | Not needed: it happens before anything is lined up | eitprocessing's loader |

In every case the part kept runs from the start time up to, but not
including, the end time. Leave the end out to keep everything up to the end
of the recording.

## Which one cuts ventilator data?

It depends on which file the ventilator data came in, because that decides
its clock.

| Ventilator data came from... | Cut it with |
|---|---|
| the EMG file (e.g. airway pressure on a Biopac export) | #1 `emg.slice_recording`, together with the EMG |
| the EIT `.bin` file | #2 `eit.slice_recording`, together with the EIT |
| its own file (loaded with `source="ventilator"`) | #3 `ventilator.slice_recording` |

`slice_ventilator` refuses ventilator data from the EIT or EMG file: cutting
it on its own would move it out of line with the recording it came with.

Each standalone ventilator recording can have its own start time
(`"ventilator:<name>"` in `session.start_times`), so `slice_ventilator` can
cut one of them without moving the others.

## Things to keep in mind

- **Cut recordings before preprocessing.** `slice_eit` and `slice_ventilator`
  refuse to run once that modality has been preprocessed, since the
  preprocessed results would still cover the whole recording. `slice_emg`
  should also be run before `preprocess_emg`.
- **Cut samples are gone** from the loaded recording. Reload the file to get
  them back.
- **Time mode of `eit.slice_signal` uses the signal's own time values.** For
  data read from an EIT file that is the time of day stored in the file (a
  Draeger file can start at 36528.6 s), not seconds from the start. The
  recording steps (#1-#3) always use seconds from the start.
- **Frames are chosen by their time stamps** in `slice_eit`. Draeger time
  stamps are not perfectly even, so 100 s can hold a few frames more or fewer
  than 100 s times the sampling rate.
- **Signals added when loading keep the full recording.** Cutting a recording
  does not shorten signals already stored in `session.signals`.
- **Old step names still work.** `emg.slice` is now `emg.slice_recording`, and
  `eit.slice` is now `eit.slice_signal`. Specs using the old names run
  unchanged.

## Examples

Line up an EIT recording and a standalone ventilator export, then cut both
to the same stretch of time:

```python
session.load_eit("recording.bin", vendor="draeger")
session.load_ventilator("ventilator.txt", source="ventilator")

# The ventilator was started 40 s before the EIT.
session.synchronize_raw_modalities(
    offset_seconds={"eit": 0.0, "ventilator": -40.0}, reference_modality="eit"
)

session.slice_eit(600.0, 900.0)         # 10 to 15 minutes into the EIT file
session.slice_ventilator(640.0, 940.0)  # the same stretch on the ventilator's clock

# Both cut recordings now start at 600 s on the shared clock.
```

In a workflow:

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

Cut one signal for a detection window, leaving the recording whole:

```yaml
- id: detection_window
  uses: eit.slice_signal
  in: { signal: global_impedance }
  with: { start: 0, end: 1550, mode: index }
  out: { result: detection_signal }
```

## Where the code lives

| What | File |
|---|---|
| Session methods `slice_emg`, `slice_eit`, `slice_ventilator`: what is cut together, start times, provenance record | `src/m3resp/core/session.py` |
| Cutting one recording: turning start/end times into sample numbers, and keeping those samples | `src/m3resp/modalities/emg.py`, `eit.py`, `ventilator.py` (`sample_window` / `frame_window`, `keep_samples` / `keep_frames`), with `TimeWindow` in `modalities/time_window.py` |
| The eitprocessing call that cuts EIT data | `src/m3resp/adapters/eitprocessing_adapter/__init__.py` (`EITProcessingAdapter.slice_sequence`) |
| Cutting one signal | `src/m3resp/processing/slicing.py` (`slice_by_index`, `slice_by_time`) |
| Workflow steps | `src/m3resp/workflows/steps/{emg,eit,ventilator}/slicing.py` |
| Start times | `src/m3resp/synchronization/start_times.py` |
