# The session

A {term}`session` (`M3Session`) holds the recordings and all the results of one analysis. Make one session for each set of recordings that were made at the same time on one patient. Each method that you call stores its result in the session, so the next method in your workflow can use it.

## Make a session and load recordings

```python
from m3resp import M3Session

session = M3Session()
session.load_eit("recording.bin", vendor="draeger")
session.load_emg("recording.Poly5")
session.load_ventilator("ventilator.txt", source="ventilator")
```

Each load method stores the loaded {term}`recording` in `session.raw`, under `"eit"`, `"emg"` or `"ventilator"`.

## Process the recordings

Each processing method reads its input from the session and writes its results back to the session:

```python
session.preprocess_eit()
session.detect_eit_breaths()
print(len(session.events["eit_breaths"]))
```

M3Resp reads EIT files, sums global and regional impedance, slices EIT data, finds the breathing rate, the heart rate and the breaths, filters the EIT data, computes the TIV and the EELI of each breath, and finds the breaths, the TIV and the EELI of each pixel with its own code. The lung region calculations use `eitprocessing` through the adapter. EMG calculations use `resurfemg`, and ventilator calculations are part of M3Resp. The results are held in M3Resp objects.

## Where the session stores the results

| Attribute | Filled by | Contains |
|---|---|---|
| `session.signals` | `preprocess_eit`, `preprocess_emg`, `preprocess_ventilator` | {doc}`Signals <signals>` |
| `session.events` | `detect_eit_breaths`, `detect_emg_breaths`, `detect_ventilator_breaths`, `add_events` | Lists of {doc}`breaths and events <events-and-breaths>`, by name |
| `session.parameter_results` | `preprocess_eit`, `postprocess_emg`, `compute_breath_timing_parameters` | {doc}`Parameters <parameters>` |
| `session.interval_data` | `preprocess_eit` and the EIT workflow steps | Values per breath, for example TIV and EELI |
| `session.pixel_masks` | The EIT lung region workflow steps | {doc}`Pixel masks <pixel-maps>` |
| `session.quality` | `preprocess_eit`, `postprocess_emg` | {doc}`Quality flags <quality>` |
| `session.linked_breaths` | `link_breaths` | {doc}`Linked breaths <synchronization>` |
| `session.start_times` | `synchronize_raw_modalities`, the `slice_*` methods | The {term}`start time` of each recording on the shared clock, in seconds |
| `session.sync_methods` | `synchronize_raw_modalities`, `synchronize_multimodal_breaths`, `skip_synchronization` | How each recording was synchronized: `"manual"` or `"none"` |
| `session.provenance` | Each session method | A log of the method calls (see {doc}`provenance`) |
| `session.processing_history` | Each workflow step | The processing steps of a workflow run (see {doc}`provenance`) |
| `session.datamodel` | You, with `session.datamodel = DataModelRecorder(session)` | The stored data model (see {doc}`provenance`) |

To find the results of one {term}`modality`, use `for_modality`:

```python
for signal in session.signals.for_modality("eit"):
    print(signal.name, signal.unit)
```

## The result dictionaries

The session also keeps the results in the format returned by the outside packages and stores them in four dictionaries: `session.raw`, `session.processed`, `session.events` and `session.parameters`. In each dictionary, you can access the specific modality's results by the modality's own key: for example `session.processed["eit"]` or `session.parameters["emg"]`.

The collections in the [table above](session.md#where-the-session-stores-the-results) (`signals`, `parameter_results`, `interval_data`, `pixel_masks`, `quality`, `linked_breaths`) use the same objects for all modalities, making it easy to compare modalities and to export results.

`preprocess_eit` can receive custom preprocessing functions as input: if you pass it a function of your own with `preprocess=`, M3Resp stores its output in `session.processed` instead than in the collections, which will therefore not contain anything for that specific step.

## Ventilator data

The ventilator is a {term}`modality`, as EIT and EMG are:

```python
session.load_ventilator("ventilator.txt", source="ventilator")
session.preprocess_ventilator()      # airway pressure, flow and volume
session.detect_ventilator_breaths()  # session.events["ventilator_breaths"]
```

- `load_ventilator` reads multi-channel files (for example a Biopac `.txt` export) with the EMG loader, and EIT `.bin` files with the EIT loader.
- `preprocess_ventilator` takes the airway pressure, flow and volume channels. It filters them only if you give a cutoff, for example `lowpass_hz=20.0`. The result keeps the original values too.
- Each ventilator {term}`signal` has `modality="ventilator"` and a `category`: `"airway_pressure"`, `"airflow"` or `"volume"` (see {doc}`signals`).
- `detect_ventilator_breaths` finds the breaths in the volume signal.

Use `source="ventilator"` for a ventilator file with its own clock. Ventilator data inside an EIT or EMG file uses the clock of that file (see {doc}`synchronization`).

## All session methods

| Method | What it does |
|---|---|
| `load_eit(path, vendor=...)` | Loads an EIT recording |
| `load_emg(path)` | Loads an EMG recording |
| `load_ventilator(path, source=..., name=...)` | Loads a ventilator recording |
| `slice_eit(start_seconds, end_seconds=None)` | Keeps only a time window of the EIT recording. Run it before `preprocess_eit` (see {doc}`slicing`). |
| `slice_emg(start_seconds, end_seconds=None)` | Keeps only a time window of the EMG recording. Run it before `preprocess_emg`. |
| `slice_ventilator(start_seconds, end_seconds=None)` | Keeps only a time window of each separate ventilator recording. Run it before `preprocess_ventilator`. |
| `preprocess_eit(variant=None, ...)` | Filters the EIT data and calculates rates, TIV and EELI |
| `preprocess_emg(variant=None, ...)` | Filters the EMG channel and calculates the {term}`envelope` |
| `preprocess_ventilator(variant=None, ...)` | Takes the airway pressure, flow and volume channels |
| `detect_eit_breaths(variant=None)` | Stores the EIT breaths in `session.events["eit_breaths"]` |
| `detect_emg_breaths(variant=None, ...)` | Finds the EMG breaths and stores them in `session.events["emg_breaths"]` |
| `detect_ventilator_breaths(variant=None, ...)` | Finds the ventilator breaths and stores them in `session.events["ventilator_breaths"]` |
| `postprocess_emg(...)` | Calculates the EMG parameters and quality flags |
| `add_events(name, events)`, `get_events(name)` | Stores or reads a list of events by name |
| `synchronize_raw_modalities(...)` | Sets the start time of each recording on the shared clock |
| `skip_synchronization()` | Records that the recordings started together |
| `synchronize_multimodal_breaths(...)` | Shifts the detected breaths by an extra offset |
| `link_breaths(time_tolerance=0.5)` | Puts the breaths of different modalities that belong together into linked breaths |
| `compute_breath_timing_parameters(...)` | Calculates the timing between modalities from the linked breaths |
| `run_preset(name, config=...)` | Runs the `"eit"`, `"emg"` or `"multimodal"` {term}`preset` |
| `export_summary(output_dir)` | Writes all results to a folder (see {doc}`../how-to/export-results`) |

Continue with {doc}`signals`, or follow a {doc}`tutorial <../tutorials/index>`.
