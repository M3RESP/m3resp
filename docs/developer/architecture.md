# Architecture

This page shows how the M3Resp code is organized: the two kinds of data objects, the folders of the code, and the folder for each kind of new code.

## The path of the data

A recording goes through an adapter into the session. The data model is a second, optional record of the same results:

```text
recording file
    |
    v
adapter (EITProcessingAdapter, ReSurfEMGAdapter, VentilatorAdapter)
    |   calls eitprocessing or resurfemg, and changes the results into Layer 1 objects
    v
Layer 1: objects of a running analysis (m3resp.data)
    Signal, ParameterResult, QualityFlag, BreathEvent, IntervalData, LinkedBreath, ...
    |
    v   only with session.datamodel = DataModelRecorder(session)
Layer 2: records of a finished analysis (m3resp.datamodel)
    Case, RecordingSession, SignalStream, ProcessingRun, DerivedFeature, QualityAnnotation, ...
    |
    v
DataModelStore -> validate_store() and export_store()
```

## Layer 1: objects of a running analysis

The objects in `m3resp.data` exist in memory while the code runs. Each session method and each workflow step makes them or reads them. They are small dataclasses, so a step can make an object before all its details are known.

The {doc}`concept guides <../concepts/index>` describe each object, its units and its time axis.

## Layer 2: records of a finished analysis

The records in `m3resp.datamodel` are a checked description of an analysis: which recordings, which processing runs, and which results. A `DataModelStore` keeps them in one table per record type. When you add a record, the store checks that each reference points to an existing record.

- `validate_store(store)` checks the whole store. With `require_complete=True`, it also checks that units, sampling rates, start times and file checksums are filled in.
- `export_store(store, path)` checks the store, then writes one JSON file per table.

Other programs can read the exported files to find which recording, settings and processing run gave a result.

## From Layer 1 to Layer 2

A session makes only Layer 1 objects. To also fill Layer 2, attach a recorder:

```python
from m3resp import DataModelRecorder

session.datamodel = DataModelRecorder(session)
```

The session then sends each provenance record to the recorder. The workflow engine sends the result of each workflow run. The recorder changes them into Layer 2 records. See {doc}`../concepts/provenance`.

## Folders of the code

```text
src/m3resp/
├── core/               M3Session, errors, provenance records, metadata, path helpers
├── data/               Layer 1: Signal, TimeSeries, events and breaths, IntervalData, ParameterResult,
│                           QualityFlag, LinkedBreath, PixelMap, ProcessingStep, units, categories
├── datamodel/          Layer 2: records, DataModelStore, DataModelRecorder, validate_store, export_store
├── adapters/           the connection to eitprocessing and resurfemg (see adapters.md)
│   ├── eitprocessing_adapter/
│   ├── resurfemg_adapter/    split by task: ECG, baseline, quality, defaults
│   └── ventilator_adapter/   reads ventilator data from EMG, EIT and ventilator files
├── processing/         signal tools for all modalities: filters, peaks, windows and envelopes,
│                           intervals, metrics, ECG removal, ventilator waveforms, slicing,
│                           interpolation onto other time points, captures (optional extra results)
├── synchronization/    start times, offsets, breath linking, breath timing parameters
├── modalities/         recording types per modality, loaders, modality names, time windows
├── io.py               load_eit and load_emg without a session
├── presets/            the "eit", "emg" and "multimodal" presets and their registry
├── workflows/          the workflow engine: files, checks, compiler, run, manifest, WorkflowService
│   └── steps/          the registered steps: eit/, emg/, ventilator/, sync.py, metrics.py, export.py
├── export/             export_session_summary and the table rows of the CSV files
├── visualization/      session overview, synchronization and EIT figures
├── synthetic/          synthetic EIT, EMG and ventilator recordings for tests and examples
└── __main__.py         the m3resp command line
```

## Where new code goes

| You add | Put it in |
|---|---|
| A function of `eitprocessing` or `resurfemg` | A method on the adapter, which changes the result into Layer 1 objects with `to_signals`, `to_parameters` or `to_quality_flags`. See {doc}`adapters`. |
| A new calculated value | A `ParameterResult`, or an `IntervalData` for one value per breath. |
| A signal tool that more than one modality uses | `processing/` |
| A step for workflow files | A module in `workflows/steps/`, with `@register_step`. See {doc}`../how-to/add-a-step`. |
| A preset | `presets/`, registered with `register_preset`. See {doc}`preset-contracts`. |
| A synchronization method or a measure across modalities | `synchronization/` |
| A new record type | `datamodel/entities.py`, and its conversion in `datamodel/recorder.py` |

Add a new Layer 1 class only for a result that is not a named value with a unit. For a new method that `eitprocessing` and `resurfemg` do not have, open an issue first.

{doc}`project-history` describes the stages of the project and the planned changes to this structure.

Continue with {doc}`adapters`.
