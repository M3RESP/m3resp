# Provenance and processing history

M3Resp keeps records of how it made each result, so you can check and repeat an analysis. There are three records. M3Resp writes the first two always. You turn on the third when you need it.

| Record | Where | Written by | Use it to |
|---|---|---|---|
| Provenance log | `session.provenance` | Each session method, also outside a workflow | See which methods you called, with which settings |
| Processing history | `session.processing_history` | Each step of a workflow run | See what a workflow did, and repeat a step |
| Data model | `session.datamodel.store` | A `DataModelRecorder` that you attach | Keep a checked, complete record of a finished analysis |

## The provenance log

Each session method adds one `ProvenanceRecord` to `session.provenance`:

```python
session.load_eit("recording.bin", vendor="draeger")
print(session.provenance[0])
# ProvenanceRecord(action='load_eit', modality='eit', parameters={'path': 'recording.bin', 'vendor': 'draeger'}, timestamp='2026-10-15T21:44:02.041920+00:00')
```

| Field | Contents |
|---|---|
| `action` | The name of the method |
| `modality` | The modality, or `None` |
| `parameters` | The settings of the call |
| `timestamp` | The time of the call, in UTC (ISO 8601) |

## The processing history

When a {term}`workflow` runs, each {term}`step` adds one `ProcessingStep` to `session.processing_history`. A session method that you call yourself is only in the provenance log.

| Field | Contents |
|---|---|
| `name` | The name of the step, for example `"eit.pixel_tiv"` |
| `input_keys`, `output_keys` | The names of the results that the step read and wrote |
| `parameters` | The settings of the step |
| `software`, `version` | `"m3resp"` and its version |
| `optional_package_versions` | The installed version of each outside package that the step uses, for example `resurfemg` and `eitprocessing`. `None` if the package is not installed. |
| `timestamp` | The time of the step |
| `status` | `"succeeded"`, `"failed"` or `"cancelled"` |

The name, the settings and the input and output names are enough to repeat a step. The package versions are also necessary: the same step with the same settings can give a different result after an update of an outside package. M3Resp records these versions only here.

## The data model

The data model is a set of checked records, as in a database: `Case`, `RecordingSession`, `SignalStream`, `DataFile`, `ProcessingRun`, `DerivedFeature`, `QualityAnnotation` and more. Attach a recorder to the session to fill it:

```python
from m3resp import DataModelRecorder, export_store, validate_store

session.datamodel = DataModelRecorder(session)
# ... run workflows or session methods ...

print(validate_store(session.datamodel.store))  # the list of problems, empty if none
export_store(session.datamodel.store, "results/datamodel")
```

- A workflow run gives one `ProcessingRun`, and each session method call also gives one. The processing history has one `ProcessingStep` per step in a workflow.
- Each `ProcessingRun` has a `name` and a `kind`: `"workflow"` for a workflow run, `"step"` for a recorded `ProcessingStep`, or `"session_action"` for a session method call. `version` can hold the version of the workflow or the step. `run_time` is a Unix time in seconds. `parameter_file_ids` links the exported array files to the run.
- Each `SignalStream` records how its recording was synchronized: `sync_method` and `time_offset_ms` (see {doc}`synchronization`).
- The quality flags in the results of a workflow run become `QualityAnnotation` records.

## Check the data model

`export_store` checks the store before it writes. If a check fails, it writes nothing and raises `DataModelValidationError` with all problems. With `validate=False`, it writes the store with no checks.

`validate_store(store, require_complete=True)` and `export_store(store, path, require_complete=True)` also check that each record has all details of a finished analysis: units, sampling rate, start time and file checksums. During the processing, some of these details are not known yet. For example, a file gets its checksum only when M3Resp writes it. So a store from a running session usually fails this check. See {doc}`../how-to/export-results` for the export.

Continue with the {doc}`glossary`.
