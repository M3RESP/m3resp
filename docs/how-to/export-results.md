# How to export the results of a session

## Write all results to a folder

To write all results of a {term}`session` to files, call `export_summary` with the path of your results folder:

```python
session.export_summary("results")
```

M3Resp makes the folder if it does not exist. Files with the same names in the folder are replaced. M3Resp writes a file only when the session has the results for it.

## Files in the results folder

| File | Contents |
|---|---|
| `summary.json` | The session metadata, the quality flags, `session.parameters`, the processing history and the {term}`start time` of each {term}`recording`. `session.parameters` holds the results of `postprocess_emg` and the settings of synchronization and of each cut to a {term}`time window`. |
| `<name>.csv` | One file for each list of {term}`events <event>` in `session.events`, for example `eit_breaths.csv` and `emg_breaths.csv`. |
| `parameters.csv` | The values in `session.parameters`, one row per value. |
| `session_metadata.json` | The session metadata. |
| `processing_history.json` | The list of processing steps that made the results (see {doc}`../concepts/provenance`). |
| `signals_manifest.csv` | One row per {term}`signal` in `session.signals`, with its name, unit and {term}`sampling rate` (see {doc}`../concepts/signals`). |
| `parameter_results.csv` | One row per {term}`parameter` in `session.parameter_results`. This includes the breath timing parameters between modalities (see {doc}`../concepts/parameters`). |
| `parameter_result_arrays.npz` | The parameters with an array as value, for example the ECG gate mask of the EMG. The row in `parameter_results.csv` gives the key of the array. |
| `interval_data.csv` | One row per breath for each value per breath, for example the TIV and EELI of the EIT: the start time, the end time and the value (see {doc}`../concepts/events-and-breaths`). |
| `interval_data_metadata.json` | The name, unit, method and settings of each value per breath. The `result_index` column in `interval_data.csv` points to its entry. |
| `interval_data_arrays.npz` | The values per breath that are arrays, for example a pixel TIV map per breath. The first axis of each array is the breath. The columns `array_key` and `array_index` in `interval_data.csv` point to the array. |
| `pixel_masks.csv` | One row per pixel mask in `session.pixel_masks`, for example a lung mask (see {doc}`../concepts/pixel-maps`). |
| `pixel_masks.npz` | The pixel grid of each mask, under the `array_key` of its row. NaN marks the pixels outside the mask. |
| `quality_flags.csv` | One row per {term}`quality flag` in `session.quality` (see {doc}`../concepts/quality`). |
| `linked_breaths.csv` | One row per {term}`linked breath` in `session.linked_breaths` (see {doc}`../concepts/synchronization`). |

## Write only some of the files

To select the files, use {func}`m3resp.export.export_session_summary`. Set a file group to `False` to leave it out:

```python
from m3resp.export import export_session_summary

export_session_summary(
    session,
    "results",
    summary_json=True,       # summary.json
    event_csvs=True,         # one CSV file per event list
    parameters_csv=True,     # parameters.csv
    postprocessing=True,     # the postprocess_emg results in summary.json and parameters.csv
    structured_export=False, # all other files in the table above
)
```

## Link the array files to a workflow run

If the session has a `DataModelRecorder` (see {doc}`../concepts/provenance`), give the identifier of the workflow run. M3Resp then links the `.npz` files to that `ProcessingRun`:

```python
session.export_summary("results", processing_run_id=result.processing_run_id)
```

Here `result` is the `WorkflowResult` from {func}`m3resp.run_workflow`.

## Export the data model

`export_summary` does not write the data model of a `DataModelRecorder`. To write it, use {func}`m3resp.export_store`. It writes one JSON file per table, for example `cases.json`, `sessions.json` and `processing_runs.json`:

```python
from m3resp import export_store

export_store(session.datamodel.store, "results/datamodel")
```

`export_store` checks the store first. If a check fails, it writes no file and raises `DataModelValidationError` with the list of problems.

| Call | What it does |
|---|---|
| `export_store(store, path)` | Checks that every record points to records that exist, and that no time window ends before it starts. Then it writes the files. A store recorded from a session passes these checks. |
| `export_store(store, path, require_complete=True)` | Also checks that every record has the details of a finished analysis: units, sampling rate, start time and file checksums. During processing, some of these details are not known yet, so a store recorded from a session usually fails this check. |
| `export_store(store, path, validate=False)` | Writes the files with no checks. |

To get the list of problems with no export, call {func}`m3resp.validate_store`.

## Export from a workflow file

In a {term}`workflow` file, the `outputs:` section writes the same files at the end of the run. See [Workflows](../workflows.md).
