# Tutorial: exporting a session

`session.export_summary(output_dir)` writes both the Stage 1 outputs and, by
default, the Milestone 2.6 structured files.

```python
session.export_summary("results/")
```

## Files written

Always (Stage 1, unless individually disabled - see below):

| File | Contents |
|---|---|
| `summary.json` | Metadata, quality, `session.parameters` (legacy dict), provenance. |
| `<event_list_name>.csv` | One CSV per non-empty entry in `session.events` (e.g. `eit_breaths.csv`, `emg_breaths.csv`). |
| `parameters.csv` | Row-shaped view of the legacy `session.parameters` dict. |

Additionally, by default (Milestone 2.6 structured export, skipped for empty
collections rather than writing empty files):

| File | Contents |
|---|---|
| `session_metadata.json` | `session.metadata`. |
| `signals_manifest.csv` | One row per `Signal` in `session.signals` (see [../concepts/signals.md](../concepts/signals.md)). |
| `parameter_results.csv` | One row per scalar `ParameterResult` in `session.parameter_results` - includes per-modality parameters *and* any `session.compute_multimodal_parameters()` results (see [../concepts/parameters.md](../concepts/parameters.md)). |
| `parameter_result_arrays.npz` | Array-valued `ParameterResult`s (e.g. EMG gate masks), written to a shared archive instead of a CSV cell. |
| `interval_data.csv` | One row per interval (usually a breath) for each `IntervalData` in `session.interval_data`, e.g. EIT TIV and EELI: the breath's start and end time and its value (see [../concepts/events-and-breaths.md](../concepts/events-and-breaths.md#values-per-interval-or-event)). |
| `interval_data_metadata.json` | One entry per `IntervalData` result: its name, unit, method, archive key and metadata (the settings it was made with, the axes of its arrays, ...). Each CSV row's `result_index` points to its entry. |
| `interval_data_arrays.npz` | Values per breath that are arrays, e.g. a pixel TIV map per breath. Each result is one array whose first axis runs over the breaths; the CSV row's `array_key` and `array_index` point to its slice. |
| `pixel_masks.csv` | One row per `PixelMask` in `session.pixel_masks`, e.g. lung-space masks, with its metadata (see [../concepts/pixel-maps.md](../concepts/pixel-maps.md)). |
| `pixel_masks.npz` | The (row, column) grid of each mask, under the row's `array_key`. NaN marks pixels left out. |
| `quality_flags.csv` | One row per `QualityFlag` in `session.quality` (see [../concepts/quality.md](../concepts/quality.md)). |
| `linked_breaths.csv` | One row per `LinkedBreath` in `session.linked_breaths` (see [../concepts/synchronization.md](../concepts/synchronization.md)). |
| `processing_history.json` | `session.provenance` (see [../concepts/provenance.md](../concepts/provenance.md)). |

## Toggles

```python
session.export_summary(
    "results/",
    summary_json=True,
    event_csvs=True,
    parameters_csv=True,
    postprocessing=True,       # include emg_postprocessing in summary.json's "parameters"
    structured_export=True,    # the Milestone 2.6 files above
    processing_run_id=None,    # links the .npz array files to a ProcessingRun
)
```

Pass `structured_export=False` to get only the Stage 1 files. `processing_run_id`
(typically `PipelineResult.processing_run_id`, from a `m3resp.run_pipeline(...)`
call - see [../pipelines.md](../pipelines.md)) links the array files
(`parameter_result_arrays.npz`, `interval_data_arrays.npz`, `pixel_masks.npz`)
to the `ProcessingRun` that produced them, listed in its `parameter_file_ids`,
when a `DataModelRecorder` is attached. Omit it for a manual export with no
associated pipeline run; the files are still written, just not linked.

## Exporting the persisted (Layer 2) data model

If `session.datamodel = DataModelRecorder(session)` was attached (see
[../concepts/provenance.md](../concepts/provenance.md)), export that
separately - it is not part of `export_summary`:

```python
from m3resp import export_store

export_store(session.datamodel.store, "results/datamodel/")
```

`export_store` writes one JSON file per entity table (`cases.json`,
`sessions.json`, `processing_runs.json`, ...).

It checks the store first (`validate_store`): every record must point at
records that exist, and no time window may end before it starts. If a check
fails, **nothing is written** and a `DataModelValidationError` lists every
problem:

| Option | What it does |
|---|---|
| `export_store(store, path)` | Checks that the records link up, then writes. A store recorded from a session passes. |
| `export_store(store, path, require_complete=True)` | Also checks that every record has the details a finished dataset needs (units, sampling rate, start time, file checksums). While data is being processed some of these are not known yet: the clock time at which a recording started is often missing, and a file only gets a checksum once it is written. So a store recorded from a session usually does not pass this check yet. |
| `export_store(store, path, validate=False)` | Writes without checking. |

To see the problems without exporting, call `validate_store(store)`; it
returns them as a list.

## Declarative pipelines

Running a pipeline through `m3resp.run_pipeline(spec, session=...)` (the
YAML/JSON engine, see [../pipelines.md](../pipelines.md)) can also trigger
export automatically via the spec's `outputs:` section - `export.*` steps
and automatic export share one resolved output directory per run.
