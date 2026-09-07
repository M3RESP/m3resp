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
| `parameter_result_arrays.npz` | Array-valued `ParameterResult`s (e.g. regional ventilation maps), written to a shared archive instead of a CSV cell. |
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
    processing_run_id=None,    # links parameter_result_arrays.npz to a ProcessingRun
)
```

Pass `structured_export=False` to get only the Stage 1 files. `processing_run_id`
(typically `PipelineResult.processing_run_id`, from a `m3resp.run_pipeline(...)`
call - see [../pipelines.md](../pipelines.md)) links the array archive to the
`ProcessingRun` that produced it when a `DataModelRecorder` is attached; omit
it for a manual export with no associated pipeline run.

## Exporting the persisted (Layer 2) data model

If `session.datamodel = DataModelRecorder(session)` was attached (see
[../concepts/provenance.md](../concepts/provenance.md)), export that
separately - it is not part of `export_summary`:

```python
from m3resp import export_store, validate_store

validate_store(session.datamodel.store)
export_store(session.datamodel.store, "results/datamodel/")
```

`export_store` writes one JSON file per entity table (`cases.json`,
`sessions.json`, `processing_runs.json`, ...).

## Declarative pipelines

Running a pipeline through `m3resp.run_pipeline(spec, session=...)` (the
YAML/JSON engine, see [../pipelines.md](../pipelines.md)) can also trigger
export automatically via the spec's `outputs:` section - `export.*` steps
and automatic export share one resolved output directory per run.
