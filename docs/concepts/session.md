# `M3Session`

`M3Session` (`src/m3resp/core/session.py`) is the central object: it holds
both the Stage 1 dict-based state (`raw`, `processed`, `events`,
`parameters` - unchanged since Stage 1) and the Stage 2 typed collections
added on top of it.

| Attribute | Populated by | Contains |
|---|---|---|
| `session.signals` | `preprocess_eit`/`preprocess_emg` (default adapter path) | [`Signal`](signals.md) |
| `session.events` | `detect_eit_breaths`/`detect_emg_breaths`/`add_events` | [`BreathEvent`/`Event`](events-and-breaths.md) lists, keyed by name |
| `session.parameter_results` | `preprocess_eit`/`postprocess_emg`/`compute_multimodal_parameters` | [`ParameterResult`](parameters.md) |
| `session.quality` | `preprocess_eit`/`postprocess_emg` | [`QualityFlag`](quality.md) |
| `session.linked_breaths` | `session.link_breaths()` | [`LinkedBreath`](synchronization.md) |
| `session.provenance` | every instrumented session method | [`ProvenanceRecord`](provenance.md) |
| `session.processing_history` | every step run through `m3resp.workflows` | [`ProcessingStep`](provenance.md) |
| `session.datamodel` | opt-in: `session.datamodel = DataModelRecorder(session)` | Layer 2 entities, see [provenance.md](provenance.md) |

A custom `preprocess=callable` passed to `preprocess_eit` bypasses the typed
collections (its output shape isn't guaranteed to match what the conversion
methods expect) - everything else about the session keeps working.

## Method overview

| Method | Does |
|---|---|
| `load_eit(path, vendor=..., **kwargs)` | Load an EIT recording via `EITProcessingAdapter`. |
| `load_emg(path, **kwargs)` | Load an EMG/ventilator recording via `ReSurfEMGAdapter`. |
| `preprocess_eit(variant=None, **kwargs)` | Filter/derive EIT signals; populates `signals`/`parameter_results`/`quality`. |
| `preprocess_emg(variant=None, **kwargs)` | Filter/derive EMG signals; populates `signals`. |
| `synchronize_raw_modalities(...)` | Align raw signals across modalities before processing. |
| `detect_eit_breaths(variant=None, **kwargs)` | Detect EIT breaths -> `session.events["eit_breaths"]`. |
| `detect_emg_breaths(variant=None, **kwargs)` | Detect EMG breaths -> `session.events["emg_breaths"]`. |
| `add_events(name, events)` / `get_events(name, default=None)` | Store/retrieve a named event list directly. |
| `postprocess_emg(**kwargs)` | Compute EMG features/quality; populates `parameter_results`/`quality`. |
| `align_modalities(method="manual_offset", offset_seconds=..., reference_modality=...)` | Shift already-detected event lists onto a common time axis. |
| `link_breaths(time_tolerance=0.5)` | Match breaths across modalities into [`LinkedBreath`](synchronization.md) objects. |
| `compute_multimodal_parameters(...)` | Compute timing-delay/duration-difference/event-agreement [`ParameterResult`](parameters.md)s from `session.linked_breaths`. |
| `run_pipeline(name, config=...)` | Run a built-in `"eit"`/`"emg"`/`"multimodal"` preset - see [pipeline-contracts.md](../developer/pipeline-contracts.md). |
| `export_summary(output_dir)` | Write the structured export - see [export-results tutorial](../tutorials/export-results.md). |

## See also

- [tutorials/](../tutorials/) for end-to-end walkthroughs.
- [../developer/architecture.md](../developer/architecture.md) for how these
  pieces fit into the package layout.
