# `M3Session`

`M3Session` (`src/m3resp/core/session.py`) is the central object: it holds
both the Stage 1 dict-based state (`raw`, `processed`, `events`,
`parameters` - unchanged since Stage 1) and the Stage 2 typed collections
added on top of it.

## Plain-language overview

`M3Session` is a class, meaning a blueprint for creating an object (a bundle
of data plus the functions that work on that data). Think of one
`M3Session` instance as a folder for one recording session: you create it
once, then every step you run (loading a file, cleaning it up, detecting
breaths, computing numbers) stores its result as an attribute on that same
object, so later steps can use what earlier steps produced.

`M3Session` itself does not do the scientific computation. It is an
orchestrator: a coordinator that calls other pieces (adapters, helper
functions) in the right order and keeps track of the results. The actual
signal-processing work is delegated to `EITProcessingAdapter` and
`ReSurfEMGAdapter`, two adapters (wrapper objects that translate calls from
`m3resp` into calls on an outside library, so the rest of the codebase never
has to know that outside library's details directly).

### Stage 1 dicts vs Stage 2 typed collections

Stage 1 stores results in plain dictionaries (`raw`, `processed`, `events`,
`parameters`; a dictionary is a lookup table of key/value pairs, like a
labeled filing cabinet). Whatever an adapter returns gets filed under a
string key such as `"eit"` or `"emg_breaths"`, with no fixed shape, so EIT
and EMG results can look completely different from each other.

Stage 2 adds a second, parallel set of attributes (`signals`,
`parameter_results`, `quality`, `linked_breaths`) built from fixed, shared
types (`Signal`, `ParameterResult`, `QualityFlag`, `BreathEvent`), so EIT and
EMG data can be compared and displayed using the same shape. These are
populated by each adapter's `to_signals`/`to_parameters`/`to_quality_flags`
conversion methods and are additive: the Stage 1 dicts keep working exactly
as before, side by side with the new typed collections.

### What Stage 3 changes

Stage 3's job is to port the underlying algorithms natively into `m3resp`
(native meaning implemented directly in this codebase, not borrowed from an
outside library), removing the dependency on the upstream `eitprocessing`
and `resurfemg` packages. Once an operation is fully native, there is no
more upstream object to keep around for compatibility, so the Stage 1 dicts
are expected to be retired for that operation, leaving the typed
collections as the only representation. Public `M3Session` method names and
signatures are a stable contract (a promise about names, inputs, and
outputs that other code can rely on) and do not change: `load_eit`,
`preprocess_eit`, `detect_eit_breaths`, and the rest keep working exactly as
they do today. What changes is invisible to the caller: inside each method,
the call to an adapter that wraps an outside library is swapped, one
operation at a time, for a call to `m3resp`'s own native implementation.
The Stage 3 GUI (graphical user interface, the visual app a user clicks
through) is required to talk only to `M3Session` and the workflow step
registry, never directly to an adapter or an outside library, so this swap
happens underneath it without anything downstream needing to change.

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
