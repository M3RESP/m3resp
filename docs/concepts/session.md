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
| `session.signals` | `preprocess_eit`/`preprocess_emg`/`preprocess_ventilator` (default adapter path) | [`Signal`](signals.md) |
| `session.events` | `detect_eit_breaths`/`detect_emg_breaths`/`detect_ventilator_breaths`/`add_events` | [`BreathEvent`/`Event`](events-and-breaths.md) lists, keyed by name |
| `session.parameter_results` | `preprocess_eit`/`postprocess_emg`/`compute_multimodal_parameters` | [`ParameterResult`](parameters.md) |
| `session.quality` | `preprocess_eit`/`postprocess_emg` | [`QualityFlag`](quality.md) |
| `session.linked_breaths` | `session.link_breaths()` | [`LinkedBreath`](synchronization.md) |
| `session.provenance` | every instrumented session method | [`ProvenanceRecord`](provenance.md) |
| `session.processing_history` | every step run through `m3resp.workflows` | [`ProcessingStep`](provenance.md) |
| `session.datamodel` | opt-in: `session.datamodel = DataModelRecorder(session)` | Layer 2 entities, see [provenance.md](provenance.md) |

A custom `preprocess=callable` passed to `preprocess_eit` bypasses the typed
collections (its output shape isn't guaranteed to match what the conversion
methods expect) - everything else about the session keeps working.

### Ventilator data

Ventilator data is being promoted to a peer modality alongside EIT and EMG.
`session.load_ventilator(path)` mirrors `load_eit`/`load_emg`: it stores a
`VentilatorRecording` on `session.ventilator` and under
`session.raw["ventilator"]` (plus the legacy `raw["vent"]` alias, pointing at
the same object).

Processing goes through `VentilatorAdapter`, which is notable for wrapping **no
upstream library**. Neither `eitprocessing` nor `resurfemg` implements
ventilator preprocessing - that is precisely why ventilator channels used to be
consumed unfiltered - so its defaults are native, built on
`m3resp.processing.filters` and `m3resp.processing.peaks`. In that sense the
ventilator path is already where Stage 3 is taking the other two.

`VentilatorAdapter.preprocess()` splits the recording into pressure/flow/volume
and low-passes each channel (20 Hz by default, clamped below Nyquist; pass
`lowpass_hz=None` to disable). The unfiltered arrays stay available under
`"raw"`, mirroring how the EMG bundle keeps `raw_channel` alongside `filtered`
and `envelope`. The cutoff is a conservative anti-noise default rather than a
clinical parameter: respiratory content sits below roughly 5 Hz, so 20 Hz
leaves breath morphology - including the sharp pressure upstroke that Pocc
quality assessment measures - untouched.

`to_signals()` tags every channel `modality="ventilator"` with its physical
quantity in `category` (`airway_pressure`/`airflow`/`volume`), which is what
lets one device contribute three distinguishable signals. See
[signals.md](signals.md) for why those are separate fields.

Loading is the one part that still delegates: ventilator channels usually
arrive in the same multi-channel file as the sEMG (e.g. a Biopac export), so
`session.load_ventilator` reads through `session.emg_adapter` unless you pass
`ventilator_adapter=` to `M3Session(...)`. Injecting one EMG loader therefore
covers both modalities.

`session.preprocess_ventilator()` and `session.detect_ventilator_breaths()`
complete the chain, taking the same `variant`/`overwrite` arguments as their
EIT/EMG counterparts:

```python
session.load_ventilator("recording.txt")
session.preprocess_ventilator()               # or lowpass_hz=None to skip filtering
session.detect_ventilator_breaths()           # -> session.events["ventilator_breaths"]
session.link_breaths()                        # matched against EIT/EMG breaths
```

Ventilator breath detection is now a method of its own;
`postprocess_emg` still populates `session.events["ventilator_breaths"]` exactly
as before.

## Method overview

| Method | Does |
|---|---|
| `load_eit(path, vendor=..., **kwargs)` | Load an EIT recording via `EITProcessingAdapter`. |
| `load_emg(path, **kwargs)` | Load an EMG recording via `ReSurfEMGAdapter`. |
| `load_ventilator(path, **kwargs)` | Load a ventilator recording -> `session.ventilator`, `session.raw["ventilator"]`. |
| `preprocess_eit(variant=None, **kwargs)` | Filter/derive EIT signals; populates `signals`/`parameter_results`/`quality`. |
| `preprocess_emg(variant=None, **kwargs)` | Filter/derive EMG signals; populates `signals`. |
| `preprocess_ventilator(variant=None, **kwargs)` | Split and filter ventilator pressure/flow/volume; populates `signals`. |
| `synchronize_raw_modalities(...)` | Align raw signals across modalities before processing. |
| `detect_eit_breaths(variant=None, **kwargs)` | Detect EIT breaths -> `session.events["eit_breaths"]`. |
| `detect_emg_breaths(variant=None, **kwargs)` | Detect EMG breaths -> `session.events["emg_breaths"]`. |
| `detect_ventilator_breaths(variant=None, **kwargs)` | Detect ventilator breaths -> `session.events["ventilator_breaths"]`. |
| `add_events(name, events)` / `get_events(name, default=None)` | Store/retrieve a named event list directly. |
| `postprocess_emg(**kwargs)` | Compute EMG features/quality; populates `parameter_results`/`quality`. |
| `synchronize_multimodal_breaths(method="manual_offset", offset_seconds=..., reference_modality=...)` | Shift already-detected event lists onto a common time axis. |
| `link_breaths(time_tolerance=0.5)` | Match breaths across modalities into [`LinkedBreath`](synchronization.md) objects. |
| `compute_multimodal_parameters(...)` | Compute timing-delay/duration-difference/event-agreement [`ParameterResult`](parameters.md)s from `session.linked_breaths`. |
| `run_pipeline(name, config=...)` | Run a built-in `"eit"`/`"emg"`/`"multimodal"` preset - see [pipeline-contracts.md](../developer/pipeline-contracts.md). |
| `export_summary(output_dir)` | Write the structured export - see [export-results tutorial](../tutorials/export-results.md). |

## See also

- [Tutorials](../tutorials/index.md) for end-to-end walkthroughs.
- [../developer/architecture.md](../developer/architecture.md) for how these
  pieces fit into the package layout.
