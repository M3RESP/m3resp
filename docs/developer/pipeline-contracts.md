# Pipeline contracts

`m3resp` has two different mechanisms that both happen to be called "run a
pipeline" - this page is about the smaller one, `Pipeline`/presets. For the declarative YAML/JSON step-registry engine
(`m3resp.workflows`), see [../pipelines.md](../pipelines.md).

## `Pipeline` (`src/m3resp/presets/base.py`)

```python
class Pipeline(ABC):
    name: str

    @abstractmethod
    def run(
        self, session: M3Session, *, config: PipelineConfig | None = None
    ) -> M3Session:
        ...
```

`PipelineConfig` is `Mapping[str, Mapping[str, Any]]` - per-method keyword
arguments, keyed by the session method name a concrete `Pipeline` calls (e.g.
`{"preprocess": {...}, "detect_breaths": {...}}`).

A concrete `Pipeline.run` is a **named shortcut for a fixed sequence of
`M3Session` method calls** - not a second execution engine. It doesn't
implement any scientific logic itself; it calls `session.preprocess_eit()`,
`session.detect_eit_breaths()`, and so on, in a fixed order. Those methods -
not `Pipeline` - populate the typed collections (`session.signals`,
`session.parameter_results`, `session.quality`) and record provenance.

## Built-in presets and the registry

| Preset | `name` | Calls |
|---|---|---|
| `EITPipeline` | `"eit"` | `session.preprocess_eit(...)`, `session.detect_eit_breaths(...)` |
| `EMGPipeline` | `"emg"` | `session.preprocess_emg(...)`, `session.detect_emg_breaths(...)`, `session.postprocess_emg(...)` |
| `MultimodalPipeline` | `"multimodal"` | `session.synchronize_raw_modalities(...)`, `session.align_modalities(...)` |

`presets/registry.py` maps each `name` to its class via `register_pipeline`/
`get_pipeline`. `M3Session.run_pipeline(name, config=...)` looks the preset up
and calls it:

```python
session.run_pipeline("eit")
session.run_pipeline("emg", config={"preprocess": {"variant": "native"}})
session.run_pipeline("multimodal")
```

There is deliberately no `BatchPipeline` yet - nothing in the current test
suite or examples needs one; add it in `presets/` following the same shape
when a real batch-processing use case appears.

## Why two mechanisms, not one

- `m3resp.run_pipeline(spec, session=...)` (module-level) runs a fully
  custom YAML/JSON step-list spec built from individually composable steps
  (`eit.mdn_filter`, `emg.ecg_gating`, ...) - the Stage 1
  `m3resp.workflows` engine, documented in [../pipelines.md](../pipelines.md).
  Most of these steps take a `session` binding and populate the typed
  collections and record provenance through the same `M3Session._record()`
  seam the `Pipeline` presets use (see `_record_step` in each modality's
  `_shared.py`) - `eit.roi_amplitude_lungspace`, `emg.ecg_gating`, and so on
  all do this. The exception is the small set of pure per-breath feature
  steps (e.g. `emg.time_to_peak`, `ventilator.features`) that operate on
  already-extracted arrays with no natural collection to write to, and so
  stay stateless. Use this for bespoke or batch workflows where the exact
  sequence of operations varies per project.
- `session.run_pipeline("eit" | "emg" | "multimodal", config=...)` (a method
  on `M3Session`, this page) runs one of the small, built-in `Pipeline`
  presets, each a fixed sequence of calls to the session's own
  already-instrumented methods. Use this for the common case of running one
  modality end-to-end with default behavior.

No new execution machinery lives in `presets/` - building a second, parallel
step-execution engine there would duplicate `m3resp.workflows` for no
benefit, and would need its own copy of the typed-collection/provenance
instrumentation the session methods already have.

## Adding a new preset

1. Add a class in `presets/*.py` implementing `Pipeline.run`, calling only
   existing `M3Session` methods (do not put scientific logic here).
2. Register it: `register_pipeline("my_name", MyPipeline)` in
   `presets/registry.py`.
3. Every option the underlying session methods accept is reachable through
   `config` - do not hardcode a value the method already exposes as a
   keyword argument.
