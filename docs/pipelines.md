# Declarative pipelines

M3Resp workflows can be assembled from a YAML or JSON **spec** instead of custom
Python. A spec is an ordered list of named *steps*; each step reads named
artifacts from a shared context, runs one operation, and writes named artifacts
back. The engine wires steps together by explicit bindings, validates the spec
before running, and records provenance.

This replaces both the hand-written per-workflow preprocess functions (such as
the old `_preprocess_rotarc_eit`) and the imperative switch-driven runner: the
legacy `config.yaml` switches are now *compiled* into a spec
(`m3resp.pipeline.compile_config`) and executed by the same engine.

## Running a pipeline

```python
from m3resp import run_pipeline

result = run_pipeline("my_pipeline.yaml")     # path to YAML or JSON, a dict, or a PipelineSpec
print(result.value("breath_duration_cv"))     # read a produced artifact
print(result.outputs)                          # every artifact produced by the spec
result.session                                 # the backing M3Session (events, parameters, export)
```

Discover the available steps:

```python
from m3resp import available_steps
for name, summary in available_steps().items():
    print(name, "—", summary)
```

## Spec format

```yaml
name: rotarc-breath-duration
inputs:                       # values referenced elsewhere with "@name"
  eit_file: /data/subject.bin
  vendor: draeger
steps:
  - uses: eit.load            # step name from the registry
    with: { file: "@eit_file", vendor: "@vendor" }
  - uses: eit.slice
    in:  { signal: raw_eit }  # parameter name -> context key it reads
    with: { start: 5373, end: 6159, mode: index }
    out: { result: selected_eit }   # natural output name -> context key it writes
  - uses: eit.detect_rates
    in:  { signal: selected_eit }
  ...
```

- **`uses`** — the registered step name.
- **`in`** — overrides where a step parameter reads from (parameter → context key).
  Omit it to use the step's default binding.
- **`with`** — static parameters. A value of `"@name"` resolves to `inputs.name`.
- **`out`** — renames a step's natural outputs to other context keys.

The engine statically rejects a spec if any step reads a context key that no
earlier step (or `inputs`) produces, so wiring mistakes fail fast and clearly.

A worked example ships at
[`examples/ROTARC_example/breath-duration.pipeline.yaml`](../examples/ROTARC_example/breath-duration.pipeline.yaml);
it reconstructs the ROTARC breath-duration CV calculation with no custom Python.

## Adding a step

Register a thin wrapper around one operation. Declare what context keys it reads
(mapped onto its parameters) and writes:

```python
from m3resp.pipeline import register_step

@register_step(
    "metric.interval_cv",
    reads={"intervals": "breath_intervals"},   # param -> default context key
    writes=("cv", "mean", "std", "n"),          # natural output names
)
def interval_cv(intervals):
    ...
    return {"cv": cv, "mean": mean, "std": std, "n": n}
```

A step returns a mapping of `{output_name: value}`. `out:` in the spec can rename
those outputs into other context keys. Built-in steps live under
`m3resp/pipeline/steps/` and defer their upstream (`eitprocessing`/`resurfemg`)
imports to call time, so the package installs without the optional modality
dependencies.

## Architecture

| Component | Responsibility |
|---|---|
| `pipeline/registry.py` | `@register_step` + the global step registry. |
| `pipeline/context.py` | `PipelineContext` blackboard wrapping an `M3Session`. |
| `pipeline/spec.py` | Parse/validate a YAML or JSON spec into one model. |
| `pipeline/engine.py` | `run_pipeline` — bind, validate, execute, record provenance. |
| `pipeline/steps/` | Built-in EIT, EMG, metric, and export steps. |
| `pipeline/compile_config.py` | Compile legacy `WorkflowConfig` switches into a spec. |
