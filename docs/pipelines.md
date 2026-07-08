# Pipelines

M3Resp workflows are written as YAML specs and executed by the pipeline engine.
A spec is an ordered list of named steps. Each step reads named artifacts from a
shared context, runs one operation, and writes its outputs back. The engine
validates that every step's inputs are produced before it runs, so wiring
mistakes are caught immediately with a clear error.

## Running a pipeline

From the command line:

```bash
m3resp run path/to/pipeline.yaml
```

From Python:

```python
from m3resp import run_spec, run_pipeline

# Run a file end-to-end (handles outputs: section automatically)
result = run_spec("path/to/pipeline.yaml")

# Run a spec dict or file with more control
result = run_pipeline("path/to/pipeline.yaml")
print(result.value("cv"))     # read a produced artifact by context key
print(result.outputs)         # dict of every artifact the spec produced
result.session                # the backing M3Session (events, parameters)
```

List the available built-in steps:

```bash
m3resp steps
```

Or from Python:

```python
from m3resp import available_steps
for name, summary in available_steps().items():
    print(f"{name}: {summary}")
```

## Spec format

A minimal spec looks like this:

```yaml
name: my-pipeline

inputs:
  eit_file: /data/subject.bin   # referenced in steps as "@eit_file"
  vendor: draeger

outputs:
  dir: output/results
  timestamped: true
  summary_json: true
  event_csvs: true

steps:
  - uses: eit.load
    with: { file: "@eit_file", vendor: "@vendor" }

  - uses: eit.detect_rates
    in:  { signal: raw_eit }        # override which context key feeds this param
    with: { subject_type: adult }

  - uses: eit.mdn_filter
    in:  { signal: raw_eit }

  - uses: eit.global_impedance
    in:  { signal: filtered_eit }

  - uses: eit.detect_breaths
    in:  { signal: global_impedance }

  - uses: metric.interval_cv
    in:  { intervals: breath_intervals }
    out: { cv: breath_duration_cv }   # rename output into a different context key
```

The four keys for each step:

- **`uses`** — the registered step name.
- **`in`** — maps a step's parameter name to the context key it reads from.
  Leave it out to use the step's default binding.
- **`with`** — static parameters passed directly to the step function. A string
  value starting with `@` is resolved from `inputs`.
- **`out`** — renames a step's natural output names to different context keys.
  Useful when you want to run the same step twice and keep both outputs.

### Top-level sections

- **`name`** — a label for the pipeline (used in log output).
- **`inputs`** — named values that steps can reference with `@name`.
- **`outputs`** — controls where and what gets exported after the pipeline
  finishes. If `dir` is set and no explicit export step is present, the engine
  calls `export_session_summary` automatically.
- **`experiment`** — study metadata used by `export.rotarc_result` for output
  file naming (`subject_id`, `mode`, `timepoint`, `run_identifier`, `selection`).

## Built-in steps

Run `m3resp steps` to see the full list with descriptions. The main groups are:

| Prefix | What it covers |
|---|---|
| `eit.*` | Load, slice, filter, detect rates/breaths, compute TIV/EELI/pixel TIV |
| `emg.*` | Load (EMG/ventilator), preprocess, detect breaths, compute per-function baseline/event-detection/feature/quality-assessment postprocessing steps |
| `session.*` | Cross-modality operations (raw signal synchronization) |
| `metric.*` | Reduction steps (e.g. coefficient of variation of intervals) |
| `export.*` | Write results to disk (scalar files, JSON, session summary, ROTARC result) |

## Adding a custom step

Register a function with `@register_step`. Declare the context keys it reads
(mapped onto its parameter names) and the output names it returns:

```python
from m3resp.pipeline import register_step

@register_step(
    "my.step",
    reads={"signal": "global_impedance"},   # param name → default context key
    writes=("result",),                      # output names returned in the dict
)
def my_step(signal, *, threshold: float = 0.5) -> dict:
    return {"result": signal.values > threshold}
```

The function receives bound keyword arguments and returns a mapping of
`{output_name: value}`. Steps that produce no outputs return `{}` or `None`.
Import the optional modality packages inside the function body so the package
installs cleanly without them.

## Example specs

Two worked examples ship with the repository:

- [`examples/ROTARC_example/breath-duration.pipeline.yaml`](../examples/ROTARC_example/breath-duration.pipeline.yaml) —
  computes breath-duration CV from EIT data and writes a ROTARC-style result
  file.
- [`examples/multimodal_example/multimodal.pipeline.yaml`](../examples/multimodal_example/multimodal.pipeline.yaml) —
  loads EIT, EMG, and ventilator signals, synchronizes them, processes each
  modality, and exports session summaries.

Run them from the repository root:

```bash
m3resp run examples/ROTARC_example/breath-duration.pipeline.yaml
m3resp run examples/multimodal_example/multimodal.pipeline.yaml
```

Update `inputs.eit_file` in the ROTARC example before running it — it contains
a site-specific path.

## Architecture

| File | Responsibility |
|---|---|
| `pipeline/registry.py` | `@register_step` decorator and the global step registry. |
| `pipeline/spec.py` | Parse and validate a YAML or JSON spec into typed dataclasses. |
| `pipeline/context.py` | `PipelineContext` — the shared artifact store wrapping an `M3Session`. |
| `pipeline/engine.py` | `run_pipeline` and `run_spec` — bind arguments, validate, execute. |
| `pipeline/steps/` | Built-in steps for EIT, EMG, metrics, session ops, and export. |
| `pipeline/utils.py` | Shared utilities: signal slicing, JSON writing, summary logging. |
| `pipeline/summaries.py` | Compact session summaries written to the log after a run. |
