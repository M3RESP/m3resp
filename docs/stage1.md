# M3Resp Stage 1

Stage 1 establishes `m3resp` as the integration layer for multimodal respiratory workflows. It gives users one place to run, summarize, and export EIT and EMG analyses, while the scientific processing code stays in the upstream packages.

## What m3resp does

`m3resp` provides:

- A `M3Session` object that holds raw and processed signals, events, and parameters for a single recording session.
- Adapters around `eitprocessing` and `resurfemg` so those packages can be swapped or upgraded without touching user code.
- A declarative pipeline engine: describe your workflow in a YAML spec, run it with `m3resp run pipeline.yaml`, and get structured outputs without writing custom Python.
- Common event dataclasses (`BreathEvent`, `Event`) shared across modalities.
- Manual raw-signal synchronization before processing.
- CSV, JSON, and figure export helpers.

## What stays upstream

Modality algorithms live in the upstream packages:

- EIT algorithms in `eitprocessing`.
- EMG algorithms in `resurfemg`.

The dependency direction is strictly one-way:

```text
m3resp → eitprocessing
m3resp → resurfemg
```

`eitprocessing` and `resurfemg` do not depend on `m3resp`.

## Package structure

```text
src/m3resp/
├── core/          Session, events, exceptions, provenance, metadata
├── adapters/      EITProcessingAdapter, ReSurfEMGAdapter
├── modalities/    Top-level load helpers (load_eit, load_emg)
├── export/        session_export, tables
├── visualization/ Session overview and synchronization plots
└── workflows/      Declarative engine, step registry, built-in steps
    ├── steps/     eit.*, emg.*, metric.*, session.*, export.*
    ├── engine.py  run_pipeline, run_spec, validate_spec
    ├── spec.py    YAML/JSON parser
    ├── registry.py  @register_step
    ├── context.py   PipelineContext (shared artifact blackboard)
    ├── utils.py     Signal slicing, JSON writing, summary logging
    └── summaries.py Session summaries for post-run logging
```

## Optional dependencies

The adapters import optional packages lazily, so the base install works without the modality packages:

```bash
pip install m3resp          # core + pipeline only
pip install "m3resp[eit]"   # adds eitprocessing
pip install "m3resp[emg]"   # adds resurfemg
pip install "m3resp[all]"   # both modality integrations
```

For local development, the modality packages are installed from the M3Resp organization forks:

```text
eitprocessing @ git+https://github.com/M3RESP/eitprocessing.git@m3resp-integration
resurfemg     @ git+https://github.com/M3RESP/ReSurfEMG.git@m3resp-integration
```

## Public API

```python
from m3resp import (
    M3Session,          # session orchestration
    run_pipeline,       # run a spec dict or file
    run_spec,           # run a spec file end-to-end (CLI entry point)
    load_spec,          # parse a YAML/JSON spec into a PipelineSpec
    register_step,      # register a custom step
    available_steps,    # list registered steps
    PipelineResult,     # result object returned by run_pipeline / run_spec
    BreathEvent,        # common breath event dataclass
    Event,              # generic event dataclass
    load_eit,           # load an EIT recording
    load_emg,           # load an EMG recording
)
```

## Running workflows

The primary way to run a workflow is via the CLI:

```bash
m3resp run pipeline.yaml
```

Or from Python:

```python
from m3resp import run_spec
result = run_spec("pipeline.yaml")
print(result.outputs)
```

See [pipelines.md](pipelines.md) for the full spec format and how to write your own steps.

## Tests

Run the test suite with:

```bash
pytest
```

Tests cover the pipeline engine, EIT and EMG session operations, adapters, synchronization, visualization, and export. Tests that require sample data files or optional modality packages are skipped automatically when those are not present.

## Out of scope

Stage 1 intentionally does not include:

- A GUI or dashboard.
- A merger of `eitprocessing` and `resurfemg`.
- A final production data model.
- Production-grade workflow scheduling.
