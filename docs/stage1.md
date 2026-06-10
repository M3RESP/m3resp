# M3Resp Stage 1

Stage 1 establishes `m3resp` as the integration layer for multimodal respiratory
workflows. It gives users one place to configure, run, summarize, visualize, and
export EIT and EMG workflows while keeping the scientific modality code in the
upstream packages.

Stage 1 does not merge the upstream EIT and EMG codebases.

## Package Boundaries

Use `m3resp` for:

- `M3Session` orchestration;
- typed YAML workflow configuration;
- automatic workflow selection;
- common event models;
- adapters around upstream packages;
- synchronization and alignment;
- summary, CSV, JSON, and figure export;
- runnable examples and tests.

Keep modality algorithms upstream:

- EIT algorithms in `eitprocessing`;
- EMG algorithms in `resurfemg`.

The dependency direction stays one-way:

```text
m3resp -> eitprocessing
m3resp -> resurfemg
```

`eitprocessing` and `resurfemg` should not depend on `m3resp`.

## Optional Dependencies

The adapters import optional packages lazily. This allows:

```bash
pip install m3resp
```

for session/export work, and:

```bash
pip install "m3resp[all]"
```

when the upstream packages are available.

For local development, the current optional integrations are installed from the
M3Resp organization forks:

```text
eitprocessing @ git+https://github.com/M3RESP/eitprocessing.git@m3resp-integration
resurfemg @ git+https://github.com/M3RESP/ReSurfEMG.git@m3resp-integration
```

## Public Entry Points

The public API exposes:

- `m3resp.M3Session` for direct orchestration;
- `m3resp.load_workflow_config` for loading typed workflow config;
- `m3resp.workflows.run_workflow` for automatic configured workflow execution;
- `m3resp.workflows.select_workflow` for reading the selected workflow kind;
- `run_eit_workflow`, `run_emg_workflow`, and `run_multimodal_workflow` for
  both configured and lightweight direct calls.

Direct path-based workflow calls preserve the original lightweight API and
return `M3Session`:

```python
import os

from m3resp.workflows import run_eit_workflow

session = run_eit_workflow(
    os.path.join("path", "to", "eit_file.bin"), vendor="draeger"
)
```

Passing `config=` runs the YAML-configured path and returns `WorkflowResult`.

## Configured Workflows

`examples/config.yaml` is the canonical Stage 1 workflow config shape. The
package loader resolves relative paths against the repository root when it can
find one:

```bash
python examples/workflow.py
```

```python
import os

from m3resp import load_workflow_config
from m3resp.workflows import run_workflow

cfg = load_workflow_config(os.path.join("examples", "config.yaml"))
result = run_workflow(config=cfg)
```

Configured workflows return a `WorkflowResult` with:

- `session`: the populated `M3Session`;
- `summary`: compact workflow summary values;
- `output_dir`: the selected output directory;
- `figures`: saved figure paths keyed by filename.

`run_workflow` selects the configured workflow from module flags:

- `eit: true` and `emg: true` runs the combined workflow.
- `eit: true` and `emg: false` runs the EIT workflow, regardless of `vent`.
- `eit: false` and `emg: true` runs the EMG workflow, with `vent` used for EMG
  postprocessing when enabled.
- `eit: false` and `emg: false` is rejected, even if `vent` is enabled.

## Configuration Shape

The Stage 1 config is represented by `WorkflowConfig` and includes:

- `modules`: `eit`, `emg`, and `vent` toggles;
- `eit`: input file, vendor, EIT processing settings, filter settings, and EIT
  output switches;
- `emg`: input file, preprocessing settings, breath detection settings, and
  selectable ReSurfEMG postprocessing functions;
- `vent`: optional ventilator waveform file used by EMG postprocessing;
- `alignment`: method plus `manual_offset_seconds`;
- `output`: directories for combined, EIT-only, and EMG-only runs;
- `results`: artifact switches for summary JSON, parameter CSV, event CSVs,
  postprocessing output, and figures.

EIT processing now uses `eit.processing.outputs` for result-producing switches:

```yaml
eit:
  processing:
    filter:
      enabled: true
      mode: mdn
    outputs:
      rates: true
      breath_intervals: true
      continuous_tiv: true
      eeli: true
      pixel_tiv: true
      filtered_data: true
      global_impedance: true
```

`continuous_tiv`, `eeli`, and `pixel_tiv` require `breath_intervals: true`.

EMG postprocessing exposes function groups from `resurfemg`:

```yaml
emg:
  processing:
    postprocessing:
      enabled: true
      functions:
        baseline:
          moving_baseline: true
          slopesum_baseline: true
        event_detection:
          detect_ventilator_breath: true
          detect_emg_breaths: true
        features:
          amplitude: false
        quality_assessment:
          detect_non_consecutive_manoeuvres: true
```

The configured runner validates known dependencies between selected
postprocessing functions, such as baseline-dependent features and quality
assessment functions.

## Artifacts And Summaries

Configured workflows can export:

- `summary.json`;
- `parameters.csv`;
- event CSV files such as `eit_breaths.csv` and `emg_breaths.csv`;
- EMG postprocessing outputs;
- figures such as `overview.png`, `synchronization.png`,
  `eit-processing.png`, and `eit-rate-detection.png` when the required data and
  plotting dependencies are available.

Artifact export is controlled by the top-level `results` section:

```yaml
results:
  summary_json: true
  parameters_csv: true
  event_csvs: true
  postprocessing: true
  figures: true
```

Summary helpers are available through `m3resp.workflows`:

- `summarize_eit`;
- `summarize_emg`;
- `summarize_emg_postprocessing`;
- `summarize_multimodal`.

## Validation And Tests

The Stage 1 tests cover:

- workflow config path resolution;
- loading nested EIT, EMG, and result switches;
- configured EIT, EMG, and multimodal workflow execution with fake adapters;
- automatic workflow selection;
- export switch behavior;
- EMG postprocessing dependency validation;
- preservation of direct path-based workflow calls returning `M3Session`.

Run the focused tests with:

```bash
pytest tests/test_configured_workflows.py
```

## Out Of Scope

Stage 1 intentionally does not include:

- a full data model rewrite;
- a GUI or dashboard;
- a merger of `eitprocessing` and `resurfemg`;
- production-grade workflow scheduling.
