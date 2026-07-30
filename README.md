# M3Resp

M3Resp is the integration layer for multimodal respiratory processing across
EIT, EMG, and ventilator signals. It provides a unified API on top of
`eitprocessing` and `resurfemg` while keeping those packages independent — no
code is copied, and users can continue to install and use them directly.

## Documentation

The documentation source is available in [docs/](docs/index.md), including
getting-started instructions, tutorials, concepts, pipeline reference material,
and the public Python API. The website is configured for Read the Docs; its
hosted URL will be added here after the M3Resp project has been imported there.

## Install

For local development:

```bash
pip install -e ".[dev]"
```

Install optional modality integrations as needed:

```bash
pip install -e ".[eit]"          # adds eitprocessing
pip install -e ".[emg]"          # adds resurfemg
pip install -e ".[all,dev]"      # everything, plus dev tools
```

The `[dev]` extra includes `pytest`, `ruff`, and `mypy`. The `[eit]` and
`[emg]` extras add the upstream modality packages; the base install works
without them.

The modality packages currently install from the M3Resp organization forks:

```text
eitprocessing @ git+https://github.com/M3RESP/eitprocessing.git@m3resp-integration
resurfemg     @ git+https://github.com/M3RESP/ReSurfEMG.git@m3resp-integration
```

To use a different branch, change the branch name after `@` in
[pyproject.toml](pyproject.toml).

## Running pipelines

Workflows are described as YAML specs and run with the CLI:

```bash
m3resp run path/to/pipeline.yaml
```

List the available steps:

```bash
m3resp steps
```

Two examples ship with the repository:

```bash
m3resp run examples/ROTARC_example/breath-duration.pipeline.yaml
m3resp run examples/multimodal_example/multimodal.pipeline.yaml
```

The ROTARC example computes breath-duration variability from EIT data and writes
a result file in the ROTARC format. The multimodal example loads EIT, EMG, and
ventilator signals, synchronizes them, processes each modality, and exports
session summaries.

The multimodal example references synthetic data files under
`data/source/synthetic/`. The ROTARC example contains a site-specific EIT file
path — update `inputs.eit_file` in that YAML before running it on another
machine.

### Output paths

`outputs.dir` in a spec is resolved relative to the spec file, not the working
directory. For example, the ROTARC spec uses:

```yaml
outputs:
  dir: ../../output/rotarc-breath-duration
```

which writes to `output/rotarc-breath-duration/` relative to the repository
root when run from there. Absolute paths are also accepted.

### Python API

Run a spec from Python:

```python
from m3resp import run_spec

result = run_spec("examples/ROTARC_example/breath-duration.pipeline.yaml")
print(result.outputs)
```

For lower-level control, use `M3Session` directly:

```python
from m3resp import M3Session

session = M3Session()

session.load_eit(os.path.join("path", "to", "eit_file"), vendor="sentec")
session.load_emg(os.path.join("path", "to", "emg_file"))

session.synchronize_raw_modalities(
    method="manual_offset",
    offset_seconds={"eit": 0.0, "emg": 2.0},
    reference_modality="eit",
)

session.preprocess_eit()
session.preprocess_emg()
session.detect_eit_breaths()
session.detect_emg_breaths()

session.export_summary("results/")
```

See [docs/pipelines.md](docs/pipelines.md) for the full spec format, the list
of built-in steps, and how to add your own.

### Stage 2: shared data model and named pipelines

On top of the Stage 1 API above, `m3resp` also provides a shared multimodal
data model (`m3resp.data`: `Signal`, `ParameterResult`, `QualityFlag`,
`LinkedBreath`, ...), typed collections on `M3Session`
(`session.signals`, `session.parameter_results`, `session.quality`,
`session.linked_breaths`), named one-call presets
(`session.run_pipeline("eit" | "emg" | "multimodal")`), cross-modality
synchronization helpers, structured export, and a persisted/audit data model
for querying and exporting what a session did. See
[docs/stage2.md](docs/stage2.md) for the full architecture and
[docs/migration.md](docs/migration.md) if you're migrating code that calls
`eitprocessing`/`resurfemg` directly.

## Development

Install the pre-commit hooks once:

```bash
pip install -e ".[dev]"
pre-commit install
```

The hooks run Ruff linting and formatting before each commit. Run them manually
at any time:

```bash
pre-commit run --all-files
```

Run the tests:

```bash
pytest
```

## What m3resp provides

- A `M3Session` API for loading, processing, and exporting multimodal sessions.
- Adapters for `eitprocessing` and `resurfemg` that can be swapped or upgraded
  without touching user code, converting their output into shared
  `Signal`/`ParameterResult`/`QualityFlag` objects.
- A declarative YAML/JSON pipeline engine and `m3resp` CLI, plus named
  one-call presets (`session.run_pipeline("eit" | "emg" | "multimodal")`).
- Common event dataclasses (`BreathEvent`, `Event`) shared across modalities,
  and `LinkedBreath` for matching breaths across modalities.
- Manual-offset and timestamp-derived synchronization, plus signal resampling.
- CSV, JSON, and figure export helpers, including a structured export of the
  full typed session state.
- A persisted, validated data model (`m3resp.datamodel`) for querying and
  auditing what a session did.
- ROTARC and multimodal pipeline examples.
- Regression tests (`tests/regression/`) pinning the adapters as thin,
  numerically identical wrappers around `eitprocessing`/`resurfemg`.

## Repository layout

```text
M3RESP/
├── eitprocessing   EIT processing algorithms
├── ReSurfEMG       Respiratory EMG algorithms
└── m3resp          This package (integration layer)
```

`eitprocessing` and `resurfemg` do not depend on `m3resp`.
