# M3Resp

M3Resp is the umbrella integration package for multimodal respiratory
processing across EIT, EMG, and ventilator signals.

Stage 1 keeps the existing scientific packages independent:

- `eitprocessing` remains the standalone EIT processing package.
- `ReSurfEMG` remains the standalone respiratory EMG processing project, with
  the Python import package `resurfemg`.
- `m3resp` provides a unified API on top of both packages.

This repository is the new integration layer. It should use adapters, not copied
code, so existing users can continue to install and import the original packages.

## Install

For local development:

```bash
pip install -e ".[dev]"
```

Install optional modality integrations as needed:

```bash
pip install -e ".[eit]"
pip install -e ".[emg]"
pip install -e ".[all,dev]"
```

These commands install this repository in editable mode with optional dependency
groups:

- `pip install -e ...` installs the local checkout in editable mode, so Python
  imports the code from this working tree and local source changes are available
  immediately without reinstalling the package.
- `".[eit]"` means "install the current package plus the `eit` optional
  dependencies". In this project, that adds the EIT integration package
  `eitprocessing`.
- `".[emg]"` means "install the current package plus the `emg` optional
  dependencies". In this project, that adds the EMG integration package
  `resurfemg` and `ipywidgets`.
- `".[all,dev]"` combines optional groups. It installs all modality integrations
  (`eitprocessing`, `resurfemg`, and `ipywidgets`) plus development tools such as
  `pytest`, `pytest-cov`, `ruff`, and `mypy`.

During Stage 1, the optional modality integrations install from the M3Resp
organization forks:

```text
eitprocessing @ git+https://github.com/M3RESP/eitprocessing.git@m3resp-integration
resurfemg @ git+https://github.com/M3RESP/ReSurfEMG.git@m3resp-integration
```

To use another branch, change the branch name after `@` in
[pyproject.toml](pyproject.toml). For example:

```text
git+https://github.com/M3RESP/eitprocessing.git@feature-branch
```

## Running Pipelines

M3Resp workflows are now usually run from declarative YAML pipeline specs. A
spec lists the processing steps, their inputs, and the export settings, so an
example can be run without writing a custom Python script.

List the available pipeline steps:

```bash
python -m m3resp steps
```

Run a pipeline spec:

```bash
python -m m3resp run path/to/pipeline.yaml
```

After installation, the console script is equivalent:

```bash
m3resp run path/to/pipeline.yaml
```

The shipped examples are:

```bash
python -m m3resp run examples/ROTARC_example/breath-duration.pipeline.yaml
python -m m3resp run examples/multimodal_example/multimodal.pipeline.yaml
```

The ROTARC example computes breath-duration variability from EIT data and writes
a ROTARC-style subject result file plus summary exports. The multimodal example
loads EIT, EMG, and ventilator signals, synchronizes raw signals, processes each
modality, aligns detected events, and exports session summaries.

Run the examples from the repository root. The multimodal example references the
synthetic files committed under `data/source/synthetic/...`; the ROTARC example
contains a site-specific absolute EIT path, so update `inputs.eit_file` in that
YAML before running it on another machine.

### Example Output Paths

Pipeline `outputs.dir` values are resolved relative to the pipeline file, not
relative to the shell directory where the command is run. For example,
`examples/ROTARC_example/breath-duration.pipeline.yaml` contains:

```yaml
outputs:
  dir: output/rotarc-breath-duration
```

When run from the repository root, that writes under:

```text
examples/ROTARC_example/output/rotarc-breath-duration
```

To write to the repository-level `output` directory from that example, set:

```yaml
outputs:
  dir: ../../output/rotarc-breath-duration
```

Absolute output paths are also accepted. Input paths are passed through from the
spec, so relative input paths should be written with the intended working
directory in mind.

### Python API

Pipeline specs can also be run from Python:

```python
import os

from m3resp import run_spec

result = run_spec(
    os.path.join("examples", "ROTARC_example", "breath-duration.pipeline.yaml")
)

print(result.outputs)
print(result.session)
```

For lower-level control, use `M3Session` directly:

```python
import os

from m3resp import M3Session

session = M3Session()

session.load_eit(os.path.join("path", "to", "eit_file"), vendor="sentec")
session.load_emg(os.path.join("path", "to", "emg_file"))

session.synchronize_raw_modalities(
    method="manual_offset",
    offset_seconds={"eit": 0.0, "emg": 0.0},
    reference_modality="eit",
)

session.preprocess_eit()
session.preprocess_emg()

session.detect_eit_breaths()
session.detect_emg_breaths()

session.export_summary(os.path.join("results"))
```

## Development Checks

Install the local pre-commit hooks once after installing the development extra:

```bash
pip install -e ".[dev]"
pre-commit install
```

The hooks run Ruff linting and Ruff formatting automatically before each commit.
You can run the same checks manually with:

```bash
pre-commit run --all-files
```

## Stage 1 Scope

Stage 1 provides:

- a small `M3Session` API;
- common event dataclasses;
- adapters for `eitprocessing` and `resurfemg`;
- basic manual synchronization;
- a declarative YAML/JSON pipeline engine and CLI;
- compiled configured workflows for legacy YAML settings;
- CSV and JSON export helpers;
- ROTARC and multimodal pipeline examples;
- focused tests for the core scientific workflow behavior.

It does not attempt a full code merge, final data model, GUI, dashboard, or
1.0 release.

## Repository Relationship

The intended organization layout is:

```text
M3RESP/
├── eitprocessing
├── ReSurfEMG
└── m3resp
```

Dependency direction for Stage 1 remains one-way:

```text
m3resp -> eitprocessing
m3resp -> resurfemg
```

`eitprocessing` and `resurfemg` do not depend on `m3resp`.
