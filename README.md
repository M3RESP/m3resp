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

## Example

Run the configured workflow from YAML:

```bash
python examples/workflow.py
```

Or call the same wrapper from Python:

```python
import os

from m3resp.workflows import run_workflow

result = run_workflow(config=os.path.join("examples", "config.yaml"))

session = result.session
print(result.summary)
print(result.output_dir)
```

The configured workflow uses `examples/config.yaml` for modality toggles, input
files, alignment settings, and output directories. `run_workflow` selects the
workflow from `modules`: EIT+EMG runs the combined workflow, EIT-only runs the
EIT workflow, and EMG-only runs the EMG workflow. `vent` is used with EMG
postprocessing when EMG is enabled.

For lower-level control, use `M3Session` directly:

```python
import os

from m3resp import M3Session

session = M3Session()

session.load_eit(os.path.join("path", "to", "eit_file"), vendor="sentec")
session.load_emg(os.path.join("path", "to", "emg_file"))

session.preprocess_eit()
session.preprocess_emg()

session.detect_eit_breaths()
session.detect_emg_breaths()

session.align_modalities(method="manual_offset", offset_seconds=0.0)
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
- CSV and JSON export helpers;
- minimal examples and tests.

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
