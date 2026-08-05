# M3Resp <!-- omit in toc -->

## Introduction

M3Resp provides reproducible workflows for processing multimodal respiratory recordings. It brings together electrical impedance tomography (EIT), respiratory electromyography (EMG), and ventilator signals through one scientist-friendly interface.

M3Resp is an integration layer on top of [`eitprocessing`](https://github.com/M3RESP/eitprocessing) and [`resurfemg`](https://github.com/M3RESP/ReSurfEMG). The modality packages remain independent: their code is not copied into M3Resp, and they can still be installed and used directly. M3Resp adds shared data structures, cross-modality synchronization, reproducible YAML/JSON workflows, quality and provenance records, and structured exports.

| Badges | |
| :-- | :-- |
| Repository | [![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/M3RESP/m3resp) |
| License | [![Apache-2.0 license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) |
| Citation | [![CITATION.cff](https://img.shields.io/badge/citation-CITATION.cff-blue)](CITATION.cff) |
| GitHub Actions | [![Tests](https://github.com/M3RESP/m3resp/actions/workflows/tests.yml/badge.svg)](https://github.com/M3RESP/m3resp/actions/workflows/tests.yml) [![Lint](https://github.com/M3RESP/m3resp/actions/workflows/lint.yml/badge.svg)](https://github.com/M3RESP/m3resp/actions/workflows/lint.yml) [![Documentation](https://github.com/M3RESP/m3resp/actions/workflows/docs.yml/badge.svg)](https://github.com/M3RESP/m3resp/actions/workflows/docs.yml) |
| Python support | ![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg) ![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg) |
| Linting | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |

## Contents <!-- omit in toc -->

- [Introduction](#introduction)
- [Installation](#installation)
  - [Install from source](#install-from-source)
  - [Optional modality support](#optional-modality-support)
  - [Developer install](#developer-install)
- [Quick start](#quick-start)
  - [Command-line workflows](#command-line-workflows)
  - [Python API](#python-api)
- [What M3Resp provides](#what-m3resp-provides)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [Package relationships](#package-relationships)

## Installation

We recommend installing M3Resp in a dedicated virtual environment. See the Python Packaging Authority's guide to [installing packages with `venv`](https://packaging.python.org/en/latest/guides installing-using-pip-and-virtual-environments/) or the conda documentation on [getting started](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

M3Resp is not yet published on PyPI. Install it from the repository as described below.

### Install from source

The base package supports ventilator data, the shared data model, and the workflow engine:

```bash
git clone https://github.com/M3RESP/m3resp.git
cd m3resp
python -m pip install -e .  # or `uv sync`
```

### Optional modality support

Install only the integrations needed for your recordings:

```bash
python -m pip install -e ".[eit]"  # add eitprocessing
python -m pip install -e ".[emg]"  # add resurfemg
python -m pip install -e ".[all]"  # add both modalities
```

The modality extras currently use the M3Resp integration branches of the upstream packages. See [pyproject.toml](pyproject.toml) for the exact versions.

### Developer install

For tests, linting, formatting, and type checking:

```bash
git clone https://github.com/M3RESP/m3resp.git
cd m3resp
python -m pip install -e ".[all,dev]"  # or `uv sync --extra all --extra dev`
pre-commit install
```

## Quick start

### Command-line workflows

M3Resp workflows are described in YAML or JSON. List the available processing steps and run a workflow with:

```bash
m3resp steps
m3resp run path/to/pipeline.yaml
```

For example, the multimodal workflow included in this repository loads EIT, EMG, and ventilator signals, synchronizes them, processes each modality, and exports session summaries:

```bash
m3resp run examples/multimodal_example/multimodal.pipeline.yaml
```

Paths in `outputs.dir` are resolved relative to the workflow file. Input files, sampling rates, synchronization choices, and all other scientific settings remain explicit in the workflow.

### Python API

The same workflow can be run from Python:

```python
from m3resp import run_spec

result = run_spec("examples/multimodal_example/multimodal.pipeline.yaml")
print(result.outputs)
```

For interactive or lower-level use, work with an `M3Session`:

```python
import os

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
session.export_summary(os.path.join("path", "to", "results"))
```

## What M3Resp provides

- A common `M3Session` interface for loading, processing, and exporting
  multimodal respiratory recordings.
- Thin adapters around `eitprocessing` and `resurfemg`, with regression tests
  that check numerical agreement with the upstream packages.
- Shared, validated representations of signals, parameters, breaths, quality
  flags, and processing history.
- Manual-offset and timestamp-derived synchronization, signal resampling, and
  breath matching across modalities.
- Reproducible YAML/JSON workflows, a command-line interface, and named
  EIT-only, EMG-only, and multimodal pipelines.
- CSV, JSON, figure, and full-session structured exports with provenance.

## Documentation

The [M3Resp documentation](docs/index.md) contains:

- [getting-started instructions](docs/getting-started.md);
- [EIT, EMG, multimodal, and export tutorials](docs/tutorials/index.md);
- [scientific concept guides](docs/concepts/index.md);
- the [workflow specification and built-in steps](docs/pipelines.md);
- the [public Python API](docs/api/index.md); and
- [migration guidance](docs/migration.md) for existing `eitprocessing` and
  `resurfemg` users.

The documentation website is configured for Read the Docs. Its hosted address will be added here when the project is published.

## Contributing

Contributions, questions, and suggestions are welcome. Please read the
[contribution guidelines](CONTRIBUTING.md) and
[Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

Run the local checks with:

```bash
pytest
pre-commit run --all-files
```

Changes to modality-specific scientific processing generally belong in `eitprocessing` or `resurfemg`; synchronization, shared data, workflow, and export changes belong in M3Resp. The contribution guide explains this boundary in more detail.

## Citation

If you use M3Resp in research, please cite the software using [CITATION.cff](CITATION.cff). Also cite the underlying modality packages used in your workflow.

## Package relationships

```text
M3RESP/
├── eitprocessing   EIT processing algorithms
├── ReSurfEMG       Respiratory EMG algorithms
└── m3resp          Multimodal integration and workflow layer
```

`eitprocessing` and `resurfemg` do not depend on `m3resp`.
