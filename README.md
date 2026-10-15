# M3Resp: multimodal respiratory signal analysis

| Badges | |
| :-- | :-- |
| Repository | [![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/M3RESP/m3resp) |
| License | [![Apache-2.0 license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) |
| Citation | [![CITATION.cff](https://img.shields.io/badge/citation-CITATION.cff-blue)](CITATION.cff) |
| GitHub Actions | [![Tests](https://github.com/M3RESP/m3resp/actions/workflows/tests.yml/badge.svg)](https://github.com/M3RESP/m3resp/actions/workflows/tests.yml) [![Lint](https://github.com/M3RESP/m3resp/actions/workflows/lint.yml/badge.svg)](https://github.com/M3RESP/m3resp/actions/workflows/lint.yml) [![Documentation](https://github.com/M3RESP/m3resp/actions/workflows/docs.yml/badge.svg)](https://github.com/M3RESP/m3resp/actions/workflows/docs.yml) |
| Python support | ![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg) ![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg) |
| Linting | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |

M3Resp is a Python package for the analysis of respiratory recordings from three modalities: electrical impedance tomography (EIT), electromyography of the respiratory muscles (EMG) and ventilator signals (airway pressure, flow and volume). It puts the recordings on one shared clock, processes each modality, finds the breaths, and matches each breath across the modalities. Each analysis is a workflow file that lists every step and every setting. You can run the same file again and get the same results.

- Documentation: [docs/](docs/index.md)
- Source code: https://github.com/M3RESP/m3resp
- Bug reports and questions: https://github.com/M3RESP/m3resp/issues
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Citation: [CITATION.cff](CITATION.cff)

## Main features

- One `M3Session` object that holds the recordings, signals, breaths, parameters, quality flags and processing history of one analysis.
- Loaders for EIT files (for example Dräger `.bin`), EMG files (`.Poly5` and Biopac `.txt`) and ventilator files.
- EMG processing: filters, ECG removal, envelope, baseline and breath detection.
- EIT processing: filters, global impedance, breath detection, tidal impedance variation (TIV), end-expiratory lung impedance (EELI) and pixel maps.
- Ventilator processing: breath detection and occlusion manoeuvres (Pocc).
- Synchronization of recordings with an offset that you enter, and matching of breaths across modalities.
- Workflows in YAML or JSON, with a command-line tool to list the steps and run a workflow.
- Three presets for a fast start: EIT only, EMG only and multimodal.
- Export of breaths, parameters, quality flags and figures to CSV, JSON and PNG files, with a record of each processing step.

## Quick example

The repository has synthetic EIT, EMG and ventilator recordings in `tests/data`. This command runs a full multimodal workflow on them:

```bash
m3resp run examples/multimodal_example/multimodal.workflow.yaml
```

The workflow loads the three recordings and synchronizes them. Then it finds the breaths in each modality and calculates the parameters. The results go to a new folder in `output/multimodal-summary/`, with one CSV file for each breath list, a `parameters.csv` file, a `summary.json` file and figures.

You can do the same analysis step by step in Python:

```python
import os

from m3resp import M3Session

data = os.path.join("tests", "data")

session = M3Session()
session.load_eit(os.path.join(data, "m3resp_multimodal_1_eit_draeger.bin"), vendor="draeger")
session.load_emg(os.path.join(data, "m3resp_multimodal_1_emg.Poly5"))

# The EMG recording started 2 s before the EIT recording.
session.synchronize_raw_modalities(offset_seconds={"emg": -2.0})

session.preprocess_eit()
session.preprocess_emg(channel=1)
session.detect_eit_breaths()
session.detect_emg_breaths()
session.export_summary(os.path.join("output", "my-first-analysis"))
```

To see all workflow steps, run `m3resp steps`.

## Installation

M3Resp is not on PyPI yet. Install it from this repository, in a new virtual environment (see the guides for [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)).

The example recordings in `tests/data` are stored with [Git LFS](https://git-lfs.com). Install `git-lfs` and run `git lfs install` once, before you clone the repository.

```bash
git clone https://github.com/M3RESP/m3resp.git
cd m3resp
python -m pip install -e ".[all]"    # or: uv sync --extra all
```

The `[all]` option installs the EIT and EMG packages. To install only the part that you need, use one of these options:

| Command | Adds | Use it for |
| --- | --- | --- |
| `python -m pip install -e .` | nothing | the data objects, workflows, synchronization, export, and ventilator data in Biopac `.txt` files |
| `python -m pip install -e ".[eit]"` | `eitprocessing` | EIT recordings, and the ventilator data in EIT `.bin` files |
| `python -m pip install -e ".[emg]"` | `resurfemg` | EMG processing, and EMG or ventilator recordings in `.Poly5` files |
| `python -m pip install -e ".[all]"` | both | all of the above |

### Development version

To change the code or run the tests, install the development tools too:

```bash
python -m pip install -e ".[all,dev]"    # or: uv sync --extra all --extra dev
pre-commit install
```

## Dependencies

M3Resp needs Python 3.12 or newer. The tests run on Python 3.12, 3.13 and 3.14.

The base package needs:

- [NumPy](https://numpy.org)
- [SciPy](https://scipy.org)
- [pandas](https://pandas.pydata.org)
- [pydantic](https://docs.pydantic.dev) 2 or newer
- [PyYAML](https://pyyaml.org)
- [loguru](https://github.com/Delgan/loguru)

The EIT and EMG options also need:

- [eitprocessing](https://github.com/M3RESP/eitprocessing), for EIT
- [ReSurfEMG](https://github.com/M3RESP/ReSurfEMG), for EMG

The options install both packages from their `m3resp-integration` branch. [pyproject.toml](pyproject.toml) gives the exact sources.

## Documentation

The documentation is in the [docs](docs/index.md) folder:

- [Getting started](docs/getting-started.md): installation and a first workflow.
- [Tutorials](docs/tutorials_src/): EIT only, EMG only, and EIT, EMG and ventilator together. Each tutorial is a Python script that you can run.
- [How-to guides](docs/how-to/index.md): the processing settings, synchronization, breath timing and export of results.
- [Concepts](docs/concepts/index.md): the scientific data objects, their units and time axes, and a [glossary](docs/concepts/glossary.md).
- [Workflows](docs/workflows.md): the workflow file format and the list of steps.
- [API reference](docs/api/index.md): the Python functions and classes.
- [Migration guide](docs/developer/migration.md): how to move from `eitprocessing` or `resurfemg` code to M3Resp.

## Relation to eitprocessing and ReSurfEMG

M3Resp is part of the M3RESP project, together with two packages for one modality each:

```text
M3RESP/
├── eitprocessing   EIT methods
├── ReSurfEMG       respiratory EMG methods
└── m3resp          synchronization, multimodal analysis and workflows
```

M3Resp uses `eitprocessing` for EIT and `resurfemg` for EMG. Both packages are independent: you can install and use them without M3Resp. Some processing functions in M3Resp come from ReSurfEMG code; [NOTICE.md](NOTICE.md) lists them.

New methods for one modality go into `eitprocessing` or `resurfemg`. Synchronization, the shared data objects, workflows and export go into M3Resp.

## Contributing

We welcome contributions, questions and suggestions. Read the [contribution guide](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) before you start.

To run the tests and the code checks:

```bash
pytest
pre-commit run --all-files
```

## Citation

If you use M3Resp in your research, cite the software with the data in [CITATION.cff](CITATION.cff). Also cite `eitprocessing` and `resurfemg` if your analysis uses EIT or EMG.

## License

M3Resp is released under the [Apache License 2.0](LICENSE). [NOTICE.md](NOTICE.md) lists the third-party code in M3Resp and its licenses.
