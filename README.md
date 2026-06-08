# M3Resp

M3Resp is the umbrella integration package for multimodal respiratory
processing across EIT, EMG, and later ventilator signals.

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

During Stage 1, the optional modality integrations install from the M3Resp
organization forks:

```text
eitprocessing @ git+https://github.com/M3Resp-org/eitprocessing.git@main
resurfemg @ git+https://github.com/M3Resp-org/ReSurfEMG.git@main
```

To use another branch, change the branch name after `@` in
[pyproject.toml](pyproject.toml). For example:

```text
git+https://github.com/M3Resp-org/eitprocessing.git@feature-branch
```

## Example

```python
from m3resp import M3Session

session = M3Session()

session.load_eit("path/to/eit_file", vendor="sentec")
session.load_emg("path/to/emg_file")

session.preprocess_eit()
session.preprocess_emg()

session.detect_eit_breaths()
session.detect_emg_breaths()

session.align_modalities(method="manual_offset", offset_seconds=0.0)
session.export_summary("results/")
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
M3Resp-org/
├── eitprocessing
├── ReSurfEMG
└── m3resp
```

Dependency direction should stay one-way:

```text
m3resp -> eitprocessing
m3resp -> resurfemg
```

`eitprocessing` and `resurfemg` should not depend on `m3resp`.
