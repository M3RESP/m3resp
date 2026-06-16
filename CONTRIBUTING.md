# Contributing

M3Resp is the integration layer for multimodal respiratory workflows. Keep
modality-specific processing in the upstream packages whenever possible:

- EIT-specific changes belong in `eitprocessing`.
- EMG-specific changes belong in `ReSurfEMG` / `resurfemg`.
- Cross-modality API, synchronization, export, session state, and the pipeline
  engine belong here.

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Making changes

Before opening a pull request, make sure the tests pass:

```bash
pytest
```

The pre-commit hooks run Ruff linting and formatting automatically. Install them
once after setting up the dev environment:

```bash
pip install -e ".[dev]"
pre-commit install
```

Run the checks manually at any time:

```bash
pre-commit run --all-files
```

## Types of contributions

We welcome contributions of all kinds:

1. **Questions** — open an issue and add the "Question" label.
2. **Bug reports** — open an issue with enough context to reproduce the problem
   (commit hash, dependency versions, OS, and a minimal example if possible).
3. **Code changes** — open a pull request. For anything beyond a small fix,
   open an issue first so we can discuss the approach.
4. **Documentation** — improvements to the docs are always welcome.

## What belongs here vs upstream

If you are unsure whether a change belongs in `m3resp` or in one of the upstream
packages, open an issue and we will figure it out together. The general rule:
if the change is about how EIT or EMG signals are processed scientifically, it
belongs upstream. If it is about how those results are orchestrated, exported,
or combined across modalities, it belongs here.
