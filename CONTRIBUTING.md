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

## Documentation

Install the documentation tools and build the website locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to preview the result. The documentation
workflow runs the same strict build for every pull request, so broken links,
invalid cross-references, and rendering warnings must be fixed before merging.

Add explanatory pages and tutorials as Markdown files under `docs/`. Include
each new page in the nearest section's `toctree` so readers can reach it from
the navigation.

The `docs/gallery_examples/` directory is reserved for future executable
Sphinx-Gallery examples. When adding the first example:

1. Use a `.py` file with narrative comments in Sphinx-Gallery format.
2. Use deterministic seeds and data that can be generated with the base
   package; do not require private recordings or write into the repository.
3. Add any required plotting package to the `docs` extra.
4. Add `generated/gallery/index` to the main documentation navigation.
5. Run the strict documentation build and check the generated `.py` and
   notebook downloads.

Read the Docs builds the default branch as `latest`. Non-prerelease Git tags
can be activated as stable and numbered documentation versions in the Read the
Docs project settings. An organization administrator must import the repository
and enable pull-request previews before the hosted site becomes available.

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
