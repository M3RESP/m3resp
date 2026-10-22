# Getting started

## Installation

M3Resp currently supports Python 3.12, 3.13, and 3.14. For a local source checkout,
install the base package with:

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
python -m pip install -e .
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv sync
```
:::
::::

:::

:::{tab-item} uv
:sync: uv

```bash
uv sync
```

:::
::::

Install the modality integrations that you need:

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
python -m pip install -e ".[eit]"       # EIT processing
python -m pip install -e ".[emg]"       # EMG processing
python -m pip install -e ".[all]"       # Both integrations
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv sync --extra eit  # EIT processing
uv sync --extra emg  # EMG processing
uv sync --extra all  # Both integrations
```
:::
::::

`uv sync` installs only the extras named in the command and removes the
others, so name every extra you need, for example
`uv sync --extra all --extra dev`.

:::

:::{tab-item} uv
:sync: uv

```bash
uv sync --extra eit  # EIT processing
uv sync --extra emg  # EMG processing
uv sync --extra all  # Both integrations
```

:::
::::

`uv sync` installs only the extras named in the command and removes the
others, so name every extra you need, for example
`uv sync --extra all --extra dev`.

The base installation remains useful without either optional integration. It
provides the shared data model, workflow specification and validation tools,
synchronization helpers, synthetic EIT generation, and export utilities.

## Run a workflow

Workflows are described in YAML or JSON:

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
m3resp run examples/multimodal_example/multimodal.workflow.yaml
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv run m3resp run examples/multimodal_example/multimodal.workflow.yaml
```
:::
::::

:::

:::{tab-item} uv
:sync: uv

```bash
uv run m3resp run examples/multimodal_example/multimodal.workflow.yaml
```

:::
::::

List the registered processing steps:

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
m3resp steps
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv run m3resp steps
```
:::
::::

:::

:::{tab-item} uv
:sync: uv

```bash
uv run m3resp steps
```

:::
::::

The example workflow references recording files that are not distributed with
the source repository. Update its input paths before running it with local
measurements. Paths inside a workflow file are resolved relative to that file.

## Use the Python API

```python
from m3resp import load_spec, run_spec

spec = load_spec("path/to/workflow.yaml")
result = run_spec(spec)
print(result.outputs)
```

For direct session control:

```python
import os

from m3resp import M3Session

session = M3Session()
session.load_eit(os.path.join("path", "to", "recording.bin"), vendor="draeger")
```

Continue with the [workflow reference](workflows.md), or choose an
[end-to-end tutorial](tutorials/index.md).
