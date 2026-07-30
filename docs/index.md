---
html_theme.sidebar_primary.remove: true
html_theme.sidebar_secondary.remove: true
---

# M3Resp documentation

:::::{div} m3-hero
:::{div} m3-eyebrow
MULTIMODAL RESPIRATORY RESEARCH · v{{ m3resp_version }}
:::

## Respiratory signals,

::::{div} m3-modalities
:::{div} m3-modality
**EIT**
:::

:::{div} m3-modality
**EMG**
:::

:::{div} m3-modality
**VENT**
:::
::::

:::{div} m3-title-continuation
one reproducible workflow.
:::

Bring EIT (Electrical impedance tomography), respiratory EMG (Respiratory muscle activity), and ventilator (Pressure, flow, and volume) recordings into a shared scientific workflow.

::::{div} m3-actions
:::{button-ref} getting-started
:ref-type: doc
:color: primary
:class: m3-primary-button

Get started
:::

:::{button-ref} tutorials/index
:ref-type: doc
:color: light
:outline:
:class: m3-secondary-button

Explore tutorials
:::
::::
:::::

## Choose your path

Start with a guided workflow or go directly to the pipeline and API references.

::::{grid} 1 2 3 3
:gutter: 3
:class-container: m3-card-grid

:::{grid-item-card} New to M3Resp?
:link: getting-started
:link-type: doc
:class-card: m3-path-card

Install the package, understand optional modality integrations, and run your
first pipeline.

**Start here →**
:::

:::{grid-item-card} Process respiratory data
:link: tutorials/index
:link-type: doc
:class-card: m3-path-card

Follow focused EIT, EMG, multimodal, and result-export walkthroughs.

**View tutorials →**
:::

:::{grid-item-card} Design a pipeline
:link: pipelines
:link-type: doc
:class-card: m3-path-card

Define reproducible workflows in YAML, inspect available steps, and validate
inputs before processing.

**Pipeline reference →**
:::
::::

:::::{div} m3-next-step
### From first recording to structured results

Use the tutorials for complete workflows, the concept guides to understand each
scientific object, and the public API when integrating M3Resp into analysis
software.

::::{div} m3-next-actions
:::{button-ref} concepts/index
:ref-type: doc
:color: primary
:outline:

Concept guides
:::

:::{button-ref} api/index
:ref-type: doc
:color: primary
:outline:

Public API
:::
::::
:::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Using M3Resp

getting-started
tutorials/index
concepts/index
pipelines
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

migration
api/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contributing

developer/index
project-history
```
