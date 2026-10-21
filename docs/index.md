---
html_theme.sidebar_primary.remove: true
html_theme.sidebar_secondary.remove: true
---

# M3Resp documentation

:::::::{div} m3-hero
::::::{div} m3-hero-copy
:::{div} m3-eyebrow
// MULTIMODAL RESPIRATORY RESEARCH · v{{ m3resp_version }}
:::

## Respiratory signal integration for reproducible multimodal analysis.

Bring electrical impedance tomography (EIT), respiratory muscle electromyography (EMG) and ventilator recordings of pressure, flow and volume into one shared scientific workflow.

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
::::::

::::::{div} m3-hero-visual
```{image} _static/m3-signal-monitor-light.svg
:alt: Schematic of three respiratory signals on one shared time axis. EIT global impedance, respiratory muscle EMG and ventilator airway pressure each show four breaths, and dashed lines mark the start of each breath in all three signals.
:class: m3-monitor only-light
```

```{image} _static/m3-signal-monitor.svg
:alt: Schematic of three respiratory signals on one shared time axis. EIT global impedance, respiratory muscle EMG and ventilator airway pressure each show four breaths, and dashed lines mark the start of each breath in all three signals.
:class: m3-monitor only-dark
```
::::::

::::::{div} m3-stats
:::{div} m3-stat
Modalities

EIT · EMG · VENT
:::

:::{div} m3-stat
Workflow files

YAML · JSON
:::

:::{div} m3-stat
Python

3.12 or newer
:::

:::{div} m3-stat
License

Apache-2.0
:::
::::::
:::::::

## How a workflow runs

A workflow is a YAML file with an ordered list of steps. A typical multimodal workflow goes through these five stages.

::::::{div} m3-stages
:::::{div} m3-stage
:::{div} m3-stage-index
01
:::

**Load**

Read the EIT, EMG and ventilator recordings into one session.

`eit.load` `emg.load` `ventilator.load`
:::::

:::::{div} m3-stage
:::{div} m3-stage-index
02
:::

**Synchronize**

Put the start time of each recording on a shared clock.

`sync.raw_modalities`
:::::

:::::{div} m3-stage
:::{div} m3-stage-index
03
:::

**Clean**

Remove the heart signal and filter each signal.

`eit.mdn_filter` `emg.ecg_gating`
:::::

:::::{div} m3-stage
:::{div} m3-stage-index
04
:::

**Detect breaths**

Find the breaths in each signal.

`eit.detect_breaths` `emg.detect_breaths` `ventilator.detect_breaths`
:::::

:::::{div} m3-stage
:::{div} m3-stage-index
05
:::

**Compute and export**

Calculate breath parameters and write the results to files.

`eit.eeli` `emg.time_product` `export.session_summary`
:::::
::::::

## Choose your path

Start with the installation, a tutorial or the workflow reference.

::::{grid} 1 2 3 3
:gutter: 3
:class-container: m3-card-grid

:::{grid-item-card} New to M3Resp?
:link: getting-started
:link-type: doc
:class-card: m3-path-card

Install M3Resp and its EIT and EMG packages, and run your first workflow.

**Start here →**
:::

:::{grid-item-card} Process respiratory data
:link: tutorials/index
:link-type: doc
:class-card: m3-path-card

Follow a complete EIT, EMG or multimodal analysis of the synthetic recordings.

**View tutorials →**
:::

:::{grid-item-card} Design a workflow
:link: workflows
:link-type: doc
:class-card: m3-path-card

Write a workflow file, list the processing steps and check the file before a run.

**Workflow reference →**
:::
::::

## How these pages are organized

- {doc}`Tutorials <tutorials/index>` take you step by step through one complete analysis. Start here if you are new to M3Resp.
- {doc}`How-to guides <how-to/index>` show how to do one task, for example to change the EMG filter or to export the results.
- {doc}`Concepts <concepts/index>` explain what each scientific object is and why M3Resp works the way it does.
- The {doc}`workflow reference <workflows>` and the {doc}`API reference <api/index>` describe each workflow step and each Python function and class.

## Help and citation

- {doc}`get-help` shows how to read an error message and where to ask a question or report a problem.
- {doc}`cite` gives the citation data of M3Resp and of the EIT and EMG packages.

::::::{div} m3-next-step
:::::{div} m3-next-copy
### From first recording to structured results

The concept guides explain each scientific object. Use the public API to call M3Resp from your own analysis code.

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

:::{button-link} https://github.com/M3RESP/m3resp#documentation
:color: primary
:outline:

Docs on GitHub
:::
::::
:::::

:::::{div} m3-terminal
```bash
# List the available processing steps
m3resp steps

# Run the example multimodal workflow
m3resp run examples/multimodal_example/multimodal.workflow.yaml
```
:::::
::::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Using M3Resp

getting-started
tutorials/index
how-to/index
concepts/index
workflows
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

api/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Help

get-help
cite
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contributing

developer/index
```
