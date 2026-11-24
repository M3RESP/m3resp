---
html_theme.sidebar_secondary.remove: true
sd_hide_title: true
---

# M3Resp: respiratory signal integration for multimodal analysis

<div id="hero">

<div id="hero-left">
  <h2>M3Resp</h2>
  <h3>Respiratory signal integration for multimodal analysis</h3>
  <p>M3Resp is a Python library that brings electrical impedance tomography (EIT), respiratory muscle EMG and ventilator recordings into one workflow. It synchronizes the recordings, finds the breaths in each signal and calculates breath parameters.</p>

<div class="homepage-button-container">
  <div class="homepage-button-container-row">
      <a href="getting-started.html" class="homepage-button primary-button">Get Started</a>
      <a href="tutorials/index.html" class="homepage-button secondary-button">See Tutorials</a>
  </div>
  <div class="homepage-button-container-row">
      <a href="api/index.html" class="homepage-button-link">See API Reference →</a>
  </div>
</div>
</div>

<div id="hero-right">

::::::{grid} 1 2 2 2
:gutter: 3

:::::{grid-item-card}
:link: tutorials/10_eit_only
:link-type: doc
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Breaths found in the EIT global impedance
:::

:::{image} _static/miniatures/eit_global_impedance.png
:alt: EIT global impedance with a green line at each breath that M3Resp found.
:::
:::::

:::::{grid-item-card}
:link: concepts/pixel-maps
:link-type: doc
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Map of the mean tidal impedance variation (TIV) of each EIT pixel
:::

:::{image} _static/miniatures/eit_pixel_tiv_map.png
:alt: Map of the mean TIV of each EIT pixel, with the two lungs in light blue and white.
:::
:::::

:::::{grid-item-card}
:link: concepts/parameters
:link-type: doc
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
TIV of each breath during the recording
:::

:::{image} _static/miniatures/eit_tiv.png
:alt: TIV of each breath against time.
:::
:::::

:::::{grid-item-card}
:link: tutorials/20_emg_only
:link-type: doc
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Raw EMG with the heart signal and the breaths that M3Resp found
:::

:::{image} _static/miniatures/emg_raw.png
:alt: Raw EMG signal with heart beats and a red line at each breath.
:::
:::::

:::::{grid-item-card}
:link: tutorials/20_emg_only
:link-type: doc
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
EMG envelope after the heart signal is removed
:::

:::{image} _static/miniatures/emg_envelope.png
:alt: EMG envelope with one peak at each breath and a red line at each breath.
:::
:::::

:::::{grid-item-card}
:link: how-to/synchronize-recordings
:link-type: doc
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Ventilator pressure on the shared clock of the session
:::

:::{image} _static/miniatures/ventilator_pressure_synchronized.png
:alt: Ventilator airway pressure after synchronization.
:::
:::::
::::::

</div>
</div>

# Key Features

:::::{grid} 1 1 2 2
:gutter: 5

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/features/feature_recordings.svg
:alt: Four coloured shapes.
:::

:::{div} key-features-text
<strong>Three Recording Types</strong><br/>
Loads EIT, respiratory muscle EMG and ventilator recordings into one {doc}`session <concepts/session>`.
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/features/feature_synchronization.svg
:alt: Three signals with dashed lines at the same times.
:::

:::{div} key-features-text
<strong>One Shared Clock</strong><br/>
Puts the recordings on one time axis, so that you can compare the breaths of each signal. See {doc}`concepts/synchronization`.
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/features/feature_cleaning.svg
:alt: A noisy signal above a clean signal.
:::

:::{div} key-features-text
<strong>Signal Cleaning</strong><br/>
Removes the heart signal (ECG) from the EMG, filters each signal and calculates the EMG envelope.
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/features/feature_breaths.svg
:alt: A breathing signal with a mark at the start of each breath.
:::

:::{div} key-features-text
<strong>Breaths and Parameters</strong><br/>
Finds the breaths in each signal and calculates {doc}`breath parameters <concepts/parameters>`, for example TIV and the end-expiratory lung impedance (EELI).
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/features/feature_pixel_maps.svg
:alt: A grid of coloured squares with two bright areas.
:::

:::{div} key-features-text
<strong>EIT Pixel Maps</strong><br/>
Calculates EIT parameters for each pixel and shows them as a {doc}`map of the lungs <concepts/pixel-maps>`.
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/features/feature_workflows.svg
:alt: Three steps in a column above three result files.
:::

:::{div} key-features-text
<strong>Workflow Files</strong><br/>
Runs the steps of a {doc}`YAML workflow file <workflows>` from the command line and writes the results to files, with a {doc}`record of each step <concepts/provenance>`.
:::
::::
:::::

# Support M3Resp

::::{grid} 1 1 2 2

:::{grid-item}

<h3>Contributions</h3>

Contributions and bug reports are welcome at [the GitHub repository](https://github.com/M3RESP/m3resp). The {doc}`developer guide <developer/index>` helps you through the process. For questions, see {doc}`get-help`.
:::

:::{grid-item}

<h3>Citation</h3>

If you use M3Resp in a paper, please cite it. See {doc}`cite` for the citation in APA and BibTeX format.
:::
::::

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
