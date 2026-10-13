# Stage 3 GUI mockup

A static HTML mockup of the Stage 3 desktop GUI, with four tabs: 1 · Prepare, 2 · Design, 3 · Check and 4 · Results. It shows the planned workflow before the real GUI exists. The page opens straight from disk over `file://` and needs no server. All state lives in the page, so a reload starts it again. The only exception is the choice "Show this card when 2 · Design opens", which the browser keeps.

## Open it

```bash
open tools/mockup_tool/stage3_gui_mockup.html        # macOS
xdg-open tools/mockup_tool/stage3_gui_mockup.html    # Linux
```

A double-click on the file also works.

`stage3_gui_mockup.html` links the files under `src/`, so the mockup is a folder. To give it to someone, send the whole `tools/mockup_tool/` folder or a zip of it. The HTML file alone shows an empty page.

## Real and illustrative parts

These parts come from m3resp itself:

- The default graph on 2 · Design follows `examples/*/multimodal-full.pipeline.yaml`: 15 of its 45 steps.
- The step list of the palette, and the summaries, parameters, units, limits, choices and ports in the help box, the Inspector and the node window, come from the step registry (`describe_steps()`). The file `src/js/data/step-info.js` holds them; see "Step list from the registry" below.
- The EIT images and impedance curves on 1 · Prepare are computed from the `.bin` recording under `data/source/synthetic/` with `eitprocessing.datahandling.loading.load_eit_data`.
- 3 · Check and 4 · Results show the output files of a past `multimodal-full-summary` run (`run_manifest.json`, `parameter_results.csv`, `quality_flags.csv` and others).

These parts are illustrative:

- EMG and ventilator waveforms, quality flag numbers, and the data previews on 2 · Design (in the node window and in Available data). A change of a setting gives a different preview shape, but the shape does not show the real effect of that setting.
- The run on 2 · Design. The steps run one after the other with a fixed time each. A step fails when none of its inputs is linked; `emg_ecg_gating` always gives the warning seen in a real run.
- The ports of the 15 steps on the default graph come from a hand-written list (`STEP_DEFS` in `graph-data.js`). For many EIT steps this list has a named signal input where the registry reads the signal through the session. Steps added from the palette get the registry ports.
- The two entries under Recent on the welcome card, and the greyed-out step `emg.ecg_wavelet_denoising` (an example of a missing optional package).

## What each tab does

**1 · Prepare.** Load a file per modality ("Load file…" opens a real file picker), set the raw synchronization offsets, and cut a working window out of the recording. Named sequences of time windows go into a library that 2 · Design can use.

**2 · Design.** A node editor over the step registry, after the canvas of Orange Data Mining.

- Each step is a circle in its modality colour, with a category icon and its name under it. Inputs are dots on the left edge, outputs dots on the right edge, and each link shows the name of its data. "Detailed" in the toolbar shows each step as a card with all port names; the cards replace the link labels.
- The palette on the left has two halves. The upper half, Steps, has two tabs: Operations (all registry steps) and Workflows (ready-made chains of steps). The lower half, Available data, lists the data made on the canvas and the sequences saved in 1 · Prepare. Each list has a search box, a switch between icons and a list, and sections that open one at a time.
- The right panel has three parts, from the top: Help (explains the palette item under the mouse), Edge legend (what the link colours, arrows, ports, step colours and run badges mean) and the Inspector.
- A click on the head of a part (Steps, Available data, Help, Edge legend, Inspector) closes or opens it. Drag the bar under an open part to change its size against the next open part; a double-click on the bar makes the two equal. One open part alone fills its panel.
- A click on a palette step adds it next to the selected step; a drag puts it where it is dropped. A double-click or right-click on empty canvas opens a quick menu of steps. A link dropped on empty canvas opens the same menu with only the steps that take that data type, and the new step arrives connected. A link dropped on a step with several matching inputs asks which input to use.
- A click on a step shows its parameters in the Inspector. A double-click opens the node window: parameters on the left, a preview of an output and the linked inputs on the right. With "Apply automatically" on, each change goes to the step at once; with it off, changes wait for "Apply".
- "▶ Run pipeline" runs the steps in graph order with a progress ring on each step and a line per step in the Run log. Badges show done, warning, failed and out of date. A change of a setting or a link makes the step and every step after it out of date; with "Auto-run" on, those steps run again at once. "Validate" fills the Problems list without a run, and a click on a problem shows its step.
- Steps never land on top of each other. A new step goes to the nearest free place, an opened workflow pushes the other steps aside, and a dropped step slides to a free place. A workflow from the Workflows tab arrives as one box; a click opens it, and "Pack" on its frame closes it again.
- "⇶ Arrange" puts the steps in columns by graph order, with one lane per modality. "Examples…" opens the welcome card: empty canvas, open a pipeline file, the multimodal-full example, recent files.
- ✎ adds a note and ↗ draws an arrow on the canvas. Notes and arrows move and zoom with the canvas.
- "?" in the zoom controls lists the mouse and key actions.

**3 · Check.** Quality flags and the run report. "Confirm & continue" opens 4 · Results.

**4 · Results.** Parameters, saved values and plots.

## Step list from the registry

`src/js/data/step-info.js` holds the name, summary, modality, category, inputs, outputs, parameters and optional packages of every registered step. Make it again after a change to the step registry. Run this from the repository root, in the environment where m3resp is installed:

```bash
.venv/bin/python tools/mockup_tool/export_step_info.py
```

## Developing

Edit the files under `src/`; `stage3_gui_mockup.html` is made from them by `build.js`, so changes made by hand in that file are lost at the next build. The HTML file links the CSS and JS files, so after a change to `src/css/` or `src/js/` a browser reload is enough. Run the build after a change to `src/html/`, or after adding, removing or moving a source file:

```bash
node build.js
```

`build.js` uses only Node built-ins. It writes the `<link>` and `<script src>` tags in the order of its file lists and copies `src/html/**` into the page. The scripts are classic scripts that share one global scope, so the order of `JS_FILES` matters: code that runs while a file loads can use only what earlier files declared. Code that runs later, after a click for example, can use any file.

```
src/
  css/     base, prepare (four files), design, popover, results
  html/    one file per tab: shell, prepare, design, check, results
  js/data/ eit-frame-data.js, review-figures.js (data as global variables),
           step-info.js (made by export_step_info.py)
  js/core/ tabs.js: tab switching, the Check → Results gate
  js/design/
    graph-data.js       default graph, step data helpers, node geometry
    selection.js        multi-select and box select
    canvas-zoom.js      zoom, drawing of nodes, Compact / Detailed switch
    node-drag.js        node drag, links, workflow groups
    layout.js           free places, push aside, glide animation
    arrange.js          "Arrange"
    annotations.js      notes and arrows
    palette-common.js   parts shared by the three palette lists
    palette-panel.js    adding a step, the Available data panel
    help-box.js         help box in the right panel
    palette-panes.js    parts of both side panels: open and close, sizes
    toolbox.js          the Operations tab
    data-preview.js     data previews
    param-form.js       parameter form of the Inspector and the node window
    node-window.js      node window
    workflow-blocks.js  the Workflows tab, inserting a workflow
    connections.js      links by drag, clear and load, Inspector
    quick-menu.js       quick menu and link chooser
    run-state.js        run, Auto-run, Validate, Problems, Run log
    welcome.js          welcome card
  js/prepare/, js/results/  code of the other tabs
```

The page uses classic scripts and loads its data as `.js` files that declare global variables, because browsers block `fetch()` and `type="module"` scripts over `file://`.

The generated `stage3_gui_mockup.html` is committed, so the mockup opens without a build.

### Checking a change

The mockup has no test suite. Check that the page shows no console errors, and that the tabs you did not change render the same DOM as before:

```bash
node build.js
google-chrome --headless=new --disable-gpu --dump-dom \
  "file://$PWD/stage3_gui_mockup.html" > /tmp/after.html
```

Make the same dump before the change and compare the two. Remove `<script>`, `<style>`, `<link>` and comment nodes from both first, so the comparison looks at the rendered content only. For 2 · Design, click through the changed feature in a browser.
