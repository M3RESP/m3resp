# M3Resp Stage 3

Stage 3 is not implemented yet. This page is a rough picture of its destination and boundaries. Treat it as a planning document (a description of intent, not working code yet), not a finished spec (a precise, binding technical description): exact module boundaries (where one folder/package's responsibility ends and another's begins) "may be adjusted during the Stage 3 architecture work" per the plan itself.

Stage 1 ([stage1.md](stage1.md)) made `m3resp` a thin wrapper (a layer of code that mostly just forwards calls to another library, adding little of its own) around `eitprocessing`/`resurfemg`. Stage 2 ([stage2.md](stage2.md)) unified the public surface (the set of names/functions other code is meant to call, as opposed to internal details that can change freely): `M3Session`, native `Signal`/`ParameterResult`/`QualityFlag`/`BreathEvent` types (native meaning implemented directly in this codebase, not borrowed from an outside library), and the declarative workflow engine (a system where you describe *what* steps to run in a YAML/config file, rather than writing the step-by-step Python code yourself). Stage 2 still delegates the actual scientific computation to those two upstream (external, maintained-elsewhere) packages behind adapters (wrapper objects that translate calls from `m3resp` into calls on an outside library). Stage 3 removes that delegation: the EIT and EMG algorithms move into `m3resp` itself, and the project stops depending on `eitprocessing`/`resurfemg` at runtime.

## Goal

Make `m3resp` standalone (able to run fully on its own, without needing another *specialist* package installed): EIT and EMG readers, processing, metrics, quality behavior, and GUI-facing services (the backend functions a graphical user interface calls) all live in `m3resp`, with no production runtime dependency (something the finished, shipped software needs installed to run) on `eitprocessing` or `resurfemg`. The general scientific stack (`numpy`/`scipy`) stays a normal dependency either way; only the two dedicated EIT/EMG packages go away. A GUI can then be built on top of `m3resp` without ever importing either upstream package.

## What "merging EIT and EMG" actually means

The whole point of `m3resp` is to merge EIT and EMG analysis into one integrated tool: one `M3Session` object holds both modalities' data for the same recording, one pipeline engine runs both modalities' steps from the same YAML spec, and one set of shared data types (`Signal`, `ParameterResult`, `QualityFlag`, `BreathEvent`) lets EIT and EMG results be compared, time-aligned, and linked breath-by-breath. That integration is real and is the reason the project exists.

What Stage 3 does *not* do is fuse the two algorithm libraries' source code into one shared implementation. `eitprocessing`'s EIT algorithms and `resurfemg`'s EMG algorithms solve different scientific problems (impedance imaging vs. muscle electrical signals), so once ported natively they still land in separate, modality-owned folders (`src/m3resp/eit/` vs `src/m3resp/emg/`), not merged into one indistinguishable algorithm library. The "merge" is at the framework level (one session, one pipeline engine, one set of data types), not at the level of the two libraries' internal math becoming the same code.

The two libraries do already share a small handful of genuinely modality-neutral building blocks, already factored out into `m3resp.processing` in Stage 2 and reused by both sides:

- `m3resp.processing.filters` - generic Butterworth filtering (lowpass/highpass/bandpass/bandstop), used by both EIT and EMG signal cleanup.
- `m3resp.processing.peaks` - generic peak detection in an array, used for both EIT breath detection and EMG peak/ECG detection.
- `m3resp.processing.windows` - rolling-window and envelope calculations, used by EMG baseline steps and reusable anywhere a moving average/window is needed.
- `m3resp.processing.intervals` - baseline-crossing and onset/offset detection, used to turn a continuous signal into discrete start/end events for a breath or other interval, regardless of modality.
- `m3resp.processing.metrics` - generic per-breath metrics (e.g. integrating a signal over a time interval), once a breath interval already exists, regardless of which modality detected it.

These are the only things that get shared as one common implementation; everything else, vendor readers, breath detection specifics, filtering choices, ROI (region-of-interest) definitions, clinical quality thresholds, stays owned by `eit/` or `emg/` and is not merged between the two.

## What moves, and where

Stage 2 already deliberately keeps modality-specific behavior (code that only makes sense for one kind of data, EIT or EMG, not both) out of generic helpers (shared utility code meant to serve many callers), so the target layout for native code is modality-owned (each folder belongs to and is only about one modality) rather than one big shared "processing" bucket:

```text
src/m3resp/eit/io/           EIT vendor readers (Draeger, Sentec, Timpel)
src/m3resp/eit/processing/   rate, breath, MDN, EELI, TIV, pixel behavior
src/m3resp/eit/roi/          EIT lung-space and ROI behavior
src/m3resp/emg/io/           EMG and ventilator readers
src/m3resp/emg/processing/   filtering, ECG removal, events, metrics, quality
src/m3resp/gui/              GUI built only on public m3resp services
```

Only the genuinely reusable, scientifically-neutral primitives listed above live outside these modality-owned packages. Anything that encodes EIT- or EMG-specific policy (a rule or default choice specific to one modality: vendor quirks, clinical thresholds, ROI definitions) stays under `eit/` or `emg/`, not in a generic (shared, modality-unaware) module (a single Python file/package).

Suggested native port order (the sequence to reimplement things in, since later pieces build on earlier ones), starting with EIT:

1. Vendor loading and normalization (Draeger, Sentec, Timpel; normalization here means putting different vendors' raw formats into one consistent shape).
2. Native EIT containers (the data-holding objects/classes) and global impedance.
3. Rate and global breath detection.
4. Filtering, including MDN.
5. Continuous EELI and TIV.
6. Pixel-breath and pixel-TIV calculations.
7. TIV, amplitude, watershed, and size-filtered ROI behavior.
8. Native plotting/view models needed by the GUI.

EMG follows the same approach in its own Stage 3 plan, sharing only the primitives listed above whose equivalence (producing the same result) with `resurfemg` has already been proven in Stage 2 regression tests (automated tests that re-check old behavior still holds after a change). In both cases, backend operations are replaced one at a time, keeping the public workflow contract (the promised names/inputs/outputs other code relies on) unchanged throughout.

## What stays as-is

Stage 2 built several things specifically so Stage 3 would not need to touch them. See [developer/architecture.md](developer/architecture.md#stage-3-outlook-what-evolves-what-stays-what-goes) for the full breakdown; in short:

- `M3Session`'s public method names and signatures (a signature is a function's name plus its expected inputs/outputs; keeping it the same means existing calling code keeps working unmodified).
- The Layer 1 runtime objects (`Signal`, `ParameterResult`, `QualityFlag`, `BreathEvent`, `LinkedBreath`), meaning the lightweight data objects created fresh each time code runs, as opposed to anything saved to disk.
- `m3resp.workflows`, the declarative step-registry engine (the system, described above, that runs named, YAML-described steps), which stays the canonical (the one official, intended-to-be-used) public module across both Stage 2 and Stage 3.
- The Layer 2 persisted/audit data model (`m3resp.datamodel`): a "persisted" record is one meant to be saved and looked up later; an "audit" record is one built so someone can later verify exactly what was done.
- The provenance schema (a record of what produced a result and how, with a fixed set of fields that record always has), in particular the stable `metadata.operation` identifier (a fixed label naming which operation was run, so tools can rely on it not changing) used by workflows and the GUI.

## Dependency transition

- `eitprocessing`/`resurfemg` stop being a production/runtime extra. A plain `pip install m3resp` and the distributed GUI must not install or import either package.
- The reference packages (the original external libraries, kept around only for comparison purposes) move to a development/reference-test-only extra, used solely to run an optional comparison suite (test group) against the frozen Stage 2 golden fixtures (saved, trusted example inputs/outputs used as the standard to compare new results against).
- Stage 2 characterization fixtures (test data that pins down exactly how the old, upstream-backed behavior worked, so it can be checked for a match later) must remain runnable in Stage 3 without installing the upstream packages at all.

## GUI boundary

The GUI is new in Stage 3 and must:

- Discover operations and controls from the `m3resp` step/capability registry (a lookup list of every available operation and what it needs, that code can query at runtime), not by importing scientific implementation classes directly.
- Call an `m3resp` application/service API, never an adapter or a scientific implementation class.
- Exchange only native data objects and serializable configuration (meaning it can be converted to a plain, portable format like JSON, with nothing exotic like a live database connection or a running process attached).
- Show units, parameter limits, warnings, provenance, and unavailable-input reasons, all supplied by `m3resp`.
- Support loading, preprocessing, event detection, metrics, quality review, visualization, and export for both EIT and EMG.
- Remain testable with synthetic native fixtures (artificially generated sample data), with no upstream package installed.

The GUI must never import from `m3resp.adapters`, `eitprocessing`, or `resurfemg`, and must never receive an upstream `Sequence`, `EITData`, `SparseData`, `PixelMask`, or ReSurfEMG object (all of these are data types belonging to the outside libraries, not to `m3resp`). GUI-specific state (data that only matters for how the screen currently looks, like which button is selected) must not leak back into scientific processing functions (i.e. the science code must stay usable and correct even with no GUI running at all).

### Three sections, all interactive

The GUI is organized into three sections, corresponding to the three stages of a session's lifecycle. All three are fully interactive (able to change session or spec state through the service API), not read-only viewers with an editor bolted onto just one of them:

1. **Data preparation** - load recordings, inspect raw signals, and configure per-session choices (vendor selection, channel/ROI assignment, session metadata) before any pipeline runs. Edits here go through the service API and mutate `M3Session` state the same way any other GUI action does.
2. **Workflow design** - assemble and edit the pipeline spec that will run against the prepared session, using the node-based editor described in ["Node-based workflow design panel"](#node-based-workflow-design-panel) below.
3. **Results review** - inspect metrics, quality flags, provenance, and plots produced by a run, with the ability to edit quality flags, adjust thresholds, and re-trigger the affected downstream steps, rather than only viewing static output.

Because every section supports editing, the same boundary rules apply throughout: every edit is expressed as a call into the `m3resp` service API (never a direct mutation of an adapter or scientific object), every change is discoverable through the step/capability registry, and every result carries the provenance needed to explain what produced it and why re-running is safe. A section being "for review" or "for preparation" does not exempt it from these rules - it changes which service calls are exposed there, not whether the boundary applies.

## Node-based workflow design panel

**The workflow-design section (above) is decided to be a node-based editor; the implementation details below (exact panel layout, phased delivery order) remain open and are not part of Stage 3's completion gate.** This section records the feasibility check behind that decision, so a reviewer can see what the GUI boundary above makes possible without having to re-derive it. No Stage 3 completion-gate item depends on any particular detail below becoming true on any particular timeline; it is written to be self-contained, so everything needed to evaluate the approach is here.

The question asked was whether the workflow-design section could be a *node-based editor* (a canvas where each operation is a box and the connections between boxes show which operation's output feeds which operation's input, as in Blender's shader editor or LabVIEW), built with a fully free front-end library, rather than a conventional form-and-button interface. The answer is yes, and most of the required backend already exists, because a pipeline spec is already a data-flow graph written in list form. See ["Front-end library choice"](#front-end-library-choice) below for which library and why.

### Why the existing engine already fits

Measured against the live registry (60 registered steps as of this writing), the metadata a node editor needs is largely present:

- Every one of the 60 steps declares typed input/output artifacts (`StepArtifact.artifact_type`, roughly 35 distinct types such as `signal`, `index_array`, `roi_mask`). These are exactly the typed connection points ("ports"/"handles") a node editor draws and type-checks.
- 45 of 60 steps declare full static-parameter metadata (`StepParameter`: type, unit, minimum/maximum, choices, default, advanced flag), which is enough to generate a settings panel for a selected node automatically, with no per-operation front-end code.
- `modality` and `category` already group operations for a palette (the menu of available nodes).
- The compiler already performs artifact-type compatibility checking, including the `ANY_ARTIFACT_TYPE` passthrough exemption, so "may these two ports be connected?" is an existing backend question, not a new front-end one.
- `PipelineService` already returns only JSON-safe dictionaries, and already supports live progress (`EventSink`) and cancellation (`CancellationToken`).
- Every `Diagnostic` already carries `step_id`, so a validation error can be attached to the exact node on the canvas with no additional plumbing.

This is a direct consequence of the GUI boundary rules above, not a coincidence: a registry-driven node editor is close to the most literal possible implementation of "discover operations and controls from the `m3resp` step/capability registry".

### How data would move between nodes

In node-editor terminology the values passed along connections are often called *tokens*. Here a token is simply an existing `StepArtifact`: the metadata already attached to every step's inputs and outputs, carrying `artifact_type` plus `unit`, `axes`, `shape_hint`, `required`, `public`, and `compatibility_only`. No new concept is required. Four rules follow from the GUI boundary above:

- **Connection points are typed and colour-coded by `artifact_type`.** The roughly 35 types should be grouped into a smaller number of colour families (signal-like, index/event-like, mask-like, result-like, bundle, path, scalar, and the `any` passthrough); 35 distinct colours would be noise rather than information.
- **Connection validity is checked twice, and the backend is the authority.** The front end can give immediate feedback from the same metadata, but the binding check remains the compiler's existing artifact-type comparison, re-run on save. The front-end check is a convenience, never the source of truth. The `ANY_ARTIFACT_TYPE` passthrough exemption (used by steps such as `eit.slice`, which return whatever type they were given) must be honoured on whichever side declares it.
- **Artifacts marked `compatibility_only` are hidden by default**, behind an explicit toggle. These are the temporary upstream/adapter objects that Stage 3 excludes from the public result contract; showing them by default would invite building pipelines against them.
- **Tokens are connection metadata, not payloads.** What crosses to the front end is the type/shape summary already produced by `summarize_output_value()` - never a NumPy array, never an upstream `Sequence`/`EITData`/ReSurfEMG object. This is the same JSON-safe contract `PipelineService` already enforces, and the UI must not bypass it. Displaying actual signal data for plotting would need a separate, explicit, downsampled preview call, kept distinct from the graph representation.

### Two findings that constrain any such UI

These are properties of the current engine that a reviewer should be aware of independently of whether the UI is ever built.

**1. Connections must be resolved by position, not by matching names.** Steps communicate through a shared blackboard (a single namespace of named values that steps read from and write to) rather than through explicit wiring, and a spec may rebind the same name to a different value partway through a run. In [../examples/multimodal_full/multimodal-full.pipeline.yaml](../examples/multimodal_full/multimodal-full.pipeline.yaml) the key `processed_emg` refers to one value before the ECG-gating step and a different value after it. Drawing a connection wherever two steps mention the same name would therefore produce wrong connections. The correct rule, which `collect_diagnostics()` in the engine already implements while validating, is that each read binds to the *most recent preceding* writer of that name. Any graph conversion must reuse that existing logic rather than reimplement it, so that the drawn graph can never disagree with validation.

**2. `session` is an undeclared dependency channel.** 39 of the 60 steps take the `M3Session` as an input, and in the multimodal example only about 10 of roughly 45 steps declare an explicit `in:` binding at all; the rest coordinate by mutating shared session state. Two consequences: rendered literally, the canvas would show one `session` node connected to 39 others, obscuring the meaningful structure; and, more importantly, genuine ordering constraints between session-mutating steps are expressed nowhere in the spec, so **reordering such steps into a broken sequence cannot currently be caught at compile time**.

The second point is worth noting on its own merits, separate from any UI. Adding optional `session_reads`/`session_writes` declarations to `register_step` would make those hidden dependencies explicit and checkable. That would be additive, backward-compatible metadata, following the same pattern by which `parameters` and `input_artifacts` were backfilled during Stage 2, and it would benefit the engine whether or not a node UI is ever built. It is *not* proposed as Stage 3 work here, only recorded as an option.

### What is missing

- There is no spec writer. `load_spec()` parses YAML/JSON into a `PipelineSpec`; nothing serializes a `PipelineSpec` back out. Editing a graph and saving it requires one.
- Because session-mediated ordering constraints are undeclared (finding 2), converting a graph back to an ordered step list cannot rely on topological sorting (ordering purely by data dependencies) alone; the author's step order would have to be preserved explicitly until those dependencies are declared.
- The 15 steps still lacking parameter metadata would need backfilling before a UI could offer them for editing safely.

### Shape of the work, if it were ever taken on

The substantive deliverable is a backend module, not a front end. A spec-to-graph and graph-to-spec conversion layer (nodes carrying operation id, static parameters, and opaque canvas coordinates; connections carrying source and target node, the two connection point names, and the context key the value flowed through) would be pure Python, JSON-safe, and testable with no user interface present. It would reuse the engine's existing producer-tracking rather than restate it, per finding 1. Alongside it, the missing spec writer must keep `@name` input references unresolved and relative paths relative to the spec root, and would lose the extensive hand-written comments in the example specs unless a comment-preserving YAML library were adopted - so it should write to a new file rather than overwrite an authored one. Canvas coordinates belong in the spec's existing free-form `metadata` block, so that a hand-written spec with no such block still opens (falling back to automatic layout).

Exposing this to a front end needs only a thin local HTTP or IPC layer over `PipelineService`: list capabilities, validate, compile, run with a progress stream, and import/export a spec. That layer belongs in `src/m3resp/gui/` and is bound by the same prohibition as the rest of the GUI - it must never import `m3resp.adapters`, `eitprocessing`, or `resurfemg`, and an import test should enforce that, as the completion gate already requires.

### Front-end library choice

[React Flow](https://reactflow.dev) (the `xyflow` project) was the first library considered: its core is MIT-licensed, but the project sells a paid "Pro" subscription (templates, some examples, priority support) alongside the free core, which is a bad fit for a project that wants no paywalled tier anywhere in its toolchain, even an optional one. The alternative chosen instead is [`@projectstorm/react-diagrams`](https://github.com/projectstorm/react-diagrams): MIT-licensed end to end, no Pro tier, no paid plugins, React- and TypeScript-native, and actively maintained. It provides the same building blocks a node editor needs - a diagram engine, typed node/port/link models, a canvas widget, and hooks for custom node/link rendering and connection validation - so the mapping onto the registry metadata described above (typed connection points, a settings panel generated from `StepParameter`, click-to-focus diagnostics) carries over unchanged; only the library name changes, not the design. If `react-diagrams`' maintenance pace or feature set turns out not to hold up during implementation, [Rete.js](https://retejs.org) is the fallback - its core framework is also MIT, though a handful of its *advanced* plugins are CC-BY-NC-SA (non-commercial) and would have to be avoided or reimplemented.

A conventional React and TypeScript build would surround the canvas with four panels: an operation palette grouped by `modality` and `category` and greyed out by capability state, the canvas itself, a settings panel generated from `StepParameter` (including `mutually_exclusive_parameters` as mutually-exclusive groups), and a problems list driven by `Diagnostic` with click-to-focus on the offending node. TypeScript types should be generated from the existing dataclasses rather than hand-written, so the two sides cannot drift. The hard rule is that no scientific logic, default, unit, or threshold may live in the front end: if the UI needs something the registry cannot tell it, that indicates missing registry metadata, not a missing front-end feature.

A sensible order of work would be: the conversion module and its round-trip test first, with no UI at all; then a read-only viewer that renders an existing spec; then editing; then run execution with live per-node status; then per-node result preview. The first increment is the honest go/no-go test - if the round trip cannot be made to hold over the existing examples, the graph model is wrong and no front-end library choice would compensate. It also has standalone value, since a graph view of a 45-step multimodal spec is useful for documentation and provenance regardless of whether an editor is ever built.

### Questions that would need answering first

None of these are blocking anything today, but a reviewer should know they are open rather than settled:

- **Comment preservation on save,** and whether that justifies an additional YAML dependency.
- **Metadata completeness.** The 15 steps without parameter metadata would have to be backfilled before an editor could offer them safely. A node editor makes it easy to assemble pipelines that are plausible but invalid, so the registry's constraint metadata becomes load-bearing in a way it is not today.

## Completion gate

Stage 3 is done only when:

- The full EIT, EMG, and multimodal (multiple data types combined, here EIT + EMG together) workflows pass using native implementations.
- Characterization tests demonstrate parity (an exact or tolerance-bounded match) with the frozen Stage 2 reference behavior.
- No production source import or dependency (no line of shipped code that imports or requires) references `eitprocessing` or `resurfemg`.
- The GUI runs end to end (from first user action through to a finished result, with every step in between working) against native `m3resp` services only.

## Out of scope / not yet decided

- The exact internal folder layout under `eit/`/`emg/` may still change during Stage 3 architecture work; the ownership boundaries (which package is responsible for which behavior) are the firm part, not the file paths.
- Whether `EITProcessingAdapter`/`ReSurfEMGAdapter` are deleted outright or kept as a thin regression/reference-comparison harness (a minimal test setup built only to run comparisons, not part of the real product) is undecided; adapter injection (passing in a specific adapter object, typically for testing) is expected to remain available for regression tests even after production code stops calling it.
- Stage 3 does not fuse `eitprocessing`'s and `resurfemg`'s algorithms into one shared implementation, see ["What 'merging EIT and EMG' actually means"](#what-merging-eit-and-emg-actually-means) above; it ports each modality's validated behavior into its own `eit`/`emg` module, sharing only the small set of scientifically-neutral primitives listed there.
- Whether `m3resp.processing` primitives (`filters`, `windows`, `ecg`, ...) gain a `Signal`/`TimeSeries`-accepting convenience layer, so a caller can pass one object instead of an array plus `sample_frequency` separately, is undecided. Today these stay deliberately dependency-free (no import of `m3resp.data`), since most current call sites operate on raw arrays - sometimes upstream `eitprocessing`/`resurfemg` objects - before any `Signal`/`TimeSeries` exists; revisit this once Stage 3's native EIT/EMG containers make it clearer what's actually available at each call site. If this does get built, the shape discussed on [PR #24](https://github.com/M3RESP/m3resp/pull/24) is a thin adapter *on top of* the existing low-level functions (e.g. `apply_to_signal(lowpass_filter, signal)` or a `signal_handler_wrapper` decorator) living in `m3resp.data`, which already depends on `m3resp.processing` - not a change to the low-level functions themselves, which would reintroduce the upward dependency this entry rules out.

## See also

- [stage1.md](stage1.md) - the wrapper layer Stage 3 eventually removes the runtime dependency on.
- [stage2.md](stage2.md) - the contracts and native types Stage 3 builds on.
- [developer/architecture.md](developer/architecture.md) - package map and a detailed per-piece Stage 3 evolution table.
