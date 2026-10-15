# Public API

This page lists the classes and functions that you import from `m3resp`. Each name links to a page with its arguments, its return value and the units. The page of a class or function also lists the tutorials that use it.

See the {doc}`../concepts/index` for what the objects mean.

## Sessions and workflows

An `M3Session` holds the recordings and the results of one analysis. The presets and the workflow functions run the processing steps.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.M3Session
   m3resp.Preset
   m3resp.EITPreset
   m3resp.EMGPreset
   m3resp.MultimodalPreset
   m3resp.WorkflowResult
   m3resp.available_presets
   m3resp.get_preset
   m3resp.register_preset
   m3resp.load_spec
   m3resp.run_workflow
   m3resp.run_spec
   m3resp.available_steps
   m3resp.register_step
   m3resp.workflows.WorkflowSpec
   m3resp.workflows.StepSpec
   m3resp.workflows.ValidationReport
   m3resp.workflows.CompiledWorkflow
   m3resp.workflows.WorkflowService
   m3resp.workflows.collect_diagnostics
   m3resp.workflows.compile_workflow
   m3resp.workflows.validate_workflow
   m3resp.workflows.validate_spec
```

## Signals, results and quality

These objects hold the processed signals, the calculated parameters, the quality checks and the record of each processing step.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.TimeSeries
   m3resp.Signal
   m3resp.SignalCollection
   m3resp.ParameterResult
   m3resp.ParameterResultCollection
   m3resp.QualityFlag
   m3resp.QualityReport
   m3resp.ProcessingStep
   m3resp.ProcessingHistory
```

## Events and synchronization

These objects and functions hold breaths, events and values per breath, and put the recordings on one clock.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.Event
   m3resp.Interval
   m3resp.BreathEvent
   m3resp.EventData
   m3resp.IntervalData
   m3resp.IntervalDataCollection
   m3resp.PixelMap
   m3resp.PixelMask
   m3resp.PixelMaskCollection
   m3resp.pixel_breath_events
   m3resp.LinkedBreath
   m3resp.coerce_event
   m3resp.coerce_events
   m3resp.coerce_interval
   m3resp.coerce_intervals
   m3resp.coerce_breath_event
   m3resp.coerce_breath_events
   m3resp.event_to_dict
   m3resp.compute_offsets_from_timestamps
   m3resp.link_breaths_by_time
   m3resp.resample_signal
```

## Data model

The data model is a set of checked records of a finished analysis. See {doc}`../concepts/provenance`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.Case
   m3resp.ClinicalEvent
   m3resp.DataFile
   m3resp.DataModelRecorder
   m3resp.DataModelStore
   m3resp.DerivedFeature
   m3resp.Device
   m3resp.ProcessingRun
   m3resp.QualityAnnotation
   m3resp.RecordingSession
   m3resp.SignalStream
   m3resp.export_store
   m3resp.validate_store
```

## Export

This function writes the results of a session to CSV, JSON and array files.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.export.export_session_summary
```

## Load a recording

These functions load one recording without a session.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.io.load_eit
   m3resp.io.load_emg
```

## Plots

These functions make figures with Matplotlib.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.visualization.plot_eit_processing_summary
   m3resp.visualization.plot_session_overview
   m3resp.visualization.plot_synchronization_comparison
```

## Synthetic data

These classes and functions make synthetic EIT, EMG and ventilator recordings for tests and examples.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.synthetic.SyntheticGeneratorConfig
   m3resp.synthetic.SyntheticDataset
   m3resp.synthetic.SyntheticRecord
   m3resp.synthetic.RespiratoryPatternConfig
   m3resp.synthetic.EITGeneratorConfig
   m3resp.synthetic.EMGGeneratorConfig
   m3resp.synthetic.VentilatorGeneratorConfig
   m3resp.synthetic.generate_synthetic_dataset
   m3resp.synthetic.generate_realistic_eit_signal
   m3resp.synthetic.load_synthetic_generator_config
```
