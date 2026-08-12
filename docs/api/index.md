# Public API

This reference covers names intentionally exported for researchers and
application authors. Modules and helpers that are not listed here remain
internal and may change without notice.

## Sessions and pipelines

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.M3Session
   m3resp.Pipeline
   m3resp.EITPipeline
   m3resp.EMGPipeline
   m3resp.MultimodalPipeline
   m3resp.PipelineResult
   m3resp.available_pipelines
   m3resp.get_pipeline
   m3resp.register_pipeline
   m3resp.load_spec
   m3resp.run_pipeline
   m3resp.run_spec
   m3resp.available_steps
   m3resp.register_step
   m3resp.workflows.PipelineSpec
   m3resp.workflows.StepSpec
   m3resp.workflows.ValidationReport
   m3resp.workflows.CompiledPipeline
   m3resp.workflows.PipelineService
   m3resp.workflows.collect_diagnostics
   m3resp.workflows.compile_pipeline
   m3resp.workflows.validate_pipeline
   m3resp.workflows.validate_spec
```

## Signals, results, and quality

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

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.Event
   m3resp.BreathEvent
   m3resp.LinkedBreath
   m3resp.coerce_event
   m3resp.coerce_breath_event
   m3resp.coerce_breath_events
   m3resp.event_to_dict
   m3resp.compute_offsets_from_timestamps
   m3resp.link_breaths_by_time
   m3resp.resample_signal
```

## Persisted data model

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

## Modality loaders

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.io.load_eit
   m3resp.io.load_emg
```

## Synthetic data

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
