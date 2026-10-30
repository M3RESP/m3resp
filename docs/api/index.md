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
   m3resp.interpolate_to_time
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
   m3resp.PixelBreaths
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

## Read EIT files

These functions read one EIT file of a Draeger, Sentec or Timpel device. Each returns an `EITFileData` with the images, the waveforms, the marks of the device and, for Timpel, the breaths found by the device. `read_eit_file` picks the reader from the device maker, and `join_eit_files` joins the data of files recorded one after the other.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.eit.io.read_eit_file
   m3resp.eit.io.read_draeger
   m3resp.eit.io.read_sentec
   m3resp.eit.io.read_timpel
   m3resp.eit.io.EITFileData
   m3resp.modalities.eit.EITRecording
   m3resp.eit.io.join_eit_files
```

## EIT calculations

`global_impedance` sums the pixels of each EIT image into the global impedance waveform. `regional_impedance` and `functional_impedance` sum the pixels inside a mask. Each of these three functions returns a `Signal`. `detect_rates` finds the breathing rate and the heart rate of the EIT images, in Hz, from their frequency spectra. `find_breaths` finds the breaths of an impedance waveform and returns them as `BreathEvent`s. `RateDetectionSettings` and `BreathDetectionSettings` hold the settings of these two functions. `mdn_filter` removes the heart signal from EIT images or a waveform with the MDN (multiple digital notch) filter and returns a `Signal`; `mdn_frequency_bands` gives the frequency bands that it removes. `tidal_impedance_variation` and `end_expiratory_lung_impedance` give the TIV and the EELI of each breath of an impedance waveform, as an `IntervalData`; the breaths are `BreathEvent`s of any modality, on the time axis of the waveform. `TIV_METHODS` lists the three TIV methods. `find_pixel_breaths` finds the breaths of each pixel of EIT images from the global breaths and returns a `PixelBreaths`. `pixel_tidal_impedance_variation` and `pixel_end_expiratory_lung_impedance` give one `PixelMap` of TIV or EELI per breath, with the times of the global breaths or of the breaths of each pixel. `minute_image` gives the mean pixel TIV over the breaths inside a time window.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.eit.processing.global_impedance
   m3resp.eit.processing.regional_impedance
   m3resp.eit.processing.functional_impedance
   m3resp.eit.processing.detect_rates
   m3resp.eit.processing.RateDetectionSettings
   m3resp.eit.processing.find_breaths
   m3resp.eit.processing.BreathDetectionSettings
   m3resp.eit.processing.mdn_filter
   m3resp.eit.processing.mdn_frequency_bands
   m3resp.eit.processing.tidal_impedance_variation
   m3resp.eit.processing.end_expiratory_lung_impedance
   m3resp.eit.processing.find_pixel_breaths
   m3resp.eit.processing.pixel_tidal_impedance_variation
   m3resp.eit.processing.pixel_end_expiratory_lung_impedance
   m3resp.eit.processing.minute_image
   m3resp.eit.processing.slicing.slice_file_data
   m3resp.processing.slicing.slice_by_index
   m3resp.processing.slicing.slice_by_time
   m3resp.processing.slicing.time_slice_indices
```

## EIT regions of interest

These functions return a `PixelMask` with 1 for the pixels of a region and NaN for the other pixels. `geometric_mask` gives a fixed part of the image: the ventral or dorsal half, the anatomical right or left half, one of four layers or one of four quadrants (`GEOMETRIC_REGIONS` lists the 12 names). `tiv_lungspace`, `amplitude_lungspace` and `watershed_lungspace` find the functional lung space from the global breaths: the pixels with a large mean TIV, the pixels with a large mean amplitude, or both together with the watershed method. `filter_mask_by_size` keeps the groups of connected pixels of a mask that have a minimum number of pixels. See {doc}`../concepts/pixel-maps`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.eit.roi.geometric_mask
   m3resp.eit.roi.tiv_lungspace
   m3resp.eit.roi.amplitude_lungspace
   m3resp.eit.roi.watershed_lungspace
   m3resp.eit.roi.filter_mask_by_size
```

## Plots

These functions make figures with Matplotlib.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.visualization.plot_eit_processing_summary
   m3resp.visualization.plot_rate_detection
   m3resp.visualization.plot_session_overview
   m3resp.visualization.plot_synchronization_comparison
```

## Pixel map pictures

`plot_pixel_map` draws a `PixelMap` or a `PixelMask` with Matplotlib, in the style of its kind, with a colour bar. `add_region_labels` writes the region numbers on a map of kind `"integer"`. `pixel_map_style` gives the style of a kind: the colours, the rule for the ends of the colour scale and the colour bar. `scale_limits` and `pixel_colors` give the ends of the colour scale and the colour of each pixel without Matplotlib. See {doc}`../concepts/pixel-maps`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.visualization.pixel_maps.plot_pixel_map
   m3resp.visualization.pixel_maps.add_region_labels
   m3resp.visualization.pixel_map_style.PixelMapStyle
   m3resp.visualization.pixel_map_style.pixel_map_style
   m3resp.visualization.pixel_map_style.style_of
   m3resp.visualization.pixel_map_style.scale_limits
   m3resp.visualization.pixel_map_style.pixel_colors
```

## Plot data for the GUI

These functions give EIT results as dictionaries of numbers, text, lists and None, which `json.dumps` writes as they are. A NaN pixel is None. Each picture comes with its style.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   m3resp.visualization.plot_data.pixel_map_plot_data
   m3resp.visualization.plot_data.pixel_mask_plot_data
   m3resp.visualization.plot_data.eit_image_plot_data
   m3resp.visualization.plot_data.eit_images_plot_data
   m3resp.visualization.plot_data.lungspace_plot_data
   m3resp.visualization.plot_data.rate_spectrum_plot_data
   m3resp.visualization.plot_data.region_label_positions
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
