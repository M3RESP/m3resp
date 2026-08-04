"""Coverage for the Phase 1 step metadata backfilled onto ``emg.py`` (see
plan/stage2/3_pipeline_structure_implementation_plan.md, "Concrete Next
Actions" item 4: emg.py is the last and largest module in the sequence).

Checks the declared metadata against the real code it describes, not just
that a description exists.
"""

from __future__ import annotations

import json

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.workflows.registry import describe_step, describe_steps

_EMG_STEP_NAMES = [
    "emg.load",
    "emg.preprocess",
    "emg.detect_breaths",
    "emg.peak_indices",
    "emg.moving_baseline",
    "emg.slopesum_baseline",
    "emg.ecg_detect_peaks",
    "emg.ecg_estimated_subtraction",
    "emg.ecg_gating",
    "emg.ecg_wavelet_denoising",
    "emg.interpeak_dist",
    "emg.onoffpeak_baseline_crossing",
    "emg.onoffpeak_slope_extrapolation",
    "emg.time_to_peak",
    "emg.pseudo_slope",
    "emg.amplitude",
    "emg.time_product",
    "emg.area_under_baseline",
    "emg.respiratory_rate",
    "emg.snr_pseudo",
    "emg.percentage_under_baseline",
    "emg.detect_local_high_aub",
    "emg.detect_extreme_time_products",
    "emg.evaluate_bell_curve_error",
    "emg.evaluate_event_timing",
    "emg.evaluate_respiratory_rates",
]

#: Ventilator steps, split out of the `emg.*` namespace once the
#: ventilator became a modality in its own right. Their functions still
#: live in `workflows/steps/emg/`; only the registered ids moved.
_VENTILATOR_STEP_NAMES = [
    "ventilator.load",
    "ventilator.channels",
    "ventilator.detect_breaths",
    "ventilator.find_occluded_breaths",
    "ventilator.pocc_intervals",
    "ventilator.pocc_time_product",
    "ventilator.pocc_quality",
    "ventilator.respiratory_rate",
    "ventilator.detect_non_consecutive_manoeuvres",
    "ventilator.normalize_breaths",
]

_ALL_STEP_NAMES = _EMG_STEP_NAMES + _VENTILATOR_STEP_NAMES

_QUALITY_STEP_NAMES = [
    "ventilator.pocc_quality",
    "emg.interpeak_dist",
    "emg.snr_pseudo",
    "emg.percentage_under_baseline",
    "emg.detect_local_high_aub",
    "emg.detect_extreme_time_products",
    "ventilator.detect_non_consecutive_manoeuvres",
    "emg.evaluate_bell_curve_error",
    "emg.evaluate_event_timing",
    "emg.evaluate_respiratory_rates",
]


def test_all_thirty_six_emg_and_ventilator_steps_are_registered_and_described():
    emg = {d.name for d in describe_steps(prefix="emg.")}
    ventilator = {d.name for d in describe_steps(prefix="ventilator.")}
    assert emg == set(_EMG_STEP_NAMES)
    assert ventilator == set(_VENTILATOR_STEP_NAMES)
    assert len(_ALL_STEP_NAMES) == 36


def test_every_emg_step_declares_some_metadata_and_is_json_safe():
    for name in _ALL_STEP_NAMES:
        description = describe_step(name)
        assert description.description, name
        assert (
            description.parameters
            or description.output_artifacts
            or description.input_artifacts
        ), name
        json.dumps(description.as_dict())


def test_ten_quality_steps_are_all_categorized_quality():
    for name in _QUALITY_STEP_NAMES:
        assert describe_step(name).category == "quality", name


def test_ecg_removal_alternatives_are_mutually_cross_referenced():
    gating = describe_step("emg.ecg_gating")
    wavelet = describe_step("emg.ecg_wavelet_denoising")
    ees = describe_step("emg.ecg_estimated_subtraction")
    assert set(gating.alternatives) == {
        "emg.ecg_wavelet_denoising",
        "emg.ecg_estimated_subtraction",
    }
    assert set(wavelet.alternatives) == {
        "emg.ecg_gating",
        "emg.ecg_estimated_subtraction",
    }
    assert set(ees.alternatives) == {"emg.ecg_gating", "emg.ecg_wavelet_denoising"}


def test_ecg_estimated_subtraction_is_native_not_resurfemg():
    # Unlike the other ECG-removal steps, EES is m3resp-native (see
    # emg.py's _upstream_metadata(source_package="m3resp", ...) call site).
    assert describe_step("emg.ecg_estimated_subtraction").optional_packages == ()
    for name in ("emg.ecg_gating", "emg.ecg_wavelet_denoising", "emg.ecg_detect_peaks"):
        assert describe_step(name).optional_packages == ("resurfemg",), name


def test_ecg_gating_fill_method_choices_match_adapter_validation():
    from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter

    description = describe_step("emg.ecg_gating")
    fill_method = next(p for p in description.parameters if p.name == "fill_method")
    assert fill_method.choices == (0, 1, 2, 3)
    assert fill_method.default == 1
    # Cross-check against the adapter's own runtime validation, not just
    # the docstring/comment.
    assert (
        ReSurfEMGAdapter.gate_ecg.__doc__ and "0" in ReSurfEMGAdapter.gate_ecg.__doc__
    )


def test_snr_pseudo_flags_are_optional_matching_conditional_behavior():
    description = describe_step("emg.snr_pseudo")
    flags = next(
        a for a in description.output_artifacts if a.name == "snr_pseudo_flags"
    )
    assert flags.required is False


def test_pocc_quality_threshold_defaults_match_function_signature():
    import inspect

    from m3resp.workflows.steps.ventilator import pocc_quality

    sig_defaults = {
        name: p.default
        for name, p in inspect.signature(pocc_quality).parameters.items()
        if name.endswith("_threshold")
    }
    description = describe_step("ventilator.pocc_quality")
    declared_defaults = {
        p.name: p.default
        for p in description.parameters
        if p.name.endswith("_threshold")
    }
    assert declared_defaults == sig_defaults


def test_describe_steps_prefix_filter_covers_emg_module():
    names = {d.name for d in describe_steps(prefix="emg.")}
    assert names == set(_EMG_STEP_NAMES)


def test_describe_steps_prefix_filter_separates_ventilator_from_emg():
    # The rename's whole point: ventilator steps are discoverable under their
    # own prefix rather than buried in the EMG namespace.
    names = {d.name for d in describe_steps(prefix="ventilator.")}
    assert names == set(_VENTILATOR_STEP_NAMES)
    assert not names & set(_EMG_STEP_NAMES)


def test_all_ninety_six_built_in_steps_still_describe_without_error():
    descriptions = describe_steps()
    assert len(descriptions) == 60
    with_metadata = [
        d
        for d in descriptions
        if d.parameters or d.output_artifacts or d.input_artifacts
    ]
    assert len(with_metadata) == 60
