"""Coverage for the Phase 1 step metadata backfilled onto ``eit.py`` (see
plan/stage2/3_pipeline_structure_implementation_plan.md, "Concrete Next
Actions" item 4: eit.py is the module after metrics/session/sync/export).

Checks the declared metadata against the real code it describes, not just
that a description exists.
"""

from __future__ import annotations

import json

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.workflows.registry import describe_step, describe_steps

_EIT_STEP_NAMES = [
    "eit.load",
    "eit.load_custom",
    "eit.slice_signal",
    "eit.slice_recording",
    "eit.detect_rates",
    "eit.mdn_filter",
    "eit.butterworth_filter",
    "eit.global_impedance",
    "eit.regional_impedance",
    "eit.functional_impedance",
    "eit.detect_breaths",
    "eit.normalize_breaths",
    "eit.continuous_tiv",
    "eit.eeli",
    "eit.pixel_tiv",
    "eit.pixel_eeli",
    "eit.pixel_breaths",
    "eit.roi_tiv_lungspace",
    "eit.roi_amplitude_lungspace",
    "eit.roi_watershed",
    "eit.roi_filter_by_size",
]


def test_all_twenty_one_eit_steps_are_registered_and_described():
    descriptions = describe_steps(prefix="eit.")
    assert {d.name for d in descriptions} == set(_EIT_STEP_NAMES)


def test_every_eit_step_declares_some_metadata_and_is_json_safe():
    for name in _EIT_STEP_NAMES:
        description = describe_step(name)
        assert description.description, name
        assert (
            description.parameters
            or description.output_artifacts
            or description.input_artifacts
        ), name
        json.dumps(description.as_dict())


#: EIT steps that run on m3resp code only.
_M3RESP_ONLY_STEPS = (
    "eit.slice_signal",
    "eit.detect_rates",
    "eit.detect_breaths",
    "eit.normalize_breaths",
    "eit.continuous_tiv",
    "eit.eeli",
    "eit.pixel_breaths",
    "eit.pixel_tiv",
    "eit.pixel_eeli",
)


def test_eit_steps_declare_the_packages_they_use():
    for name in _EIT_STEP_NAMES:
        expected = () if name in _M3RESP_ONLY_STEPS else ("eitprocessing",)
        assert describe_step(name).optional_packages == expected, name


def test_eit_load_vendor_choices_match_adapter_error_message():
    description = describe_step("eit.load")
    vendor = next(p for p in description.parameters if p.name == "vendor")
    assert vendor.choices == ("draeger", "sentec", "timpel")


def test_eit_slice_mode_choices_match_slice_signal_by_mode():
    description = describe_step("eit.slice_signal")
    mode = next(p for p in description.parameters if p.name == "mode")
    assert mode.choices == ("index", "time")
    assert mode.default == "index"

    from m3resp.processing.slicing import slice_signal_by_mode

    for choice in mode.choices:
        try:
            slice_signal_by_mode(object(), start=0, end=1, slicing_mode=choice)
        except ValueError as exc:
            raise AssertionError(
                f"{choice!r} rejected by slice_signal_by_mode"
            ) from exc
        except (TypeError, AttributeError):
            pass  # got past the slicing_mode check; failed later on the fake signal


def test_eit_detect_rates_subject_type_choices_match_function_signature():
    from typing import get_type_hints

    from m3resp.workflows.steps.eit import detect_rates

    # `from __future__ import annotations` makes annotations strings at
    # runtime; get_type_hints() resolves the real Literal[...] object.
    annotation = get_type_hints(detect_rates)["subject_type"]
    description = describe_step("eit.detect_rates")
    subject_type = next(p for p in description.parameters if p.name == "subject_type")
    assert set(subject_type.choices or ()) == set(annotation.__args__)


def test_eit_pixel_breaths_phase_correction_choices_match_allowed_set():
    from m3resp.eit.processing.pixel_breath import PHASE_CORRECTION_MODES

    description = describe_step("eit.pixel_breaths")
    param = next(p for p in description.parameters if p.name == "phase_correction_mode")
    # Every accepted mode is offered, the empty one included: it is a real
    # option meaning "no phase correction", so a GUI built from `choices` must
    # be able to select it.
    assert set(param.choices or ()) == set(PHASE_CORRECTION_MODES)
    assert None in (param.choices or ())


def test_eit_pixel_breaths_phase_correction_options_have_one_definition():
    """The choices, the runtime check and the type hint cannot drift apart."""

    from typing import get_args, get_type_hints

    from m3resp.eit.processing.pixel_breath import PHASE_CORRECTION_MODES
    from m3resp.workflows.steps.eit.pixel import pixel_breaths

    annotation = get_type_hints(pixel_breaths)["phase_correction_mode"]
    description = describe_step("eit.pixel_breaths")
    param = next(p for p in description.parameters if p.name == "phase_correction_mode")

    assert set(get_args(annotation)) == set(PHASE_CORRECTION_MODES)
    assert set(param.choices or ()) == set(PHASE_CORRECTION_MODES)


def test_eit_roi_filter_by_size_connectivity_choices_are_one_or_two():
    description = describe_step("eit.roi_filter_by_size")
    connectivity = next(p for p in description.parameters if p.name == "connectivity")
    assert connectivity.choices == (1, 2)
    assert connectivity.default == 1


def test_roi_threshold_parameters_are_bounded_zero_to_one():
    for name, param_name in [
        ("eit.roi_tiv_lungspace", "threshold"),
        ("eit.roi_amplitude_lungspace", "threshold"),
        ("eit.roi_watershed", "threshold_fraction"),
    ]:
        description = describe_step(name)
        param = next(p for p in description.parameters if p.name == param_name)
        assert param.minimum == 0.0
        assert param.maximum == 1.0
        assert param.default == 0.15


_IMAGE_ARTIFACTS = {
    ("eit.load", "raw_eit"),
    ("eit.load", "raw_pixel_impedance_signal"),
    ("eit.mdn_filter", "signal"),
    ("eit.mdn_filter", "filtered_eit"),
    ("eit.butterworth_filter", "signal"),
    ("eit.butterworth_filter", "filtered_eit_signal"),
    ("eit.detect_rates", "signal"),
    ("eit.global_impedance", "signal"),
    ("eit.regional_impedance", "signal"),
    ("eit.functional_impedance", "signal"),
    ("eit.pixel_tiv", "eit_data"),
    ("eit.pixel_breaths", "eit_data"),
    ("eit.roi_tiv_lungspace", "eit_data"),
}
_WAVEFORM_ARTIFACTS = {
    ("eit.load", "raw_global_impedance"),
    ("eit.load", "raw_global_impedance_signal"),
    ("eit.global_impedance", "global_impedance"),
    ("eit.regional_impedance", "regional_impedance_signal"),
    ("eit.functional_impedance", "functional_impedance"),
    ("eit.detect_breaths", "signal"),
    ("eit.continuous_tiv", "signal"),
    ("eit.eeli", "signal"),
    ("eit.pixel_tiv", "signal"),
    ("eit.pixel_breaths", "timing_data"),
    ("eit.roi_watershed", "timing_data"),
}


def test_eit_images_and_impedance_waveforms_have_their_own_labels():
    labels = {}
    for name in _EIT_STEP_NAMES:
        description = describe_step(name)
        for artifact in (*description.input_artifacts, *description.output_artifacts):
            labels[name, artifact.name] = artifact
    for key in _IMAGE_ARTIFACTS:
        assert labels[key].artifact_type == "eit_pixel_signal", key
        assert not labels[key].compatibility_only, key
    for key in _WAVEFORM_ARTIFACTS:
        assert labels[key].artifact_type == "eit_impedance_waveform", key
        assert not labels[key].compatibility_only, key
    assert "signal" not in {artifact.artifact_type for artifact in labels.values()}
