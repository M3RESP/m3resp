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
    "eit.slice",
    "eit.detect_rates",
    "eit.mdn_filter",
    "eit.butterworth_filter",
    "eit.global_impedance",
    "eit.detect_breaths",
    "eit.normalize_breaths",
    "eit.continuous_tiv",
    "eit.eeli",
    "eit.pixel_tiv",
    "eit.pixel_breaths",
    "eit.roi_tiv_lungspace",
    "eit.roi_amplitude_lungspace",
    "eit.roi_watershed",
    "eit.roi_filter_by_size",
]


def test_all_sixteen_eit_steps_are_registered_and_described():
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


def test_every_eit_step_declares_the_optional_eitprocessing_dependency():
    for name in _EIT_STEP_NAMES:
        assert describe_step(name).optional_packages == ("eitprocessing",), name


def test_eit_load_vendor_choices_match_adapter_error_message():
    description = describe_step("eit.load")
    vendor = next(p for p in description.parameters if p.name == "vendor")
    assert vendor.choices == ("draeger", "sentec", "timpel")


def test_eit_slice_mode_choices_match_slice_signal_by_mode():
    description = describe_step("eit.slice")
    mode = next(p for p in description.parameters if p.name == "mode")
    assert mode.choices == ("index", "time")
    assert mode.default == "index"

    from m3resp.workflows.utils import slice_signal_by_mode

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
    from m3resp.workflows.steps.eit import _ALLOWED_PIXEL_BREATH_PHASE_MODES

    description = describe_step("eit.pixel_breaths")
    param = next(p for p in description.parameters if p.name == "phase_correction_mode")
    # The allowed set also includes Python None, which is not a `choices` entry
    # (choices are the string values; null is accepted as a separate case).
    assert set(param.choices or ()) == _ALLOWED_PIXEL_BREATH_PHASE_MODES - {None}


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
