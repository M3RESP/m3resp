"""`M3Session.preprocess_ventilator` / `detect_ventilator_breaths`.

These complete the ventilator's promotion to a peer modality: it now has the
same load -> preprocess -> detect chain as EIT and EMG, with the same
`variant`/`overwrite` semantics, instead of riding along as keyword arguments
to `postprocess_emg`.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters import ReSurfEMGAdapter
from m3resp.core.exceptions import (
    MissingModalityDataError,
    VariantAlreadyExistsError,
)
from m3resp.core.session import M3Session

FS = 100.0
N = 1000


def _payload() -> dict:
    time = np.arange(N) / FS
    wave = np.sin(2 * np.pi * 0.25 * time) + 0.2 * np.sin(2 * np.pi * 40 * time)
    return {
        "array": np.vstack([wave * 10, wave * 5, wave]),
        "metadata": {
            "fs": FS,
            "labels": ["Paw", "Flow", "Volume"],
            "units": ["cmH2O", "L/min", "L"],
        },
    }


def _loaded_session() -> M3Session:
    session = M3Session(
        emg_adapter=ReSurfEMGAdapter(loader=lambda path, **kwargs: _payload()),
    )
    session.load_ventilator("subject.txt")
    return session


class TestPreprocessVentilator:
    def test_returns_the_channel_bundle(self):
        result = _loaded_session().preprocess_ventilator()
        assert {"pressure", "flow", "volume", "fs"} <= set(result)

    def test_requires_a_loaded_recording(self):
        with pytest.raises(MissingModalityDataError, match="load_ventilator"):
            M3Session().preprocess_ventilator()

    def test_updates_the_recordings_channel_fields(self):
        session = _loaded_session()
        session.preprocess_ventilator()

        assert session.ventilator.pressure is not None
        assert session.ventilator.flow is not None
        assert session.ventilator.volume is not None
        assert session.ventilator.fs == FS

    def test_mirrors_the_default_variant_onto_processed(self):
        session = _loaded_session()
        result = session.preprocess_ventilator()
        assert session.processed["ventilator"] is result
        assert session.processed_variants["ventilator"]["default"] is result

    def test_records_provenance(self):
        session = _loaded_session()
        session.preprocess_ventilator()
        assert session.provenance[-1].action == "preprocess_ventilator"
        assert session.provenance[-1].modality == "ventilator"

    def test_forwards_adapter_options(self):
        session = _loaded_session()
        result = session.preprocess_ventilator(lowpass_hz=None)
        assert result["filter"]["lowpass_hz"] is None


class TestPreprocessVentilatorVariants:
    """Same semantics as `preprocess_eit`/`preprocess_emg`."""

    def test_named_variants_coexist(self):
        session = _loaded_session()
        session.preprocess_ventilator(variant="raw_ish", lowpass_hz=None)
        session.preprocess_ventilator(variant="smooth", lowpass_hz=5.0)

        variants = session.processed_variants["ventilator"]
        assert set(variants) == {"raw_ish", "smooth"}
        assert variants["raw_ish"]["filter"]["lowpass_hz"] is None
        assert variants["smooth"]["filter"]["lowpass_hz"] == 5.0

    def test_a_named_variant_does_not_touch_processed(self):
        session = _loaded_session()
        session.preprocess_ventilator(variant="smooth")
        assert "ventilator" not in session.processed

    def test_rewriting_a_variant_raises(self):
        session = _loaded_session()
        session.preprocess_ventilator()
        with pytest.raises(VariantAlreadyExistsError, match="already exists"):
            session.preprocess_ventilator()

    def test_overwrite_allows_replacing(self):
        session = _loaded_session()
        session.preprocess_ventilator()
        replaced = session.preprocess_ventilator(overwrite=True, lowpass_hz=5.0)
        assert session.processed["ventilator"] is replaced

    def test_session_wide_allow_overwrite_is_honored(self):
        session = _loaded_session()
        session.allow_overwrite = True
        session.preprocess_ventilator()
        session.preprocess_ventilator()  # must not raise


class TestTypedCollections:
    def test_ventilator_signals_reach_the_session(self):
        # Ventilator data never landed in `session.signals` before the
        # ventilator became a peer modality.
        session = _loaded_session()
        session.preprocess_ventilator()
        assert len(session.signals.for_modality("ventilator")) == 6

    def test_each_channel_is_retrievable_by_category(self):
        session = _loaded_session()
        session.preprocess_ventilator()

        for category in ("airway_pressure", "airflow", "volume"):
            found = session.signals.for_category(category)
            assert len(found) == 2  # raw + processed
            assert {s.modality for s in found} == {"ventilator"}

    def test_signals_accumulate_across_variants(self):
        session = _loaded_session()
        session.preprocess_ventilator(variant="a")
        session.preprocess_ventilator(variant="b")
        assert len(session.signals.for_modality("ventilator")) == 12


class TestDetectVentilatorBreaths:
    def test_detects_and_stores_breaths(self):
        session = _loaded_session()
        session.preprocess_ventilator()
        breaths = session.detect_ventilator_breaths()

        assert breaths is session.events["ventilator_breaths"]
        assert 2 <= len(breaths) <= 3

    def test_breaths_use_the_canonical_modality(self):
        session = _loaded_session()
        session.preprocess_ventilator()
        breaths = session.detect_ventilator_breaths()
        assert {breath.modality for breath in breaths} == {"ventilator"}

    def test_preprocesses_on_the_fly_when_not_already_done(self):
        # Raw data has no split channels, so detection would otherwise fail;
        # preprocessing on demand matches how `postprocess_emg` accepts a raw
        # ventilator recording.
        session = _loaded_session()
        breaths = session.detect_ventilator_breaths()
        assert len(breaths) >= 2

    def test_requires_a_loaded_recording(self):
        with pytest.raises(MissingModalityDataError, match="load_ventilator"):
            M3Session().detect_ventilator_breaths()

    def test_records_provenance(self):
        session = _loaded_session()
        session.detect_ventilator_breaths()
        assert session.provenance[-1].action == "detect_ventilator_breaths"

    def test_a_custom_detector_is_forwarded(self):
        session = _loaded_session()
        session.preprocess_ventilator()
        breaths = session.detect_ventilator_breaths(detector=lambda bundle, **kw: [7])
        assert len(breaths) == 1
        assert breaths[0].peak_time == pytest.approx(7 / FS)


class TestDetectVentilatorBreathsVariants:
    def test_variant_events_are_stored_under_their_own_key(self):
        session = _loaded_session()
        session.preprocess_ventilator(variant="smooth", lowpass_hz=5.0)
        session.detect_ventilator_breaths(variant="smooth")

        assert "ventilator_breaths:smooth" in session.events
        assert "ventilator_breaths" not in session.events

    def test_an_unknown_variant_raises(self):
        session = _loaded_session()
        with pytest.raises(MissingModalityDataError, match="preprocess_ventilator"):
            session.detect_ventilator_breaths(variant="nope")


class TestLinkingAcrossModalities:
    def test_detected_breaths_link_under_the_ventilator_key(self):
        session = _loaded_session()
        session.preprocess_ventilator()
        session.detect_ventilator_breaths()

        linked = session.link_breaths()
        assert linked
        assert all("ventilator" in link.breaths for link in linked)
