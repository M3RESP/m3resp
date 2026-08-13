"""`Signal.category` as an axis independent of `Signal.modality`.

`modality` answers "which device produced this?", `category` answers "what
physical quantity is it?". They used to be one field, which made
"ventilator device, volume quantity" inexpressible - see
`m3resp.data.categories` for the full rationale.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.data import (
    ParameterResult,
    ProcessingStep,
    QualityFlag,
    Signal,
    SignalCollection,
)
from m3resp.data.categories import (
    category_aliases,
    normalize_category,
    register_category_alias,
    reset_category_aliases,
    save_category_aliases,
)
from m3resp.data.collections import ParameterResultCollection, QualityReport
from m3resp.datamodel.recorder import _signal_type_for, _stream_key


def _signal(**kwargs):
    return Signal(values=np.zeros(4), time=np.arange(4.0), **kwargs)


class TestCategoryNormalization:
    def test_known_alias_is_canonicalized(self):
        assert _signal(modality="ventilator", category="paw").category == (
            "airway_pressure"
        )
        assert _signal(modality="ventilator", category="flow").category == "airflow"

    def test_matching_is_case_and_whitespace_insensitive(self):
        assert normalize_category("  PAW ") == "airway_pressure"

    def test_unknown_category_is_kept_verbatim(self):
        # The vocabulary is open, and unlike ParameterResult.metric_type there
        # is no second field preserving the caller's original label - so an
        # unrecognized category must survive rather than become None.
        signal = _signal(modality="ventilator", category="jet_driving_pressure")
        assert signal.category == "jet_driving_pressure"

    def test_absent_category_stays_none(self):
        assert _signal(modality="eit").category is None

    def test_esophageal_pressure_is_not_collapsed_into_airway(self):
        # Bare "pressure" resolving to airway is a convenience; a named
        # non-airway pressure must never be silently relabelled.
        assert normalize_category("pressure") == "airway_pressure"
        assert normalize_category("pes") == "esophageal_pressure"

    def test_category_applies_to_parameters_and_flags_too(self):
        parameter = ParameterResult(
            name="pocc_time_product", value=1.0, modality="ventilator", category="paw"
        )
        flag = QualityFlag(
            name="q",
            passed=True,
            severity="info",
            modality="ventilator",
            category="paw",
        )
        assert parameter.category == "airway_pressure"
        assert flag.category == "airway_pressure"


class TestCustomVocabulary:
    def test_registered_alias_resolves(self):
        try:
            register_category_alias("pdi", "transdiaphragmatic_pressure")
            assert normalize_category("Pdi") == "transdiaphragmatic_pressure"
        finally:
            reset_category_aliases()

    def test_reset_drops_registrations(self):
        register_category_alias("pdi", "transdiaphragmatic_pressure")
        reset_category_aliases()
        assert normalize_category("pdi") is None

    def test_round_trips_through_a_yaml_file(self, tmp_path):
        # This is the route for adopting an externally-maintained taxonomy
        # (e.g. eitprocessing's shared catalogue) without vendoring a copy.
        try:
            register_category_alias("pdi", "transdiaphragmatic_pressure")
            path = save_category_aliases(tmp_path / "categories.yaml")
            reset_category_aliases()
            assert normalize_category("pdi") is None

            from m3resp.data.categories import load_category_aliases

            load_category_aliases(path)
            assert normalize_category("pdi") == "transdiaphragmatic_pressure"
            # Built-in defaults are extended, never replaced.
            assert normalize_category("paw") == "airway_pressure"
        finally:
            reset_category_aliases()

    def test_only_custom_entries_are_saved_by_default(self, tmp_path):
        try:
            register_category_alias("pdi", "transdiaphragmatic_pressure")
            path = save_category_aliases(tmp_path / "categories.yaml")
            import yaml

            payload = yaml.safe_load(path.read_text())
            assert payload == {"pdi": "transdiaphragmatic_pressure"}
        finally:
            reset_category_aliases()

    def test_aliases_snapshot_is_a_copy(self):
        snapshot = category_aliases()
        snapshot["bogus"] = "bogus"
        assert normalize_category("bogus") is None


class TestFilteringByEitherAxis:
    def _collection(self):
        collection = SignalCollection()
        for category in ("airway_pressure", "airflow", "volume"):
            collection.add(_signal(modality="ventilator", category=category))
        collection.add(_signal(modality="eit", category="impedance"))
        return collection

    def test_for_modality_returns_every_channel_of_that_device(self):
        assert len(self._collection().for_modality("ventilator")) == 3

    def test_for_category_selects_one_quantity(self):
        selected = self._collection().for_category("airway_pressure")
        assert len(selected) == 1
        assert selected[0].modality == "ventilator"

    def test_for_category_accepts_an_alias(self):
        assert len(self._collection().for_category("paw")) == 1

    def test_the_two_axes_are_independent(self):
        # One device spans several categories, and one category can be reached
        # without naming the device at all.
        collection = self._collection()
        assert len(collection.for_modality("eit")) == 1
        assert len(collection.for_category("impedance")) == 1

    def test_parameters_and_flags_filter_by_category(self):
        parameters = ParameterResultCollection()
        parameters.add(
            ParameterResult(
                name="ptp", value=1.0, modality="ventilator", category="airway_pressure"
            )
        )
        parameters.add(
            ParameterResult(
                name="vt", value=2.0, modality="ventilator", category="volume"
            )
        )
        report = QualityReport()
        report.add(
            QualityFlag(
                name="q",
                passed=True,
                severity="info",
                modality="ventilator",
                category="airway_pressure",
            )
        )

        assert len(parameters.for_modality("ventilator")) == 2
        assert len(parameters.for_category("airway_pressure")) == 1
        assert len(report.for_category("airway_pressure")) == 1


class TestPersistedStreamTypeResolution:
    """Layer 2 keys streams by a name fusing device and quantity, so resolving
    one needs both Layer 1 axes."""

    @pytest.mark.parametrize(
        ("category", "expected"),
        [
            ("airway_pressure", "ventilator_pressure"),
            ("airflow", "ventilator_flow"),
            ("volume", "ventilator_volume"),
            ("tidal_volume", "ventilator_volume"),
        ],
    )
    def test_ventilator_channels_resolve_to_distinct_stream_types(
        self, category, expected
    ):
        signal = _signal(modality="ventilator", category=category)
        assert _signal_type_for(signal) == expected

    def test_ventilator_volume_is_reachable(self):
        # Regression: with modality as the only axis, every ventilator signal
        # resolved to "ventilator_pressure" and this vocabulary entry could
        # never be produced by any code path.
        signal = _signal(modality="ventilator", category="volume")
        assert _signal_type_for(signal) == "ventilator_volume"

    def test_uncategorized_ventilator_signal_is_skipped_not_guessed(self):
        # Guessing pressure here is what silently mislabelled volume and flow.
        assert _signal_type_for(_signal(modality="ventilator")) is None

    def test_eit_and_emg_resolve_without_a_category(self):
        assert _signal_type_for(_signal(modality="eit")) == "eit_waveform"
        assert _signal_type_for(_signal(modality="emg")) == "emg_raw"

    def test_emg_splits_on_processing_state_not_category(self):
        # Raw trace and envelope are both electrical potentials.
        raw = _signal(modality="emg", category="electrical_potential")
        envelope = _signal(
            modality="emg",
            category="electrical_potential",
            processing_state="processed",
        )
        assert _signal_type_for(raw) == "emg_raw"
        assert _signal_type_for(envelope) == "emg_envelope"

    def test_unknown_modality_resolves_to_nothing(self):
        assert _signal_type_for(_signal()) is None


class TestRecordingMultiChannelDevices:
    """A ventilator emits several streams from one device, so the recorder must
    key them by both axes."""

    def _recorder(self):
        from m3resp.core.session import M3Session
        from m3resp.datamodel import DataModelRecorder, DataModelStore

        session = M3Session()
        store = DataModelStore()
        return DataModelRecorder(session, store), store

    def test_each_ventilator_channel_gets_its_own_stream(self):
        recorder, store = self._recorder()
        for category in ("airway_pressure", "airflow", "volume"):
            recorder.record_signal(_signal(modality="ventilator", category=category))

        streams = list(store.signal_streams.values())
        assert len(streams) == 3
        assert {stream.signal_type for stream in streams} == {
            "ventilator_pressure",
            "ventilator_flow",
            "ventilator_volume",
        }

    def test_channels_share_one_device(self):
        recorder, store = self._recorder()
        for category in ("airway_pressure", "airflow", "volume"):
            recorder.record_signal(_signal(modality="ventilator", category=category))

        device_ids = {stream.device_id for stream in store.signal_streams.values()}
        assert len(device_ids) == 1

    def test_a_parameter_attaches_to_its_own_category_stream(self):
        # Regression: keyed by modality alone, all three channels collapsed to
        # one cache entry and every derived feature was attributed to whichever
        # channel happened to be recorded last.
        recorder, _store = self._recorder()
        streams_by_category = {
            category: recorder.record_signal(
                _signal(modality="ventilator", category=category)
            )
            for category in ("airway_pressure", "airflow", "volume")
        }
        run = recorder.record_processing_step(ProcessingStep(name="pocc"))

        feature = recorder.record_parameter(
            ParameterResult(
                name="pocc_time_product",
                value=1.0,
                modality="ventilator",
                category="airway_pressure",
            ),
            processing_run_id=run.processing_run_id,
        )
        assert feature.source_signal_ids == [
            streams_by_category["airway_pressure"].signal_id
        ]


class TestStreamKey:
    def test_both_axes_produce_distinct_keys(self):
        keys = {
            _stream_key("ventilator", category)
            for category in ("airway_pressure", "airflow", "volume")
        }
        assert len(keys) == 3

    def test_falls_back_to_bare_modality_without_a_category(self):
        # Keeps the pre-category keys for EIT/EMG, which have one stream each.
        assert _stream_key("eit", None) == "eit"
