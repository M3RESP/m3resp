"""Ventilator channels identified by quantity, origin and key - not by quantity alone.

A study can record several pressures at once (a Draeger pressure pod reports
airway, esophageal, transpulmonary and gastric pressure alongside the
ventilator's own airway pressure), and the same quantity can arrive from more
than one instrument. Keying channels by physical quantity collapsed those onto
one slot and hard-coded the vocabulary to pressure/flow/volume; these tests
cover the resolution that replaces it.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters.ventilator_adapter import (
    CHANNEL_CATEGORIES,
    DEFAULT_CHANNEL_UNITS,
    VentilatorAdapter,
    channel_aliases,
    normalize_channel_label,
    primary_channel,
    register_channel_alias,
    reset_channel_aliases,
    resolve_channel_name,
    resolve_channels,
    split_channels,
)
from m3resp.core.exceptions import UnresolvedChannelError

FS = 100.0
N = 600


@pytest.fixture(autouse=True)
def _restore_aliases():
    # The alias registry is process-wide, like `data.categories`'.
    yield
    reset_channel_aliases()


def _wave(scale: float = 1.0) -> np.ndarray:
    return scale * np.sin(2 * np.pi * 0.25 * np.arange(N) / FS)


def _payload(labels, units=None, rows=None) -> dict:
    rows = rows if rows is not None else [_wave(i + 1) for i in range(len(labels))]
    metadata: dict = {"fs": FS, "labels": list(labels)}
    if units is not None:
        metadata["units"] = list(units)
    return {"array": np.vstack(rows), "metadata": metadata}


class TestLabelNormalization:
    def test_vendor_tags_are_stripped(self):
        assert normalize_channel_label("airway_pressure_(timpel)") == "airway pressure"
        assert normalize_channel_label("global_impedance_(raw)") == "global impedance"

    def test_a_pod_tag_survives(self):
        # `(pod)` marks a physically separate transducer, so it must not be
        # normalized away: it is what tells a pod airway pressure apart from
        # the ventilator's own.
        assert normalize_channel_label("airway pressure (pod)") == (
            "airway pressure (pod)"
        )

    def test_matching_is_case_and_separator_insensitive(self):
        assert resolve_channel_name("Airway_Pressure") == "airway_pressure"
        assert resolve_channel_name("PAW") == "airway_pressure"

    def test_an_unknown_label_resolves_to_nothing(self):
        assert resolve_channel_name("thermistor") is None


class TestVocabulary:
    def test_every_pressure_has_its_own_channel(self):
        for channel in (
            "airway_pressure",
            "esophageal_pressure",
            "transpulmonary_pressure",
            "gastric_pressure",
        ):
            assert channel in CHANNEL_CATEGORIES

    def test_pressures_map_to_distinct_categories(self):
        categories = {
            CHANNEL_CATEGORIES[name]
            for name in (
                "airway_pressure",
                "esophageal_pressure",
                "transpulmonary_pressure",
                "gastric_pressure",
            )
        }
        # Four pressures, four categories - none collapses onto airway.
        assert len(categories) == 4
        assert CHANNEL_CATEGORIES["airway_pressure"] == "airway_pressure"


class TestAliasRegistry:
    def test_a_site_can_register_its_own_naming(self):
        register_channel_alias("Pmus_proximal", "airway_pressure")
        assert resolve_channel_name("pmus proximal") == "airway_pressure"

    def test_the_old_airway_pressure_name_is_refused(self):
        # An old alias file pointing at "pressure" must not quietly define a
        # second channel apart from the airway pressure.
        with pytest.raises(ValueError, match="airway_pressure"):
            register_channel_alias("Ptrach", "pressure")
        assert "pressure" not in CHANNEL_CATEGORIES

    def test_an_old_alias_file_changes_nothing(self, tmp_path):
        # Every entry is checked before any is registered, so a file with one
        # old entry leaves the active map as it was, even with replace=True.
        from m3resp.adapters.ventilator_adapter import load_channel_aliases

        register_channel_alias("Pmus_proximal", "airway_pressure")
        before = channel_aliases()
        path = tmp_path / "old_channels.yaml"
        path.write_text("a label: flow\nptrach: pressure\nz label: volume\n")

        with pytest.raises(ValueError, match="airway_pressure"):
            load_channel_aliases(path, replace=True)
        assert channel_aliases() == before

    def test_registering_a_new_channel_defines_it(self):
        # A physical quantity not among the seven built-ins - a new
        # instrument's channel - can be registered directly rather than
        # requiring a code change to this module first.
        register_channel_alias("Whatever", "co2_waveform")
        assert CHANNEL_CATEGORIES["co2_waveform"] == "co2_waveform"
        assert resolve_channel_name("whatever") == "co2_waveform"

    def test_a_new_channel_without_a_category_defaults_to_its_own_name(self):
        register_channel_alias("Whatever", "co2_waveform")
        assert CHANNEL_CATEGORIES["co2_waveform"] == "co2_waveform"

    def test_a_new_channel_can_be_given_an_explicit_category(self):
        register_channel_alias("Whatever", "co2_waveform", category="co2")
        assert CHANNEL_CATEGORIES["co2_waveform"] == "co2"

    def test_a_new_channel_can_be_given_a_unit(self):
        register_channel_alias("Whatever", "co2_waveform", unit="%")
        assert DEFAULT_CHANNEL_UNITS["co2_waveform"] == "%"

    def test_a_new_channel_has_no_default_unit_unless_given_one(self):
        register_channel_alias("Whatever", "co2_waveform")
        assert "co2_waveform" not in DEFAULT_CHANNEL_UNITS

    def test_a_newly_registered_channel_can_be_requested(self):
        register_channel_alias("Whatever", "co2_waveform")
        bundle = split_channels(
            _payload(["Paw", "Flow", "Volume", "Whatever"]),
            channels=("airway_pressure", "co2_waveform"),
        )
        assert "co2_waveform" in bundle["channels"]

    def test_category_and_unit_on_an_existing_channel_update_it(self):
        register_channel_alias("Ptrach", "airway_pressure", category="airway_pressure")
        assert CHANNEL_CATEGORIES["airway_pressure"] == "airway_pressure"

    def test_reset_undoes_a_newly_registered_channel(self):
        register_channel_alias("Whatever", "co2_waveform", unit="%")
        reset_channel_aliases()
        assert "co2_waveform" not in CHANNEL_CATEGORIES
        assert "co2_waveform" not in DEFAULT_CHANNEL_UNITS
        assert resolve_channel_name("whatever") is None

    def test_reset_restores_the_defaults(self):
        register_channel_alias("Pmus_proximal", "airway_pressure")
        reset_channel_aliases()
        assert resolve_channel_name("pmus proximal") is None

    def test_a_registered_alias_is_used_when_splitting(self):
        register_channel_alias("Ptrach", "airway_pressure")
        bundle = split_channels(_payload(["Ptrach", "Flow", "Volume"]))
        assert bundle["labels"]["airway_pressure"] == "Ptrach"

    def test_aliases_can_be_round_tripped_through_a_file(self, tmp_path):
        from m3resp.adapters.ventilator_adapter import (
            load_channel_aliases,
            save_channel_aliases,
        )

        register_channel_alias("Ptrach", "airway_pressure")
        path = save_channel_aliases(tmp_path / "channels.yaml")
        reset_channel_aliases()
        assert resolve_channel_name("ptrach") is None

        load_channel_aliases(path)
        assert resolve_channel_name("ptrach") == "airway_pressure"
        # Only the custom entry is written; the built-ins still apply.
        assert resolve_channel_name("airway pressure") == "airway_pressure"
        assert "ptrach" in channel_aliases()


class TestResolutionByName:
    def test_labels_decide_the_column_not_the_position(self):
        # Volume first, pressure last: positional defaults would get all three
        # wrong. Only name matching gets this right.
        bundle = split_channels(_payload(["Volume", "Flow", "Paw"]))
        assert bundle["channel_indices"] == {
            "airway_pressure": 2,
            "flow": 1,
            "volume": 0,
        }

    def test_an_unlabelled_recording_falls_back_to_positions(self):
        payload = {"array": np.vstack([_wave(i + 1) for i in range(3)]), "fs": FS}
        bundle = split_channels(payload["array"], fs=FS)
        assert bundle["channel_indices"] == {
            "airway_pressure": 0,
            "flow": 1,
            "volume": 2,
        }

    def test_a_labelled_recording_never_hands_on_an_unnamed_column(self):
        # Airway pressure plus two EMG channels in one file: flow and volume
        # are missing, not columns 1 and 2 (which hold EMG).
        payload = _payload(["Paw", "EMGdi", "EMGsc"])
        with pytest.raises(UnresolvedChannelError, match="flow"):
            split_channels(payload)
        bundle = split_channels(payload, channels=("airway_pressure",))
        assert bundle["channel_indices"] == {"airway_pressure": 0}

    def test_labels_that_name_nothing_known_are_read_by_position_with_a_warning(
        self,
    ):
        with pytest.warns(UserWarning, match="column position"):
            bundle = split_channels(_payload(["P", "F", "V"]))
        assert bundle["channel_indices"] == {
            "airway_pressure": 0,
            "flow": 1,
            "volume": 2,
        }

    def test_an_explicit_index_overrides_a_label_match(self):
        bundle = split_channels(
            _payload(["Paw", "Flow", "Volume"]), airway_pressure_channel=2
        )
        assert bundle["channel_indices"]["airway_pressure"] == 2

    def test_an_index_for_a_channel_not_being_read_is_an_error(self):
        # Ignoring it would take the channel from another column unannounced,
        # e.g. an index given under the old name "pressure".
        with pytest.raises(ValueError, match="pressure"):
            split_channels(
                _payload(["Paw", "Flow", "Volume"]), channel_indices={"pressure": 2}
            )
        with pytest.raises(ValueError, match="flow"):
            split_channels(
                _payload(["Paw", "Flow", "Volume"]),
                channels=("airway_pressure",),
                flow_channel=1,
            )

    def test_a_requested_channel_that_is_absent_is_an_error(self):
        with pytest.raises(UnresolvedChannelError, match="esophageal_pressure"):
            split_channels(
                _payload(["Paw", "Flow", "Volume"]),
                channels=("airway_pressure", "esophageal_pressure"),
            )

    def test_the_error_names_the_labels_it_saw(self):
        with pytest.raises(UnresolvedChannelError, match="Thermistor"):
            split_channels(
                _payload(["Thermistor", "Flow", "Volume"]),
                channels=("esophageal_pressure",),
            )


class TestSeveralPressures:
    LABELS = (
        "airway pressure",
        "flow",
        "volume",
        "esophageal pressure (pod)",
        "transpulmonary pressure (pod)",
        "gastric pressure/auxiliary pressure (pod)",
    )

    def test_all_five_pressures_can_be_read_from_one_recording(self):
        bundle = split_channels(
            _payload(self.LABELS),
            channels=(
                "airway_pressure",
                "esophageal_pressure",
                "transpulmonary_pressure",
                "gastric_pressure",
            ),
        )
        assert set(bundle["channels"]) == {
            "airway_pressure",
            "esophageal_pressure",
            "transpulmonary_pressure",
            "gastric_pressure",
        }

    def test_each_pressure_keeps_its_own_category(self):
        bundle = split_channels(
            _payload(self.LABELS),
            channels=("airway_pressure", "esophageal_pressure", "gastric_pressure"),
        )
        assert bundle["categories"] == {
            "airway_pressure": "airway_pressure",
            "esophageal_pressure": "esophageal_pressure",
            "gastric_pressure": "gastric_pressure",
        }

    def test_two_airway_pressures_both_survive(self):
        # The ventilator's own airway pressure and the pod's copy. Before, one
        # overwrote the other; now the first keeps the bare key and the second
        # is distinguished by the tag that separates them.
        bundle = split_channels(
            _payload(["airway pressure", "flow", "volume", "airway pressure (pod)"])
        )
        assert "airway_pressure" in bundle["channels"]
        assert "airway_pressure__pod" in bundle["channels"]
        assert bundle["labels"]["airway_pressure"] == "airway pressure"
        assert bundle["labels"]["airway_pressure__pod"] == "airway pressure (pod)"

    def test_two_airway_pressures_share_a_category(self):
        bundle = split_channels(
            _payload(["airway pressure", "flow", "volume", "airway pressure (pod)"])
        )
        assert bundle["categories"]["airway_pressure"] == "airway_pressure"
        assert bundle["categories"]["airway_pressure__pod"] == "airway_pressure"

    def test_the_first_airway_pressure_is_the_primary_one(self):
        bundle = split_channels(
            _payload(["airway pressure", "flow", "volume", "airway pressure (pod)"])
        )
        assert bundle["primary"]["airway_pressure"] == "airway_pressure"
        assert primary_channel(bundle, "airway_pressure") == "airway_pressure"
        assert primary_channel(bundle, "airway_pressure") == "airway_pressure"

    def test_the_two_pressures_hold_different_data(self):
        rows = [_wave(10.0), _wave(5.0), _wave(1.0), _wave(7.0)]
        bundle = split_channels(
            _payload(
                ["airway pressure", "flow", "volume", "airway pressure (pod)"],
                rows=rows,
            )
        )
        assert np.allclose(bundle["airway_pressure"], rows[0])
        assert np.allclose(bundle["airway_pressure__pod"], rows[3])


class TestOriginDistinguishesDevices:
    def test_origin_is_recorded_per_channel(self):
        specs = resolve_channels(
            ["airway pressure", "flow", "volume"], origin="draeger_bin"
        )
        assert {spec.origin for spec in specs} == {"draeger_bin"}

    def test_a_pod_channel_records_the_pod_as_its_origin(self):
        specs = resolve_channels(
            ["esophageal pressure (pod)"],
            requested=("esophageal_pressure",),
            origin="draeger_bin",
        )
        assert specs[0].origin == "pod"

    def test_the_vendor_label_is_never_discarded(self):
        specs = resolve_channels(
            ["airway_pressure_(timpel)", "flow_(timpel)", "volume_(timpel)"]
        )
        assert [spec.label for spec in specs] == [
            "airway_pressure_(timpel)",
            "flow_(timpel)",
            "volume_(timpel)",
        ]

    def test_units_are_kept_as_reported_and_not_converted(self):
        # Draeger reports volume in mL, Timpel in L. Neither is rescaled.
        draeger = split_channels(
            _payload(
                ["airway pressure", "flow", "volume"], units=["mbar", "L/min", "mL"]
            )
        )
        timpel = split_channels(
            _payload(
                ["airway_pressure_(timpel)", "flow_(timpel)", "volume_(timpel)"],
                units=["cmH2O", "L/s", "L"],
            )
        )
        assert draeger["units"]["volume"] == "mL"
        assert timpel["units"]["volume"] == "L"


class TestPreprocessingCarriesEveryChannel:
    LABELS = ("airway pressure", "flow", "volume", "esophageal pressure (pod)")

    def _bundle(self, **kwargs):
        return VentilatorAdapter().preprocess(
            _payload(self.LABELS, units=["mbar", "L/min", "mL", "mbar"]),
            channels=("airway_pressure", "flow", "volume", "esophageal_pressure"),
            **kwargs,
        )

    def test_every_resolved_channel_is_filtered_when_a_cutoff_is_given(self):
        processed = self._bundle(lowpass_hz=20.0)
        assert set(processed["filtered"]) == {
            "airway_pressure",
            "flow",
            "volume",
            "esophageal_pressure",
        }

    def test_the_unfiltered_channels_stay_available(self):
        processed = self._bundle(lowpass_hz=20.0)
        assert set(processed["raw"]) == set(processed["filtered"])

    def test_signals_are_emitted_for_every_channel(self):
        signals = VentilatorAdapter().to_signals(self._bundle())
        assert {signal.channel for signal in signals} == {
            "airway_pressure",
            "flow",
            "volume",
            "esophageal_pressure",
        }

    def test_the_esophageal_signal_is_not_labelled_airway(self):
        signals = VentilatorAdapter().to_signals(self._bundle())
        esophageal = [s for s in signals if s.channel == "esophageal_pressure"]
        assert esophageal
        assert {s.category for s in esophageal} == {"esophageal_pressure"}

    def test_breath_detection_uses_the_primary_volume_channel(self):
        adapter = VentilatorAdapter()
        breaths = adapter.detect_breaths(self._bundle())
        assert breaths


class TestTwoAirwayPressuresReachTheSignalCollection:
    LABELS = ("airway pressure", "flow", "volume", "airway pressure (pod)")

    def test_both_are_emitted_as_signals(self):
        adapter = VentilatorAdapter()
        processed = adapter.preprocess(_payload(self.LABELS))
        signals = adapter.to_signals(processed)
        channels = {signal.channel for signal in signals}
        assert {"airway_pressure", "airway_pressure__pod"} <= channels

    def test_a_category_query_returns_both(self):
        from m3resp.data.collections import SignalCollection

        adapter = VentilatorAdapter()
        collection = SignalCollection()
        for signal in adapter.to_signals(adapter.preprocess(_payload(self.LABELS))):
            collection.add(signal)

        airway = collection.for_category("airway_pressure")
        # Two channels, neither overwriting the other.
        assert {signal.channel for signal in airway} == {
            "airway_pressure",
            "airway_pressure__pod",
        }
        assert len(airway) == 2

    def test_the_pod_pressure_records_where_it_came_from(self):
        adapter = VentilatorAdapter()
        signals = adapter.to_signals(adapter.preprocess(_payload(self.LABELS)))
        pod = [s for s in signals if s.channel == "airway_pressure__pod"]
        assert {s.source for s in pod} == {"pod"}


class TestBackwardsCompatibility:
    def test_the_three_standard_keys_answer(self):
        bundle = split_channels(_payload(["Paw", "Flow", "Volume"]))
        for name in ("airway_pressure", "flow", "volume"):
            assert bundle[name].shape == (N,)

    def test_a_bundle_without_primary_still_resolves_a_channel(self):
        assert primary_channel({"volume": np.zeros(3)}, "volume") == "volume"

    def test_unknown_requested_channels_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown ventilator channel"):
            split_channels(_payload(["Paw", "Flow", "Volume"]), channels=("tiv",))
