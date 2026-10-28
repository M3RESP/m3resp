"""Alias maps for unit strings (`m3resp.data.units`) and metric names
(`m3resp.data.metrics`).

Both modules keep a process-wide active map, so every test starts and ends
with the built-in defaults.
"""

from __future__ import annotations

import json

import pytest
import yaml

from m3resp.data import metrics, units


@pytest.fixture(autouse=True)
def _reset_alias_maps():
    units.reset_unit_aliases()
    metrics.reset_metric_aliases()
    yield
    units.reset_unit_aliases()
    metrics.reset_metric_aliases()


# Both modules expose the same API under different names; parametrizing over
# them keeps the two in step.
ALIAS_APIS = [
    pytest.param(
        {
            "normalize": units.normalize_unit,
            "register": units.register_unit_alias,
            "aliases": units.unit_aliases,
            "reset": units.reset_unit_aliases,
            "save": units.save_unit_aliases,
            "load": units.load_unit_aliases,
            "defaults": units._DEFAULT_UNIT_ALIASES,
        },
        id="units",
    ),
    pytest.param(
        {
            "normalize": metrics.normalize_metric_name,
            "register": metrics.register_metric_alias,
            "aliases": metrics.metric_aliases,
            "reset": metrics.reset_metric_aliases,
            "save": metrics.save_metric_aliases,
            "load": metrics.load_metric_aliases,
            "defaults": metrics._DEFAULT_METRIC_ALIASES,
        },
        id="metrics",
    ),
]


class TestNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("uV", "µV"), (" MicroVolts ", "µV"), ("mv", "mV"), ("A.U", "a.u.")],
    )
    def test_known_unit_aliases_are_canonicalized(self, raw, expected):
        assert units.normalize_unit(raw) == expected

    def test_unknown_unit_is_returned_unchanged(self):
        assert units.normalize_unit("cmH2O") == "cmH2O"

    def test_unit_none_stays_none(self):
        assert units.normalize_unit(None) is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("RR", "respiratory_rate"),
            (" breath_rate ", "respiratory_rate"),
            ("tiv", "tidal_impedance_variation"),
            ("VT", "tidal_volume"),
        ],
    )
    def test_known_metric_aliases_are_canonicalized(self, raw, expected):
        assert metrics.normalize_metric_name(raw) == expected

    def test_unknown_metric_returns_none(self):
        """Unlike units, an unknown metric name is not echoed back, so a
        custom label is never tagged as a standard metric."""
        assert metrics.normalize_metric_name("my_custom_index") is None

    def test_metric_none_stays_none(self):
        assert metrics.normalize_metric_name(None) is None


@pytest.mark.parametrize("api", ALIAS_APIS)
class TestRegistry:
    def test_register_is_case_and_whitespace_insensitive(self, api):
        api["register"]("  NewAlias ", "canonical")
        assert api["normalize"]("newalias") == "canonical"
        assert api["aliases"]()["newalias"] == "canonical"

    def test_aliases_returns_a_copy(self, api):
        snapshot = api["aliases"]()
        snapshot["mutated"] = "x"
        assert "mutated" not in api["aliases"]()

    def test_reset_drops_registrations(self, api):
        api["register"]("temporary", "x")
        api["reset"]()
        assert api["aliases"]() == api["defaults"]

    @pytest.mark.parametrize("suffix", [".json", ".yaml"])
    def test_save_only_custom_writes_additions_and_overrides(
        self, api, tmp_path, suffix
    ):
        overridden = next(iter(api["defaults"]))
        api["register"]("added", "new_canonical")
        api["register"](overridden, "overridden_canonical")

        path = api["save"](tmp_path / f"aliases{suffix}")

        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
        assert payload == {
            "added": "new_canonical",
            overridden: "overridden_canonical",
        }

    def test_save_full_map(self, api, tmp_path):
        api["register"]("added", "new_canonical")

        path = api["save"](tmp_path / "aliases.json", only_custom=False)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {**api["defaults"], "added": "new_canonical"}

    @pytest.mark.parametrize("suffix", [".json", ".yml"])
    def test_save_then_load_round_trips(self, api, tmp_path, suffix):
        api["register"]("added", "new_canonical")
        path = api["save"](tmp_path / f"aliases{suffix}")
        api["reset"]()

        active = api["load"](path)

        assert active["added"] == "new_canonical"
        assert api["normalize"]("ADDED") == "new_canonical"

    def test_load_merges_by_default(self, api, tmp_path):
        api["register"]("adhoc", "kept")
        path = tmp_path / "aliases.json"
        path.write_text(json.dumps({"from_file": "loaded"}), encoding="utf-8")

        active = api["load"](path)

        assert active["adhoc"] == "kept"
        assert active["from_file"] == "loaded"

    def test_load_replace_resets_to_defaults_first(self, api, tmp_path):
        api["register"]("adhoc", "dropped")
        path = tmp_path / "aliases.json"
        path.write_text(json.dumps({"from_file": "loaded"}), encoding="utf-8")

        active = api["load"](path, replace=True)

        assert "adhoc" not in active
        assert active == {**api["defaults"], "from_file": "loaded"}

    def test_load_empty_yaml_keeps_active_map(self, api, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")

        assert api["load"](path) == api["defaults"]

    def test_load_rejects_non_mapping(self, api, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- uv\n- mv\n", encoding="utf-8")

        with pytest.raises(TypeError, match="must be a mapping"):
            api["load"](path)
