"""Tests for `m3resp.processing.captures.capture_value`."""

from __future__ import annotations

from m3resp.processing.captures import capture_value


def test_stores_the_value_under_the_key():
    captures: dict = {}

    capture_value(captures, "sample_frequency", 20.0)

    assert captures == {"sample_frequency": 20.0}


def test_a_second_value_replaces_the_first():
    captures: dict = {"order": 2}

    capture_value(captures, "order", 4)

    assert captures == {"order": 4}


def test_append_to_list_collects_every_value():
    captures: dict = {}

    capture_value(captures, "frequency_bands", (49.0, 51.0), append_to_list=True)
    capture_value(captures, "frequency_bands", (99.0, 101.0), append_to_list=True)

    assert captures == {"frequency_bands": [(49.0, 51.0), (99.0, 101.0)]}
