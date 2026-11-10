"""Tests of the sign breaths of the pixel breath search in the EIT steps."""

from __future__ import annotations

import numpy as np

import m3resp.workflows.steps.eit._shared as shared
from m3resp.core.session import M3Session
from m3resp.data import Signal
from m3resp.eit.processing import BreathDetectionSettings


def _waveform() -> Signal:
    """A 60 s breathing waveform at 0.25 Hz, sampled at 20 Hz."""

    time = np.arange(1200) / 20.0
    return Signal(
        values=np.sin(2 * np.pi * 0.25 * time) + 3.0,
        time=time,
        sample_frequency=20.0,
        unit="AU",
        name="global impedance",
        modality="eit",
    )


def test_the_default_settings_give_no_sign_breaths() -> None:
    sign_breaths = shared._sign_breaths_used_by(
        BreathDetectionSettings(), _waveform(), M3Session()
    )

    assert sign_breaths is None


def test_the_sign_breaths_are_found_once_per_signal_and_session(monkeypatch) -> None:
    calls = []
    find_breaths = shared.find_breaths

    def counting_find_breaths(signal, settings):
        calls.append(settings)
        return find_breaths(signal, settings)

    monkeypatch.setattr(shared, "find_breaths", counting_find_breaths)
    waveform = _waveform()
    session = M3Session()
    custom = BreathDetectionSettings(minimum_duration=1.5)

    first = shared._sign_breaths_used_by(custom, waveform, session)
    second = shared._sign_breaths_used_by(custom, waveform, session)

    assert first is second
    assert len(first) == 14
    assert calls == [BreathDetectionSettings()]
