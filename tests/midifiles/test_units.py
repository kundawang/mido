# SPDX-FileCopyrightText: 2017 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import pytest

from mido.midifiles.units import (
    SMPTE_FPS,
    bpm2tempo,
    second2tick,
    smpte_second2tick,
    smpte_tick2second,
    tempo2bpm,
    tick2second,
)


def test_tempo2bpm():
    # default tempo (500000 ms per quarter note)
    assert tempo2bpm(500000) == 120
    assert tempo2bpm(1000000) == 60
    # x/4 time signature: 4 beats (quarter notes) per bar
    # 60 bpm: 1 beat / sec = 1 quarter note / sec: 1 sec / quarter note
    assert tempo2bpm(1000000, time_signature=(4, 4)) == 60
    # 120 bpm: 2 beat / sec = 2 quarter note / sec: 0.5 sec / quarter note
    assert tempo2bpm(500000, time_signature=(4, 4)) == 120
    assert tempo2bpm(500000, time_signature=(3, 4)) == 120  # 3/4 is the same
    # x/2 time signature: 2 beats (half notes) per bar
    # 60 bpm: 1 beat / sec = 2 quarter note / sec: 0.5 sec / quarter note
    assert tempo2bpm(500000, time_signature=(2, 2)) == 60
    # 120 bpm: 2 beat / sec = 4 quarter note / sec: 0.25 sec / quarter note
    assert tempo2bpm(250000, time_signature=(2, 2)) == 120
    assert tempo2bpm(250000, time_signature=(3, 2)) == 120  # 3/2 is the same
    # x/8 time signature: 8 beats (eighth notes) per bar
    # 60 bpm: 1 beat / sec = 0.5 quarter note / sec: 2 sec / quarter note
    assert tempo2bpm(2000000, time_signature=(8, 8)) == 60
    # 120 bpm: 2 beat / sec = 1 quarter note / sec: 1 sec / quarter note
    assert tempo2bpm(1000000, time_signature=(8, 8)) == 120
    assert tempo2bpm(1000000, time_signature=(8, 8)) == 120  # 6/8 is the same


def test_bpm2tempo():
    # default tempo (500000 ms per quarter note)
    assert bpm2tempo(60) == 1000000
    assert bpm2tempo(120) == 500000
    # x/4 time signature: 4 beats (quarter notes) per bar
    # 60 bpm: 1 beat / sec = 1 quarter note / sec: 1 sec / quarter note
    assert bpm2tempo(60, time_signature=(4, 4)) == 1000000
    # 120 bpm: 2 beat / sec = 2 quarter note / sec: 0.5 sec / quarter note
    assert bpm2tempo(120, time_signature=(4, 4)) == 500000
    assert bpm2tempo(120, time_signature=(3, 4)) == 500000  # 3/4 is the same
    # x/2 time signature: 2 beats (half notes) per bar
    # 60 bpm: 1 beat / sec = 2 quarter note / sec: 0.5 sec / quarter note
    assert bpm2tempo(60, time_signature=(2, 2)) == 500000
    # 120 bpm: 2 beat / sec = 4 quarter note / sec: 0.25 sec / quarter note
    assert bpm2tempo(120, time_signature=(2, 2)) == 250000
    assert bpm2tempo(120, time_signature=(3, 2)) == 250000  # 3/2 is the same
    # x/8 time signature: 8 beats (eighth notes) per bar
    # 60 bpm: 1 beat / sec = 0.5 quarter note / sec: 2 sec / quarter note
    assert bpm2tempo(60, time_signature=(8, 8)) == 2000000
    # 120 bpm: 2 beat / sec = 1 quarter note / sec: 1 sec / quarter note
    assert bpm2tempo(120, time_signature=(8, 8)) == 1000000
    assert bpm2tempo(120, time_signature=(8, 8)) == 1000000  # 6/8 is the same


# TODO: these tests could be improved with better test values such as
# edge cases.
def test_tick2second():
    # default tempo (500000 ms per quarter note)
    assert tick2second(1, ticks_per_beat=100, tempo=500000) == 0.005
    assert tick2second(2, ticks_per_beat=100, tempo=100000) == 0.002


def test_second2tick():
    # default tempo (500000 ms per quarter note)
    assert second2tick(0.001, ticks_per_beat=100, tempo=500000) == 0
    assert second2tick(0.004, ticks_per_beat=100, tempo=500000) == 1
    assert second2tick(0.005, ticks_per_beat=100, tempo=500000) == 1
    assert second2tick(0.0015, ticks_per_beat=100, tempo=100000) == 2
    assert second2tick(0.0025, ticks_per_beat=100, tempo=100000) == 2


# One frame lasts 1 / fps seconds and is split into ticks_per_frame
# ticks, so one tick lasts 1 / (fps * ticks_per_frame) seconds.
@pytest.mark.parametrize(
    ('fps', 'ticks_per_frame', 'expected_seconds_per_tick'),
    [
        (24, 40, 1.0 / 960),
        (25, 40, 1.0 / 1000),
        (29.97, 40, 1.0 / (29.97 * 40)),
        (30, 40, 1.0 / 1200),
    ],
)
def test_smpte_tick2second(fps, ticks_per_frame,
                           expected_seconds_per_tick):
    assert smpte_tick2second(0, fps, ticks_per_frame) == 0
    assert smpte_tick2second(1, fps, ticks_per_frame) == \
        pytest.approx(expected_seconds_per_tick)
    assert smpte_tick2second(120, fps, ticks_per_frame) == \
        pytest.approx(120 * expected_seconds_per_tick)


def test_smpte_whole_frames_exact_seconds():
    # At 25 fps / 40 ticks per frame a whole frame (40 ticks) is
    # exactly 1/25 = 0.04 seconds; 1000 ticks make one second.
    assert smpte_tick2second(40, 25, 40) == pytest.approx(0.04)
    assert smpte_tick2second(1000, 25, 40) == pytest.approx(1.0)
    assert smpte_tick2second(1200, 30, 40) == pytest.approx(1.0)
    assert smpte_tick2second(960, 24, 40) == pytest.approx(1.0)


def test_smpte_29_97_is_drop_frame_rate():
    # The nominal 29.97 fps rate is 30000 / 1001.
    assert smpte_tick2second(2997, 29.97, 1) == pytest.approx(100.0)


@pytest.mark.parametrize('fps', SMPTE_FPS)
def test_smpte_second2tick_is_inverse(fps):
    for tick in (0, 1, 40, 120, 1000):
        seconds = smpte_tick2second(tick, fps, 40)
        assert smpte_second2tick(seconds, fps, 40) == tick


def test_smpte_second2tick_rounding():
    # Half a tick at 24 fps / 40 tpf rounds to zero.
    half_tick = 0.5 / (24 * 40)
    assert smpte_second2tick(half_tick, 24, 40) == 0
    assert smpte_second2tick(half_tick * 3, 24, 40) == 2


@pytest.mark.parametrize('fps', [23.976, 29, 0, -24, '24', None])
def test_smpte_invalid_frame_rate(fps):
    with pytest.raises(ValueError):
        smpte_tick2second(1, fps, 40)
    with pytest.raises(ValueError):
        smpte_second2tick(1.0, fps, 40)


@pytest.mark.parametrize('ticks_per_frame', [0, -1, -40])
def test_smpte_non_positive_ticks_per_frame(ticks_per_frame):
    with pytest.raises(ValueError):
        smpte_tick2second(1, 24, ticks_per_frame)


def test_smpte_ticks_per_frame_overflow():
    with pytest.raises(OverflowError):
        smpte_tick2second(1, 24, 256)


def test_smpte_ticks_per_frame_wrong_type():
    with pytest.raises(TypeError):
        smpte_tick2second(1, 24, 40.0)


def test_smpte_negative_tick():
    with pytest.raises(ValueError):
        smpte_tick2second(-1, 24, 40)


def test_smpte_negative_second():
    with pytest.raises(ValueError):
        smpte_second2tick(-0.1, 24, 40)
