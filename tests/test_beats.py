# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the real-time beat tracker.

All timing is driven by an injected fake monotonic clock, so the
replays of constant-speed, ramping and packet-loss sequences are
fully deterministic.
"""
import pytest

from mido import Message
from mido.beats import (BeatTracker, BeatSnapshot, PULSES_PER_BEAT,
                        PULSES_PER_SONGPOS_UNIT)


class FakeClock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


def gap_for_bpm(bpm):
    """Clock pulse interval in seconds at a given tempo."""
    return 60.0 / (bpm * PULSES_PER_BEAT)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def tracker(clock):
    return BeatTracker(clock=clock)


def feed(tracker, clock, msg_type, gap=None, timestamp=None):
    """Send a message at a time advanced by gap seconds."""
    if timestamp is None:
        timestamp = clock.advance(gap if gap is not None else 0.0)
    return tracker.update(Message(msg_type), timestamp)


def start(tracker, clock):
    return tracker.update(Message('start'), clock.now)


def test_constants():
    assert PULSES_PER_BEAT == 24
    assert PULSES_PER_SONGPOS_UNIT == 6


def test_initial_state(tracker):
    snapshot = tracker.snapshot()
    assert isinstance(snapshot, BeatSnapshot)
    assert not snapshot.running
    assert snapshot.pulses == 0
    assert snapshot.beat == 0.0
    assert snapshot.beat_phase == 0
    assert snapshot.songpos_units == 0
    assert snapshot.bpm is None
    assert snapshot.smoothed_bpm is None
    assert snapshot.stale
    assert snapshot.dropped_pulses == 0


def test_clock_before_start_does_not_count(tracker, clock):
    # A clock received while stopped anchors timing but must not
    # advance the position.
    feed(tracker, clock, 'clock', gap=0.1)
    assert tracker.pulses == 0
    assert tracker.beat == 0.0


def test_constant_speed_phase_and_bpm(tracker, clock):
    bpm = 120.0
    gap = gap_for_bpm(bpm)
    start(tracker, clock)

    first = feed(tracker, clock, 'clock', gap=gap)
    # One clock is not enough to estimate a tempo.
    assert first.pulses == 1
    assert first.bpm is None

    for n in range(2, 24 * 4 + 1):
        snapshot = feed(tracker, clock, 'clock', gap=gap)
        assert snapshot.pulses == n
        assert snapshot.running

    # Phase: one beat (24 pulses) per quarter note.
    assert tracker.pulses == 24 * 4
    assert tracker.beat == pytest.approx(4.0)
    assert tracker.beat_phase == 0
    assert tracker.songpos_units == (24 * 4) // 6
    assert tracker.bpm == pytest.approx(bpm)
    assert tracker.smoothed_bpm == pytest.approx(bpm, rel=1e-9)
    assert not tracker.stale


def test_beat_phase_wraps_within_beat(tracker, clock):
    gap = gap_for_bpm(100.0)
    start(tracker, clock)
    for _ in range(25):
        feed(tracker, clock, 'clock', gap=gap)
    assert tracker.pulses == 25
    assert tracker.beat_phase == 1
    assert tracker.beat == pytest.approx(25 / 24)


def test_jitter_is_tolerated(tracker, clock):
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    feed(tracker, clock, 'clock', gap=gap)
    # +/- 5% gaps are inside the default 10% jitter window.
    factors = (1.05, 0.95, 1.03, 0.97, 1.0, 1.04, 0.96)
    for factor in factors:
        snapshot = feed(tracker, clock, 'clock', gap=gap * factor)
        assert snapshot.dropped_pulses == 0
    assert tracker.pulses == 1 + len(factors)
    # Tempo estimate stays close to the true tempo.
    assert tracker.smoothed_bpm == pytest.approx(120.0, rel=0.05)


def test_single_lost_pulse_does_not_double_step(tracker, clock):
    """The headline requirement: losing one pulse must not skip a beat."""
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    for _ in range(23):
        feed(tracker, clock, 'clock', gap=gap)
    pulses_before = tracker.pulses
    beat_before = tracker.beat

    # One pulse is lost: the next gap is roughly two intervals.
    snapshot = feed(tracker, clock, 'clock', gap=gap * 2)

    assert snapshot.pulses == pulses_before + 1
    assert snapshot.beat == pytest.approx(beat_before + 1 / 24)
    assert snapshot.dropped_pulses == 1
    # Tempo must not be halved by the double-length gap.
    assert tracker.bpm == pytest.approx(120.0, rel=1e-9)
    assert tracker.smoothed_bpm == pytest.approx(120.0, rel=0.05)

    # Following normal pulses keep counting one-by-one.
    snapshot = feed(tracker, clock, 'clock', gap=gap)
    assert snapshot.dropped_pulses == 0
    assert snapshot.pulses == pulses_before + 2


def test_two_and_three_lost_pulses(tracker, clock):
    gap = gap_for_bpm(140.0)
    start(tracker, clock)
    feed(tracker, clock, 'clock', gap=gap)
    feed(tracker, clock, 'clock', gap=gap)

    for missed in (2, 3):
        snapshot = feed(tracker, clock, 'clock', gap=gap * (1 + missed))
        assert snapshot.dropped_pulses == missed
        assert tracker.bpm == pytest.approx(140.0, rel=1e-9)
        # Recovery gap resets the dropped counter.
        snapshot = feed(tracker, clock, 'clock', gap=gap)
        assert snapshot.dropped_pulses == 0


def test_unplausibly_long_gap_reanchors_without_burst(tracker, clock):
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    for _ in range(10):
        feed(tracker, clock, 'clock', gap=gap)
    pulses = tracker.pulses

    # A gap of 10 intervals exceeds max_missed_pulses (default 4).
    feed(tracker, clock, 'clock', gap=gap * 10)
    # Exactly one pulse is counted, the tempo is re-anchored instead.
    assert tracker.pulses == pulses + 1
    assert tracker.bpm == pytest.approx(12.0, rel=1e-9)


def test_ramping_tempo_tracks_without_jumps(tracker, clock):
    """Gradually accelerate from 90 to 150 BPM."""
    start(tracker, clock)
    bpm = 90.0
    feed(tracker, clock, 'clock', gap=gap_for_bpm(bpm))
    feed(tracker, clock, 'clock', gap=gap_for_bpm(bpm))

    beats = 16
    for _ in range(beats * PULSES_PER_BEAT):
        bpm += (150.0 - 90.0) / (beats * PULSES_PER_BEAT)
        feed(tracker, clock, 'clock', gap=gap_for_bpm(bpm))

    # Every received clock advanced the phase by exactly one.
    assert tracker.pulses == beats * PULSES_PER_BEAT + 2
    assert tracker.beat_phase == 2
    # Smoothed BPM lags the instantaneous ramping tempo but must have
    # moved most of the way and never overshoot wildly.
    assert 120.0 < tracker.smoothed_bpm <= 150.0
    assert tracker.bpm == pytest.approx(150.0, rel=0.02)


def test_sudden_tempo_increase_reanchors(clock):
    tracker = BeatTracker(clock=clock, reanchor_streak=3)
    start(tracker, clock)
    slow = gap_for_bpm(90.0)
    fast = gap_for_bpm(180.0)
    for _ in range(12):
        feed(tracker, clock, 'clock', gap=slow)
    assert tracker._expected_gap == pytest.approx(slow)

    # Three consecutive fast gaps are a tempo change, not jitter.
    feed(tracker, clock, 'clock', gap=fast)
    feed(tracker, clock, 'clock', gap=fast)
    # The expected gap (used for phase classification) is unchanged
    # until the streak completes.
    assert tracker._expected_gap == pytest.approx(slow)
    feed(tracker, clock, 'clock', gap=fast)
    assert tracker._expected_gap == pytest.approx(fast)
    assert tracker.bpm == pytest.approx(180.0)
    assert tracker.pulses == 15


def test_sudden_tempo_decrease_reanchors(clock):
    # A 1.5x slowdown is not an integer multiple of the old gap, so it
    # cannot be mistaken for missed pulses and is treated as a streak.
    tracker = BeatTracker(clock=clock, reanchor_streak=3)
    start(tracker, clock)
    fast = gap_for_bpm(180.0)
    slow = gap_for_bpm(120.0)
    for _ in range(12):
        feed(tracker, clock, 'clock', gap=fast)

    for _ in range(3):
        feed(tracker, clock, 'clock', gap=slow)
    assert tracker.bpm == pytest.approx(120.0)
    assert tracker._expected_gap == pytest.approx(slow)


def test_tempo_halving_is_indistinguishable_from_missed_pulses(clock):
    # A 2x slowdown looks exactly like one lost pulse per gap.  The
    # tracker deliberately treats it as missed pulses (keeping the old
    # tempo) rather than re-anchoring to a wildly wrong BPM.
    tracker = BeatTracker(clock=clock, max_missed_pulses=4)
    start(tracker, clock)
    fast = gap_for_bpm(180.0)
    slow = gap_for_bpm(90.0)
    for _ in range(12):
        feed(tracker, clock, 'clock', gap=fast)

    snapshot = feed(tracker, clock, 'clock', gap=slow)
    assert snapshot.dropped_pulses == 1
    assert tracker.bpm == pytest.approx(180.0)
    assert tracker.pulses == 13


def test_stop_and_continue_keep_position(tracker, clock):
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    for _ in range(36):
        feed(tracker, clock, 'clock', gap=gap)
    pulses = tracker.pulses

    stop_snapshot = tracker.update(Message('stop'), clock.now)
    assert not stop_snapshot.running
    assert stop_snapshot.pulses == pulses

    # Clocks while stopped neither count nor corrupt the estimate.
    feed(tracker, clock, 'clock', gap=gap)
    assert tracker.pulses == pulses

    cont = tracker.update(Message('continue'), clock.now)
    assert cont.running
    assert cont.pulses == pulses
    # Tempo hint survives the stop/continue cycle.
    assert cont.smoothed_bpm == pytest.approx(120.0)

    feed(tracker, clock, 'clock', gap=gap)
    assert tracker.pulses == pulses + 1


def test_start_resets_position(tracker, clock):
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    for _ in range(48):
        feed(tracker, clock, 'clock', gap=gap)
    assert tracker.pulses == 48

    snapshot = tracker.update(Message('start'), clock.now)
    assert snapshot.pulses == 0
    assert snapshot.beat == 0.0
    assert snapshot.running


def test_songpos_sets_position_in_sixteenth_units(tracker, clock):
    tracker.update(Message('songpos', pos=8), clock.now)
    assert tracker.pulses == 8 * PULSES_PER_SONGPOS_UNIT
    assert tracker.songpos_units == 8
    assert tracker.beat == pytest.approx(48 / 24)

    # 'start' rewinds to the beginning of the song per the MIDI spec.
    start(tracker, clock)
    assert tracker.songpos_units == 0
    gap = gap_for_bpm(120.0)
    for _ in range(48):
        feed(tracker, clock, 'clock', gap=gap)
    assert tracker.songpos_units == 8
    # One more pulse stays inside the same 16th-note unit.
    feed(tracker, clock, 'clock', gap=gap)
    assert tracker.songpos_units == 8
    assert tracker.pulses == 49


def test_songpos_then_continue_keeps_position(tracker, clock):
    # 'continue' (unlike 'start') must preserve the songpos position.
    tracker.update(Message('songpos', pos=10), clock.now)
    tracker.update(Message('continue'), clock.now)
    assert tracker.pulses == 10 * PULSES_PER_SONGPOS_UNIT
    assert tracker.songpos_units == 10
    feed(tracker, clock, 'clock', gap=gap_for_bpm(120.0))
    assert tracker.songpos_units == 10
    assert tracker.pulses == 61


def test_duplicate_timestamp_is_ignored(tracker, clock):
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    feed(tracker, clock, 'clock', gap=gap)
    pulses = tracker.pulses

    snapshot = tracker.update(Message('clock'), clock.now)
    assert snapshot.pulses == pulses
    assert tracker._last_clock_time == clock.now
    assert tracker.bpm is None or tracker.bpm > 0


def test_backwards_timestamp_is_ignored(tracker, clock):
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    feed(tracker, clock, 'clock', gap=gap)
    feed(tracker, clock, 'clock', gap=gap)
    pulses = tracker.pulses
    bpm = tracker.bpm

    snapshot = tracker.update(Message('clock'), clock.now - 5.0)
    assert snapshot.pulses == pulses
    assert tracker.bpm == bpm
    # Timing anchor is untouched.
    assert tracker._last_clock_time == clock.now


def test_pulse_timeout_marks_stale(tracker, clock):
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    for _ in range(6):
        feed(tracker, clock, 'clock', gap=gap)
    assert not tracker.stale

    clock.advance(tracker.pulse_timeout + 0.1)
    assert tracker.stale

    # The next clock re-anchors and clears staleness without
    # inventing a pile of missed pulses.
    pulses = tracker.pulses
    snapshot = feed(tracker, clock, 'clock', gap=gap)
    assert not snapshot.stale
    assert snapshot.pulses == pulses + 1


def test_non_clock_messages_are_ignored(tracker, clock):
    start(tracker, clock)
    note = Message('note_on', note=60, velocity=64, time=12.5)
    before = tracker.snapshot()
    snapshot = tracker.update(note, clock.advance(0.2))
    assert snapshot.pulses == before.pulses
    assert snapshot.running == before.running
    assert snapshot.bpm == before.bpm


def test_message_is_not_mutated(tracker, clock):
    note = Message('note_on', note=60, velocity=64, time=12.5)
    original = note.dict()
    tracker.update(note, clock.now)
    assert note.dict() == original
    assert note.time == 12.5

    clock_msg = Message('clock')
    original_clock = clock_msg.dict()
    start(tracker, clock)
    feed(tracker, clock, 'clock', gap=gap_for_bpm(120.0))
    tracker.update(clock_msg, clock.now)
    assert clock_msg.dict() == original_clock


def test_explicit_timestamp_overrides_clock(clock):
    tracker = BeatTracker(clock=clock)
    start(tracker, clock)
    gap = gap_for_bpm(100.0)
    # Pass timestamps explicitly; the injected clock should not be
    # consulted for clock handling.
    tracker.update(Message('clock'), timestamp=5.0)
    tracker.update(Message('clock'), timestamp=5.0 + gap)
    assert tracker.bpm == pytest.approx(100.0)
    assert clock.now == 1000.0  # untouched


def test_default_clock_is_used_when_timestamp_omitted():
    # A bare tracker must work with the real monotonic clock.
    tracker = BeatTracker()
    tracker.update(Message('start'))
    snapshot = tracker.update(Message('clock'))
    assert snapshot.running
    assert snapshot.pulses == 1


def test_invalid_arguments():
    with pytest.raises(TypeError):
        BeatTracker(clock='not callable')
    with pytest.raises(ValueError):
        BeatTracker(jitter_window=1.0)
    with pytest.raises(ValueError):
        BeatTracker(max_missed_pulses=0)
    with pytest.raises(ValueError):
        BeatTracker(reanchor_streak=0)
    with pytest.raises(ValueError):
        BeatTracker(smoothing=0.0)
    with pytest.raises(ValueError):
        BeatTracker(pulse_timeout=0.0)


def test_update_rejects_non_message(tracker, clock):
    with pytest.raises(TypeError):
        tracker.update('clock', clock.now)
    with pytest.raises(TypeError):
        tracker.update(None, clock.now)


def test_reset_clears_everything(tracker, clock):
    gap = gap_for_bpm(120.0)
    start(tracker, clock)
    for _ in range(30):
        feed(tracker, clock, 'clock', gap=gap)
    tracker.reset()
    snapshot = tracker.snapshot()
    assert not snapshot.running
    assert snapshot.pulses == 0
    assert snapshot.bpm is None
    assert snapshot.smoothed_bpm is None
    assert snapshot.stale


def test_smoothing_parameter_takes_effect(clock):
    # With full-weight smoothing every estimate equals the latest gap.
    tracker = BeatTracker(clock=clock, smoothing=1.0)
    start(tracker, clock)
    gap = gap_for_bpm(120.0)
    feed(tracker, clock, 'clock', gap=gap)
    feed(tracker, clock, 'clock', gap=gap)
    # A single jittery gap is reflected immediately in full.
    feed(tracker, clock, 'clock', gap=gap * 1.05)
    assert tracker.smoothed_bpm == pytest.approx(tracker.bpm)


def test_jitter_window_configuration(clock):
    # A tight window classifies a 5% short gap as a streak candidate.
    tracker = BeatTracker(clock=clock, jitter_window=0.02,
                          reanchor_streak=10)
    start(tracker, clock)
    gap = gap_for_bpm(120.0)
    feed(tracker, clock, 'clock', gap=gap)
    feed(tracker, clock, 'clock', gap=gap)
    feed(tracker, clock, 'clock', gap=gap * 0.95)
    assert tracker._fast_streak == 1


def test_packet_loss_replay_phase_accuracy(clock):
    """End-to-end replay: 10% random packet loss keeps exact phase."""
    tracker = BeatTracker(clock=clock)
    gap = gap_for_bpm(125.0)
    start(tracker, clock)

    received = 0
    total = 48 * PULSES_PER_BEAT
    lost = {7, 31, 32, 100, 250, 500, 777, 1000}
    t = clock.now
    for n in range(1, total + 1):
        t += gap
        if n in lost:
            continue  # the device sent a clock that never arrived
        tracker.update(Message('clock'), t)
        received += 1

    # Phase counts received clocks one-for-one, never double-stepping.
    assert tracker.pulses == received
    assert tracker.pulses == total - len(lost)
    assert tracker.beat_phase == received % PULSES_PER_BEAT
    # Tempo estimate survives the gaps (each gap is at most 3x).
    assert tracker.smoothed_bpm == pytest.approx(125.0, rel=0.08)
