# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io
import math

import pytest

from mido import Message, MetaMessage, MidiFile, MidiTrack
from mido.midifiles import (DEFAULT_SKIP_CONTROLS,
                            simplify_control_changes, simplify_midi_file)


def cc(control, value, time=0, channel=0):
    return Message('control_change', channel=channel, control=control,
                   value=value, time=time)


def curve(track, control=1, channel=0):
    """Extract (absolute tick, value) points for one controller."""
    now = 0
    points = []
    for msg in track:
        now += msg.time
        if (msg.type == 'control_change' and msg.control == control
                and msg.channel == channel):
            points.append((now, msg.value))
    return points


def min_error_to_curve(points, t, v, time_error):
    """Independent reconstruction: smallest value error when
    reconstructing (t, v) from the simplified curve by linear
    interpolation, allowing a time shift of up to time_error ticks."""
    if len(points) == 1:
        return abs(points[0][1] - v)
    best = math.inf
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        lo = max(t - time_error, t0)
        hi = min(t + time_error, t1)
        if lo > hi:
            continue

        def interp(tc):
            return v0 + (v1 - v0) * (tc - t0) / (t1 - t0)

        flo = interp(lo) - v
        fhi = interp(hi) - v
        if flo == 0 or (flo < 0) != (fhi < 0):
            return 0.0
        best = min(best, abs(flo), abs(fhi))
    return best


def assert_within_tolerance(original, simplified, value_error, time_error):
    """Every original point must be reconstructable from the simplified
    curve within (value_error, time_error)."""
    for t, v in original:
        error = min_error_to_curve(simplified, t, v, time_error)
        assert error <= value_error + 1e-6


def make_midifile(tracks, type=1, ticks_per_beat=480):
    mid = MidiFile(type=type, ticks_per_beat=ticks_per_beat)
    mid.tracks.extend(tracks)
    return mid


# --- dense smooth curve ------------------------------------------------

def test_dense_smooth_curve_shrinks_and_stays_within_tolerance():
    # A smooth sine-like automation sweep, 10 messages per beat.
    n = 400
    track = MidiTrack()
    for i in range(n):
        value = round(64 + 60 * math.sin(i / 40))
        track.append(cc(1, value, time=48))
    track.append(MetaMessage('end_of_track', time=0))

    original = curve(track)
    new_track = simplify_control_changes(track, value_error=1.0,
                                         time_error=10)
    simplified = curve(new_track)

    # Endpoints kept, massive reduction.
    assert simplified[0] == original[0]
    assert simplified[-1] == original[-1]
    assert len(simplified) < len(original) // 4

    # Independent reconstruction of every original point.
    assert_within_tolerance(original, simplified, 1.0, 10)


def test_linear_ramp_collapses_to_endpoints():
    track = MidiTrack([cc(1, i, time=10) for i in range(128)])
    new_track = simplify_control_changes(track, value_error=0.5)
    assert curve(new_track) == [(10, 0), (1280, 127)]


def test_zero_tolerance_keeps_everything_not_exactly_redundant():
    track = MidiTrack([cc(1, v, time=10) for v in [0, 1, 3, 4, 5]])
    new_track = simplify_control_changes(track, value_error=0,
                                         time_error=0)
    # (20,1) breaks the line from (10,0), and (30,3) breaks the line
    # from (20,1); only (40,4) is exactly redundant with (30,3)-(50,5).
    assert curve(new_track) == [(10, 0), (20, 1), (30, 3), (50, 5)]


# --- step curve --------------------------------------------------------

def test_step_curve_keeps_every_jump():
    values = [0] * 20 + [127] * 20 + [10] * 20 + [10, 11, 12, 13]
    track = MidiTrack([cc(1, v, time=24) for v in values])
    new_track = simplify_control_changes(track, value_error=2.0,
                                         time_error=5)
    points = curve(new_track)
    # The two jumps (0->127, 127->10) must survive with both sides.
    # (First point is at tick 24 since times are deltas.)
    assert (20 * 24, 0) in points
    assert (21 * 24, 127) in points
    assert (40 * 24, 127) in points
    assert (41 * 24, 10) in points
    # And the reconstruction still brackets the steps.
    assert_within_tolerance(curve(track), points, 2.0, 5)


def test_single_point_curve_is_kept():
    track = MidiTrack([cc(1, 64, time=100),
                       MetaMessage('end_of_track', time=0)])
    new_track = simplify_control_changes(track, value_error=100)
    assert curve(new_track) == [(100, 64)]


def test_two_point_curve_is_kept():
    track = MidiTrack([cc(1, 0, time=0), cc(1, 127, time=480)])
    new_track = simplify_control_changes(track, value_error=100)
    assert curve(new_track) == [(0, 0), (480, 127)]


# --- duplicate ticks ----------------------------------------------------

def test_duplicate_ticks_collapse_to_last_value():
    track = MidiTrack([cc(1, 10, time=0),
                       cc(1, 20, time=0),
                       cc(1, 30, time=0),
                       cc(1, 90, time=100)])
    new_track = simplify_control_changes(track, value_error=0.5)
    # Only the last value at tick 0 survives.
    assert curve(new_track) == [(0, 30), (100, 90)]


# --- tolerance validation ----------------------------------------------

@pytest.mark.parametrize('bad', [-1, -0.5, float('nan'), float('inf'),
                                 '1', None, True])
def test_invalid_value_error_raises(bad):
    with pytest.raises(ValueError):
        simplify_control_changes(MidiTrack(), value_error=bad)


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), 'x', False])
def test_invalid_time_error_raises(bad):
    with pytest.raises(ValueError):
        simplify_control_changes(MidiTrack(), time_error=bad)


@pytest.mark.parametrize('bad', [[-1], [128], [1.5], ['1']])
def test_invalid_skip_controls_raises(bad):
    with pytest.raises(ValueError):
        simplify_control_changes(MidiTrack(), skip_controls=bad)


# --- bank select and skip_controls --------------------------------------

def test_bank_select_is_untouched_by_default():
    track = MidiTrack([cc(0, 0, time=0), cc(32, 0, time=0),
                       cc(0, 1, time=10), cc(32, 1, time=0),
                       cc(0, 2, time=10), cc(32, 2, time=0)])
    new_track = simplify_control_changes(track, value_error=10,
                                         time_error=10)
    assert list(new_track) == list(track)
    assert DEFAULT_SKIP_CONTROLS == (0, 32)


def test_skip_controls_can_be_changed():
    track = MidiTrack([cc(0, 0, time=0), cc(0, 1, time=10),
                       cc(0, 2, time=10)])
    new_track = simplify_control_changes(track, value_error=10,
                                         skip_controls=())
    assert curve(new_track, control=0) == [(0, 0), (20, 2)]


# --- ordering guarantees -------------------------------------------------

def test_other_messages_and_same_tick_order_preserved():
    track = MidiTrack([
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        cc(1, 0, time=0),
        cc(1, 1, time=10),
        MetaMessage('track_name', name='a', time=0),
        cc(1, 2, time=10),
        Message('note_off', channel=0, note=60, velocity=0, time=5),
        cc(1, 3, time=5),
    ])
    new_track = simplify_control_changes(track, value_error=10)

    # The ramp collapses to its endpoints; everything else stays in
    # order, with identical absolute times.
    def signature(tr):
        now = 0
        sig = []
        for msg in tr:
            now += msg.time
            sig.append((now, msg.type, getattr(msg, 'note', None),
                        getattr(msg, 'value', None)))
        return sig

    assert signature(new_track) == [
        (0, 'note_on', 60, None),
        (0, 'control_change', None, 0),
        (10, 'track_name', None, None),
        (25, 'note_off', 60, None),
        (30, 'control_change', None, 3),
    ]


def test_other_tracks_are_not_touched():
    dense = MidiTrack([cc(1, i % 4, time=10) for i in range(100)])
    other = MidiTrack([cc(2, i % 4, time=10) for i in range(100)])
    mid = make_midifile([dense, other])
    new_mid = simplify_midi_file(mid, value_error=2, time_error=5)

    assert len(new_mid.tracks[0]) < 100
    # Same simplification applied per track; crucially the source
    # tracks are unchanged.
    assert len(mid.tracks[0]) == 100
    assert len(mid.tracks[1]) == 100
    assert list(new_mid.tracks[1]) == list(simplify_control_changes(
        other, value_error=2, time_error=5))


def test_source_file_object_is_not_modified():
    track = MidiTrack([cc(1, i, time=10) for i in range(50)])
    mid = make_midifile([track])
    before = [list(t) for t in mid.tracks]
    new_mid = simplify_midi_file(mid, value_error=5)
    assert [list(t) for t in mid.tracks] == before
    assert new_mid is not mid
    assert new_mid.filename is None
    assert new_mid.ticks_per_beat == mid.ticks_per_beat
    assert new_mid.type == mid.type


# --- 14 bit controller pairs ---------------------------------------------

def test_14bit_pairs_simplified_as_one_curve():
    # Smooth 14 bit sweep on controllers 1 (MSB) and 33 (LSB).
    track = MidiTrack()
    for i in range(200):
        value14 = round(8192 + 8000 * math.sin(i / 30))
        track.append(cc(1, (value14 >> 7) & 0x7f, time=24))
        track.append(cc(33, value14 & 0x7f, time=0))

    new_track = simplify_control_changes(track, value_error=20.0,
                                         time_error=10, msb=True)

    def curve14(tr):
        now = 0
        msb = lsb = 0
        points = []
        for msg in tr:
            now += msg.time
            if msg.type == 'control_change':
                if msg.control == 1:
                    msb = msg.value
                elif msg.control == 33:
                    lsb = msg.value
                else:
                    continue
                points.append((now, (msb << 7) | lsb))
        # Same-tick pairs: keep only the last (complete) value.
        out = []
        for p in points:
            if out and out[-1][0] == p[0]:
                out[-1] = p
            else:
                out.append(p)
        return out

    original14 = curve14(track)
    simplified14 = curve14(new_track)
    assert len(simplified14) < len(original14) // 4
    assert simplified14[0][0] == original14[0][0]
    assert simplified14[-1][0] == original14[-1][0]
    assert abs(simplified14[0][1] - original14[0][1]) <= 20
    assert_within_tolerance(original14, simplified14, 20.0, 10)


def test_14bit_kept_point_reemits_both_halves():
    # MSB changes at tick 0, then a long flat LSB wiggle: the kept
    # endpoint must restate both MSB and LSB.
    track = MidiTrack([cc(1, 5, time=0)])
    track += MidiTrack(cc(33, v, time=10) for v in [0, 1, 0, 1, 0])
    new_track = simplify_control_changes(track, value_error=200.0,
                                         time_error=0, msb=True)
    msgs = [m for m in new_track if m.type == 'control_change']
    # Endpoints only, each restating both halves.
    assert len(msgs) == 4
    assert [(m.control, m.value) for m in msgs[:2]] == [(1, 5), (33, 0)]
    assert [(m.control, m.value) for m in msgs[2:]] == [(1, 5), (33, 0)]


def test_14bit_msb_only_curve_does_not_invent_lsb():
    track = MidiTrack([cc(7, v, time=10) for v in [0, 1, 2, 3]])
    new_track = simplify_control_changes(track, value_error=300.0,
                                         msb=True)
    controls = [m.control for m in new_track
                if m.type == 'control_change']
    assert set(controls) == {7}  # no controller 39 appears


def test_14bit_bank_select_pair_skipped():
    track = MidiTrack([cc(0, 0, time=0), cc(32, 0, time=0),
                       cc(0, 1, time=10), cc(32, 1, time=0)])
    new_track = simplify_control_changes(track, value_error=1000,
                                         msb=True)
    assert list(new_track) == list(track)


# --- output size, read-back ----------------------------------------------

def test_output_size_shrinks_and_reads_back():
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='automation', time=0))
    for i in range(2000):
        value = round(64 + 63 * math.sin(i / 100))
        track.append(cc(11, value, time=12))
    track.append(Message('note_on', channel=0, note=64, velocity=100,
                         time=480))
    track.append(Message('note_off', channel=0, note=64, velocity=0,
                         time=480))
    track.append(MetaMessage('end_of_track', time=0))
    mid = make_midifile([track])

    new_mid = simplify_midi_file(mid, value_error=1.0, time_error=20)

    def to_bytes(m):
        buf = io.BytesIO()
        m.save(file=buf)
        return buf.getvalue()

    original_bytes = to_bytes(mid)
    simplified_bytes = to_bytes(new_mid)
    assert len(simplified_bytes) < len(original_bytes) // 2

    # Read the simplified file back and compare message by message.
    read_back = MidiFile(file=io.BytesIO(simplified_bytes))
    assert read_back.ticks_per_beat == mid.ticks_per_beat
    assert len(read_back.tracks) == 1
    assert list(read_back.tracks[0]) == list(new_mid.tracks[0])

    # The note and meta messages survive the round trip.
    types = [m.type for m in read_back.tracks[0]]
    assert types.count('note_on') == 1
    assert types.count('note_off') == 1
    assert types.count('track_name') == 1
    assert types[-1] == 'end_of_track'
