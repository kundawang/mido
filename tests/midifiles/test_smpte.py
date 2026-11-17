# SPDX-FileCopyrightText: 2024 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io
import struct

import pytest

from mido.messages import Message
from mido.midifiles.meta import MetaMessage
from mido.midifiles.midifiles import (
    DEFAULT_TICKS_PER_BEAT,
    MidiFile,
    MidiTrack,
    decode_division,
    encode_division,
)

# SMPTE frame rate code stored in the upper (negative) byte of the
# division word, mapped to the frame rate in frames per second.
FRAME_RATES = [
    (-24, 24),
    (-25, 25),
    (-29, 29.97),
    (-30, 30),
]

TICKS_PER_FRAME = 40

# Each message below is separated by 120 ticks.
DELTA = 120


def make_track(tempo=1_000_000):
    """Build a track with two notes and a set_tempo event.

    Layout (all delta times are 120 ticks except end_of_track):

        note_on   d=120
        set_tempo d=120
        note_off  d=120
        end       d=0

    The tempo change must affect PPQ files but be ignored by SMPTE
    files.
    """
    track = b'MTrk' + struct.pack('>L', 4 + 7 + 4 + 4)
    track += bytes([DELTA]) + bytes.fromhex('90 40 40')
    track += bytes([DELTA]) + b'\xff\x51\x03' + tempo.to_bytes(3, 'big')
    track += bytes([DELTA]) + bytes.fromhex('80 40 40')
    track += b'\x00\xff\x2f\x00'
    return track


def make_file_bytes(division, track):
    header = b'MThd' + struct.pack('>L', 6)
    header += struct.pack('>HHh', 1, 1, division)
    return header + track


def read_smpte_file(code, ticks_per_frame=TICKS_PER_FRAME, track=None):
    division = (code << 8) | ticks_per_frame
    data = make_file_bytes(division, track or make_track())
    return MidiFile(file=io.BytesIO(data))


# ---------------------------------------------------------------------
# Division word encoding / decoding
# ---------------------------------------------------------------------

@pytest.mark.parametrize(('code', 'fps'), FRAME_RATES)
def test_encode_division_smpte(code, fps):
    division = encode_division(fps=fps, ticks_per_frame=TICKS_PER_FRAME)
    assert division == (code << 8) | TICKS_PER_FRAME
    assert division < 0


def test_encode_division_ppq():
    assert encode_division(ticks_per_beat=480) == 480
    assert encode_division() == DEFAULT_TICKS_PER_BEAT


@pytest.mark.parametrize(('code', 'fps'), FRAME_RATES)
def test_decode_division_smpte(code, fps):
    division = (code << 8) | TICKS_PER_FRAME
    assert decode_division(division) == \
        ('smpte', fps, TICKS_PER_FRAME)


def test_decode_division_ppq():
    assert decode_division(480) == ('ppq', 480)


@pytest.mark.parametrize('division', [
    (-23 << 8) | 40,   # invalid frame rate code
    (-26 << 8) | 40,
    (-31 << 8) | 40,
    (-1 << 8) | 40,    # frame code -1 is not a valid SMPTE rate
])
def test_decode_division_invalid_frame_rate(division):
    with pytest.raises(ValueError):
        decode_division(division)


def test_decode_division_zero_ticks_per_frame():
    with pytest.raises(ValueError):
        decode_division((-24 << 8) | 0)
    # As a signed 16-bit word 0x8000 decodes to -32768:
    # frame code -128, zero tpf.
    with pytest.raises(ValueError):
        decode_division(-32768)


def test_decode_division_zero_word():
    with pytest.raises(ValueError):
        decode_division(0)


def test_encode_division_overflow():
    with pytest.raises(OverflowError):
        encode_division(ticks_per_beat=0x8000)
    with pytest.raises(OverflowError):
        encode_division(fps=24, ticks_per_frame=256)


def test_encode_division_rejects_bad_smpte_values():
    with pytest.raises(ValueError):
        encode_division(fps=23.976, ticks_per_frame=40)
    with pytest.raises(ValueError):
        encode_division(fps=24, ticks_per_frame=0)
    with pytest.raises(TypeError):
        encode_division(fps=24, ticks_per_frame=40.0)


def test_encode_division_rejects_bad_ppq_values():
    with pytest.raises(ValueError):
        encode_division(ticks_per_beat=0)
    with pytest.raises(ValueError):
        encode_division(ticks_per_beat=-1)
    with pytest.raises(TypeError):
        encode_division(ticks_per_beat=480.0)


# ---------------------------------------------------------------------
# Reading hand-crafted SMPTE files
# ---------------------------------------------------------------------

@pytest.mark.parametrize(('code', 'fps'), FRAME_RATES)
def test_read_smpte_header_attributes(code, fps):
    mid = read_smpte_file(code)
    assert mid.is_smpte
    assert mid.fps == fps
    assert mid.ticks_per_frame == TICKS_PER_FRAME
    assert mid.ticks_per_beat is None
    assert mid.division == (code << 8) | TICKS_PER_FRAME


@pytest.mark.parametrize(('code', 'fps'), FRAME_RATES)
def test_smpte_tick_times_ignore_tempo(code, fps):
    # The track contains a set_tempo event, but SMPTE timing is
    # derived purely from the frame rate.
    mid = read_smpte_file(code)
    seconds_per_tick = 1.0 / (fps * TICKS_PER_FRAME)
    expected_delta = DELTA * seconds_per_tick

    times = [msg.time for msg in mid]
    assert times[:3] == pytest.approx(
        [expected_delta, expected_delta, expected_delta])
    assert times[-1] == 0

    # MidiFile.length uses the same time base as iteration.
    assert mid.length == pytest.approx(3 * expected_delta)


def test_smpte_29_97_exact_seconds():
    # One second at 29.97 fps (30000/1001) with 40 ticks per frame
    # contains 29.97 * 40 = 1198.8 ticks.
    mid = read_smpte_file(-29)
    assert mid.length == pytest.approx(
        3 * DELTA / (29.97 * TICKS_PER_FRAME))
    # The nominal 29.97 fps is the NTSC rate 30000 / 1001; the float
    # literal 29.97 only approximates the exact rational value.
    assert (29.97 * 40 * (1001.0 / 1_200_000)
            == pytest.approx(1.0, rel=1e-4))


def test_smpte_yields_all_messages():
    mid = read_smpte_file(-24)
    messages = list(mid)
    assert [msg.type for msg in messages] == [
        'note_on', 'set_tempo', 'note_off', 'end_of_track']
    # Non-zero deltas are yielded as seconds; the final zero delta
    # stays the integer 0, matching the existing PPQ behavior.
    assert [msg.time for msg in messages][-1] == 0
    assert all(isinstance(msg.time, float) for msg in messages[:-1])
    # Copies are yielded, not the track messages themselves.
    assert messages[0] is not mid.tracks[0][0]


def test_ppq_file_still_uses_tempo():
    data = make_file_bytes(480, make_track(tempo=1_000_000))
    mid = MidiFile(file=io.BytesIO(data))
    assert not mid.is_smpte
    assert mid.ticks_per_beat == 480
    assert mid.fps is None

    times = [msg.time for msg in mid]
    # Default tempo (500000 us) applies until the set_tempo message
    # has been yielded; afterwards 60 BPM (1000000 us) applies.
    assert times[0] == pytest.approx(0.125)
    assert times[1] == pytest.approx(0.125)
    assert times[2] == pytest.approx(0.25)
    assert mid.length == pytest.approx(0.5)


# ---------------------------------------------------------------------
# Writing / round trips
# ---------------------------------------------------------------------

@pytest.mark.parametrize(('code', 'fps'), FRAME_RATES)
def test_smpte_round_trip(code, fps):
    mid = read_smpte_file(code)

    buf = io.BytesIO()
    mid.save(file=buf)
    raw = buf.getvalue()

    # The header division word survives the round trip.
    assert struct.unpack('>h', raw[12:14])[0] == mid.division

    buf.seek(0)
    reloaded = MidiFile(file=buf)
    assert reloaded.is_smpte
    assert reloaded.fps == fps
    assert reloaded.ticks_per_frame == TICKS_PER_FRAME
    assert reloaded.tracks == mid.tracks
    assert reloaded.length == pytest.approx(mid.length)


def test_ppq_round_trip_preserves_division():
    mid = MidiFile(type=1, ticks_per_beat=120, tracks=[
        MidiTrack([
            Message('note_on', note=64, time=0),
            Message('note_off', note=64, time=120),
            MetaMessage('end_of_track', time=0),
        ]),
    ])

    buf = io.BytesIO()
    mid.save(file=buf)
    raw = buf.getvalue()
    assert struct.unpack('>h', raw[12:14])[0] == 120

    buf.seek(0)
    reloaded = MidiFile(file=buf)
    assert not reloaded.is_smpte
    assert reloaded.ticks_per_beat == 120
    assert reloaded.fps is None
    assert reloaded.ticks_per_frame is None


def test_construct_smpte_file_and_save():
    mid = MidiFile(type=1, fps=25, ticks_per_frame=80, tracks=[
        MidiTrack([MetaMessage('end_of_track', time=0)]),
    ])
    assert mid.division == (-25 << 8) | 80

    buf = io.BytesIO()
    mid.save(file=buf)
    assert struct.unpack('>h', buf.getvalue()[12:14])[0] == \
        (-25 << 8) | 80


def test_repr_round_trips_smpte():
    mid = MidiFile(type=1, fps=24, ticks_per_frame=40)
    rebuilt = eval(repr(mid))  # noqa: S307
    assert rebuilt.is_smpte
    assert rebuilt.division == mid.division


def test_repr_round_trips_ppq():
    mid = MidiFile(type=1, ticks_per_beat=123)
    rebuilt = eval(repr(mid))  # noqa: S307
    assert rebuilt.ticks_per_beat == 123
    assert not rebuilt.is_smpte


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_invalid_frame_rate_code_in_file():
    data = make_file_bytes((-23 << 8) | 40, make_track())
    with pytest.raises(ValueError):
        MidiFile(file=io.BytesIO(data))


def test_zero_ticks_per_frame_in_file():
    data = make_file_bytes((-24 << 8) | 0, make_track())
    with pytest.raises(ValueError):
        MidiFile(file=io.BytesIO(data))


def test_zero_division_word_in_file():
    data = make_file_bytes(0, make_track())
    with pytest.raises(ValueError):
        MidiFile(file=io.BytesIO(data))


def test_constructor_rejects_invalid_fps():
    with pytest.raises(ValueError):
        MidiFile(fps=23.976, ticks_per_frame=40)


def test_constructor_rejects_zero_ticks_per_frame():
    with pytest.raises(ValueError):
        MidiFile(fps=24, ticks_per_frame=0)


def test_constructor_rejects_ticks_per_frame_overflow():
    with pytest.raises(OverflowError):
        MidiFile(fps=24, ticks_per_frame=256)


def test_constructor_rejects_ppq_overflow():
    with pytest.raises(OverflowError):
        MidiFile(ticks_per_beat=32768)


def test_constructor_rejects_non_positive_ppq():
    with pytest.raises(ValueError):
        MidiFile(ticks_per_beat=0)


def test_ppq_and_fps_are_mutually_exclusive():
    with pytest.raises(ValueError):
        MidiFile(ticks_per_beat=480, fps=24, ticks_per_frame=40)


def test_fps_requires_ticks_per_frame():
    with pytest.raises(ValueError):
        MidiFile(fps=24)


def test_ticks_per_frame_without_fps_rejected():
    with pytest.raises(ValueError):
        MidiFile(ticks_per_frame=40)


def test_negative_tick_rejected_on_save():
    mid = MidiFile(type=1, fps=24, ticks_per_frame=40, tracks=[
        MidiTrack([
            Message('note_on', note=64, time=-1),
        ]),
    ])
    with pytest.raises(ValueError):
        mid.save(file=io.BytesIO())


def test_smpte_default_constructor_has_no_tempo_dependency():
    # Even an extreme tempo event must not change SMPTE playback time.
    mid = read_smpte_file(-30)
    baseline = mid.length
    other = read_smpte_file(-30, track=make_track(tempo=100))
    assert other.length == pytest.approx(baseline)
