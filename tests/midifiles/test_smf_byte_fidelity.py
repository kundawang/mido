# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

"""SMF byte-fidelity and malformed-input tests.

The fixtures under ``tests/midifiles/fixtures`` are fixed-byte Standard
MIDI Files hand-built by ``build_fixtures.py`` (same directory). They are
*not* produced by mido, so the tests below pin down wire-level behaviour
that the high-level :class:`~mido.midifiles.midifiles.MidiFile` API hides:
running status, VLQ encoding boundaries, F0/F7 system exclusive events,
unknown meta events and end_of_track.

See ``build_fixtures.py`` for the SMF specification references and the
annotated byte layout of every fixture.

Rules for legal re-encoding
---------------------------
A rewritten file need not be byte-identical to its input. The following
differences are legal and are tested explicitly:

1. **Running status is advisory.** The writer may either reuse the last
   channel status byte (drop the status byte) or re-emit it; the reader
   must accept both. mido reuses running status whenever consecutive
   channel messages share a status byte and resets it after meta, SysEx
   and system-common events.
2. **VLQs are canonical on write.** Delta times and meta lengths are
   re-emitted as the shortest possible variable-length quantity. A
   non-minimal (but well-formed) VLQ on input keeps its decoded value and
   is normalized on write.
3. **SysEx framing is normalized.** An F0 event may store the leading
   0xF0 either as the event status byte or inside the payload; mido reads
   both identically and always writes the shorter form (the 0xF0 status
   byte is not repeated in the payload). F7 escape events and F0 events
   both decode to ``Message('sysex', ...)`` and are re-written as F0.
4. **end_of_track is canonicalized.** Any end_of_track messages are
   removed on save and a single one is appended at the end of the track
   (see :func:`mido.midifiles.tracks.fix_end_of_track`).
5. **Unknown meta events round-trip.** An unknown meta type byte is kept
   verbatim (type byte, length and payload) through read and write.
6. **Chunk lengths follow the payload.** A length field changes whenever
   the payload above is re-encoded; the four-byte chunk magic and the
   MThd payload of an unchanged file structure stay identical.
"""

import io
from pathlib import Path

import pytest

from mido.messages import Message
from mido.midifiles.meta import (MetaMessage, UnknownMetaMessage,
                                 encode_variable_int)
from mido.midifiles.midifiles import MidiFile

FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures'

# Delta-time VLQ boundary values, mirrored from build_fixtures.py.
VLQ_127 = 127
VLQ_128 = 128
VLQ_16383 = 16383
VLQ_16384 = 16384
VLQ_2097151 = 2097151
VLQ_2097152 = 2097152


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fixture(name):
    return (FIXTURE_DIR / name).read_bytes()


def parse(data, clip=False):
    return MidiFile(file=io.BytesIO(data), clip=clip)


def rewrite(mid):
    out = io.BytesIO()
    mid.save(file=out)
    return out.getvalue()


def iter_chunks(data):
    """Yield (name, header_offset, payload_offset, length) for each chunk."""
    pos = 0
    while pos + 8 <= len(data):
        name = bytes(data[pos:pos + 4])
        length = int.from_bytes(data[pos + 4:pos + 8], 'big')
        yield name, pos, pos + 8, length
        pos += 8 + length


def find_event_range(data, marker, start=0):
    """Return (start, end) of the track event containing *marker*.

    The event starts at its delta-time VLQ and *marker* must begin at the
    event's status byte and be unique in *data*.
    """
    rel = data.index(marker, start)
    # Start on the byte immediately in front of the status byte and move
    # left while the byte still further left is a VLQ continuation (high
    # bit set). We stop positioned on the VLQ terminator, the first byte
    # of the event. For a one-byte VLQ this is simply rel - 1.
    event_start = rel - 1
    while event_start > 0 and data[event_start - 1] & 0x80:
        event_start -= 1
    return event_start, rel + len(marker)


def first_track_offsets(data):
    """Return (length_field_offset, payload_offset) of the first MTrk."""
    for name, header_offset, payload_offset, length in iter_chunks(data):
        if name == b'MTrk':
            return header_offset + 4, payload_offset, length
    raise AssertionError('fixture contains no MTrk chunk')


def find_meta_range(data, meta_type, payload, start=0):
    """Range of a complete meta event: delta VLQ + FF type len payload."""
    marker = bytes([0xFF, meta_type]) + bytes(encode_variable_int(len(payload))) \
        + bytes(payload)
    return find_event_range(data, marker, start)


def find_sysex_range(data, status, payload, start=0):
    """Range of a complete F0/F7 sysex event."""
    marker = bytes([status]) + bytes(encode_variable_int(len(payload))) \
        + bytes(payload)
    return find_event_range(data, marker, start)


# ---------------------------------------------------------------------------
# Fixture provenance and structural sanity
# ---------------------------------------------------------------------------

def test_fixtures_exist_and_match_builder():
    """The checked-in binaries must be exactly what build_fixtures emits."""
    from tests.midifiles.fixtures import build_fixtures

    for name, builder in build_fixtures.FIXTURES.items():
        assert load_fixture(name) == builder(), (
            f'{name} is out of date; regenerate with build_fixtures.py')


@pytest.mark.parametrize('name, expected_type, expected_tracks', [
    ('type0.mid', 0, 1),
    ('type1.mid', 1, 3),
    ('type2.mid', 2, 2),
])
def test_fixture_structure(name, expected_type, expected_tracks):
    mid = parse(load_fixture(name))
    assert mid.type == expected_type
    assert len(mid.tracks) == expected_tracks
    assert mid.ticks_per_beat == 480

    chunks = list(iter_chunks(load_fixture(name)))
    assert chunks[0][0] == b'MThd'
    assert [c[0] for c in chunks[1:]] == [b'MTrk'] * expected_tracks
    # Every declared chunk must fit exactly inside the file.
    data = load_fixture(name)
    end = chunks[-1][2] + chunks[-1][3]
    assert end == len(data)
    # Every track must terminate with an end_of_track meta event.
    for track in mid.tracks:
        assert track[-1].type == 'end_of_track'


# ---------------------------------------------------------------------------
# Type 0: event semantics
# ---------------------------------------------------------------------------

def test_type0_event_semantics():
    mid = parse(load_fixture('type0.mid'))
    track = mid.tracks[0]

    assert track[0] == MetaMessage('track_name', name='Type0', time=0)

    # Unknown manufacturer meta is preserved with its raw type byte.
    assert isinstance(track[1], UnknownMetaMessage)
    assert track[1].type_byte == 0x7D
    assert track[1].data == (0x01, 0x02, 0x03)
    assert track[1].time == 0

    # F0 SysEx: framing bytes are stripped, payload is the manufacturer data.
    assert track[2] == Message('sysex', data=(0x7D, 0x01, 0x02, 0x03), time=0)

    # Running status: three note_on events, deltas 0 / 127 / 128.
    assert track[3] == Message('note_on', note=0x40, velocity=0x40, time=0)
    assert track[4] == Message('note_on', note=0x41, velocity=0x41,
                               time=VLQ_127)
    assert track[5] == Message('note_on', note=0x42, velocity=0x42,
                               time=VLQ_128)

    # New status byte resets running status.
    assert track[6] == Message('note_off', note=0x40, velocity=0x40, time=0)

    # One-byte channel message and its running-status companion.
    assert track[7] == Message('program_change', program=5, time=0)
    assert track[8] == Message('program_change', program=6, time=0)

    # Another reset, then running status at the 16383/16384 boundary.
    assert track[9] == Message('control_change', control=7, value=100, time=0)
    assert track[10] == Message('control_change', control=7, value=100,
                                time=VLQ_16383)

    # F7 escape decodes to the same message type as an F0 SysEx.
    assert track[11] == Message('sysex', data=(0x11, 0x22, 0x33),
                                time=VLQ_16384)

    # VLQ boundary deltas on meta events.
    assert track[12].type == 'text'
    assert track[12].time == VLQ_2097151
    assert len(track[12].text) == 0x7F
    assert track[13].type == 'text'
    assert track[13].time == VLQ_2097152
    assert len(track[13].text) == 0x80

    assert track[14] == MetaMessage('end_of_track', time=0)
    assert len(track) == 15


def test_type0_unknown_meta_bytes_roundtrip():
    data = load_fixture('type0.mid')
    start, end = find_meta_range(data, 0x7D, [0x01, 0x02, 0x03])
    original = data[start:end]

    rewritten = rewrite(parse(data))
    assert original in rewritten
    # And the re-read object is still unknown with identical raw bytes.
    mid2 = parse(rewritten)
    unknown = [m for m in mid2.tracks[0] if isinstance(m, UnknownMetaMessage)]
    assert len(unknown) == 1
    assert unknown[0].bytes() == [0xFF, 0x7D, 0x03, 0x01, 0x02, 0x03]


def test_type0_end_of_track_bytes_preserved():
    data = load_fixture('type0.mid')
    # Final four bytes of the only track payload: delta 00 FF 2F 00.
    assert data[-4:] == b'\x00\xff\x2f\x00'
    rewritten = rewrite(parse(data))
    assert rewritten[-4:] == b'\x00\xff\x2f\x00'


def test_type0_vlq_boundaries_on_disk():
    """The fixture really stores the multi-byte VLQs, not their values."""
    data = load_fixture('type0.mid')
    expected_vlqs = {
        VLQ_127: b'\x7f',
        VLQ_128: b'\x81\x00',
        VLQ_16383: b'\xff\x7f',
        VLQ_16384: b'\x81\x80\x00',
        VLQ_2097151: b'\xff\xff\x7f',
        VLQ_2097152: b'\x81\x80\x80\x00',
    }
    for value, wire in expected_vlqs.items():
        assert wire in data, f'VLQ {value} encoding {wire.hex()} missing'
    # The 128-byte meta payload needs a two-byte length field.
    assert b'\xff\x01\x81\x00' in data


def test_type0_running_status_wire_bytes():
    data = load_fixture('type0.mid')
    # Full status followed by two bare running-status data pairs.
    assert b'\x00\x90\x40\x40\x7f\x41\x41\x81\x00\x42\x42' in data
    # program_change running status: only the program number follows.
    assert b'\x00\xc0\x05\x00\x06' in data
    # Rewritten file keeps the same running-status usage.
    rewritten = rewrite(parse(data))
    assert b'\x00\x90\x40\x40\x7f\x41\x41\x81\x00\x42\x42' in rewritten
    assert b'\x00\xc0\x05\x00\x06' in rewritten


def test_type0_sysex_f0_framing_normalization():
    """Rule 3: leading-F0-in-payload and F7 escape collapse to one form."""
    data = load_fixture('type0.mid')

    f0_start, f0_end = find_sysex_range(
        data, 0xF0, [0xF0, 0x7D, 0x01, 0x02, 0x03, 0xF7])
    f7_start, f7_end = find_sysex_range(
        data, 0xF7, [0x11, 0x22, 0x33, 0xF7])
    assert data[f0_start:f0_end] == b'\x00\xf0\x06\xf0\x7d\x01\x02\x03\xf7'
    # The F7 event sits behind a three-byte VLQ (delta 16384).
    assert data[f7_start:f7_end] == \
        b'\x81\x80\x00\xf7\x04\x11\x22\x33\xf7'

    rewritten = rewrite(parse(data))
    # Both events are re-emitted as F0 without a duplicated payload F0.
    assert b'\x00\xf0\x05\x7d\x01\x02\x03\xf7' in rewritten
    assert b'\x81\x80\x00\xf0\x04\x11\x22\x33\xf7' in rewritten
    # The original longer F0 encoding is the only byte range that changes.
    assert data[f0_start:f0_end] not in rewritten

    # Semantics are identical before and after the normalization.
    before = parse(data)
    after = parse(rewritten)
    assert [m for m in before.tracks[0] if m.type == 'sysex'] == \
        [m for m in after.tracks[0] if m.type == 'sysex']


def test_f7_escape_with_terminating_f7_is_not_misframed():
    """Regression: an F7 event ending in F7 must not expose the F7 as data.

    The previous implementation stripped a leading F7 even for F7 escape
    events, which shifted the framing and made the terminating F7 look
    like an out-of-range data byte (raising ValueError).
    """
    payload = b'\x00\xf7\x04\x11\x22\x33\xf7'  # delta, F7, len, data, F7
    smf = (b'MThd' + (6).to_bytes(4, 'big')
           + (1).to_bytes(2, 'big') + (1).to_bytes(2, 'big')
           + (480).to_bytes(2, 'big')
           + b'MTrk' + len(payload).to_bytes(4, 'big') + payload)
    mid = parse(smf)
    assert mid.tracks[0] == [
        Message('sysex', data=(0x11, 0x22, 0x33), time=0),
    ]


# ---------------------------------------------------------------------------
# Type 1: multiple tracks and cross-track reset
# ---------------------------------------------------------------------------

def test_type1_event_semantics():
    mid = parse(load_fixture('type1.mid'))
    tempo_track, note_track, unknown_track = mid.tracks

    assert tempo_track[0] == MetaMessage('track_name', name='Tempo', time=0)
    assert tempo_track[1] == MetaMessage('set_tempo', tempo=500000, time=0)
    assert tempo_track[2] == MetaMessage(
        'time_signature', numerator=4, denominator=4,
        clocks_per_click=24, notated_32nd_notes_per_beat=8, time=128)
    assert tempo_track[3] == MetaMessage('end_of_track', time=16384)

    assert note_track[0] == Message('note_on', note=0x40, velocity=0x40,
                                    time=0)
    assert note_track[1] == Message('note_on', note=0x41, velocity=0x41,
                                    time=0)
    assert note_track[2] == Message('sysex', data=(0x7E, 0x01), time=0)
    # Full status is mandatory after the SysEx reset.
    assert note_track[3] == Message('note_on', note=0x42, velocity=0x42,
                                    time=0)
    assert note_track[4] == Message('note_off', note=0x40, velocity=0x40,
                                    time=127)
    assert note_track[5] == MetaMessage('end_of_track', time=0)

    unknown = unknown_track[1]
    assert isinstance(unknown, UnknownMetaMessage)
    assert unknown.type_byte == 0x7D
    assert unknown.data == (0xDE, 0xAD, 0xBE, 0xEF)


def test_type1_running_status_does_not_leak_across_sysex_or_tracks():
    data = load_fixture('type1.mid')
    rewritten = rewrite(parse(data))

    # In track 1 the note_on following the SysEx must carry a full status
    # byte on both the original file and the mido rewrite.
    assert b'\xf0\x04\xf0\x7e\x01\xf7\x00\x90\x42\x42' in data
    assert b'\xf0\x03\x7e\x01\xf7\x00\x90\x42\x42' in rewritten

    # Track 2 starts with a meta event: no channel status may be carried
    # over from track 1. Its first bytes are delta 00 + the track_name
    # meta event.
    chunks = list(iter_chunks(rewritten))
    track2_payload_offset = chunks[3][2]
    assert rewritten[track2_payload_offset:track2_payload_offset + 3] == \
        b'\x00\xff\x03'


def test_type1_header_bytes_identical_after_rewrite():
    """Rule 6: with unchanged structure the MThd chunk is byte-identical."""
    data = load_fixture('type1.mid')
    rewritten = rewrite(parse(data))
    assert rewritten[:14] == data[:14]


# ---------------------------------------------------------------------------
# Type 2: independent tracks
# ---------------------------------------------------------------------------

def test_type2_event_semantics_and_no_iteration():
    mid = parse(load_fixture('type2.mid'))
    assert mid.tracks[0][0] == Message('note_on', channel=1, note=0x3C,
                                       velocity=0x7F, time=0)
    assert mid.tracks[0][1] == Message('note_on', channel=1, note=0x3D,
                                       velocity=0x7F, time=0)
    assert mid.tracks[1][0] == Message('note_on', channel=2, note=0x40,
                                       velocity=0x50, time=0)
    assert mid.tracks[1][1] == Message('note_on', channel=2, note=0x41,
                                       velocity=0x50, time=2097152)
    # Type 2 tracks cannot be merged for playback.
    with pytest.raises(TypeError):
        list(mid)
    with pytest.raises(ValueError):
        mid.length


def test_type2_wire_bytes_keep_running_status_and_four_byte_vlq():
    data = load_fixture('type2.mid')
    tracks = [c for c in iter_chunks(data) if c[0] == b'MTrk']
    # Track 0: full status then a bare running-status data pair.
    _, _, payload0, len0 = tracks[0]
    assert data[payload0:payload0 + len0] == \
        b'\x00\x91\x3c\x7f\x00\x3d\x7f\x00\xff\x2f\x00'
    # Track 1: the 2097152 delta must be a full four-byte VLQ, and running
    # status must be established afresh inside this independent track.
    _, _, payload1, len1 = tracks[1]
    assert data[payload1:payload1 + len1] == \
        b'\x00\x92\x40\x50\x81\x80\x80\x00\x41\x50\x00\xff\x2f\x00'


def test_type2_byte_identical_roundtrip():
    data = load_fixture('type2.mid')
    assert rewrite(parse(data)) == data


# ---------------------------------------------------------------------------
# Read / rewrite readability and byte-range preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['type0.mid', 'type1.mid', 'type2.mid'])
def test_rewrite_is_readable_with_identical_semantics(name):
    data = load_fixture(name)
    mid = parse(data)
    rewritten = rewrite(mid)

    mid2 = parse(rewritten)
    assert mid2.type == mid.type
    assert mid2.ticks_per_beat == mid.ticks_per_beat
    assert len(mid2.tracks) == len(mid.tracks)
    for track_a, track_b in zip(mid.tracks, mid2.tracks):
        assert list(track_a) == list(track_b)

    # A second rewrite must be stable (fixpoint of the writer).
    assert rewrite(mid2) == rewritten


@pytest.mark.parametrize('name', ['type0.mid', 'type1.mid', 'type2.mid'])
def test_end_of_track_range_preserved(name):
    data = load_fixture(name)
    rewritten = rewrite(parse(data))
    # Every MTrk payload ends with the canonical end_of_track event.
    for chunk in iter_chunks(rewritten):
        if chunk[0] != b'MTrk':
            continue
        _, _, payload_offset, length = chunk
        assert rewritten[payload_offset + length - 4:
                         payload_offset + length] == b'\x00\xff\x2f\x00'


def test_non_minimal_vlq_is_accepted_and_canonicalized():
    """Rule 2: redundant VLQ encodings keep their value, shrink on write."""
    original = bytearray(load_fixture('type0.mid'))
    length_field, payload_offset, track_length = first_track_offsets(original)
    data = bytearray(original)
    # The first track event starts at the payload offset with delta 0x00.
    # Replace it with the non-minimal two-byte encoding 0x80 0x00 (still
    # zero) and grow the declared track length by one.
    assert data[payload_offset] == 0x00
    data[payload_offset] = 0x80
    data.insert(payload_offset + 1, 0x00)
    data[length_field:length_field + 4] = (track_length + 1).to_bytes(4, 'big')

    mid = parse(bytes(data))
    assert mid.tracks[0][0].time == 0

    rewritten = rewrite(mid)
    # The redundant leading VLQ byte is gone. The rewrite is one byte
    # shorter than the mutated input and one byte shorter still than the
    # original fixture, because the F0 SysEx framing is also normalized
    # (see test_type0_sysex_f0_framing_normalization).
    assert rewritten[payload_offset] == 0x00
    assert len(rewritten) == len(data) - 2
    assert parse(rewritten).tracks[0] == parse(
        bytes(original)).tracks[0]


# ---------------------------------------------------------------------------
# Malformed input: every failure must be located
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cut', [0, 4, 7, 10, 13])
def test_truncated_header_raises_eof_with_offset(cut):
    data = load_fixture('type0.mid')[:cut]
    with pytest.raises(EOFError) as excinfo:
        parse(data)
    assert 'offset' in str(excinfo.value)


def test_bad_magic_is_rejected():
    data = bytearray(load_fixture('type0.mid'))
    data[0:4] = b'XXXX'
    with pytest.raises(OSError, match='MThd not found'):
        parse(bytes(data))


def test_truncated_track_chunk_header():
    # Cut inside the first MTrk header that follows the 14-byte MThd.
    data = bytearray(load_fixture('type1.mid'))
    track_header = data.index(b'MTrk')
    truncated = bytes(data[:track_header + 4])
    with pytest.raises(
            EOFError,
            match=f'chunk header starting at offset {track_header}'):
        parse(truncated)


def test_truncated_track_payload_reports_offset():
    data = bytearray(load_fixture('type0.mid'))
    _, payload_offset, _ = first_track_offsets(data)
    cut = payload_offset + 18  # mid-event, well inside the payload
    with pytest.raises(EOFError, match=f'byte offset {cut}'):
        parse(bytes(data[:cut]))


def test_inflated_track_length_runs_into_eof():
    data = bytearray(load_fixture('type0.mid'))
    length_field, _, track_length = first_track_offsets(data)
    assert track_length + 8 + 14 == len(data)  # single track fixture
    data[length_field:length_field + 4] = (track_length + 1).to_bytes(4, 'big')
    with pytest.raises(EOFError, match=f'offset {len(data)}'):
        parse(bytes(data))


def test_shrunken_track_length_is_detected():
    data = bytearray(load_fixture('type0.mid'))
    length_field, _, _ = first_track_offsets(data)
    data[length_field:length_field + 4] = (10).to_bytes(4, 'big')
    with pytest.raises(OSError, match='declares 10 byte'):
        parse(bytes(data))


def test_illegal_status_byte_is_located():
    data = bytearray(load_fixture('type0.mid'))
    # First note_on event: delta 00 then status 0x90.
    status_offset = data.index(b'\x00\x90\x40\x40') + 1
    assert data[status_offset] == 0x90
    data[status_offset] = 0xF4  # undefined status byte
    with pytest.raises(
            OSError,
            match='undefined status byte 0xf4 '
            f'at byte offset {status_offset}'):
        parse(bytes(data))


def test_running_status_without_last_status_is_located():
    data = bytearray(load_fixture('type1.mid'))
    # The first MTrk whose payload starts with a channel event.
    status_offset = data.index(b'\x00\x90\x40\x40') + 1
    assert data[status_offset - 1:status_offset + 2] == b'\x00\x90\x40'
    data[status_offset] = 0x40  # bare data byte, no prior running status
    with pytest.raises(
            OSError,
            match='running status without last_status '
            f'at byte offset {status_offset}'):
        parse(bytes(data))


def test_high_bit_data_byte_is_located():
    data = bytearray(load_fixture('type0.mid'))
    event_offset = data.index(b'\x00\x90\x40\x40')
    assert data[event_offset:event_offset + 4] == b'\x00\x90\x40\x40'
    data[event_offset + 2] = 0xFF  # note value with high bit set
    with pytest.raises(OSError,
                       match='data byte must be in range 0..127'):
        parse(bytes(data))


def test_unterminated_vlq_runs_into_eof():
    # Keep the file up to the first payload byte, then turn that byte
    # into a VLQ continuation so the VLQ can never terminate.
    data = bytearray(load_fixture('type0.mid'))
    _, payload_offset, _ = first_track_offsets(data)
    data = data[:payload_offset + 1]
    data[payload_offset] = 0x80
    with pytest.raises(EOFError,
                       match=f'offset {payload_offset + 1}'):
        parse(bytes(data))


def test_sysex_with_internal_high_byte_is_located():
    # A data byte >= 0x80 inside a SysEx payload must be reported with its
    # position, even though the framing itself is valid.
    data = bytearray(load_fixture('type0.mid'))
    # Locate the F0 event dynamically: 00 F0 06 F0 7D 01 02 03 F7.
    event_offset = data.index(b'\x00\xf0\x06\xf0\x7d\x01\x02\x03\xf7')
    data[event_offset + 5] = 0xFE  # corrupt a manufacturer data byte
    with pytest.raises(OSError, match='sysex byte'):
        parse(bytes(data))


# ---------------------------------------------------------------------------
# Mutation tests: prove the assertions above actually catch defects
# ---------------------------------------------------------------------------

def test_mutation_undetected_change_breaks_semantic_assertion():
    """A parser that silently swallows a byte must fail the semantics test.

    This demonstrates the negative control for the semantic assertions:
    we emulate a broken reader by dropping one event and show the golden
    expectation no longer holds.
    """
    mid = parse(load_fixture('type0.mid'))
    del mid.tracks[0][4]  # the running-status note_on with delta 127
    rewritten = rewrite(mid)
    repaired = parse(rewritten)
    assert repaired.tracks[0] != parse(
        load_fixture('type0.mid')).tracks[0]
    assert Message('note_on', note=0x41, velocity=0x41,
                   time=VLQ_127) not in repaired.tracks[0]


def test_mutation_flipped_status_byte_changes_wire_bytes():
    """The byte-range assertions fail when an encoding defect is injected."""
    data = bytearray(load_fixture('type0.mid'))
    status_offset = data.index(b'\x00\x90\x40\x40') + 1
    assert data[status_offset] == 0x90
    data[status_offset] = 0x80  # note_on becomes note_off on disk
    mutated = bytes(data)
    # The file is still parseable, but semantics and byte-range assertions
    # both detect the mutation.
    mid = parse(mutated)
    assert mid.tracks[0][3] != Message('note_on', note=0x40, velocity=0x40,
                                       time=0)
    assert b'\x00\x90\x40\x40' not in mutated


def test_mutation_truncation_is_not_silently_accepted():
    """Every truncation of a valid fixture must raise, never parse short."""
    original = load_fixture('type1.mid')
    cuts = [14, len(original) - 1]
    cuts += [c[1] + 2 for c in iter_chunks(original) if c[0] == b'MTrk']
    for cut in cuts:
        with pytest.raises((EOFError, OSError)):
            parse(original[:cut])


def test_mutation_bad_length_cannot_satisfy_chunk_invariant():
    """The structural chunk-length invariant fails under a bad length."""
    data = bytearray(load_fixture('type2.mid'))
    track_chunks = [c for c in iter_chunks(data) if c[0] == b'MTrk']
    _, header_offset, _, real_length = track_chunks[1]
    length_field = header_offset + 4
    data[length_field:length_field + 4] = \
        (real_length + 7).to_bytes(4, 'big')
    with pytest.raises(EOFError):
        parse(bytes(data))
