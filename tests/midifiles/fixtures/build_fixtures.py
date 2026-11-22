# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

"""Build fixed-byte SMF fixtures used by test_smf_byte_fidelity.py.

These files are deliberately hand-constructed at the byte level (instead of
being produced by :class:`mido.MidiFile`) so that the tests can pin down
behaviour that the high-level API hides:

* running status being used, reset by a different status byte, and reset by
  meta/SysEx events,
* delta-time and meta-length VLQs on both sides of the 1-/2-/3-/4-byte
  boundaries (127/128, 16383/16384, 2097151/2097152),
* both flavours of system exclusive escape (F0 normal and F7 continuation),
* an unknown (manufacturer-specific) meta type byte, and
* the end_of_track terminator.

Provenance
----------
The bytes follow the Standard MIDI File 1.0 specification, in particular:

* "MIDI File Format" chapter of the Complete MIDI 1.0 Detailed
  Specification (MMA, https://midi.org/standard-midi-files),
* the variable-length quantity definition (31-bit, big-endian, high bit of
  every byte set except the last),
* sections "Meta-Events" (FF xx) and "System Exclusive Events" (F0/F7).

No third-party binary file is embedded: every fixture is regenerated from
this script and the exact expected layout is annotated below. Regenerate
with::

    python3 tests/midifiles/fixtures/build_fixtures.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

# Delta-time / length VLQ boundary values.
VLQ_127 = 127          # fits in one VLQ byte:         7f
VLQ_128 = 128          # first two-byte VLQ:           81 00
VLQ_16383 = 16383      # largest two-byte VLQ:         ff 7f
VLQ_16384 = 16384      # first three-byte VLQ:         81 80 00
VLQ_2097151 = 2097151  # largest three-byte VLQ:       ff ff 7f
VLQ_2097152 = 2097152  # first four-byte VLQ:          81 80 80 00


def vlq(value):
    """Encode *value* as an SMF variable-length quantity (big-endian)."""
    if value < 0:
        raise ValueError('VLQ must be non-negative')
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.reverse()
    return bytes(out)


def chunk(name, payload):
    if len(name) != 4:
        raise ValueError('chunk name must be 4 bytes')
    return name + len(payload).to_bytes(4, 'big') + payload


def header(type_, num_tracks, ticks_per_beat):
    return chunk(b'MThd',
                 type_.to_bytes(2, 'big')
                 + num_tracks.to_bytes(2, 'big')
                 + ticks_per_beat.to_bytes(2, 'big'))


def event(delta, payload):
    """A raw track event: delta-time VLQ followed by status+data bytes."""
    return vlq(delta) + bytes(payload)


def meta(delta, meta_type, data):
    data = bytes(data)
    return event(delta, [0xFF, meta_type] + list(vlq(len(data))) + list(data))


def sysex(delta, raw_payload, status=0xF0):
    """F0/F7 system exclusive event.

    *raw_payload* is the full event payload including any F0/F7 framing
    bytes, exactly as it appears in the MTrk event.
    """
    raw_payload = bytes(raw_payload)
    return event(delta, [status] + list(vlq(len(raw_payload)))
                 + list(raw_payload))


def end_of_track(delta=0):
    return meta(delta, 0x2F, b'')


def build_type0():
    """Single-track (format 0) file exercising every byte-level feature.

    Track layout (all offsets relative to the first byte of the MTrk
    payload)::

        delta  status / event                       purpose
        -----  -----------------------------------  ------------------------
          0    FF 03 'Type0'                         known meta (track name)
          0    FF 7D 03 01 02 03                     unknown meta, type 0x7d
          0    F0 06 F0 7D 01 02 03 F7              normal SysEx, F0..F7
          0    90 40 40                              note_on (full status)
         127   41 41                                 running status note_on
         128   42 42                                 2-byte VLQ + running
          0    80 40 40                              status reset: note_off
          0    C0 05                                1-byte program_change
          0    06                                    running status (same)
          0    B0 07 64                              status reset: ctrl_change
       16383   07 64                                 2-byte max VLQ + running
       16384   F7 04 F7 11 22 33                     3-byte VLQ; F7 escape
    2097151    FF 01 <len=127> ...                   3-byte max VLQ; 1-byte
                                                     meta length boundary
    2097152    FF 01 <len=128> ...                   4-byte VLQ; 2-byte meta
                                                     length boundary
          0    FF 2F 00                              end_of_track
    """
    events = [
        meta(0, 0x03, b'Type0'),
        meta(0, 0x7D, [0x01, 0x02, 0x03]),
        sysex(0, [0xF0, 0x7D, 0x01, 0x02, 0x03, 0xF7], status=0xF0),
        # Full status, then two running-status note_ons.
        event(0, [0x90, 0x40, 0x40]),
        event(VLQ_127, [0x41, 0x41]),
        event(VLQ_128, [0x42, 0x42]),
        # Different status byte resets running status.
        event(0, [0x80, 0x40, 0x40]),
        # One-byte channel messages also support running status.
        event(0, [0xC0, 0x05]),
        event(0, [0x06]),
        # Another status change resets running status.
        event(0, [0xB0, 0x07, 0x64]),
        event(VLQ_16383, [0x07, 0x64]),
        # F7 SysEx escape (no leading F0 by definition) resets running
        # status too. The payload carries ordinary data bytes plus the
        # mandatory terminating F7.
        sysex(VLQ_16384, [0x11, 0x22, 0x33, 0xF7], status=0xF7),
        # VLQ boundary on delta time and on the meta payload length.
        meta(VLQ_2097151, 0x01, bytes(range(0x7F))),
        meta(VLQ_2097152, 0x01, bytes([0x41]) * 0x80),
        end_of_track(0),
    ]
    return header(0, 1, 480) + chunk(b'MTrk', b''.join(events))


def build_type1():
    """Multi-track (format 1) file.

    Track 0 holds tempo/timing meta events; track 1 holds channel events
    with running status reset across a SysEx event; track 2 holds an
    unknown manufacturer meta plus end_of_track.
    """
    tempo = (500000).to_bytes(3, 'big')
    track0 = b''.join([
        meta(0, 0x03, b'Tempo'),
        meta(0, 0x51, tempo),               # set_tempo
        meta(128, 0x58, [4, 2, 24, 8]),     # 4/4, delta uses 2-byte VLQ
        end_of_track(16384),                # 3-byte VLQ on EOT delta
    ])
    track1 = b''.join([
        event(0, [0x90, 0x40, 0x40]),
        event(0, [0x41, 0x41]),             # running status
        sysex(0, [0xF0, 0x7E, 0x01, 0xF7]),  # SysEx resets running status
        # Full status is mandatory here: a bare 0x42 0x42 would be a
        # "running status without last_status" error. The byte-fidelity
        # test asserts the rewriter keeps this full status byte.
        event(0, [0x90, 0x42, 0x42]),
        event(127, [0x80, 0x40, 0x40]),
        end_of_track(0),
    ])
    track2 = b''.join([
        meta(0, 0x03, b'Unknown'),
        meta(0, 0x7D, [0xDE, 0xAD, 0xBE, 0xEF]),
        end_of_track(0),
    ])
    return (header(1, 3, 480)
            + chunk(b'MTrk', track0)
            + chunk(b'MTrk', track1)
            + chunk(b'MTrk', track2))


def build_type2():
    """Format 2 file: independent one-pattern tracks, no shared tempo.

    Each track is self-contained; running status appears inside a track
    but must never leak between tracks.
    """
    track0 = b''.join([
        event(0, [0x91, 0x3C, 0x7F]),
        event(0, [0x3D, 0x7F]),             # running status within track 0
        end_of_track(0),
    ])
    track1 = b''.join([
        event(0, [0x92, 0x40, 0x50]),
        event(2097152, [0x41, 0x50]),       # 4-byte VLQ + running status
        end_of_track(0),
    ])
    return (header(2, 2, 480)
            + chunk(b'MTrk', track0)
            + chunk(b'MTrk', track1))


FIXTURES = {
    'type0.mid': build_type0,
    'type1.mid': build_type1,
    'type2.mid': build_type2,
}


def main():
    for name, builder in FIXTURES.items():
        data = builder()
        (HERE / name).write_bytes(data)
        print(f'wrote {name}: {len(data)} bytes')


if __name__ == '__main__':
    main()
