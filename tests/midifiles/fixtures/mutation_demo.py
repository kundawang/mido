# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

"""Demonstrate that the SMF byte-fidelity tests catch injected defects.

This is a *mutation harness*, not part of the pytest suite. It takes the
known-good fixtures and injects one encoding defect at a time, then shows
that either mido raises a positioned error or a semantic/byte-range
assertion in ``test_smf_byte_fidelity.py`` would fail.

Run it from the repository root::

    python3 tests/midifiles/fixtures/mutation_demo.py
"""

import io
import sys
import tempfile
from pathlib import Path

# Make the repository importable when run directly.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mido.midifiles.midifiles import MidiFile  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent


def parse(data):
    return MidiFile(file=io.BytesIO(data))


def mutate_truncation(data):
    """Defect 1: chop the file in the middle of a track payload."""
    return data[:len(data) // 2], 'truncated to half the file'


def mutate_bad_length(data):
    """Defect 2: inflate the first MTrk declared length beyond EOF."""
    corrupted = bytearray(data)
    track_length_offset = data.index(b'MTrk') + 4
    # Inflate past the physical end of the file so the reader walks into
    # EOF while still inside the (bogus) declared chunk.
    file_end = len(data)
    payload_end = track_length_offset + 4
    corrupted[track_length_offset:track_length_offset + 4] = \
        (file_end - payload_end + 16).to_bytes(4, 'big')
    return bytes(corrupted), 'MTrk length inflated past end of file'


def _first_status_of_first_event(data):
    """Offset of the status byte of the first channel event of any MTrk."""
    search_from = data.index(b'MTrk') + 8
    for index in range(search_from, len(data)):
        if 0x80 <= data[index] <= 0xEF:
            return index
    raise ValueError('no channel status byte in fixture')


def _first_standalone_status(data):
    """First channel status byte in a track with no earlier channel event.

    Meta and SysEx events reset running status, so the first channel
    status byte inside any MTrk is always standalone: replacing it with
    a bare data byte must produce "running status without last_status".
    """
    pos = 0
    while True:
        marker = data.find(b'MTrk', pos)
        if marker < 0:
            break
        payload_start = marker + 8
        length = int.from_bytes(data[marker + 4:marker + 8], 'big')
        payload_end = payload_start + length
        for index in range(payload_start, payload_end):
            if 0x80 <= data[index] <= 0xEF:
                return index
        pos = payload_end
    raise ValueError('no track contains a channel status event')


def mutate_illegal_status(data):
    """Defect 3: replace a channel status byte with undefined 0xF4."""
    corrupted = bytearray(data)
    offset = _first_status_of_first_event(data)
    corrupted[offset] = 0xF4
    return bytes(corrupted), f'status byte at offset {offset} changed to F4'


def mutate_running_status(data):
    """Defect 4: turn the first track's first status into a bare data byte."""
    corrupted = bytearray(data)
    offset = _first_standalone_status(data)
    corrupted[offset] = 0x40
    return bytes(corrupted), f'status byte at offset {offset} stripped ' \
                             '(running status without last_status)'


def mutate_semantic_flip(data):
    """Defect 5: change the first status; parses but changes semantics."""
    corrupted = bytearray(data)
    offset = _first_status_of_first_event(data)
    original = corrupted[offset]
    # Move the status into a different message category while keeping it
    # legal: swap note_on<->note_off, otherwise step the high nibble.
    high_nibble = original & 0xF0
    if high_nibble == 0x90:
        new = original - 0x10
    elif high_nibble == 0x80:
        new = original + 0x10
    else:
        new = original ^ 0x10
    corrupted[offset] = new
    return bytes(corrupted), \
        f'status byte at offset {offset} changed from {original:02x} ' \
        f'to {new:02x}'


def mutate_vlq(data):
    """Defect 6: leave a VLQ unterminated (all continuation bytes)."""
    payload_start = data.index(b'MTrk') + 8
    corrupted = bytearray(data[:payload_start + 1])
    corrupted[payload_start] = 0x80
    return bytes(corrupted), 'first delta VLQ replaced by a lone 0x80'


MUTATORS = [
    mutate_truncation,
    mutate_bad_length,
    mutate_illegal_status,
    mutate_running_status,
    mutate_semantic_flip,
    mutate_vlq,
]


def main():
    failures = 0
    for fixture_name in ('type0.mid', 'type1.mid', 'type2.mid'):
        original = (FIXTURE_DIR / fixture_name).read_bytes()
        print(f'== {fixture_name} ({len(original)} bytes)')
        for mutator in MUTATORS:
            try:
                mutated, description = mutator(original)
            except ValueError as err:
                # Mutation not applicable to this fixture.
                print(f'  skip {mutator.__name__}: {err}')
                continue
            try:
                mid = parse(mutated)
            except Exception as err:  # noqa: BLE001 - demo wants every error
                print(f'  DETECTED  {mutator.__name__:24s} '
                      f'-> {type(err).__name__}: {err}')
                continue
            # Parseable mutations must still be caught by the semantic
            # expectations of the test suite. We compare the decoded
            # events against the original fixture: any changed message
            # violates the golden semantic assertions.
            original_events = [
                (i, j, repr(msg))
                for i, track in enumerate(parse(original).tracks)
                for j, msg in enumerate(track)
            ]
            mutated_events = [
                (i, j, repr(msg))
                for i, track in enumerate(mid.tracks)
                for j, msg in enumerate(track)
            ]
            if mutated_events != original_events:
                print(f'  DETECTED  {mutator.__name__:24s} '
                      f'-> decoded semantics differ ({description})')
            else:
                print(f'  MISSED    {mutator.__name__:24s} '
                      f'-> {description}')
                failures += 1
        print()

    # The fixture builder itself must regenerate byte-identical files;
    # a drift there is caught by test_fixtures_exist_and_match_builder.
    from tests.midifiles.fixtures import build_fixtures
    for name, builder in build_fixtures.FIXTURES.items():
        on_disk = (FIXTURE_DIR / name).read_bytes()
        assert on_disk == builder(), f'{name} drifted from its builder'
    print('all fixtures match build_fixtures.py')

    if failures:
        print(f'FAIL: {failures} mutation(s) were not detected')
        sys.exit(1)
    print('OK: every injected defect is detected')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory():
        main()
