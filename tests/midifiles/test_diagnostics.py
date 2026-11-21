# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

import pytest

from mido import Message, MidiFile, MidiTrack
from mido.midifiles.diagnostics import (
    Conflict,
    ConflictReport,
    ConflictType,
    Event,
    detect_conflicts,
)


def make_file(tracks, type=1, ticks_per_beat=480):
    return MidiFile(type=type, ticks_per_beat=ticks_per_beat,
                    tracks=[MidiTrack(track) for track in tracks])


def types_in(report):
    return {conflict.type for conflict in report}


# ---------------------------------------------------------------------
# Overlapping notes
# ---------------------------------------------------------------------

def test_cross_track_overlap_is_detected():
    # Two arrangement tracks wrongly share channel 0 and hold the same
    # note at the same absolute tick.
    mid = make_file([
        [Message('note_on', channel=0, note=60, velocity=64, time=0),
         Message('note_off', channel=0, note=60, velocity=64, time=480)],
        [Message('note_on', channel=0, note=60, velocity=64, time=100),
         Message('note_off', channel=0, note=60, velocity=64, time=380)],
    ])

    report = detect_conflicts(mid)

    assert len(report) == 1
    conflict = report[0]
    assert conflict.type == ConflictType.OVERLAP
    assert conflict.channel == 0
    assert conflict.note == 60
    assert conflict.tick == 100
    assert [event.track for event in conflict.events] == [0, 1]
    assert all(event.message.type == 'note_on'
               for event in conflict.events)


def test_same_track_overlap_is_detected():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=10),
        Message('note_off', channel=0, note=60, velocity=64, time=10),
        Message('note_off', channel=0, note=60, velocity=64, time=10),
    ]])

    report = detect_conflicts(mid)

    assert ConflictType.OVERLAP in types_in(report)
    assert report.by_type(ConflictType.OVERLAP)[0].tick == 10
    assert len(report.by_type(ConflictType.OVERLAP)[0].events) == 2


def test_normal_polyphony_is_not_flagged():
    # Different pitches and different channels sounding together are
    # ordinary polyphony, not conflicts.
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_on', channel=0, note=64, velocity=64, time=0),
        Message('note_on', channel=1, note=60, velocity=64, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=240),
        Message('note_off', channel=0, note=64, velocity=64, time=0),
        Message('note_off', channel=1, note=60, velocity=64, time=0),
    ]])

    report = detect_conflicts(mid)

    assert not report
    assert str(report) == 'No conflicts found.'


def test_overlap_report_contains_all_active_notes():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=10),
        Message('note_on', channel=0, note=60, velocity=64, time=10),
    ]])

    report = detect_conflicts(mid)

    # Two overlaps; the second one must list all three sounding notes.
    overlaps = report.by_type(ConflictType.OVERLAP)
    assert len(overlaps) == 2
    assert len(overlaps[1].events) == 3
    assert types_in(report) == {ConflictType.OVERLAP,
                                ConflictType.DANGLING_NOTE}


# ---------------------------------------------------------------------
# Pairing strategies
# ---------------------------------------------------------------------

def test_repeated_note_off_pairs_lifo_by_default():
    # on, on, off leaves the oldest note dangling under LIFO pairing.
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=10),
        Message('note_off', channel=0, note=60, velocity=64, time=10),
    ]])

    report = detect_conflicts(mid)

    assert types_in(report) == {ConflictType.OVERLAP,
                                ConflictType.DANGLING_NOTE}
    dangling = report.by_type(ConflictType.DANGLING_NOTE)
    assert len(dangling) == 1
    assert dangling[0].events[0].tick == 0
    assert dangling[0].events[0].track == 0


def test_fifo_pairing_leaves_newest_note_dangling():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=10),
        Message('note_off', channel=0, note=60, velocity=64, time=10),
    ]])

    report = detect_conflicts(mid, overlap_strategy='fifo')

    assert types_in(report) == {ConflictType.OVERLAP,
                                ConflictType.DANGLING_NOTE}
    dangling = report.by_type(ConflictType.DANGLING_NOTE)
    assert len(dangling) == 1
    assert dangling[0].events[0].tick == 10


def test_extra_note_off_is_orphan_under_both_strategies():
    messages = [
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=10),
        Message('note_off', channel=0, note=60, velocity=64, time=10),
    ]
    for strategy in ('lifo', 'fifo'):
        report = detect_conflicts(make_file([messages]),
                                  overlap_strategy=strategy)
        orphans = report.by_type(ConflictType.ORPHAN_NOTE_OFF)
        assert len(orphans) == 1
        assert orphans[0].tick == 20
        assert orphans[0].events[0].message.type == 'note_off'


def test_invalid_pairing_strategy_raises():
    with pytest.raises(ValueError):
        detect_conflicts(make_file([[]]), overlap_strategy='stack')


# ---------------------------------------------------------------------
# Zero-velocity note_on and same-tick ordering
# ---------------------------------------------------------------------

def test_zero_velocity_note_on_acts_as_note_off():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_on', channel=0, note=60, velocity=0, time=100),
    ]])

    report = detect_conflicts(mid)

    assert not report.by_type(ConflictType.ORPHAN_NOTE_OFF)
    assert not report.by_type(ConflictType.DANGLING_NOTE)
    assert not report


def test_zero_velocity_note_on_without_active_note_is_orphan():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=0, time=0),
    ]])

    report = detect_conflicts(mid)

    assert types_in(report) == {ConflictType.ORPHAN_NOTE_OFF}
    assert report[0].events[0].message.velocity == 0


def test_same_tick_off_then_reopen_in_one_track_is_not_overlap():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=100),
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=100),
    ]])

    report = detect_conflicts(mid)

    assert not report


def test_same_tick_note_on_across_tracks_is_overlap():
    # Cross-track events share a tick; deterministic (track, index)
    # order still exposes the ambiguity.
    mid = make_file([
        [Message('note_on', channel=0, note=60, velocity=64, time=100),
         Message('note_off', channel=0, note=60, velocity=64, time=100)],
        [Message('note_on', channel=0, note=60, velocity=64, time=100),
         Message('note_off', channel=0, note=60, velocity=64, time=100)],
    ])

    report = detect_conflicts(mid)

    assert report.by_type(ConflictType.OVERLAP)
    assert report[0].tick == 100


# ---------------------------------------------------------------------
# Dangling notes and orphan note offs
# ---------------------------------------------------------------------

def test_dangling_note_reports_location():
    mid = make_file([[
        Message('note_on', channel=3, note=72, velocity=64, time=20),
    ]])

    report = detect_conflicts(mid)

    assert len(report) == 1
    conflict = report[0]
    assert conflict.type == ConflictType.DANGLING_NOTE
    assert conflict.tick == 20
    assert conflict.channel == 3
    assert conflict.note == 72
    assert conflict.events[0].track == 0


def test_orphan_note_off_reports_location():
    mid = make_file([
        [],
        [Message('note_off', channel=0, note=60, velocity=64, time=55)],
    ])

    report = detect_conflicts(mid)

    assert len(report) == 1
    conflict = report[0]
    assert conflict.type == ConflictType.ORPHAN_NOTE_OFF
    assert conflict.tick == 55
    assert conflict.events[0].track == 1
    assert conflict.events[0].index == 0


# ---------------------------------------------------------------------
# Drum channel
# ---------------------------------------------------------------------

def test_drum_channel_skips_note_lifecycle_checks():
    mid = make_file([[
        Message('note_on', channel=9, note=36, velocity=127, time=0),
        Message('note_on', channel=9, note=36, velocity=127, time=10),
        Message('note_off', channel=9, note=38, velocity=64, time=10),
    ]])

    report = detect_conflicts(mid)

    assert not report


def test_drum_channel_can_be_disabled():
    mid = make_file([[
        Message('note_on', channel=9, note=36, velocity=127, time=0),
    ]])

    report = detect_conflicts(mid, drum_channel=None)

    assert types_in(report) == {ConflictType.DANGLING_NOTE}


def test_drum_channel_program_conflict_is_still_reported():
    mid = make_file([
        [Message('program_change', channel=9, program=1, time=0)],
        [Message('program_change', channel=9, program=2, time=0)],
    ])

    report = detect_conflicts(mid)

    assert types_in(report) == {ConflictType.PROGRAM_CONFLICT}


# ---------------------------------------------------------------------
# Ignored channels and validation
# ---------------------------------------------------------------------

def test_ignored_channels_skip_everything():
    mid = make_file([[
        Message('program_change', channel=2, program=5, time=0),
        Message('note_on', channel=2, note=60, velocity=64, time=0),
        Message('note_on', channel=2, note=60, velocity=64, time=10),
        Message('note_off', channel=2, note=60, velocity=64, time=10),
    ]])

    report = detect_conflicts(mid, ignored_channels=[2])

    assert not report


@pytest.mark.parametrize('kwargs', [
    {'ignored_channels': [16]},
    {'ignored_channels': [-1]},
    {'ignored_channels': [True]},
    {'drum_channel': 16},
])
def test_bad_channel_raises(kwargs):
    with pytest.raises(ValueError):
        detect_conflicts(make_file([[]]), **kwargs)


# ---------------------------------------------------------------------
# Program and bank conflicts
# ---------------------------------------------------------------------

def test_cross_track_program_conflict():
    mid = make_file([
        [Message('program_change', channel=0, program=12, time=0)],
        [Message('program_change', channel=0, program=24, time=480)],
    ])

    report = detect_conflicts(mid)

    assert len(report) == 1
    conflict = report[0]
    assert conflict.type == ConflictType.PROGRAM_CONFLICT
    assert conflict.tick == 480
    assert conflict.channel == 0
    assert conflict.note is None
    assert [event.track for event in conflict.events] == [0, 1]


def test_same_program_across_tracks_is_no_conflict():
    mid = make_file([
        [Message('program_change', channel=0, program=12, time=0)],
        [Message('program_change', channel=0, program=12, time=100)],
    ])

    assert not detect_conflicts(mid)


def test_program_changes_inside_one_track_are_allowed():
    mid = make_file([[
        Message('program_change', channel=0, program=12, time=0),
        Message('program_change', channel=0, program=24, time=100),
        Message('program_change', channel=0, program=12, time=100),
    ]])

    assert not detect_conflicts(mid)


def test_bank_msb_and_lsb_conflicts_are_separate():
    mid = make_file([
        [Message('control_change', channel=0, control=0, value=1, time=0),
         Message('control_change', channel=0, control=32, value=7, time=0)],
        [Message('control_change', channel=0, control=0, value=2, time=10),
         Message('control_change', channel=0, control=32, value=8, time=10)],
    ])

    report = detect_conflicts(mid)
    bank_conflicts = report.by_type(ConflictType.BANK_CONFLICT)

    assert len(bank_conflicts) == 2
    controls = {event.message.control
                for conflict in bank_conflicts
                for event in conflict.events}
    assert controls == {0, 32}


def test_same_bank_value_repeated_does_not_re_report():
    mid = make_file([
        [Message('control_change', channel=0, control=0, value=1, time=0)],
        [Message('control_change', channel=0, control=0, value=2, time=10),
         Message('control_change', channel=0, control=0, value=2, time=10)],
    ])

    report = detect_conflicts(mid)

    assert len(report.by_type(ConflictType.BANK_CONFLICT)) == 1


# ---------------------------------------------------------------------
# File types
# ---------------------------------------------------------------------

def test_type_0_single_track_detects_internal_overlap():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=5),
        Message('note_off', channel=0, note=60, velocity=64, time=5),
    ]], type=0)

    report = detect_conflicts(mid)

    assert ConflictType.OVERLAP in types_in(report)


def test_type_2_tracks_are_analysed_independently():
    # Same channel/program in both tracks would clash in a type 1 file,
    # but type 2 sequences are independent so no cross-track conflict
    # exists. Orphan note offs are still found per track.
    mid = make_file([
        [Message('program_change', channel=0, program=1, time=0),
         Message('note_off', channel=0, note=60, velocity=64, time=10)],
        [Message('program_change', channel=0, program=2, time=0),
         Message('note_off', channel=0, note=60, velocity=64, time=20)],
    ], type=2)

    report = detect_conflicts(mid)

    assert types_in(report) == {ConflictType.ORPHAN_NOTE_OFF}
    assert {conflict.events[0].track for conflict in report} == {0, 1}
    assert {conflict.tick for conflict in report} == {10, 20}


def test_type_2_dangling_note_keeps_track_index():
    mid = make_file([
        [Message('note_on', channel=0, note=60, velocity=64, time=0)],
        [Message('note_on', channel=0, note=60, velocity=64, time=0)],
    ], type=2)

    report = detect_conflicts(mid)

    assert len(report) == 2
    assert {conflict.events[0].track for conflict in report} == {0, 1}


def test_empty_file_and_empty_tracks_are_clean():
    assert not detect_conflicts(make_file([]))
    assert not detect_conflicts(make_file([[], [], []]))


# ---------------------------------------------------------------------
# Non-mutation, ordering, API surface
# ---------------------------------------------------------------------

def test_scan_does_not_modify_or_rewrite_file():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=10),
    ]])
    buffer = io.BytesIO()
    mid.save(file=buffer)
    original_bytes = buffer.getvalue()
    original_messages = list(mid.tracks[0])

    report = detect_conflicts(mid)
    assert report  # sanity: the fixture really has a conflict

    buffer = io.BytesIO()
    mid.save(file=buffer)
    assert buffer.getvalue() == original_bytes
    assert list(mid.tracks[0]) == original_messages
    # Messages are immutable anyway; the report points at the originals.
    assert report[0].events[0].message is mid.tracks[0][0]


def test_report_is_sorted_deterministically():
    mid = make_file([
        [Message('note_off', channel=0, note=60, velocity=64, time=999)],
        [Message('note_on', channel=1, note=60, velocity=64, time=0)],
    ])

    report = detect_conflicts(mid)
    ticks = [conflict.tick for conflict in report]

    assert ticks == sorted(ticks)
    assert ticks[0] == 0


def test_report_supports_sequence_protocol():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
    ]])
    report = detect_conflicts(mid)

    assert len(report) == 1
    assert report[0] == report[-1]
    assert isinstance(next(iter(report)), Conflict)
    assert isinstance(report, ConflictReport)


def test_method_and_function_give_same_result():
    mid = make_file([
        [Message('program_change', channel=0, program=1, time=0)],
        [Message('program_change', channel=0, program=2, time=0)],
    ])

    assert mid.detect_conflicts() == detect_conflicts(mid)
    assert ('program_conflict' in mid.detect_conflicts().format())


def test_event_namedtuple_fields():
    event = Event(track=2, index=3, tick=480,
                  message=Message('note_on', channel=0, note=60))

    assert event.track == 2
    assert event.index == 3
    assert event.tick == 480
    assert event.message.note == 60


def test_other_channel_messages_are_ignored():
    # Pitchwheel, aftertouch, polytouch and unrelated CCs carry no note
    # lifecycle or patch state and must not disturb the scan.
    mid = make_file([[
        Message('pitchwheel', channel=0, pitch=100, time=0),
        Message('aftertouch', channel=0, value=5, time=0),
        Message('polytouch', channel=0, note=60, value=5, time=0),
        Message('control_change', channel=0, control=7, value=100, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=10),
    ]])

    assert not detect_conflicts(mid)
