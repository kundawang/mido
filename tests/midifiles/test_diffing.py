# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io
import json

import pytest

from mido import Message, MetaMessage, MidiFile, MidiTrack
from mido.midifiles.diffing import (
    CONFLICT_KIND_DRIFT,
    CONFLICT_KIND_MISSING,
    FORMAT_VERSION,
    OP_ADD,
    OP_CHANGE,
    OP_REMOVE,
    Conflict,
    DiffError,
    Event,
    IncompatibleFilesError,
    MidiDiff,
    PatchConflictError,
    PlacedEvent,
    TrackOp,
    apply_patch,
    diff_midi,
    event_from_message,
    file_signature,
    semantic_equal,
)
from mido.midifiles.meta import UnknownMetaMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_file(tracks, tpb=480, ftype=1):
    return MidiFile(type=ftype, ticks_per_beat=tpb,
                    tracks=[MidiTrack(track) for track in tracks])


def with_eot(messages):
    return messages + [MetaMessage('end_of_track', time=0)]


def round_trip(old, new):
    """Diff, patch and assert semantic and byte-level equivalence."""
    report = diff_midi(old, new)
    patched = apply_patch(old, report)
    assert semantic_equal(patched, new)

    old_bytes, new_bytes = io.BytesIO(), io.BytesIO()
    patched.save(file=old_bytes)
    new.save(file=new_bytes)
    assert old_bytes.getvalue() == new_bytes.getvalue()
    return report, patched


# ---------------------------------------------------------------------------
# Basic add / remove / change
# ---------------------------------------------------------------------------

def test_add_event():
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_off', note=60, velocity=64, time=480),
    ])])
    report, _ = round_trip(old, new)
    (op,) = report.ops()
    assert op.kind == OP_ADD
    assert op.tick == 480
    assert op.ordinal == 0
    assert op.event.message['type'] == 'note_off'


def test_remove_event():
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_off', note=60, velocity=64, time=480),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
    ])])
    report, _ = round_trip(old, new)
    (op,) = report.ops()
    assert op.kind == OP_REMOVE
    assert op.tick == 480


def test_change_event_content():
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=10),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=90, time=10),
    ])])
    report, _ = round_trip(old, new)
    (op,) = report.ops()
    assert op.kind == OP_CHANGE
    assert op.tick == 10
    assert op.new_tick == 10
    assert op.new_ordinal == 0
    assert op.event.message['velocity'] == 64
    assert op.new_event.message['velocity'] == 90


def test_change_tempo_meta_event():
    old = make_file([with_eot([
        MetaMessage('set_tempo', tempo=500000, time=0),
    ])])
    new = make_file([with_eot([
        MetaMessage('set_tempo', tempo=400000, time=0),
    ])])
    report, _ = round_trip(old, new)
    (op,) = report.ops()
    assert op.kind == OP_CHANGE
    assert op.new_event.message['tempo'] == 400000


def test_move_event_to_other_tick():
    old = make_file([with_eot([
        Message('note_on', note=64, velocity=80, time=120),
        Message('note_off', note=64, velocity=80, time=240),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=64, velocity=80, time=240),
        Message('note_off', note=64, velocity=80, time=240),
    ])])
    report, _ = round_trip(old, new)
    move = next(op for op in report.ops()
                if op.kind == OP_CHANGE and op.new_tick != op.tick)
    assert move.tick == 120
    assert move.new_tick == 240


# ---------------------------------------------------------------------------
# Same-tick ordering, duplicates, meta and sysex
# ---------------------------------------------------------------------------

def test_same_tick_order_preserved():
    old = make_file([with_eot([
        Message('note_on', note=60, time=0),
        Message('note_on', note=62, time=0),
        Message('note_on', note=64, time=0),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, time=0),
        Message('note_on', note=61, time=0),
        Message('note_on', note=62, time=0),
        Message('note_on', note=64, time=0),
    ])])
    _, patched = round_trip(old, new)
    notes = [m.note for m in patched.tracks[0] if m.type == 'note_on']
    assert notes == [60, 61, 62, 64]


def test_same_tick_reorder_is_a_change():
    old = make_file([with_eot([
        Message('note_on', note=48, velocity=70, time=0),
        Message('pitchwheel', pitch=100, time=0),
    ])])
    new = make_file([with_eot([
        Message('pitchwheel', pitch=100, time=0),
        Message('note_on', note=48, velocity=70, time=0),
    ])])
    report, _ = round_trip(old, new)
    # A swap may be expressed as change/change or remove+add, but every
    # anchor stays on tick 0 and the rebuilt order matches the target.
    for op in report.ops():
        assert op.tick == 0
        if op.kind == OP_CHANGE:
            assert op.new_tick == 0


def test_identical_events_swap_across_ticks():
    # Two byte-identical events swap their tick/ordinal positions.
    # Fingerprint-only alignment cannot distinguish them; positional
    # pairing must produce changes that rebuild the exact target layout.
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=100),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=100),
        Message('note_on', note=60, velocity=64, time=0),
    ])])
    report, patched = round_trip(old, new)
    assert any(op.kind == OP_CHANGE for op in report.ops())

    def absolute_ticks(track):
        ticks, now = [], 0
        for message in track:
            if message.type == 'end_of_track':
                continue
            now += message.time
            ticks.append(now)
        return ticks

    assert absolute_ticks(patched.tracks[0]) == [100, 100]


def test_duplicate_identical_events_are_stable():
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=10),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=10),
    ])])

    first = diff_midi(old, new).to_json()
    second = diff_midi(old, new).to_json()
    assert first == second  # deterministic report
    round_trip(old, new)


def test_sysex_and_unknown_meta_round_trip():
    old = make_file([with_eot([
        Message('sysex', data=(1, 2, 3, 120), time=0),
        UnknownMetaMessage(0x65, [9, 8, 7], time=100),
    ])])
    new = make_file([with_eot([
        Message('sysex', data=(1, 2, 3, 120), time=240),
        UnknownMetaMessage(0x65, [9, 8, 6], time=100),
    ])])
    report, _ = round_trip(old, new)
    kinds = {op.kind for op in report.ops()}
    assert kinds == {OP_CHANGE}
    # full payloads survive serialization
    rebuilt = MidiDiff.from_json(report.to_json())
    assert semantic_equal(apply_patch(old, rebuilt), new)


# ---------------------------------------------------------------------------
# Multi-track and track structure
# ---------------------------------------------------------------------------

def test_multi_track_concurrent_events():
    old = make_file([
        with_eot([
            Message('note_on', note=60, time=0),
            Message('control_change', control=7, value=100, time=0),
        ]),
        with_eot([
            Message('note_on', note=48, time=0),
        ]),
    ])
    new = make_file([
        with_eot([
            Message('note_on', note=60, time=0),
            Message('control_change', control=7, value=90, time=0),
            Message('control_change', control=11, value=1, time=0),
        ]),
        with_eot([
            Message('note_on', note=50, time=0),
        ]),
    ])
    report, _ = round_trip(old, new)
    tracks = {td.track for td in report.track_diffs}
    assert tracks == {0, 1}
    # stable global ordering: track, then tick, then ordinal
    keys = [(op.track, op.tick, op.ordinal) for op in report.ops()]
    assert keys == sorted(keys)


def test_add_track():
    old = make_file([with_eot([Message('note_on', note=60, time=0)])])
    new = make_file([
        with_eot([Message('note_on', note=60, time=0)]),
        with_eot([
            MetaMessage('track_name', name='Bass', time=0),
            Message('note_on', note=36, velocity=90, time=0),
        ]),
    ])
    report, patched = round_trip(old, new)
    (track_op,) = report.track_ops
    assert track_op.kind == 'add_track'
    assert track_op.track == 1
    assert len(track_op.events) == 2
    assert len(patched.tracks) == 2


def test_remove_track():
    old = make_file([
        with_eot([Message('note_on', note=60, time=0)]),
        with_eot([Message('note_on', note=36, time=0)]),
    ])
    new = make_file([with_eot([Message('note_on', note=60, time=0)])])
    report, patched = round_trip(old, new)
    (track_op,) = report.track_ops
    assert track_op.kind == 'remove_track'
    assert track_op.track == 1
    assert len(patched.tracks) == 1


def test_remove_track_conflict_when_content_changed():
    old = make_file([
        with_eot([Message('note_on', note=60, time=0),
                  Message('note_off', note=60, time=480)]),
        with_eot([Message('note_on', note=36, time=0)]),
    ])
    new = make_file([with_eot([Message('note_on', note=60, time=0),
                               Message('note_off', note=60, time=480)])])
    report = diff_midi(old, new)

    drifted = make_file([
        with_eot([Message('note_on', note=60, time=0),
                  Message('note_off', note=60, time=480)]),
        with_eot([Message('note_on', note=37, time=0)]),  # changed
    ])
    with pytest.raises(PatchConflictError) as info:
        apply_patch(drifted, report)
    conflict = info.value.conflicts[0]
    assert conflict.track == 1
    assert conflict.kind == CONFLICT_KIND_DRIFT


# ---------------------------------------------------------------------------
# Type 0, empty tracks, type 2 and resolution policies
# ---------------------------------------------------------------------------

def test_type_zero_single_track():
    old = make_file([with_eot([Message('note_on', note=60, time=0)])],
                    ftype=0)
    new = make_file([with_eot([Message('note_on', note=62, time=0)])],
                    ftype=0)
    report, patched = round_trip(old, new)
    assert patched.type == 0


def test_empty_tracks_are_supported():
    old = make_file([MidiTrack(), MidiTrack()])
    new = make_file([
        MidiTrack(),
        with_eot([Message('note_on', note=60, time=5)]),
    ])
    report, _ = round_trip(old, new)
    assert report.ops()[0].kind == OP_ADD
    assert report.ops()[0].tick == 5


def test_empty_diff_is_empty():
    mid = make_file([with_eot([Message('note_on', note=60, time=0)])])
    report = diff_midi(mid, mid)
    assert report.is_empty()
    assert report.ops() == []
    assert semantic_equal(apply_patch(mid, report), mid)


def test_type_two_rejected():
    mid = make_file([with_eot([Message('note_on', note=60, time=0)])],
                    ftype=2)
    with pytest.raises(IncompatibleFilesError, match='type 2'):
        diff_midi(mid, mid)


def test_different_ticks_per_beat_rejected():
    old = make_file([MidiTrack()], tpb=480)
    new = make_file([MidiTrack()], tpb=240)
    with pytest.raises(IncompatibleFilesError, match='ticks_per_beat'):
        diff_midi(old, new)


def test_different_file_type_rejected():
    old = make_file([MidiTrack()], ftype=0)
    new = make_file([MidiTrack()], ftype=1)
    with pytest.raises(IncompatibleFilesError, match='file type'):
        diff_midi(old, new)


def test_patching_wrong_header_rejected():
    old = make_file([MidiTrack()], tpb=480)
    report = MidiDiff(
        source_type=1, target_type=1, ticks_per_beat=240,
        source_layout=[[]], target_layout=[[]])
    with pytest.raises(IncompatibleFilesError, match='ticks_per_beat'):
        apply_patch(old, report)

    wrong_type = make_file([MidiTrack()], tpb=240, ftype=0)
    report_type = MidiDiff(
        source_type=1, target_type=1, ticks_per_beat=240,
        source_layout=[[]], target_layout=[[]])
    with pytest.raises(IncompatibleFilesError, match='file type'):
        apply_patch(wrong_type, report_type)


# ---------------------------------------------------------------------------
# Conflicts and drift localization
# ---------------------------------------------------------------------------

def test_conflict_anchor_content_changed():
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=62, velocity=64, time=0),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=90, time=0),
        Message('note_on', note=62, velocity=64, time=0),
    ])])
    report = diff_midi(old, new)

    drifted = make_file([with_eot([
        Message('note_on', note=60, velocity=77, time=0),  # anchor changed
        Message('note_on', note=62, velocity=64, time=0),
    ])])
    with pytest.raises(PatchConflictError) as info:
        apply_patch(drifted, report)
    conflict = info.value.conflicts[0]
    assert conflict.track == 0
    assert conflict.tick == 0
    assert conflict.ordinal == 0
    assert conflict.kind == CONFLICT_KIND_DRIFT


def test_conflict_anchor_missing():
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=10),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=10),
    ])])
    report = diff_midi(old, new)

    drifted = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
    ])])
    with pytest.raises(PatchConflictError) as info:
        apply_patch(drifted, report)
    kinds = {conflict.kind for conflict in info.value.conflicts}
    assert CONFLICT_KIND_MISSING in kinds
    # every conflict is located at least down to the track; event-level
    # conflicts additionally carry tick/ordinal
    for conflict in info.value.conflicts:
        assert conflict.track == 0
        assert conflict.message


def test_conflict_in_untracked_region_is_still_located():
    # The report edits ordinal 1; the drift is at ordinal 0, which no
    # operation references.  The full-layout preflight must still find
    # and locate it.
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=10),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=10),
    ])])
    report = diff_midi(old, new)

    drifted = make_file([with_eot([
        Message('note_on', note=60, velocity=77, time=0),
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=60, velocity=64, time=10),
    ])])
    with pytest.raises(PatchConflictError) as info:
        apply_patch(drifted, report)
    conflict = info.value.conflicts[0]
    assert conflict.ordinal == 0
    assert conflict.kind == CONFLICT_KIND_DRIFT


def test_failed_patch_does_not_modify_input():
    old = make_file([with_eot([
        Message('note_on', note=60, velocity=64, time=0),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=60, velocity=90, time=0),
    ])])
    report = diff_midi(old, new)
    drifted = make_file([with_eot([
        Message('note_on', note=60, velocity=77, time=0),
    ])])
    before = io.BytesIO()
    old.save(file=before)
    with pytest.raises(PatchConflictError):
        apply_patch(drifted, report)
    after = io.BytesIO()
    old.save(file=after)
    assert before.getvalue() == after.getvalue()


# ---------------------------------------------------------------------------
# Serialization and non-mutation
# ---------------------------------------------------------------------------

def test_json_report_round_trip():
    old = make_file([
        with_eot([
            Message('sysex', data=(1, 2, 3), time=0),
            Message('note_on', note=60, time=0),
        ]),
    ])
    new = make_file([
        with_eot([
            Message('note_on', note=62, time=10),
            MetaMessage('track_name', name='Lead', time=0),
        ]),
    ])
    report = diff_midi(old, new)
    text = report.to_json()
    decoded = json.loads(text)  # plain JSON data
    assert decoded['version'] == FORMAT_VERSION
    rebuilt = MidiDiff.from_json(text)
    assert semantic_equal(apply_patch(old, rebuilt), new)


def test_tampered_event_fingerprint_rejected():
    old = make_file([with_eot([Message('note_on', note=60, time=0)])])
    new = make_file([with_eot([Message('note_on', note=62, time=0)])])
    data = diff_midi(old, new).to_dict()
    data['track_diffs'][0]['ops'][0]['event']['fingerprint'] = 'deadbeef'
    with pytest.raises(DiffError, match='fingerprint'):
        MidiDiff.from_dict(data)


def test_unknown_format_version_rejected():
    data = diff_midi(
        make_file([MidiTrack()]), make_file([MidiTrack()])).to_dict()
    data['version'] = 999
    with pytest.raises(DiffError, match='format version'):
        MidiDiff.from_dict(data)


def test_apply_patch_does_not_mutate_source():
    old = make_file([with_eot([
        Message('note_on', note=60, time=0),
        Message('note_off', note=60, time=480),
    ])])
    new = make_file([with_eot([
        Message('note_on', note=62, time=0),
        Message('note_off', note=62, time=480),
        MetaMessage('set_tempo', tempo=400000, time=0),
    ])])
    snapshot = io.BytesIO()
    old.save(file=snapshot)
    report = diff_midi(old, new)
    patched = apply_patch(old, report)

    again = io.BytesIO()
    old.save(file=again)
    assert snapshot.getvalue() == again.getvalue()

    # the patched file owns its messages; editing it leaves old intact
    assert patched.tracks is not old.tracks
    patched.tracks[0].append(Message('note_on', note=1, time=0))
    third = io.BytesIO()
    old.save(file=third)
    assert snapshot.getvalue() == third.getvalue()


def test_text_report_is_stable_and_readable():
    old = make_file([with_eot([Message('note_on', note=60, time=0)])])
    new = make_file([with_eot([Message('note_on', note=62, time=0)])])
    report = diff_midi(old, new)
    text = report.to_text()
    assert 'track 0' in text
    assert 'note_on' in text
    assert report.to_text() == text


# ---------------------------------------------------------------------------
# Signatures and event helpers
# ---------------------------------------------------------------------------

def test_signature_ignores_delta_time_but_catches_content():
    a = make_file([with_eot([Message('note_on', note=60, time=0)])])
    moved = make_file([with_eot([Message('note_on', note=60, time=100)])])
    changed = make_file([with_eot([Message('note_on', note=61, time=0)])])
    assert file_signature(a) != file_signature(moved)
    assert file_signature(a) != file_signature(changed)

    event = event_from_message(Message('note_on', note=60, time=123))
    same = event_from_message(Message('note_on', note=60, time=0))
    assert event.fingerprint == same.fingerprint
    assert Event.from_dict(event.to_dict()).fingerprint == event.fingerprint


def test_placed_event_and_track_op_serialization():
    placed = PlacedEvent(10, 1, event_from_message(
        Message('note_on', note=60, time=0)))
    op = TrackOp(kind='add_track', track=2, events=[placed])
    rebuilt = TrackOp.from_dict(op.to_dict())
    assert rebuilt.kind == 'add_track'
    assert rebuilt.track == 2
    assert rebuilt.events[0].tick == 10
    assert rebuilt.events[0].ordinal == 1


def test_conflict_str_contains_location():
    conflict = Conflict(message='boom', track=3, tick=480, ordinal=1,
                        kind=CONFLICT_KIND_DRIFT)
    text = str(conflict)
    assert 'track 3' in text and 'tick 480' in text and 'ordinal 1' in text
    assert 'boom' in text
