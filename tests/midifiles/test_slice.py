# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

from pytest import raises, approx

from mido.messages import Message
from mido.midifiles.meta import MetaMessage
from mido.midifiles.midifiles import MidiFile, MidiTrack


def make_track(events):
    """Build a track from (absolute_time, message) pairs."""
    track = MidiTrack()
    now = 0
    for time, msg in events:
        track.append(msg.copy(time=time - now))
        now = time
    return track


def make_tempo_file():
    """Type 1 file with a tempo change, channel state and notes that
    cross the slice boundaries [480, 960)."""
    track0 = make_track([
        (0, MetaMessage('set_tempo', tempo=500000, time=0)),
        (480, MetaMessage('set_tempo', tempo=250000, time=0)),
    ])
    track1 = make_track([
        (0, Message('control_change', channel=0, control=0, value=1)),
        (0, Message('control_change', channel=0, control=32, value=2)),
        (0, Message('program_change', channel=0, program=5)),
        (0, Message('control_change', channel=0, control=64, value=127)),
        (0, Message('note_on', channel=0, note=60, velocity=100)),
        (240, Message('note_off', channel=0, note=60, velocity=0)),
        # Crosses the start of the slice.
        (240, Message('note_on', channel=0, note=61, velocity=101)),
        (480, Message('note_on', channel=0, note=64, velocity=102)),
        (600, Message('note_off', channel=0, note=61, velocity=0)),
        (720, Message('note_off', channel=0, note=64, velocity=0)),
        # Crosses the end of the slice.
        (720, Message('note_on', channel=0, note=65, velocity=103)),
        (1200, Message('note_off', channel=0, note=65, velocity=0)),
    ])
    return MidiFile(type=1, ticks_per_beat=480, tracks=[track0, track1])


def test_slice_truncate():
    sliced = make_tempo_file().slice(480, 960)

    assert sliced.type == 1
    assert sliced.ticks_per_beat == 480
    assert len(sliced.tracks) == 2

    # The active tempo is replayed, followed by the tempo change
    # that happens exactly at the start of the slice.
    assert sliced.tracks[0] == [
        MetaMessage('set_tempo', tempo=500000, time=0),
        MetaMessage('set_tempo', tempo=250000, time=0),
        MetaMessage('end_of_track', time=0),
    ]

    assert sliced.tracks[1] == [
        # State at the start of the slice: bank select, program,
        # then the remaining controllers.
        Message('control_change', channel=0, control=0, value=1, time=0),
        Message('control_change', channel=0, control=32, value=2, time=0),
        Message('program_change', channel=0, program=5, time=0),
        Message('control_change', channel=0, control=64, value=127, time=0),
        # Note 61 crosses the start and is re-articulated.
        Message('note_on', channel=0, note=61, velocity=101, time=0),
        # Note 64 starts exactly at the start of the slice.
        Message('note_on', channel=0, note=64, velocity=102, time=0),
        Message('note_off', channel=0, note=61, velocity=0, time=120),
        Message('note_off', channel=0, note=64, velocity=0, time=120),
        Message('note_on', channel=0, note=65, velocity=103, time=0),
        # Note 65 crosses the end and is truncated with a generated
        # note_off exactly at the end of the slice.
        Message('note_off', channel=0, note=65, velocity=64, time=240),
        MetaMessage('end_of_track', time=0),
    ]


def test_slice_keep():
    sliced = make_tempo_file().slice(480, 960, note_policy='keep')

    # The note crossing the end keeps its original note_off, so the
    # track extends past the end of the slice (tick 1200 - 480 = 720).
    assert sliced.tracks[1][-2:] == [
        Message('note_off', channel=0, note=65, velocity=0, time=480),
        MetaMessage('end_of_track', time=0),
    ]
    # Everything before that is identical to the truncated slice,
    # minus the generated note_off.
    truncated = make_tempo_file().slice(480, 960)
    assert sliced.tracks[1][:-2] == truncated.tracks[1][:-2]


def test_keep_generates_note_off_when_missing():
    track = make_track([
        (0, Message('note_on', channel=0, note=60, velocity=100)),
        (100, Message('note_on', channel=0, note=64, velocity=100)),
        # Note 64 has no note_off in the source.
    ])
    mid = MidiFile(type=1, ticks_per_beat=480, tracks=[track])

    sliced = mid.slice(50, 200, note_policy='keep')
    assert sliced.tracks[0] == [
        # Note 60 crosses the start and is re-articulated.
        Message('note_on', channel=0, note=60, velocity=100, time=0),
        Message('note_on', channel=0, note=64, velocity=100, time=50),
        # Neither note has a note_off in the source, so note_offs are
        # generated at the end of the slice even with 'keep'.
        Message('note_off', channel=0, note=60, velocity=64, time=100),
        Message('note_off', channel=0, note=64, velocity=64, time=0),
        MetaMessage('end_of_track', time=0),
    ]


def test_playback_timeline():
    # At tempo 250000 and 480 ticks per beat one tick is 0.25/480 s.
    sliced = make_tempo_file().slice(480, 960)
    messages = [msg for msg in sliced
                if not msg.is_meta and msg.type != 'end_of_track']
    # All state and note_on messages happen at time 0, then note_offs
    # at 120 and 240 ticks, and the final note_off at 480 ticks.
    times = []
    now = 0.0
    for msg in sliced:
        now += msg.time
        if msg.type.startswith('note_'):
            times.append(now)
    assert times == approx([0, 0, 0.0625, 0.125, 0.125, 0.25])
    assert sliced.length == approx(0.25)


def test_multichannel_sustain_state():
    track = make_track([
        (0, Message('control_change', channel=0, control=64, value=127)),
        (50, Message('program_change', channel=1, program=20)),
        (100, Message('control_change', channel=1, control=64, value=127)),
        # Sustain released on channel 0 before the slice.
        (200, Message('control_change', channel=0, control=64, value=0)),
        (300, Message('note_on', channel=1, note=70, velocity=90)),
        (400, Message('note_off', channel=1, note=70, velocity=0)),
    ])
    mid = MidiFile(type=1, ticks_per_beat=480, tracks=[track])

    sliced = mid.slice(300, 400)
    assert sliced.tracks[0] == [
        Message('control_change', channel=0, control=64, value=0, time=0),
        Message('program_change', channel=1, program=20, time=0),
        Message('control_change', channel=1, control=64, value=127, time=0),
        Message('note_on', channel=1, note=70, velocity=90, time=0),
        # The note_off at 400 is outside the half-open interval, so a
        # note_off is generated at the end of the slice instead.
        Message('note_off', channel=1, note=70, velocity=64, time=100),
        MetaMessage('end_of_track', time=0),
    ]


def test_channel_mode_messages_not_replayed():
    track = make_track([
        (0, Message('control_change', channel=0, control=123, value=0)),
        (0, Message('control_change', channel=0, control=7, value=100)),
        (100, Message('note_on', channel=0, note=60, velocity=90)),
        (200, Message('note_off', channel=0, note=60, velocity=0)),
    ])
    mid = MidiFile(type=1, ticks_per_beat=480, tracks=[track])

    sliced = mid.slice(100, 200)
    # "All notes off" (123) is a channel mode message and is not part
    # of the replayed state; the volume controller (7) is.
    assert sliced.tracks[0][0] == Message(
        'control_change', channel=0, control=7, value=100, time=0)
    assert all(msg.control != 123 for msg in sliced.tracks[0]
               if msg.type == 'control_change')


def test_empty_slice():
    mid = make_tempo_file()

    # Past the end of the file: only the state is replayed.
    sliced = mid.slice(2000, 3000)
    assert sliced.tracks[0] == [
        MetaMessage('set_tempo', tempo=250000, time=0),
        MetaMessage('end_of_track', time=0),
    ]
    assert sliced.tracks[1] == [
        Message('control_change', channel=0, control=0, value=1, time=0),
        Message('control_change', channel=0, control=32, value=2, time=0),
        Message('program_change', channel=0, program=5, time=0),
        Message('control_change', channel=0, control=64, value=127, time=0),
        MetaMessage('end_of_track', time=0),
    ]
    assert sliced.length == 0

    # No state and no messages at all: just an end_of_track.
    bare = MidiFile(type=1, ticks_per_beat=480, tracks=[make_track([
        (0, Message('note_on', channel=0, note=60, velocity=90)),
        (100, Message('note_off', channel=0, note=60, velocity=0)),
    ])])
    sliced = bare.slice(200, 300)
    assert sliced.tracks[0] == [MetaMessage('end_of_track', time=0)]


def test_same_tick_events_come_after_state():
    track = make_track([
        (0, Message('program_change', channel=0, program=1)),
        (100, Message('program_change', channel=0, program=2)),
        (100, Message('note_on', channel=0, note=60, velocity=90)),
        (200, Message('note_off', channel=0, note=60, velocity=0)),
    ])
    mid = MidiFile(type=1, ticks_per_beat=480, tracks=[track])

    sliced = mid.slice(100, 200)
    assert sliced.tracks[0] == [
        # Replayed state comes first.
        Message('program_change', channel=0, program=1, time=0),
        # Then the messages at exactly the start of the slice.
        Message('program_change', channel=0, program=2, time=0),
        Message('note_on', channel=0, note=60, velocity=90, time=0),
        # The note_off at tick 200 is outside the half-open interval,
        # so one is generated at the end of the slice instead.
        Message('note_off', channel=0, note=60, velocity=64, time=100),
        MetaMessage('end_of_track', time=0),
    ]


def test_slice_seconds():
    mid = make_tempo_file()
    # Tempo 500000: 480 ticks (one beat) = 0.5 s. Tempo 250000:
    # 480 ticks = 0.25 s, so tick 960 is at 0.75 s.
    by_seconds = mid.slice(0.5, 0.75, seconds=True)
    by_ticks = mid.slice(480, 960)
    assert by_seconds.tracks == by_ticks.tracks


def test_slice_type0():
    track = make_track([
        (0, MetaMessage('set_tempo', tempo=500000, time=0)),
        (0, Message('note_on', channel=0, note=60, velocity=90)),
        (100, Message('note_off', channel=0, note=60, velocity=0)),
        (200, Message('note_on', channel=0, note=62, velocity=90)),
        (300, Message('note_off', channel=0, note=62, velocity=0)),
    ])
    mid = MidiFile(type=0, ticks_per_beat=480, tracks=[track])

    sliced = mid.slice(100, 300)
    assert sliced.type == 0
    assert len(sliced.tracks) == 1
    assert sliced.tracks[0] == [
        MetaMessage('set_tempo', tempo=500000, time=0),
        # The note_off at tick 100 is inside the half-open interval.
        Message('note_off', channel=0, note=60, velocity=0, time=0),
        Message('note_on', channel=0, note=62, velocity=90, time=100),
        # The note_off at tick 300 is outside the half-open interval,
        # so one is generated at the end of the slice instead.
        Message('note_off', channel=0, note=62, velocity=64, time=100),
        MetaMessage('end_of_track', time=0),
    ]


def test_slice_type2():
    tracks = [
        make_track([
            (0, Message('note_on', channel=0, note=60, velocity=90)),
            (100, Message('note_off', channel=0, note=60, velocity=0)),
        ]),
        make_track([
            (50, Message('note_on', channel=1, note=70, velocity=90)),
            (150, Message('note_off', channel=1, note=70, velocity=0)),
        ]),
    ]
    mid = MidiFile(type=2, ticks_per_beat=480, tracks=tracks)

    # Slicing by ticks works: each track is sliced independently.
    sliced = mid.slice(0, 100)
    assert sliced.type == 2
    assert sliced.tracks[0] == [
        Message('note_on', channel=0, note=60, velocity=90, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=100),
        MetaMessage('end_of_track', time=0),
    ]
    assert sliced.tracks[1] == [
        Message('note_on', channel=1, note=70, velocity=90, time=50),
        Message('note_off', channel=1, note=70, velocity=64, time=50),
        MetaMessage('end_of_track', time=0),
    ]

    # Type 2 tracks have no shared timeline, so seconds are rejected.
    with raises(ValueError):
        mid.slice(0, 1, seconds=True)


def test_slice_defaults_to_whole_file():
    mid = make_tempo_file()
    sliced = mid.slice()
    # Slicing the whole file reproduces every message (plus a
    # normalized end_of_track per track).
    for track, sliced_track in zip(mid.tracks, sliced.tracks):
        assert ([msg for msg in sliced_track
                 if msg.type != 'end_of_track'] == list(track))


def test_invalid_interval():
    mid = make_tempo_file()
    with raises(ValueError):
        mid.slice(960, 480)
    with raises(ValueError):
        mid.slice(480, 480)
    with raises(ValueError):
        mid.slice(-1, 480)
    with raises(ValueError):
        mid.slice(480, 960, note_policy='explode')


def test_source_file_not_modified():
    mid = make_tempo_file()
    before = [[repr(msg) for msg in track] for track in mid.tracks]
    mid.slice(480, 960)
    mid.slice(480, 960, note_policy='keep')
    assert [[repr(msg) for msg in track] for track in mid.tracks] == before


def test_slice_save_roundtrip():
    sliced = make_tempo_file().slice(480, 960)

    file = io.BytesIO()
    sliced.save(file=file)
    file.seek(0)
    loaded = MidiFile(file=file)

    assert loaded.type == sliced.type
    assert loaded.ticks_per_beat == sliced.ticks_per_beat
    assert loaded.tracks == sliced.tracks
