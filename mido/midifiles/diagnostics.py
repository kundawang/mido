# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Conflict diagnostics for MIDI files.

Two arrangement tracks that accidentally use the same MIDI channel share
one voice state on playback: note on/off events from one track interfere
with the notes of the other, and program/bank changes switch the
instrument for every track on that channel.

This module scans a :class:`~mido.midifiles.midifiles.MidiFile` without
modifying it and produces a :class:`ConflictReport` describing:

* ``overlap`` -- ambiguous overlapping notes on the same channel and
  pitch (inside one track or across tracks).
* ``orphan_note_off`` -- a note off (or a zero-velocity note on, which
  is equivalent to a note off) with no matching active note.
* ``dangling_note`` -- a note still active when the end of its track is
  reached.
* ``program_conflict`` -- different program_change values sent to the
  same channel from different tracks.
* ``bank_conflict`` -- different bank_select_msb / bank_select_lsb
  control_change values sent to the same channel from different tracks.

Note events are handled as live synthesizers do: a note_on with
velocity 0 is treated as a note off. Type 0 and type 1 files are
analysed across tracks on a common tick timeline; type 2 files contain
independent, unsynchronised sequences, so every track is analysed on
its own and cross-track program/bank conflicts are not reported.
"""

from collections import namedtuple
from enum import Enum

from ..messages import Message

# Control numbers defined by the General MIDI "Bank Select" convention.
BANK_SELECT_MSB = 0
BANK_SELECT_LSB = 32

# Channel 10 (index 9) is the conventional percussion/drum channel, where
# every note is an independent one-shot sound rather than a pitched
# voice with a distinct note lifecycle.
DEFAULT_DRUM_CHANNEL = 9

PAIRING_STRATEGIES = ('lifo', 'fifo')

# Channel voice messages carrying note or patch state. Everything else
# (pitchwheel, aftertouch, polytouch, sysex, meta messages, ...) is
# irrelevant to this analysis and ignored.
_NOTE_TYPES = frozenset(('note_on', 'note_off'))
_PATCH_TYPES = frozenset(('program_change', 'control_change'))


class ConflictType(str, Enum):
    """Kinds of conflicts a report can contain.

    The string values are part of the public API and are used verbatim
    in the formatted report.
    """

    OVERLAP = 'overlap'
    ORPHAN_NOTE_OFF = 'orphan_note_off'
    DANGLING_NOTE = 'dangling_note'
    PROGRAM_CONFLICT = 'program_conflict'
    BANK_CONFLICT = 'bank_conflict'


# A single event participating in a conflict. ``message`` is the actual
# (immutable) mido Message; ``tick`` is its absolute position in ticks
# inside its track; ``track`` and ``index`` locate it in the file.
Event = namedtuple('Event', ('track', 'index', 'tick', 'message'))


class Conflict:
    """A single detected conflict.

    Attributes:
        type: a :class:`ConflictType`.
        tick: absolute tick at which the conflict is reported.
        channel: MIDI channel (0..15) involved.
        note: MIDI note number, or ``None`` for program/bank conflicts.
        events: the :class:`Event` objects involved. For an overlap this
            contains every active note plus the new note_on; for the
            other types the triggering event (and, for dangling notes,
            the note that was never closed).
    """

    __slots__ = ('channel', 'events', 'note', 'tick', 'type')

    def __init__(self, type, tick, channel, note, events):
        self.type = ConflictType(type)
        self.tick = tick
        self.channel = channel
        self.note = note
        self.events = tuple(events)

    def __repr__(self):
        return (f'Conflict(type={self.type.value!r}, tick={self.tick!r}, '
                f'channel={self.channel!r}, note={self.note!r}, '
                f'events={list(self.events)!r})')

    def __eq__(self, other):
        if not isinstance(other, Conflict):
            return NotImplemented
        return (self.type == other.type
                and self.tick == other.tick
                and self.channel == other.channel
                and self.note == other.note
                and self.events == other.events)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return hash((self.type, self.tick, self.channel, self.note))


class ConflictReport:
    """Result of a conflict scan.

    The object is an immutable sequence of :class:`Conflict` objects,
    sorted deterministically by tick, conflict type and location. It is
    falsy when no conflicts were found.
    """

    __slots__ = (
        '_conflicts',
        'drum_channel',
        'file_type',
        'ignored_channels',
        'pairing_strategy',
    )

    def __init__(self, conflicts, file_type, ignored_channels,
                 drum_channel, pairing_strategy):
        self.file_type = file_type
        self.ignored_channels = frozenset(ignored_channels)
        self.drum_channel = drum_channel
        self.pairing_strategy = pairing_strategy
        self._conflicts = tuple(conflicts)

    def __iter__(self):
        return iter(self._conflicts)

    def __len__(self):
        return len(self._conflicts)

    def __bool__(self):
        return bool(self._conflicts)

    def __getitem__(self, index):
        return self._conflicts[index]

    def __eq__(self, other):
        if not isinstance(other, ConflictReport):
            return NotImplemented
        return self._conflicts == other._conflicts

    def __repr__(self):
        return 'ConflictReport({})'.format(
            ', '.join(repr(conflict) for conflict in self._conflicts))

    def by_type(self, type):
        """Return a list of conflicts of the given :class:`ConflictType`."""
        type = ConflictType(type)
        return [conflict for conflict in self._conflicts
                if conflict.type == type]

    def format(self):
        """Return a human-readable report as a string.

        One line per conflict, for example::

            [overlap] tick=480 channel=0 note=60 tracks=[0, 1] \
events=[note_on channel=0 note=60 velocity=64 time=0 @track0:tick0, ...]
        """
        if not self._conflicts:
            return 'No conflicts found.'

        lines = []
        for conflict in self._conflicts:
            locations = ', '.join(
                f'{event.message} @track{event.track}:tick{event.tick}'
                for event in conflict.events)
            tracks = sorted({event.track for event in conflict.events})
            description = (f'[{conflict.type.value}] tick={conflict.tick} '
                           f'channel={conflict.channel}')
            if conflict.note is not None:
                description += f' note={conflict.note}'
            description += f' tracks={tracks} events=[{locations}]'
            lines.append(description)
        return '\n'.join(lines)

    def __str__(self):
        return self.format()


def _check_channel(channel, name):
    # bool is a subclass of int; reject it to avoid silent surprises.
    if not isinstance(channel, int) or isinstance(channel, bool) \
            or not 0 <= channel <= 15:
        raise ValueError(
            f'{name} must be an integer in range 0..15, got {channel!r}')


def _channel_events(tracks, note_only=False, sort=False):
    """Yield every relevant channel message with absolute tick data.

    Meta and sysex messages do not carry a channel and are skipped.
    Events keep their track and message indices so reports can point at
    the exact location in the original file.

    With ``sort=True`` the events of all tracks are merged into playback
    order (tick, then track and message index). This is the timeline of
    type 0 and 1 files. Without it, events are yielded track by track,
    which is what independent type 2 tracks need.
    """
    wanted = _NOTE_TYPES if note_only else _NOTE_TYPES | _PATCH_TYPES
    events = []
    for track_index, track in enumerate(tracks):
        tick = 0
        for message_index, message in enumerate(track):
            tick += message.time
            if isinstance(message, Message) and message.type in wanted:
                events.append(Event(track_index, message_index, tick,
                                    message))
    if sort:
        events.sort(key=lambda event: (event.tick, event.track,
                                       event.index))
    return iter(events)


def _is_note_off(message):
    """Return True for note_off and zero-velocity note_on messages."""
    return (message.type == 'note_off'
            or (message.type == 'note_on' and message.velocity == 0))


class _NoteTracker:
    """State machine for note on/off events on one (channel, note) key.

    A list is used as a stack of the currently active notes. LIFO
    pairing pops the most recent note (common for repeated notes and
    retries), FIFO pairing pops the oldest (strict chronological
    interpretation).
    """

    def __init__(self, pairing_strategy):
        self.pairing_strategy = pairing_strategy
        self.active = []

    def note_on(self, event):
        conflicts = []
        if self.active:
            # The new note_on is ambiguous while another note on the
            # same channel and pitch is still sounding. Report the whole
            # set so every involved event is visible.
            conflicts.append(Conflict(
                ConflictType.OVERLAP, event.tick,
                event.message.channel, event.message.note,
                sorted(self.active + [event],
                       key=lambda item: (item.tick, item.track,
                                         item.index))))
        self.active.append(event)
        return conflicts

    def note_off(self, event):
        if not self.active:
            return [Conflict(
                ConflictType.ORPHAN_NOTE_OFF, event.tick,
                event.message.channel, event.message.note, [event])]

        if self.pairing_strategy == 'lifo':
            self.active.pop()
        else:  # fifo
            self.active.pop(0)
        return []

    def finish(self):
        return [Conflict(
            ConflictType.DANGLING_NOTE, event.tick,
            event.message.channel, event.message.note, [event])
            for event in self.active]


class _PatchTracker:
    """Cross-track program/bank state for one channel (type 0 and 1).

    Only state changes that originate from different tracks can clash;
    repeated changes inside one track are an ordinary instrument
    switch. Each (track, kind) pair stores the value most recently sent
    by that track, so a conflict is reported the moment two tracks
    disagree.
    """

    def __init__(self, channel):
        self.channel = channel
        # kind -> {track_index: (tick, value, event)}
        self.state = {}

    def _register(self, kind, value, event):
        kind_state = self.state.setdefault(kind, {})
        own = kind_state.get(event.track)
        if own is not None and own[1] == value:
            # Same value repeated by the same track: nothing changed.
            return []

        conflicts = []
        for other_track, (other_tick, other_value, other_event) \
                in kind_state.items():
            if other_track == event.track or other_value == value:
                continue
            if kind == 'program':
                type = ConflictType.PROGRAM_CONFLICT
            else:
                type = ConflictType.BANK_CONFLICT
            conflicts.append(Conflict(
                type, event.tick, self.channel, None,
                sorted([other_event, event],
                       key=lambda item: (item.tick, item.track,
                                         item.index))))
        kind_state[event.track] = (event.tick, value, event)
        return conflicts

    def handle(self, event):
        message = event.message
        if message.type == 'program_change':
            return self._register('program', message.program, event)
        if message.type == 'control_change':
            if message.control == BANK_SELECT_MSB:
                return self._register('bank_msb', message.value, event)
            if message.control == BANK_SELECT_LSB:
                return self._register('bank_lsb', message.value, event)
        return []


def _conflict_sort_key(conflict):
    # Deterministic order even when several conflicts share a tick.
    event_key = min((event.tick, event.track, event.index)
                    for event in conflict.events)
    note = -1 if conflict.note is None else conflict.note
    return (conflict.tick, conflict.type.value, conflict.channel,
            note, event_key)


def detect_conflicts(midifile, overlap_strategy='lifo',
                     ignored_channels=None, drum_channel=DEFAULT_DRUM_CHANNEL):
    """Scan a MIDI file for channel and note conflicts.

    The file is only read; messages and tracks are never modified and
    nothing is written back.

    Args:
        midifile: a :class:`~mido.midifiles.midifiles.MidiFile`.
        overlap_strategy: how note_off events are paired with active
            note_on events when several overlap. ``'lifo'`` (the
            default) pairs a note off with the most recently started
            note, ``'fifo'`` with the oldest. Overlap detection itself
            does not depend on the strategy.
        ignored_channels: iterable of channel numbers (0..15) to skip
            completely (no note, program or bank checks).
        drum_channel: channel number (0..15) treated as the
            percussion/drum channel. Notes there are one-shot sounds, so
            note lifecycle checks (overlap, orphan note off, dangling
            note) are skipped; program/bank conflicts are still
            reported. Pass ``None`` to treat every channel as a pitched
            channel.

    Returns:
        A :class:`ConflictReport`.

    Raises:
        ValueError: for an unknown pairing strategy or an out-of-range
            channel number.
    """
    if overlap_strategy not in PAIRING_STRATEGIES:
        raise ValueError(
            f'overlap_strategy must be one of {PAIRING_STRATEGIES}, '
            f'got {overlap_strategy!r}')

    if ignored_channels is None:
        ignored_channels = frozenset()
    else:
        ignored_channels = frozenset(ignored_channels)
    for channel in ignored_channels:
        _check_channel(channel, 'ignored_channels entry')

    if drum_channel is not None:
        _check_channel(drum_channel, 'drum_channel')

    conflicts = []

    if midifile.type in (0, 1):
        conflicts.extend(_scan_synchronized(
            midifile.tracks, overlap_strategy,
            ignored_channels, drum_channel))
    elif midifile.type == 2:
        # Each track is an independent, unsynchronised sequence: analyse
        # note lifecycles per track and never compare patches across
        # tracks. Events keep their original track index.
        events_by_track = {}
        for event in _channel_events(midifile.tracks, note_only=True):
            events_by_track.setdefault(event.track, []).append(event)
        for track_events in events_by_track.values():
            conflicts.extend(_scan_notes(
                iter(track_events), overlap_strategy,
                ignored_channels, drum_channel))
    else:  # pragma: no cover - MidiFile rejects other types on load
        raise ValueError(
            f'unknown MIDI file type {midifile.type!r}')

    conflicts.sort(key=_conflict_sort_key)
    return ConflictReport(
        conflicts, midifile.type, ignored_channels,
        drum_channel, overlap_strategy)


def _scan_synchronized(tracks, overlap_strategy,
                       ignored_channels, drum_channel):
    """Type 0/1 scan: one shared tick timeline across all tracks."""
    conflicts = []

    # One playback-ordered stream for all tracks, deterministic at
    # equal ticks (track index, then message index).
    events = list(_channel_events(tracks, sort=True))

    conflicts.extend(_scan_notes(
        (event for event in events
         if event.message.type in _NOTE_TYPES),
        overlap_strategy, ignored_channels, drum_channel))

    patch_trackers = {}
    for event in events:
        if event.message.type not in _PATCH_TYPES:
            continue
        if event.message.channel in ignored_channels:
            continue
        tracker = patch_trackers.setdefault(
            event.message.channel,
            _PatchTracker(event.message.channel))
        conflicts.extend(tracker.handle(event))

    return conflicts


def _scan_notes(events, overlap_strategy, ignored_channels, drum_channel):
    """Run the note lifecycle state machine over a stream of events."""
    # (channel, note) -> _NoteTracker. Events arrive in playback order,
    # which inside a type 0/1 file is (tick, track index, message
    # index); same-tick events therefore keep deterministic track order.
    trackers = {}
    conflicts = []

    for event in events:
        channel = event.message.channel
        if channel in ignored_channels or channel == drum_channel:
            continue

        key = (channel, event.message.note)
        tracker = trackers.get(key)
        if tracker is None:
            tracker = _NoteTracker(overlap_strategy)
            trackers[key] = tracker

        if _is_note_off(event.message):
            conflicts.extend(tracker.note_off(event))
        else:
            conflicts.extend(tracker.note_on(event))

    for tracker in trackers.values():
        conflicts.extend(tracker.finish())

    return conflicts
