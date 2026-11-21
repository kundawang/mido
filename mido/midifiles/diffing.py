# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Semantic diff and patch between MIDI files.

Where a byte-level diff only shows that something in a file changed,
this module describes *musical* changes: which event was added,
removed or modified on which track, at which absolute tick, while
keeping the order of events that occur at the same tick intact.

Example:

    >>> from mido import MidiFile
    >>> from mido.midifiles.diffing import diff_midi, apply_patch
    >>> report = diff_midi(old_file, new_file)
    >>> patched = apply_patch(old_file, report)

The report is a plain-data object and can be serialized to JSON with
:meth:`MidiDiff.to_dict` / :meth:`MidiDiff.from_dict`.

Design notes
------------

* Events are identified by a triple ``(track index, absolute tick,
  ordinal)`` where *ordinal* is the position of the event among the
  events sharing the same tick.  This preserves the order of
  simultaneous events, which is meaningful in MIDI (for example a
  controller change right before a note on at the same tick).
* The event *fingerprint* is the canonical wire encoding of the
  message with the delta time stripped (``msg.bytes()``).  Meta and
  SysEx data are therefore compared byte for byte.
* ``end_of_track`` meta messages are not part of the model: they are
  structural markers that :class:`MidiFile` regenerates on every save,
  so including them would produce noise and they could never anchor an
  edit.
* Type 2 (asynchronous) files are rejected: their tracks do not share
  a tick timeline, so a semantic comparison across them is not
  meaningful.
"""

import copy
import hashlib
import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from ..messages import Message
from .meta import MetaMessage, UnknownMetaMessage
from .midifiles import MidiFile
from .tracks import MidiTrack

# Bump when the serialized report format changes in an incompatible way.
FORMAT_VERSION = 1

#: A new event was inserted into a track.
OP_ADD = 'add'
#: An event was removed from a track.
OP_REMOVE = 'remove'
#: An event was replaced in place (content and/or tick changed).
OP_CHANGE = 'change'

#: Track-level operations, namespaced so they can never collide with
#: event-level operation names.
OP_ADD_TRACK = 'add_track'
OP_REMOVE_TRACK = 'remove_track'

#: Conflict categories reported through :class:`Conflict.kind`.
CONFLICT_KIND_DRIFT = 'drift'
CONFLICT_KIND_MISSING = 'missing'

_EVENT_OPS = (OP_ADD, OP_REMOVE, OP_CHANGE)
_TRACK_OPS = (OP_ADD_TRACK, OP_REMOVE_TRACK)

# Deterministic ordering of operation kinds when everything else ties.
_KIND_RANK = {
    OP_REMOVE: 0,
    OP_CHANGE: 1,
    OP_ADD: 2,
    OP_ADD_TRACK: 0,
    OP_REMOVE_TRACK: 1,
}


class DiffError(ValueError):
    """Base class for errors raised by the diffing module."""


class IncompatibleFilesError(DiffError):
    """Raised when two files cannot be compared or patched together."""


class PatchConflictError(DiffError):
    """Raised when a report cannot be applied to a source file.

    ``conflicts`` holds one :class:`Conflict` per anchor that could not
    be resolved, so the caller can locate every problem instead of
    only the first one.
    """

    def __init__(self, conflicts: List['Conflict']):
        self.conflicts = list(conflicts)
        locations = '; '.join(str(conflict) for conflict in self.conflicts)
        super().__init__(f'patch conflicts: {locations}')


@dataclass
class Conflict:
    """A failed anchor resolution located down to track and tick."""

    message: str
    track: Optional[int] = None
    tick: Optional[int] = None
    ordinal: Optional[int] = None
    kind: Optional[str] = None

    def __str__(self):
        parts = []
        if self.kind is not None:
            parts.append(self.kind)
        if self.track is not None:
            parts.append(f'track {self.track}')
        if self.tick is not None:
            parts.append(f'tick {self.tick}')
        if self.ordinal is not None:
            parts.append(f'ordinal {self.ordinal}')
        prefix = ' '.join(parts) + ': ' if parts else ''
        return prefix + self.message


@dataclass
class Event:
    """A timestamped MIDI event with no comparison ambiguity.

    ``fingerprint`` is the hex-encoded canonical wire representation of
    the message without its delta time.  It identifies the event
    content (including full Meta/SysEx payloads) independently of when
    it occurs.
    """

    fingerprint: str
    message: dict

    def __eq__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return self.fingerprint == other.fingerprint

    def __hash__(self):
        return hash(self.fingerprint)

    def __repr__(self):
        return f'Event({self.fingerprint}) {self.describe()}'

    def describe(self) -> str:
        """Human-readable one-line description of the event content."""
        msg = self.to_message()
        return repr(msg)

    def to_message(self):
        """Rebuild an independent (mutable, delta-timed) mido message."""
        return message_from_dict(copy.deepcopy(self.message))

    def to_dict(self) -> dict:
        return {
            'fingerprint': self.fingerprint,
            'message': copy.deepcopy(self.message),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        event = cls(
            fingerprint=data['fingerprint'],
            message=copy.deepcopy(data['message']),
        )
        # Guard against hand-edited or corrupted reports.
        if event.fingerprint != fingerprint(event.to_message()):
            raise DiffError(
                'event fingerprint does not match its message payload')
        return event


@dataclass
class DiffOp:
    """A single add/remove/change operation on one track.

    Attributes:
        kind: one of :data:`OP_ADD`, :data:`OP_REMOVE`,
            :data:`OP_CHANGE`.
        track: index of the track the operation applies to.
        tick: anchor tick in the *source* file for remove/change, and
            the destination tick for add.
        ordinal: anchor ordinal (position among events sharing
            ``tick``) in the source for remove/change, and the
            destination ordinal for add.
        event: event being added or removed.
        new_tick: destination tick for a change (``None`` if the event
            stays on the same tick).
        new_ordinal: destination ordinal for a change (``None`` if it
            keeps the same ordinal).
        new_event: replacement event for a change (``None`` for
            add/remove).
    """

    kind: str
    track: int
    tick: int
    ordinal: int
    event: Optional[Event] = None
    new_tick: Optional[int] = None
    new_ordinal: Optional[int] = None
    new_event: Optional[Event] = None

    def __post_init__(self):
        if self.kind not in _EVENT_OPS:
            raise ValueError(f'invalid event operation kind: {self.kind!r}')

    @property
    def sort_key(self) -> Tuple:
        return (self.track, self.tick, _KIND_RANK[self.kind], self.ordinal)

    def to_dict(self) -> dict:
        data = {
            'kind': self.kind,
            'track': self.track,
            'tick': self.tick,
            'ordinal': self.ordinal,
        }
        if self.event is not None:
            data['event'] = self.event.to_dict()
        if self.kind == OP_CHANGE:
            data['new_tick'] = self.new_tick
            data['new_ordinal'] = self.new_ordinal
            data['new_event'] = self.new_event.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'DiffOp':
        kind = data['kind']
        op = cls(
            kind=kind,
            track=data['track'],
            tick=data['tick'],
            ordinal=data['ordinal'],
            event=Event.from_dict(data['event']) if 'event' in data else None,
            new_tick=data.get('new_tick'),
            new_ordinal=data.get('new_ordinal'),
            new_event=(Event.from_dict(data['new_event'])
                       if 'new_event' in data else None),
        )
        return op


@dataclass
class TrackDiff:
    """All event-level operations applying to one track."""

    track: int
    ops: List[DiffOp] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'track': self.track,
            'ops': [op.to_dict() for op in self.ops],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TrackDiff':
        return cls(
            track=data['track'],
            ops=[DiffOp.from_dict(op) for op in data['ops']],
        )


@dataclass
class PlacedEvent:
    """An event placed at an absolute tick with its same-tick ordinal.

    Used for whole-track operations, where the full target timeline
    must be carried by the report.
    """

    tick: int
    ordinal: int
    event: Event

    def to_dict(self) -> dict:
        return {
            'tick': self.tick,
            'ordinal': self.ordinal,
            'event': self.event.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PlacedEvent':
        return cls(
            tick=data['tick'],
            ordinal=data['ordinal'],
            event=Event.from_dict(data['event']),
        )


@dataclass
class TrackOp:
    """A whole-track add or remove operation."""

    kind: str
    track: int
    events: Optional[List[PlacedEvent]] = None

    def __post_init__(self):
        if self.kind not in _TRACK_OPS:
            raise ValueError(f'invalid track operation kind: {self.kind!r}')

    @property
    def sort_key(self) -> Tuple:
        return (self.track, _KIND_RANK[self.kind])

    def to_dict(self) -> dict:
        data = {'kind': self.kind, 'track': self.track}
        if self.events is not None:
            data['events'] = [event.to_dict() for event in self.events]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'TrackOp':
        return cls(
            kind=data['kind'],
            track=data['track'],
            events=([PlacedEvent.from_dict(event)
                     for event in data['events']]
                    if 'events' in data else None),
        )


@dataclass
class MidiDiff:
    """A complete, stable and serializable semantic diff.

    A report is *stable*: diffing the same two files repeatedly, or
    files containing identical duplicate events, yields byte-identical
    reports.  ``source_signature`` and ``target_signature`` are SHA256
    hashes of the semantic content of each side and let
    :func:`apply_patch` reject a report applied to a changed source.

    ``source_layout`` and ``target_layout`` hold the full semantic
    timeline of each side as ``(tick, ordinal, fingerprint)`` triples.
    They carry no message payloads (those live in the operations), but
    let the patcher locate drift anywhere in the file, even inside
    regions the report does not edit, rather than only reporting a
    signature mismatch.
    """

    source_type: int
    target_type: int
    ticks_per_beat: int
    source_layout: List[List[List]]
    target_layout: List[List[List]]
    track_diffs: List[TrackDiff] = field(default_factory=list)
    track_ops: List[TrackOp] = field(default_factory=list)
    version: int = FORMAT_VERSION

    @property
    def source_signature(self) -> str:
        """SHA256 of the source file's semantic layout."""
        return _layout_signature(
            self.source_type, self.ticks_per_beat, self.source_layout)

    @property
    def target_signature(self) -> str:
        """SHA256 of the target file's semantic layout."""
        return _layout_signature(
            self.target_type, self.ticks_per_beat, self.target_layout)

    def ops(self) -> List[DiffOp]:
        """All event-level operations in stable order."""
        operations = []
        for track_diff in self.track_diffs:
            operations.extend(track_diff.ops)
        return sorted(operations, key=lambda op: op.sort_key)

    def is_empty(self) -> bool:
        """True if the two files were semantically equivalent."""
        return not self.track_diffs and not self.track_ops

    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'source_type': self.source_type,
            'target_type': self.target_type,
            'ticks_per_beat': self.ticks_per_beat,
            'source_signature': self.source_signature,
            'target_signature': self.target_signature,
            'source_layout': copy.deepcopy(self.source_layout),
            'target_layout': copy.deepcopy(self.target_layout),
            'track_diffs': [td.to_dict() for td in self.track_diffs],
            'track_ops': [op.to_dict() for op in self.track_ops],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> 'MidiDiff':
        version = data.get('version', 1)
        if version != FORMAT_VERSION:
            raise DiffError(
                f'unsupported diff format version {version} '
                f'(expected {FORMAT_VERSION})')
        report = cls(
            version=version,
            source_type=data['source_type'],
            target_type=data['target_type'],
            ticks_per_beat=data['ticks_per_beat'],
            source_layout=copy.deepcopy(data['source_layout']),
            target_layout=copy.deepcopy(data['target_layout']),
            track_diffs=[TrackDiff.from_dict(td)
                         for td in data.get('track_diffs', [])],
            track_ops=[TrackOp.from_dict(op)
                       for op in data.get('track_ops', [])],
        )
        # Stored signatures are redundant hashes of the layouts; verify
        # them to reject truncated or hand-edited reports.
        for side, stored in (
                ('source', data.get('source_signature')),
                ('target', data.get('target_signature'))):
            if stored is not None:
                actual = (report.source_signature if side == 'source'
                          else report.target_signature)
                if actual != stored:
                    raise DiffError(
                        f'{side}_signature does not match the stored '
                        f'{side}_layout')
        return report

    @classmethod
    def from_json(cls, text: str) -> 'MidiDiff':
        """Rebuild a report previously written with :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def to_text(self) -> str:
        """Render a human-readable report.

        The format is stable and intended for reviewers and logs;
        :meth:`to_json` is the machine-readable form.
        """
        lines = [
            f'# MIDI semantic diff (format v{self.version})',
            f'# type {self.source_type} -> {self.target_type}, '
            f'ticks_per_beat={self.ticks_per_beat}',
            f'# source {self.source_signature[:12]} '
            f'-> target {self.target_signature[:12]}',
        ]
        if self.is_empty():
            lines.append('# no semantic changes')
        for track_diff in self.track_diffs:
            for op in track_diff.ops:
                lines.append(_format_event_op(op))
        for track_op in sorted(self.track_ops, key=lambda op: op.sort_key):
            if track_op.kind == OP_ADD_TRACK:
                lines.append(f'+ track {track_op.track} '
                             f'({len(track_op.events)} events)')
            else:
                lines.append(f'- track {track_op.track}')
        return '\n'.join(lines)


def _format_event_op(op: DiffOp) -> str:
    if op.kind == OP_ADD:
        return (f'+ track {op.track} tick {op.tick} ordinal {op.ordinal}: '
                f'{op.event.describe()}')
    if op.kind == OP_REMOVE:
        return (f'- track {op.track} tick {op.tick} ordinal {op.ordinal}: '
                f'{op.event.describe()}')
    moved = ''
    if op.new_tick != op.tick or op.new_ordinal != op.ordinal:
        moved = (f' -> tick {op.new_tick} ordinal {op.new_ordinal}')
    return (f'~ track {op.track} tick {op.tick} ordinal {op.ordinal}: '
            f'{op.event.describe()}{moved} '
            f'=> {op.new_event.describe()}')


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def fingerprint(message) -> str:
    """Return the canonical content fingerprint of a message.

    The delta time is not part of the fingerprint, so the same event at
    two different ticks produces the same fingerprint.
    """
    if isinstance(message, UnknownMetaMessage):
        data = [0xff, message.type_byte] + list(message.data)
    else:
        data = list(message.bytes())
    return bytes(data).hex()


def event_from_message(message) -> Event:
    """Build an :class:`Event` from any mido file message."""
    return Event(
        fingerprint=fingerprint(message),
        message=message.dict(),
    )


def message_from_dict(data: dict):
    """Reverse of ``message.dict()`` for file messages."""
    data = dict(data)
    message_type = data.pop('type')
    if 'type_byte' in data:
        # UnknownMetaMessage.dict() exposes type_byte and data.
        type_byte = data.pop('type_byte')
        return UnknownMetaMessage(type_byte=type_byte, **data)
    if _is_meta_type(message_type):
        return MetaMessage(message_type, **data)
    return Message(message_type, **data)


def _is_meta_type(message_type: str) -> bool:
    # Imported lazily to avoid pulling the registry into module import
    # cost; it is a plain dictionary lookup.
    from .meta import _META_SPEC_BY_TYPE
    return message_type in _META_SPEC_BY_TYPE


# ---------------------------------------------------------------------------
# Track normalization
# ---------------------------------------------------------------------------

@dataclass
class _AbsEvent:
    """An event placed on the absolute-tick timeline."""

    tick: int
    ordinal: int
    event: Event


def absolute_events(track) -> List[_AbsEvent]:
    """Convert a delta-timed MidiTrack to absolute-tick events.

    ``end_of_track`` messages are dropped (see module docstring).  The
    ordinal counts only the surviving events sharing a tick, starting
    at zero.
    """
    result = []
    now = 0
    per_tick = {}
    for message in track:
        now += message.time
        if message.type == 'end_of_track':
            continue
        ordinal = per_tick.get(now, 0)
        per_tick[now] = ordinal + 1
        result.append(_AbsEvent(
            tick=now,
            ordinal=ordinal,
            event=event_from_message(message),
        ))
    return result


def relative_track(events: List[_AbsEvent]) -> MidiTrack:
    """Rebuild a delta-timed MidiTrack from absolute-tick events."""
    track = MidiTrack()
    last_tick = 0
    for item in events:
        message = item.event.to_message()
        message.time = item.tick - last_tick
        track.append(message)
        last_tick = item.tick
    return track


def file_layout(mid: MidiFile) -> List[List[List]]:
    """Return the full semantic timeline of a file.

    The layout is a list of tracks, each a list of
    ``[tick, ordinal, fingerprint]`` triples in track order.
    """
    return [
        [
            [item.tick, item.ordinal, item.event.fingerprint]
            for item in absolute_events(track)
        ]
        for track in mid.tracks
    ]


def _layout_signature(file_type: int, ticks_per_beat: int,
                      layout: List[List[List]]) -> str:
    payload = json.dumps(
        {
            'type': file_type,
            'ticks_per_beat': ticks_per_beat,
            'tracks': layout,
        },
        sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def file_signature(mid: MidiFile) -> str:
    """Return the SHA256 semantic signature of a MidiFile."""
    return _layout_signature(mid.type, mid.ticks_per_beat, file_layout(mid))


def semantic_equal(left: MidiFile, right: MidiFile) -> bool:
    """Return True if two files contain the same semantic events."""
    if left.type != right.type or (
            left.ticks_per_beat != right.ticks_per_beat):
        return False
    if len(left.tracks) != len(right.tracks):
        return False
    return all(
        [(item.tick, item.ordinal, item.event.fingerprint)
         for item in absolute_events(left_track)]
        ==
        [(item.tick, item.ordinal, item.event.fingerprint)
         for item in absolute_events(right_track)]
        for left_track, right_track in zip(left.tracks, right.tracks)
    )


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _validate_pair(old: MidiFile, new: MidiFile) -> None:
    if not isinstance(old, MidiFile) or not isinstance(new, MidiFile):
        raise IncompatibleFilesError('diff_midi() requires two MidiFile '
                                     'objects')
    if old.type == 2 or new.type == 2:
        raise IncompatibleFilesError(
            'type 2 (asynchronous) MIDI files cannot be compared: '
            'their tracks do not share a tick timeline')
    if old.ticks_per_beat != new.ticks_per_beat:
        raise IncompatibleFilesError(
            f'ticks_per_beat differs '
            f'({old.ticks_per_beat} vs {new.ticks_per_beat}); '
            'rescale one file to a common resolution before diffing')
    if old.type != new.type:
        # Type 0 and 1 use the same event model, but a patch must
        # reproduce a specific header, so refuse rather than guess.
        raise IncompatibleFilesError(
            f'MIDI file type differs ({old.type} vs {new.type}); '
            'convert to a common type before diffing')


def diff_midi(old: MidiFile, new: MidiFile) -> MidiDiff:
    """Compute the semantic diff from ``old`` to ``new``.

    Neither input object is modified.  Raises
    :class:`IncompatibleFilesError` for type 2 files, mismatching
    ticks_per_beat or mismatching file types.
    """
    _validate_pair(old, new)

    source_layout = file_layout(old)
    target_layout = file_layout(new)

    track_diffs: List[TrackDiff] = []
    track_ops: List[TrackOp] = []

    shared = min(len(old.tracks), len(new.tracks))
    for index in range(shared):
        track_diff = _diff_track(index, old.tracks[index], new.tracks[index])
        if track_diff is not None:
            track_diffs.append(track_diff)

    if len(new.tracks) > len(old.tracks):
        for index in range(shared, len(new.tracks)):
            track_ops.append(TrackOp(
                kind=OP_ADD_TRACK,
                track=index,
                events=[PlacedEvent(item.tick, item.ordinal, item.event)
                        for item in absolute_events(new.tracks[index])],
            ))
    elif len(old.tracks) > len(new.tracks):
        for index in range(shared, len(old.tracks)):
            track_ops.append(TrackOp(
                kind=OP_REMOVE_TRACK,
                track=index,
                events=[PlacedEvent(item.tick, item.ordinal, item.event)
                        for item in absolute_events(old.tracks[index])],
            ))

    report = MidiDiff(
        source_type=old.type,
        target_type=new.type,
        ticks_per_beat=old.ticks_per_beat,
        source_layout=source_layout,
        target_layout=target_layout,
        track_diffs=track_diffs,
        track_ops=sorted(track_ops, key=lambda op: op.sort_key),
    )
    return report


def _diff_track(index: int, old_track, new_track) -> Optional[TrackDiff]:
    old_events = absolute_events(old_track)
    new_events = absolute_events(new_track)

    old_keys = [item.event.fingerprint for item in old_events]
    new_keys = [item.event.fingerprint for item in new_events]

    matcher = SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
    ops: List[DiffOp] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # An equal block only guarantees the same *sequence of
            # fingerprints*, not positional identity: when two equal
            # events swap order, pairing by position is what detects the
            # move (their tick/ordinal differ), while pairing purely by
            # content would silently leave them in the wrong slots.
            for offset in range(i2 - i1):
                old_item = old_events[i1 + offset]
                new_item = new_events[j1 + offset]
                if (old_item.tick != new_item.tick
                        or old_item.ordinal != new_item.ordinal):
                    ops.append(DiffOp(
                        kind=OP_CHANGE,
                        track=index,
                        tick=old_item.tick,
                        ordinal=old_item.ordinal,
                        event=old_item.event,
                        new_tick=new_item.tick,
                        new_ordinal=new_item.ordinal,
                        new_event=new_item.event,
                    ))
        elif tag == 'delete':
            for old_item in old_events[i1:i2]:
                ops.append(DiffOp(
                    kind=OP_REMOVE,
                    track=index,
                    tick=old_item.tick,
                    ordinal=old_item.ordinal,
                    event=old_item.event,
                ))
        elif tag == 'insert':
            for new_item in new_events[j1:j2]:
                ops.append(DiffOp(
                    kind=OP_ADD,
                    track=index,
                    tick=new_item.tick,
                    ordinal=new_item.ordinal,
                    event=new_item.event,
                ))
        else:  # replace: pair in order, leftovers are pure adds/removes
            old_slice = old_events[i1:i2]
            new_slice = new_events[j1:j2]
            paired = min(len(old_slice), len(new_slice))
            for old_item, new_item in zip(old_slice[:paired],
                                          new_slice[:paired]):
                ops.append(DiffOp(
                    kind=OP_CHANGE,
                    track=index,
                    tick=old_item.tick,
                    ordinal=old_item.ordinal,
                    event=old_item.event,
                    new_tick=new_item.tick,
                    new_ordinal=new_item.ordinal,
                    new_event=new_item.event,
                ))
            for old_item in old_slice[paired:]:
                ops.append(DiffOp(
                    kind=OP_REMOVE,
                    track=index,
                    tick=old_item.tick,
                    ordinal=old_item.ordinal,
                    event=old_item.event,
                ))
            for new_item in new_slice[paired:]:
                ops.append(DiffOp(
                    kind=OP_ADD,
                    track=index,
                    tick=new_item.tick,
                    ordinal=new_item.ordinal,
                    event=new_item.event,
                ))

    if not ops:
        return None
    return TrackDiff(track=index, ops=sorted(ops, key=lambda op: op.sort_key))


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

def apply_patch(mid: MidiFile, report: MidiDiff) -> MidiFile:
    """Apply a semantic report to a source file.

    A new :class:`MidiFile` is returned; the input object and its
    tracks and messages are never modified.  Raises
    :class:`IncompatibleFilesError` if the file header does not match
    the report and :class:`PatchConflictError` if any anchor cannot be
    resolved in the (possibly already modified) source, or if the
    rebuilt file does not match the report's target signature.
    """
    _validate_report_source(mid, report)

    conflicts: List[Conflict] = []

    # --- preflight: the source must still match the file the report
    #     was generated from.  Comparing the full layouts locates drift
    #     even in regions the report never references, and handles
    #     duplicate events correctly (order is part of the layout).
    source_conflicts = _layout_conflicts(
        actual=file_layout(mid), expected=report.source_layout,
        header='source file has changed since the report was generated')
    if source_conflicts:
        raise PatchConflictError(source_conflicts)

    result = MidiFile(
        type=report.target_type,
        ticks_per_beat=report.ticks_per_beat,
        # Deep copy: UnknownMetaMessage is mutable, so even a shared
        # message object could let a caller mutate the source through
        # the patched file.
        tracks=[MidiTrack(copy.deepcopy(track)) for track in mid.tracks],
    )

    # --- event-level operations, per shared track ----------------------
    # Every anchor is resolved against the *original* source track
    # first, so one bad anchor can never shift the ordinals used by the
    # remaining anchors.  The new timeline is only built once all
    # anchors are known to be valid.
    for track_diff in report.track_diffs:
        index = track_diff.track
        if index >= len(result.tracks):
            for op in track_diff.ops:
                conflicts.append(Conflict(
                    kind=CONFLICT_KIND_MISSING, track=index, tick=op.tick,
                    ordinal=op.ordinal,
                    message='track does not exist in source file'))
            continue
        source_groups = _group_by_tick(
            absolute_events(result.tracks[index]))
        new_groups = _rebuild_groups(track_diff.ops, source_groups, conflicts)
        if new_groups is not None:
            result.tracks[index] = relative_track(
                _flatten_groups(new_groups))

    if conflicts:
        raise PatchConflictError(conflicts)

    # --- whole-track structure, applied only after every anchor was
    #     resolved so that track indices still referred to the source
    #     layout while anchors were being checked.  Descending index
    #     order keeps earlier indices stable while mutating the list.
    for track_op in sorted(report.track_ops, key=lambda op: op.sort_key,
                           reverse=True):
        _apply_track_op(track_op, result, conflicts)

    if conflicts:
        raise PatchConflictError(conflicts)

    # --- final safety net: the result must be semantically identical
    #     to the file the report was generated from.  Layout comparison
    #     locates the first divergent event instead of only hashing.
    result_conflicts = _layout_conflicts(
        actual=file_layout(result), expected=report.target_layout,
        header='rebuilt file does not match the report target')
    if result_conflicts:
        raise PatchConflictError(result_conflicts)
    return result


def _layout_conflicts(actual: List[List[List]],
                      expected: List[List[List]],
                      header: str) -> List[Conflict]:
    """Locate every semantic difference between two file layouts."""
    conflicts = []
    if len(actual) != len(expected):
        conflicts.append(Conflict(
            kind=CONFLICT_KIND_MISSING,
            message=(f'{header}: track count differs '
                     f'({len(actual)} vs {len(expected)})')))
        return conflicts
    for track, (actual_track, expected_track) in enumerate(
            zip(actual, expected)):
        if len(actual_track) != len(expected_track):
            conflicts.append(Conflict(
                kind=CONFLICT_KIND_MISSING, track=track,
                message=(f'{header}: event count differs '
                         f'({len(actual_track)} vs {len(expected_track)})')))
            continue
        for actual_event, expected_event in zip(actual_track, expected_track):
            if actual_event != expected_event:
                tick, ordinal, fingerprint = expected_event
                conflicts.append(Conflict(
                    kind=CONFLICT_KIND_DRIFT, track=track, tick=tick,
                    ordinal=ordinal,
                    message=(f'{header}: expected event {fingerprint[:12]} '
                             f'but found {actual_event[2][:12]}')))
    return conflicts


def _validate_report_source(mid: MidiFile, report: MidiDiff) -> None:
    if not isinstance(mid, MidiFile):
        raise IncompatibleFilesError('apply_patch() requires a MidiFile')
    if not isinstance(report, MidiDiff):
        raise IncompatibleFilesError('apply_patch() requires a MidiDiff')
    if mid.type != report.source_type:
        raise IncompatibleFilesError(
            f'source file type {mid.type} does not match report '
            f'source type {report.source_type}')
    if mid.ticks_per_beat != report.ticks_per_beat:
        raise IncompatibleFilesError(
            f'source ticks_per_beat {mid.ticks_per_beat} does not match '
            f'report {report.ticks_per_beat}')


def _group_by_tick(events: List[_AbsEvent]) -> dict:
    groups = {}
    for item in events:
        groups.setdefault(item.tick, []).append(item)
    return groups


def _flatten_groups(groups: dict) -> List[_AbsEvent]:
    result = []
    for tick in sorted(groups):
        for ordinal, item in enumerate(groups[tick]):
            result.append(_AbsEvent(tick=tick, ordinal=ordinal,
                                    event=item.event))
    return result


def _resolve_anchor(op: DiffOp, groups: dict,
                    conflicts: List[Conflict]) -> Optional[_AbsEvent]:
    """Validate the (tick, ordinal, fingerprint) anchor of an op."""
    group = groups.get(op.tick)
    if not group or op.ordinal >= len(group):
        conflicts.append(Conflict(
            kind=CONFLICT_KIND_MISSING, track=op.track, tick=op.tick,
            ordinal=op.ordinal,
            message='anchor event is missing from the source file'))
        return None
    anchored = group[op.ordinal]
    if anchored.event.fingerprint != op.event.fingerprint:
        conflicts.append(Conflict(
            kind=CONFLICT_KIND_DRIFT, track=op.track, tick=op.tick,
            ordinal=op.ordinal,
            message=(
                'anchor event has changed: expected '
                f'{op.event.describe()}, found {anchored.event.describe()}')))
        return None
    return anchored


def _rebuild_groups(ops: List[DiffOp], source_groups: dict,
                    conflicts: List[Conflict]) -> Optional[dict]:
    """Validate every anchor against the source, then rebuild the groups.

    Returns ``None`` when conflicts were found (they are appended to
    ``conflicts``).  Source groups are copied, so the input timeline is
    never mutated.
    """
    # Pass 1: resolve all anchors against the untouched source.
    anchored_ops = []
    for op in ops:
        if op.kind == OP_ADD:
            anchored_ops.append((op, None))
            continue
        anchored = _resolve_anchor(op, source_groups, conflicts)
        if anchored is not None:
            anchored_ops.append((op, anchored))
    if conflicts:
        return None

    # Pass 2: start from a deep copy of the source timeline.
    new_groups = {
        tick: [_AbsEvent(tick=tick, ordinal=ordinal, event=item.event)
               for ordinal, item in enumerate(group)]
        for tick, group in source_groups.items()
    }

    # Removes and the removal half of changes first, descending by
    # ordinal within each tick so indices never shift underneath us.
    removals = [(op, anchored) for op, anchored in anchored_ops
                if op.kind in (OP_REMOVE, OP_CHANGE)]
    for op, _anchored in sorted(
            removals, key=lambda pair: (pair[0].tick, -pair[0].ordinal)):
        group = new_groups[op.tick]
        group.pop(op.ordinal)
        if not group:
            del new_groups[op.tick]

    # Pass 3: insert adds and the insertion half of changes at their
    # destination ordinals, ascending so later inserts don't shift
    # earlier destination slots.
    insertions = []
    for op, _ in anchored_ops:
        if op.kind == OP_ADD:
            insertions.append((op, op.tick, op.ordinal, op.event))
        elif op.kind == OP_CHANGE:
            insertions.append((op, op.new_tick, op.new_ordinal,
                               op.new_event))
    for op, tick, ordinal, event in sorted(
            insertions, key=lambda item: (item[1], item[2])):
        group = new_groups.setdefault(tick, [])
        if ordinal < 0 or ordinal > len(group):
            conflicts.append(Conflict(
                kind=CONFLICT_KIND_MISSING, track=op.track, tick=tick,
                ordinal=ordinal,
                message='destination ordinal is out of range'))
            continue
        group.insert(ordinal, _AbsEvent(
            tick=tick, ordinal=ordinal, event=event))

    if conflicts:
        return None
    return new_groups


def _apply_track_op(track_op: TrackOp, result: MidiFile,
                    conflicts: List[Conflict]) -> None:
    if track_op.kind == OP_REMOVE_TRACK:
        if track_op.track >= len(result.tracks):
            conflicts.append(Conflict(
                kind=CONFLICT_KIND_MISSING, track=track_op.track,
                message='track does not exist in source file'))
            return
        actual = absolute_events(result.tracks[track_op.track])
        expected = [(item.tick, item.ordinal, item.event.fingerprint)
                    for item in track_op.events]
        actual_keys = [(item.tick, item.ordinal, item.event.fingerprint)
                       for item in actual]
        if actual_keys != expected:
            conflicts.append(Conflict(
                kind=CONFLICT_KIND_DRIFT, track=track_op.track,
                message='track content has changed since the report '
                        'was generated'))
            return
        result.tracks.pop(track_op.track)
        return

    # OP_ADD_TRACK: ops are processed in descending index order, so an
    # insert at ``track`` lands in the exact slot stored in the report.
    if track_op.track > len(result.tracks):
        conflicts.append(Conflict(
            kind=CONFLICT_KIND_MISSING, track=track_op.track,
            message='track index is beyond the end of the source file'))
        return
    events = [
        _AbsEvent(tick=item.tick, ordinal=item.ordinal, event=item.event)
        for item in track_op.events
    ]
    result.tracks.insert(track_op.track, relative_track(events))
