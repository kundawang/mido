# SPDX-FileCopyrightText: 2016 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

from ..messages import Message
from .meta import MetaMessage


class MidiTrack(list):
    @property
    def name(self):
        """Name of the track.

        This will return the name from the first track_name meta
        message in the track, or '' if there is no such message.

        Setting this property will update the name field of the first
        track_name message in the track. If no such message is found,
        one will be added to the beginning of the track with a delta
        time of 0."""
        for message in self:
            if message.type == 'track_name':
                return message.name
        else:
            return ''

    @name.setter
    def name(self, name):
        # Find the first track_name message and modify it.
        for message in self:
            if message.type == 'track_name':
                message.name = name
                return
        else:
            # No track name found, add one.
            self.insert(0, MetaMessage('track_name', name=name, time=0))

    def copy(self):
        return self.__class__(self)

    def __getitem__(self, index_or_slice):
        # Retrieve item from the MidiTrack
        lst = list.__getitem__(self, index_or_slice)
        if isinstance(index_or_slice, int):
            # If an index was provided, return the list element
            return lst
        else:
            # Otherwise, construct a MidiTrack to return.
            # TODO: this make a copy of the list. Is there a better way?
            return self.__class__(lst)

    def __add__(self, other):
        return self.__class__(list.__add__(self, other))

    def __mul__(self, other):
        return self.__class__(list.__mul__(self, other))

    def __repr__(self):
        if len(self) == 0:
            messages = ''
        elif len(self) == 1:
            messages = f'[{self[0]}]'
        else:
            messages = '[\n  {}]'.format(',\n  '.join(repr(m) for m in self))
        return f'{self.__class__.__name__}({messages})'


def _to_abstime(messages):
    """Convert messages to absolute time."""
    now = 0
    for msg in messages:
        now += msg.time
        yield msg.copy(time=now)


def _to_reltime(messages):
    """Convert messages to relative time."""
    now = 0
    for msg in messages:
        delta = msg.time - now
        yield msg.copy(time=delta)
        now = msg.time


def fix_end_of_track(messages):
    """Remove all end_of_track messages and add one at the end.

    This is used by merge_tracks() and MidiFile.save()."""
    # Accumulated delta time from removed end of track messages.
    # This is added to the next message.
    accum = 0

    for msg in messages:
        if msg.type == 'end_of_track':
            accum += msg.time
        else:
            if accum:
                delta = accum + msg.time
                yield msg.copy(time=delta)
                accum = 0
            else:
                yield msg

    yield MetaMessage('end_of_track', time=accum)


def merge_tracks(tracks):
    """Returns a MidiTrack object with all messages from all tracks.

    The messages are returned in playback order with delta times
    as if they were all in one track.
    """
    messages = []
    for track in tracks:
        messages.extend(_to_abstime(track))

    messages.sort(key=lambda msg: msg.time)

    return MidiTrack(fix_end_of_track(_to_reltime(messages)))


def _is_note_on(msg):
    return msg.type == 'note_on' and msg.velocity > 0


def _is_note_off(msg):
    return msg.type == 'note_off' or (
        msg.type == 'note_on' and msg.velocity == 0)


# Channel mode messages (120..127) such as "all notes off" and
# "reset all controllers" are not part of the controller state that
# slice_track() replays at the start of a slice.
_CHANNEL_MODE_MESSAGES = range(120, 128)

# Bank select is emitted before program changes, so these are handled
# separately from the rest of the controller state.
_BANK_SELECT_CONTROLS = (0, 32)


def slice_track(track, start, end, note_policy='truncate'):
    """Return a new track with the half-open tick interval [start, end).

    The returned track contains all messages of `track` with absolute
    times in [start, end), rebased so the slice starts at time 0. The
    source track is not modified.

    To make the slice playable on its own, the state at `start` is
    replayed at time 0: the active tempo, and for each channel the
    bank select (cc 0 and 32), program change and the last value of
    every other controller (except channel mode messages 120..127).
    Notes that are sounding at `start` are re-articulated at time 0
    with their original velocity. (A note that is turned off or
    re-triggered exactly at `start` does not cross into the slice
    and is not re-articulated.)

    Notes that are still sounding at `end` are handled according to
    `note_policy`:

    - 'truncate' (default): a note_off is generated for them at the
      end of the slice.
    - 'keep': their original note_off is kept even if it falls at or
      after `end`, so the track may extend past the slice. A note_off
      is still generated at the end of the slice for notes that have
      none in the source track.

    Messages at exactly `start` are included and come after the
    replayed state; messages at exactly `end` are excluded (except
    note_offs kept by the 'keep' policy).
    """
    if note_policy not in ('truncate', 'keep'):
        raise ValueError(
            f"note_policy must be 'truncate' or 'keep', "
            f'not {note_policy!r}')
    if not 0 <= start < end:
        raise ValueError(
            f'require 0 <= start < end, got start={start}, end={end}')

    tempo = None
    programs = {}          # channel -> program
    controls = {}          # (channel, control) -> value
    sounding = {}          # (channel, note) -> note_on before start
    head_notes = None      # snapshot of sounding at the slice start
    events = []            # messages inside the slice (absolute time)
    open_notes = {}        # (channel, note) -> note_on inside the slice
    tail_offs = []         # note_offs at/after end kept by 'keep'

    for msg in _to_abstime(track):
        time = msg.time

        if msg.type == 'end_of_track':
            continue

        if time < start:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'program_change':
                programs[msg.channel] = msg.program
            elif msg.type == 'control_change':
                if msg.control not in _CHANNEL_MODE_MESSAGES:
                    controls[(msg.channel, msg.control)] = msg.value
            elif _is_note_on(msg):
                sounding[(msg.channel, msg.note)] = msg
            elif _is_note_off(msg):
                sounding.pop((msg.channel, msg.note), None)
        elif time == start:
            # Boundary events are inside the slice, but note_ons and
            # note_offs at exactly the start also decide which notes
            # cross into the slice: a note that is turned off (or
            # re-triggered) exactly at the start is not re-articulated.
            if _is_note_on(msg):
                key = (msg.channel, msg.note)
                sounding.pop(key, None)
                open_notes[key] = msg
            elif _is_note_off(msg):
                sounding.pop((msg.channel, msg.note), None)
            events.append(msg)
        else:
            if head_notes is None:
                # Notes that are sounding when the slice starts are
                # re-articulated at time 0.
                head_notes = dict(sounding)

            if time < end:
                events.append(msg)
                if _is_note_on(msg):
                    key = (msg.channel, msg.note)
                    open_notes[key] = msg
                    # The note is re-articulated inside the slice, so
                    # the carried-over note_on no longer counts as open.
                    sounding.pop(key, None)
                elif _is_note_off(msg):
                    key = (msg.channel, msg.note)
                    open_notes.pop(key, None)
                    sounding.pop(key, None)
            else:
                # At or after the end of the slice. With 'keep' the
                # original note_offs of open notes are kept; with
                # 'truncate' they are ignored and the notes are closed
                # by a generated note_off at the end of the slice.
                if note_policy == 'keep' and _is_note_off(msg):
                    key = (msg.channel, msg.note)
                    if key in open_notes or key in sounding:
                        tail_offs.append(msg)
                        open_notes.pop(key, None)
                        sounding.pop(key, None)

    if head_notes is None:
        # The slice starts at or after the last message.
        head_notes = dict(sounding)

    out = []

    # Replay the state at the start of the slice.
    if tempo is not None:
        out.append(MetaMessage('set_tempo', tempo=tempo, time=0))

    channels = set(programs)
    channels.update(channel for channel, _ in controls)
    for channel in sorted(channels):
        for control in _BANK_SELECT_CONTROLS:
            if (channel, control) in controls:
                out.append(Message('control_change', channel=channel,
                                   control=control,
                                   value=controls[(channel, control)],
                                   time=0))
        if channel in programs:
            out.append(Message('program_change', channel=channel,
                               program=programs[channel], time=0))
        for chan, control in sorted(controls):
            if (chan == channel
                    and control not in _BANK_SELECT_CONTROLS):
                out.append(Message('control_change', channel=channel,
                                   control=control,
                                   value=controls[(chan, control)],
                                   time=0))

    # Re-articulate notes that are sounding at the start.
    for channel, note in sorted(head_notes):
        out.append(head_notes[(channel, note)].copy(time=0))

    # Messages inside the slice, rebased to the slice start.
    out.extend(msg.copy(time=msg.time - start) for msg in events)

    # Notes still sounding at the end of the slice: (channel, note)
    # pairs from both before and inside the slice.
    still_sounding = sorted(set(open_notes) | set(sounding))
    if note_policy == 'keep':
        kept = {(msg.channel, msg.note) for msg in tail_offs}
    else:
        kept = set()

    # Close notes that have no note_off in the source (or all of them
    # when truncating) exactly at the end of the slice.
    out.extend(Message('note_off', channel=channel, note=note,
                       time=end - start)
               for channel, note in still_sounding
               if (channel, note) not in kept)

    if note_policy == 'keep':
        # Keep the original note_offs, possibly past the slice end.
        # These come last since they are all at or after `end`.
        out.extend(msg.copy(time=msg.time - start) for msg in tail_offs)

    # The end_of_track shares the absolute time of the last message
    # so that it gets a delta time of 0.
    out.append(MetaMessage('end_of_track',
                           time=out[-1].time if out else 0))

    return MidiTrack(_to_reltime(out))
