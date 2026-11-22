# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Simplification of control change (automation) curves in MIDI files.

Dense automation recording often writes dozens of nearly identical
control_change messages per beat.  This module removes the redundant
ones while keeping the curve within a given value and time tolerance,
so the file shrinks and the synth hears (almost) the same thing.

The main entry points are:

* :func:`simplify_control_changes` -- simplify one track.
* :func:`simplify_midi_file` -- simplify every track of a ``MidiFile``
  and return a new ``MidiFile``.  The source file/object is never
  modified.

Strategies for the corner cases (all covered by the test suite):

* Single point curves (and two point curves) are always kept as is.
* Several events for the same controller at the same tick collapse to
  the last value, which is the value a synth would end up with.  The
  surviving event keeps the position of the last of those events, so
  the order of all other messages is untouched.
* Tolerances must be finite numbers >= 0.  Anything else (negative,
  NaN, infinity, non-numbers) raises ``ValueError``.  A tolerance of 0
  means "no error allowed on this axis", so only exactly redundant
  points are removed.
* Bank select (controllers 0 and 32) is not a continuous curve --
  removing or shifting it changes program change semantics -- so it is
  skipped by default via ``skip_controls``.  Users commonly extend
  this with data entry / RPN / NRPN controllers (6, 38, 98..101).
* Abrupt (nonlinear) jumps are preserved automatically: any jump
  larger than ``value_error`` sticks out of the tolerance band and
  anchors the simplified curve on both sides.
* The relative order of all non-simplified messages, of events at the
  same tick, and of other tracks is preserved.  Only control_change
  messages belonging to a simplified curve are removed (or, in 14 bit
  mode, replaced by re-synthesized MSB/LSB pairs at the same tick).

The simplification is a greedy sequential pass with a strict
guarantee: a point is only removed if its value can be reconstructed
from the final simplified curve (by linear interpolation, allowing a
time shift of up to ``time_error`` ticks) with an error of at most
``value_error``.
"""

import math

from ..messages import Message
from .midifiles import MidiFile
from .tracks import MidiTrack

#: Controllers skipped by default: bank select MSB and LSB.
DEFAULT_SKIP_CONTROLS = (0, 32)


def _validate_tolerance(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number, got {value!r}')
    if math.isnan(value) or math.isinf(value) or value < 0:
        raise ValueError(f'{name} must be a finite number >= 0, got {value!r}')


def _validate_skip_controls(skip_controls):
    controls = frozenset(skip_controls)
    for control in controls:
        if isinstance(control, bool) or not isinstance(control, int):
            raise ValueError(f'skip_controls entries must be ints, got {control!r}')
        if not 0 <= control <= 127:
            raise ValueError(
                f'skip_controls entries must be in 0..127, got {control!r}')
    return controls


def _point_error(t, v, seg, time_error):
    """Minimum reconstruction error of point (t, v) against a segment.

    The value at ``t`` may be reconstructed by linear interpolation
    anywhere within ``time_error`` ticks of ``t`` (clamped to the
    segment).  Returns the smallest possible absolute value error.
    """
    (t0, v0), (t1, v1) = seg
    lo = max(t - time_error, t0)
    hi = min(t + time_error, t1)

    def interp(tc):
        return v0 + (v1 - v0) * (tc - t0) / (t1 - t0)

    flo = interp(lo) - v
    fhi = interp(hi) - v
    if flo == 0 or (flo < 0) != (fhi < 0):
        # The interpolation hits v somewhere inside the window.
        return 0.0
    return min(abs(flo), abs(fhi))


def _simplify_points(points, value_error, time_error):
    """Greedy sequential simplification of (tick, value) points.

    Returns the set of point indices to keep.  A point is removed only
    if it is reconstructable from the segment between its kept
    neighbours within (value_error, time_error), so every removed
    point is within tolerance of the final curve.  Endpoints are
    always kept, and any jump larger than the tolerances automatically
    anchors the curve.
    """
    n = len(points)
    if n <= 2:
        return set(range(n))

    keep = [0]
    anchor = 0
    i = 2
    while i < n:
        seg = (points[anchor], points[i])
        for j in range(anchor + 1, i):
            t, v = points[j]
            if _point_error(t, v, seg, time_error) > value_error:
                # Segment anchor..i is not representable; keep the last
                # point that still worked and restart from it.
                keep.append(i - 1)
                anchor = i - 1
                i = anchor + 2
                break
        else:
            i += 1
    keep.append(n - 1)
    return set(keep)


def _dedupe_ticks(events):
    """Collapse events at the same tick to the last one.

    ``events`` is a list of (index, tick, value) in track order.
    Returns (points, reps): points is a list of (tick, value) with
    strictly increasing ticks, reps[i] is the track index of the event
    that represents points[i].
    """
    points = []
    reps = []
    for index, tick, value in events:
        if points and points[-1][0] == tick:
            points[-1] = (tick, value)
            reps[-1] = index
        else:
            points.append((tick, value))
            reps.append(index)
    return points, reps


def _abs_times(track):
    times = []
    now = 0
    for msg in track:
        now += msg.time
        times.append(now)
    return times


def _rebuild(track, times, drop, inserts):
    """Build a new track without the dropped messages.

    ``inserts`` maps a sort key (tick, order) to a list of messages to
    insert at that absolute tick.  Order keys place inserted messages
    exactly where the removed curve events were, so the relative order
    of everything else (including same-tick events) is preserved.
    """
    entries = []
    for i, msg in enumerate(track):
        if i in drop:
            continue
        entries.append((times[i], (i, 0), msg))
    for (tick, order), msgs in inserts.items():
        for j, msg in enumerate(msgs):
            entries.append((tick, (order, 1 + j), msg))
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    new_track = MidiTrack()
    prev = 0
    for tick, _, msg in entries:
        new_track.append(msg.copy(time=tick - prev))
        prev = tick
    return new_track


def _simplify_7bit(track, times, value_error, time_error, skip_controls):
    curves = {}
    for i, msg in enumerate(track):
        if (msg.type == 'control_change'
                and msg.control not in skip_controls):
            key = (msg.channel, msg.control)
            curves.setdefault(key, []).append((i, times[i], msg.value))

    drop = set()
    for events in curves.values():
        points, reps = _dedupe_ticks(events)
        keep = _simplify_points(points, value_error, time_error)
        kept_indices = {reps[i] for i in keep}
        for index, _, _ in events:
            if index not in kept_indices:
                drop.add(index)

    return _rebuild(track, times, drop, {})


def _simplify_14bit(track, times, value_error, time_error, skip_controls):
    # Group controllers 0..31 with their LSB counterparts at +32.
    # A pair is skipped if either half is in skip_controls.
    curves = {}
    for i, msg in enumerate(track):
        if msg.type != 'control_change':
            continue
        control = msg.control
        if control < 32:
            msb_control = control
            is_lsb = False
        elif control < 64:
            msb_control = control - 32
            is_lsb = True
        else:
            continue
        if msb_control in skip_controls or msb_control + 32 in skip_controls:
            continue
        key = (msg.channel, msb_control)
        curves.setdefault(key, []).append((i, times[i], is_lsb, msg.value))

    drop = set()
    inserts = {}
    for (channel, msb_control), events in curves.items():
        # Combine MSB/LSB into a running 14 bit value.  A component
        # that has not been seen yet is assumed 0.
        msb_value = 0
        lsb_value = 0
        has_msb = False
        has_lsb = False
        combined = []
        for index, tick, is_lsb, value in events:
            if is_lsb:
                lsb_value = value
                has_lsb = True
            else:
                msb_value = value
                has_msb = True
            combined.append((index, tick, (msb_value << 7) | lsb_value))

        points, _ = _dedupe_ticks(combined)
        keep = _simplify_points(points, value_error, time_error)

        # All original events of the curve are removed; every kept
        # point is re-synthesized as an MSB/LSB pair at the same tick,
        # anchored where the curve's first event at that tick was.
        for index, _, _, _ in events:
            drop.add(index)

        first_index_at_tick = {}
        for index, tick, _, _ in events:
            first_index_at_tick.setdefault(tick, index)

        for i in sorted(keep):
            tick, value14 = points[i]
            msgs = []
            if has_msb:
                msgs.append(Message('control_change', channel=channel,
                                    control=msb_control,
                                    value=(value14 >> 7) & 0x7f, time=0))
            if has_lsb:
                msgs.append(Message('control_change', channel=channel,
                                    control=msb_control + 32,
                                    value=value14 & 0x7f, time=0))
            inserts.setdefault((tick, first_index_at_tick[tick]), []).extend(msgs)

    return _rebuild(track, times, drop, inserts)


def simplify_control_changes(track, value_error=1.0, time_error=0.0,
                             msb=False,
                             skip_controls=DEFAULT_SKIP_CONTROLS):
    """Return a new track with redundant control changes removed.

    The input track is not modified.

    ``value_error`` is the maximum allowed value deviation (in 7 bit
    units, or 14 bit units when ``msb=True``).  ``time_error`` is the
    maximum allowed deviation in ticks.  Both must be finite numbers
    >= 0; 0 means no error is allowed on that axis.

    If ``msb`` is true, controller pairs 0..31 and 32..63 on the same
    channel are treated as single 14 bit curves and re-emitted as
    MSB/LSB pairs.

    ``skip_controls`` lists controllers that are never touched.  It
    defaults to bank select (0 and 32); in 14 bit mode a pair is
    skipped if either half is listed.
    """
    _validate_tolerance('value_error', value_error)
    _validate_tolerance('time_error', time_error)
    skip_controls = _validate_skip_controls(skip_controls)

    times = _abs_times(track)
    if msb:
        return _simplify_14bit(track, times, value_error, time_error,
                               skip_controls)
    else:
        return _simplify_7bit(track, times, value_error, time_error,
                              skip_controls)


def simplify_midi_file(mid, value_error=1.0, time_error=0.0,
                       msb=False,
                       skip_controls=DEFAULT_SKIP_CONTROLS):
    """Return a new MidiFile with simplified control change curves.

    Every track is simplified with :func:`simplify_control_changes`.
    The source MidiFile (and the file it was loaded from) is not
    modified; the new file has ``filename=None`` so it cannot
    accidentally overwrite the source on ``save()``.
    """
    new = MidiFile(type=mid.type,
                   ticks_per_beat=mid.ticks_per_beat,
                   charset=mid.charset)
    for track in mid.tracks:
        new.tracks.append(simplify_control_changes(
            track, value_error=value_error, time_error=time_error,
            msb=msb, skip_controls=skip_controls))
    return new
