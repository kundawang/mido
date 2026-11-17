# SPDX-FileCopyrightText: 2017 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

from numbers import Integral

# In an SMPTE (MIDI Time Code) MIDI file the upper byte of the header
# division word is one of these negative frame rate codes and the lower
# byte holds the number of ticks per frame.
#
# 29.97 frames per second is the nominal NTSC drop-frame rate (exactly
# 30000 / 1001).
SMPTE_FPS_BY_CODE = {
    -24: 24,
    -25: 25,
    -29: 29.97,
    -30: 30,
}

# The inverse mapping, for example 24 -> -24.
SMPTE_CODE_BY_FPS = {fps: code for code, fps in SMPTE_FPS_BY_CODE.items()}

# Frame rates accepted when creating an SMPTE MIDI file.
SMPTE_FPS = tuple(SMPTE_FPS_BY_CODE.values())


def _check_smpte_resolution(fps, ticks_per_frame):
    if fps not in SMPTE_FPS_BY_CODE.values():
        raise ValueError(
            f'invalid SMPTE frame rate {fps!r} '
            f'(must be one of {", ".join(str(value) for value in SMPTE_FPS)})')
    if not isinstance(ticks_per_frame, Integral):
        raise TypeError('ticks_per_frame must be an integer')
    if ticks_per_frame <= 0:
        raise ValueError('ticks_per_frame must be a positive integer')
    if ticks_per_frame > 0xFF:
        raise OverflowError(
            f'ticks_per_frame {ticks_per_frame} does not fit in one byte')


def smpte_seconds_per_tick(fps, ticks_per_frame):
    """Return the duration in seconds of one tick in an SMPTE MIDI file.

    ``fps`` is one of the SMPTE frame rates (24, 25, 29.97 or 30) and
    ``ticks_per_frame`` is the number of delta-time ticks counted in each
    frame. Tempo events do not affect SMPTE timing.
    """
    _check_smpte_resolution(fps, ticks_per_frame)
    return 1.0 / (fps * ticks_per_frame)


def smpte_tick2second(tick, fps, ticks_per_frame):
    """Convert SMPTE delta time in ticks to seconds.

    Unlike :func:`tick2second` the tempo is irrelevant here: time is
    derived from the frame rate and the number of ticks per frame.
    """
    if not isinstance(tick, Integral):
        raise TypeError('tick must be an integer')
    if tick < 0:
        raise ValueError('tick must be non-negative')
    return tick * smpte_seconds_per_tick(fps, ticks_per_frame)


def smpte_second2tick(second, fps, ticks_per_frame):
    """Convert seconds to SMPTE delta time in ticks.

    This is the inverse of :func:`smpte_tick2second`. Normal rounding
    applies.
    """
    if second < 0:
        raise ValueError('second must be non-negative')
    _check_smpte_resolution(fps, ticks_per_frame)
    return int(round(second * fps * ticks_per_frame))


def tick2second(tick, ticks_per_beat, tempo):
    """Convert absolute time in ticks to seconds.

    Returns absolute time in seconds for a chosen MIDI file time resolution
    (ticks/pulses per quarter note, also called PPQN) and tempo (microseconds
    per quarter note).
    """
    scale = tempo * 1e-6 / ticks_per_beat
    return tick * scale


def second2tick(second, ticks_per_beat, tempo):
    """Convert absolute time in seconds to ticks.

    Returns absolute time in ticks for a chosen MIDI file time resolution
    (ticks/pulses per quarter note, also called PPQN) and tempo (microseconds
    per quarter note). Normal rounding applies.
    """
    scale = tempo * 1e-6 / ticks_per_beat
    return int(round(second / scale))


def bpm2tempo(bpm, time_signature=(4, 4)):
    """Convert BPM (beats per minute) to MIDI file tempo (microseconds per
    quarter note).

    Depending on the chosen time signature a bar contains a different number of
    beats. These beats are multiples/fractions of a quarter note, thus the
    returned BPM depend on the time signature. Normal rounding applies.
    """
    return int(round(60 * 1e6 / bpm * time_signature[1] / 4.))


def tempo2bpm(tempo, time_signature=(4, 4)):
    """Convert MIDI file tempo (microseconds per quarter note) to BPM (beats
    per minute).

    Depending on the chosen time signature a bar contains a different number of
    beats. The beats are multiples/fractions of a quarter note, thus the
    returned tempo depends on the time signature denominator.
    """
    return 60 * 1e6 / tempo * time_signature[1] / 4.
