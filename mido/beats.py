# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Real-time beat tracking from MIDI timing clock messages.

The MIDI protocol uses 24 timing clock pulses per quarter note
(see the MIDI 1.0 specification). A device sends one ``clock``
message for every pulse while it is playing, plus transport messages
(``start``, ``continue``, ``stop``) and ``songpos`` (song position
pointer) messages, whose unit is 16th notes -- that is, six clock
pulses per unit.

A clock pulse can be lost or delayed.  The :class:`BeatTracker` keeps
the phase (pulse and beat position) deterministic and free of
double-stepping by counting every received clock exactly once.  Gaps
between clocks are classified into three zones:

* a *normal* gap, inside the configurable jitter window,
* a *missed pulse* gap, where one or more pulses are assumed lost, and
* a *re-anchor* gap, too large to be explained by missed pulses,
  which is treated as a tempo change or a transport pause.

Tempo is estimated from the observed gaps and smoothed with an
exponential moving average so that the reported BPM does not jump
around on every jittery pulse.

The tracker never mutates the messages it receives.
"""
from collections import namedtuple
from time import monotonic

from .messages import Message

#: Number of MIDI timing clock pulses in a quarter note (one beat).
PULSES_PER_BEAT = 24

#: Number of clock pulses per ``songpos`` unit (a 16th note).
PULSES_PER_SONGPOS_UNIT = 6

#: Default half-width of the jitter window as a fraction of the
#: expected gap.  A gap between ``1 - jitter_window`` and
#: ``1 + jitter_window`` times the expected gap is considered normal.
DEFAULT_JITTER_WINDOW = 0.1

#: Default number of consecutive out-of-window gaps required before
#: the expected gap is re-anchored (a tempo change is assumed).
DEFAULT_REANCHOR_STREAK = 3

#: Default smoothing factor for the exponential moving average used
#: to smooth the BPM.  Lower values are smoother but react slower.
DEFAULT_SMOOTHING = 0.2

#: Default timeout in seconds after which, with no clock received,
#: the tempo is considered stale.
DEFAULT_PULSE_TIMEOUT = 2.0


class BeatSnapshot(namedtuple('BeatSnapshot',
                              ('running', 'pulses', 'beat', 'beat_phase',
                               'songpos_units', 'bpm', 'smoothed_bpm',
                               'stale', 'dropped_pulses'))):
    """Immutable state snapshot returned by :meth:`BeatTracker.update`.

    Attributes:
        running: True after a ``start`` or ``continue`` message and
            False after a ``stop`` message.
        pulses: Total number of clock pulses accumulated since the
            tracker was created (or the last :meth:`reset`).
        beat: Beat position as a float, measured in quarter notes.
        beat_phase: Phase inside the current beat in the range
            ``0..PULSES_PER_BEAT - 1`` (clock pulses).
        songpos_units: Equivalent song position pointer position in
            16th-note units (six pulses each), derived from the
            pulse count and any ``songpos`` messages received.
        bpm: Instantaneous tempo estimate in beats per minute, or
            ``None`` until two clocks have been observed.
        smoothed_bpm: Exponentially smoothed tempo estimate, or
            ``None`` until two clocks have been observed.
        stale: True when no clock has been received for longer than
            the configured pulse timeout.
        dropped_pulses: Number of pulses currently assumed to have
            been lost since the previous clock gap.
    """
    __slots__ = ()


class BeatTracker:
    """Track beat position and tempo from MIDI real-time messages.

    Messages are passed to :meth:`update` together with a receive
    timestamp from a monotonic clock.  The clock can be injected for
    testing and replay by passing a callable to the ``clock``
    argument; it defaults to :func:`time.monotonic`.

    Args:
        clock: Zero-argument callable returning monotonically
            increasing seconds, like :func:`time.monotonic`.
        jitter_window: Fractional half-width of the normal-gap
            window (default 0.1, i.e. +/- 10%).
        max_missed_pulses: Maximum number of pulses a single gap may
            be assumed to have lost before the gap is treated as a
            transport pause or tempo change (default 4).
        reanchor_streak: Number of consecutive gaps outside the
            jitter window required to re-anchor the expected gap
            (default 3).
        smoothing: Smoothing factor (alpha) for the exponential
            moving average of the BPM, between 0 and 1 (default 0.2).
        pulse_timeout: Seconds without a clock after which the tempo
            estimate is reported as stale (default 2.0).

    Example:
        >>> tracker = BeatTracker()
        >>> for message in input_port.iter_pending():
        ...     snapshot = tracker.update(message)
        ...     print(snapshot.beat, snapshot.smoothed_bpm)
    """

    def __init__(self,
                 clock=monotonic,
                 jitter_window=DEFAULT_JITTER_WINDOW,
                 max_missed_pulses=4,
                 reanchor_streak=DEFAULT_REANCHOR_STREAK,
                 smoothing=DEFAULT_SMOOTHING,
                 pulse_timeout=DEFAULT_PULSE_TIMEOUT):
        if not callable(clock):
            raise TypeError('clock must be a callable returning seconds')
        if not 0.0 <= jitter_window < 1.0:
            raise ValueError('jitter_window must be in the range [0, 1)')
        if max_missed_pulses < 1:
            raise ValueError('max_missed_pulses must be at least 1')
        if reanchor_streak < 1:
            raise ValueError('reanchor_streak must be at least 1')
        if not 0.0 < smoothing <= 1.0:
            raise ValueError('smoothing must be in the range (0, 1]')
        if pulse_timeout <= 0.0:
            raise ValueError('pulse_timeout must be positive')

        self._clock = clock
        self.jitter_window = jitter_window
        self.max_missed_pulses = max_missed_pulses
        self.reanchor_streak = reanchor_streak
        self.smoothing = smoothing
        self.pulse_timeout = pulse_timeout

        self.reset()

    def reset(self):
        """Reset all transport, position and tempo state."""
        # Transport state.
        self.running = False
        self.pulses = 0

        # Timing state.
        self._last_clock_time = None
        self._expected_gap = None
        self._bpm = None
        self._smoothed_bpm = None

        # Gap classification state.
        self._fast_streak = 0
        self._slow_streak = 0
        self._dropped_pulses = 0

    @property
    def beat(self):
        """Beat position in quarter notes (24 pulses per beat)."""
        return self.pulses / PULSES_PER_BEAT

    @property
    def beat_phase(self):
        """Pulse position inside the current beat (0..23)."""
        return self.pulses % PULSES_PER_BEAT

    @property
    def songpos_units(self):
        """Position in 16th-note units (six pulses per unit).

        This follows the accumulated pulse count, including positions
        set through ``songpos`` messages and ``start``.
        """
        return self.pulses // PULSES_PER_SONGPOS_UNIT

    @property
    def bpm(self):
        """Instantaneous tempo estimate in BPM, or None."""
        return self._bpm

    @property
    def smoothed_bpm(self):
        """Smoothed tempo estimate in BPM, or None."""
        return self._smoothed_bpm

    @property
    def stale(self):
        """True when no clock arrived within the pulse timeout."""
        if self._last_clock_time is None:
            return True
        return self._clock() - self._last_clock_time > self.pulse_timeout

    def update(self, message, timestamp=None):
        """Process one MIDI message and return a :class:`BeatSnapshot`.

        Only ``clock``, ``start``, ``continue``, ``stop`` and
        ``songpos`` messages affect the state; every other message is
        ignored but still produces a snapshot.  The message is never
        modified.

        Args:
            message: A :class:`mido.Message`.
            timestamp: Receive time in seconds from a monotonic
                clock.  When omitted, the injected clock is called.

        Returns:
            A :class:`BeatSnapshot` describing the state after the
            message was processed.
        """
        if not isinstance(message, Message):
            raise TypeError('message must be a mido.Message instance')

        if timestamp is None:
            timestamp = self._clock()

        if message.type == 'clock':
            self._handle_clock(timestamp)
        elif message.type == 'start':
            self._handle_start()
        elif message.type == 'continue':
            self.running = True
        elif message.type == 'stop':
            self.running = False
        elif message.type == 'songpos':
            self._handle_songpos(message.pos)
        # Any other message type is intentionally ignored.

        return self.snapshot()

    def snapshot(self):
        """Return the current state as an immutable :class:`BeatSnapshot`."""
        return BeatSnapshot(
            running=self.running,
            pulses=self.pulses,
            beat=self.beat,
            beat_phase=self.beat_phase,
            songpos_units=self.songpos_units,
            bpm=self._bpm,
            smoothed_bpm=self._smoothed_bpm,
            stale=self.stale,
            dropped_pulses=self._dropped_pulses,
        )

    def _handle_start(self):
        # 'start' rewinds to the beginning of the song.  The pulse
        # counter is reset, but stop/continue do not clear it.
        self.running = True
        self.pulses = 0
        self._fast_streak = 0
        self._slow_streak = 0
        self._dropped_pulses = 0
        self._reset_timing()

    def _handle_songpos(self, pos):
        # The song position pointer is measured in 16th notes.
        self.pulses = pos * PULSES_PER_SONGPOS_UNIT
        self._fast_streak = 0
        self._slow_streak = 0
        self._dropped_pulses = 0
        self._reset_timing()

    def _reset_timing(self):
        """Forget timing history but keep the last BPM as a hint."""
        self._last_clock_time = None
        self._expected_gap = None
        # _bpm and _smoothed_bpm are retained so tempo can be
        # reported immediately after continue.  They are rebuilt from
        # observed gaps as soon as new clocks arrive.

    def _handle_clock(self, timestamp):
        self._dropped_pulses = 0

        if self._last_clock_time is None:
            # First clock (or the first one after a transport event):
            # anchor the timing but do not measure a gap yet.
            self._last_clock_time = timestamp
            if self.running:
                self.pulses += 1
            return

        gap = timestamp - self._last_clock_time

        if gap <= 0.0:
            # Time went backwards or the timestamp is a duplicate.
            # Ignore the clock entirely: it must not advance the
            # phase and must not corrupt the tempo estimate.
            return

        if gap > self.pulse_timeout:
            # A gap this long means the device paused (or the pulse
            # stream genuinely stopped).  Re-anchor without feeding the
            # huge gap to the tempo estimator, so a stale BPM is kept
            # (reported as stale until fresh clocks arrive).
            self._expected_gap = None
            self._fast_streak = 0
            self._slow_streak = 0
            self._last_clock_time = timestamp
            if self.running:
                self.pulses += 1
            return

        if self._expected_gap is None:
            # Second clock ever (or first after a transport event):
            # establishes the first tempo estimate.
            self._reanchor(gap)
        else:
            self._classify_gap(gap)

        self._last_clock_time = timestamp
        if self.running:
            self.pulses += 1

    def _classify_gap(self, gap):
        ratio = gap / self._expected_gap
        low = 1.0 - self.jitter_window
        high = 1.0 + self.jitter_window

        if low <= ratio <= high:
            # Normal, jittery gap.
            self._fast_streak = 0
            self._slow_streak = 0
            self._update_tempo(gap)
        elif ratio < low:
            # Clocks arrived closer together than expected.  A single
            # short gap is treated as jitter; only a run of them is a
            # real tempo increase.
            self._fast_streak += 1
            self._slow_streak = 0
            if self._fast_streak >= self.reanchor_streak:
                self._reanchor(gap)
            else:
                self._update_tempo(gap)
        else:
            # Long gap.  Check whether it can be explained by missed
            # pulses (an integer multiple of the expected gap).
            missed = self._classify_slow_gap(ratio)
            if missed is not None:
                # The phase is not advanced by the missing pulses:
                # every received clock increments the pulse counter by
                # exactly one, so a single lost pulse can never make
                # the beat counter jump by two beats.  The missing
                # time is simply excluded from the tempo estimate.
                self._dropped_pulses = missed
                self._fast_streak = 0
                self._slow_streak = 0
                self._update_tempo(gap / (1 + missed))
            else:
                # Either too many pulses were lost to be plausible, or
                # gaps have been growing for a while (tempo decrease).
                self._fast_streak = 0
                self._slow_streak += 1
                if self._slow_streak >= self.reanchor_streak:
                    self._reanchor(gap)
                else:
                    self._update_tempo(gap)

    def _classify_slow_gap(self, ratio):
        """Return the plausible missed-pulse count for a long gap.

        Returns None when the gap does not look like an integer
        multiple of the expected gap within the jitter window, or
        when more than ``max_missed_pulses`` would be required.
        """
        for missed in range(1, self.max_missed_pulses + 1):
            multiple = 1 + missed
            if abs(ratio - multiple) <= self.jitter_window:
                return missed
        return None

    def _reanchor(self, gap):
        """Adopt a new expected gap and reset streak counters."""
        self._expected_gap = gap
        self._fast_streak = 0
        self._slow_streak = 0
        self._update_tempo(gap)

    def _update_tempo(self, estimated_gap):
        if estimated_gap <= 0.0:
            return
        bpm = 60.0 / (estimated_gap * PULSES_PER_BEAT)
        self._bpm = bpm
        if self._smoothed_bpm is None:
            self._smoothed_bpm = bpm
        else:
            alpha = self.smoothing
            self._smoothed_bpm = (
                alpha * bpm + (1.0 - alpha) * self._smoothed_bpm
            )
        if self._expected_gap is None:
            self._expected_gap = estimated_gap
