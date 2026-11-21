# SPDX-FileCopyrightText: 2026 Mido contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the optional bounded output queue in mido.ports.BaseOutput.

All tests use a controllable slow port so they do not need real MIDI
hardware and can measure throughput and ordering deterministically.
"""

import queue
import threading
import time

import pytest

from mido import Message
from mido.ports import BaseOutput, QueueAborted


class SlowPort(BaseOutput):
    """Output port whose backend write can be slowed down on demand."""

    def _open(self, delay=0.0):
        self.delay = delay
        self.written = []
        self.write_lock = threading.Lock()
        self.close_calls = 0
        self.close_from_send = False

    def _send(self, message):
        if self.delay:
            time.sleep(self.delay)
        with self.write_lock:
            self.written.append(message)

    def _close(self):
        self.close_calls += 1

    def _worker_busy(self):
        """Return True while the worker is inside _send()."""
        state = self._output_state
        if state is None:
            return False
        with state.cond:
            return state.in_flight == 1


class RecordingPort(BaseOutput):
    """Instant port that records the order of writes."""

    def _open(self):
        self.written = []
        self.write_threads = []

    def _send(self, message):
        self.written.append((message.note, message.velocity))
        self.write_threads.append(threading.get_ident())


def note(n, velocity=64):
    return Message('note_on', note=n, velocity=velocity)


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


class TestDefaults:
    def test_default_send_is_synchronous(self):
        """Without queue options send() still writes on the caller thread."""
        with RecordingPort() as port:
            assert not port.is_queued()
            assert port.queue_size is None
            assert port.pending == 0
            assert port.send(note(1)) is None

            assert [n for n, _ in port.written] == [1]
            assert port.write_threads == [threading.get_ident()]

    def test_default_returns_none(self):
        with RecordingPort() as port:
            assert port.send(note(1)) is None

    def test_queue_options_not_passed_to_backend(self):
        # _open() accepts no queue keyword arguments; if BaseOutput
        # leaked them through, constructing the port would fail.
        with SlowPort(queue_size=4) as port:
            assert port.is_queued()

    def test_zero_queue_size_is_rendezvous(self):
        with RecordingPort(queue_size=0) as port:
            assert port.is_queued()
            assert port.queue_size == 0
            assert port.send(note(1)) == 1

    def test_invalid_options(self):
        with pytest.raises(ValueError):
            SlowPort(queue_size=-1)
        with pytest.raises(ValueError):
            SlowPort(queue_size=1.5)
        with pytest.raises(ValueError):
            SlowPort(queue_size=1, queue_policy='sometimes')
        with pytest.raises(ValueError):
            SlowPort(queue_size=1, queue_timeout=-0.5)
        with pytest.raises(ValueError):
            SlowPort(queue_size=1, max_queue_size=2)
        with pytest.raises(ValueError):
            SlowPort(queue_size=1, close_timeout=-1)

    def test_send_validation_is_unchanged(self):
        with RecordingPort(queue_size=4) as port:
            with pytest.raises(TypeError):
                port.send('not a message')

        port = RecordingPort(queue_size=4)
        port.close()
        with pytest.raises(ValueError):
            port.send(note(1))


class TestSerialization:
    def test_messages_are_copied_before_send(self):
        port = RecordingPort(queue_size=4)
        message = note(60)
        port.send(message)

        # The worker has the copy; changing the original must not
        # affect what is written.
        message.note = 99
        port.flush()

        assert [n for n, _ in port.written] == [60]
        port.close()

    def test_fifo_order_single_thread(self):
        with RecordingPort(queue_size=8) as port:
            for i in range(20):
                port.send(note(i))

            port.flush()

            assert [n for n, _ in port.written] == list(range(20))

    def test_all_writes_happen_on_one_worker_thread(self):
        with RecordingPort(queue_size=8) as port:
            for i in range(50):
                port.send(note(i))
            port.flush()

            assert len(set(port.write_threads)) == 1
            assert port.write_threads[0] != threading.get_ident()

    def test_multi_thread_order_is_total_and_complete(self):
        """Concurrent senders produce every message exactly once."""
        port = RecordingPort(queue_size=16)
        thread_count = 4
        per_thread = 20

        def sender(thread_id):
            for i in range(per_thread):
                port.send(note(thread_id * 20 + i))

        threads = [threading.Thread(target=sender, args=(t,))
                   for t in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        port.flush()

        written = [n for n, _ in port.written]
        assert len(written) == thread_count * per_thread
        assert sorted(written) == sorted(
            t * 20 + i
            for t in range(thread_count)
            for i in range(per_thread))

        # Each sender's messages keep their relative order, so the
        # total order is a valid interleaving.
        for t in range(thread_count):
            own = [n for n in written if n // 20 == t]
            assert own == [t * 20 + i for i in range(per_thread)]

        port.close()

    def test_send_sequence_numbers_are_increasing(self):
        with RecordingPort(queue_size=4) as port:
            first = port.send(note(1))
            second = port.send(note(2))
            assert second == first + 1

            third = port.send(note(3))
            assert third == second + 1


class TestPolicies:
    def test_block_policy_waits_for_room(self):
        port = SlowPort(delay=0.1, queue_size=1)
        port.send(note(0))
        # Wait until the worker has taken the first message, then fill
        # the single queue slot.
        assert wait_until(lambda: port.sent_count == 0 and
                          port.last_error is None and port._worker_busy())
        port.send(note(1))

        start = time.monotonic()
        port.send(note(2))  # must wait for the worker
        elapsed = time.monotonic() - start

        assert elapsed >= 0.01
        port.flush()
        assert [m.note for m in port.written] == [0, 1, 2]
        port.close()

    def test_reject_policy_raises_full(self):
        port = SlowPort(delay=0.2, queue_size=1, queue_policy='reject')
        port.send(note(0))
        assert wait_until(lambda: port._worker_busy())
        port.send(note(1))

        with pytest.raises(queue.Full):
            port.send(note(2))

        port.close(timeout=0)

    def test_timeout_policy(self):
        port = SlowPort(delay=5.0, queue_size=1, queue_policy='timeout',
                        queue_timeout=0.1)
        port.send(note(0))
        assert wait_until(lambda: port._worker_busy())
        port.send(note(1))

        start = time.monotonic()
        with pytest.raises(queue.Full):
            port.send(note(2))
        elapsed = time.monotonic() - start

        assert 0.05 <= elapsed < 1.0
        port.close(timeout=0)

    def test_timeout_policy_succeeds_within_timeout(self):
        port = SlowPort(delay=0.1, queue_size=1, queue_policy='timeout',
                        queue_timeout=5.0)
        port.send(note(0))
        assert wait_until(lambda: port._worker_busy())
        port.send(note(1))

        # Room frees up when the worker takes the first message.
        port.send(note(2))
        port.flush()
        assert [m.note for m in port.written] == [0, 1, 2]
        port.close()


class TestPanicAndReset:
    def test_panic_discards_pending_and_jumps_queue(self):
        port = SlowPort(delay=0.02, queue_size=100)
        for i in range(100):
            port.send(note(i))

        port.panic()
        port.flush()

        notes = [m.note for m in port.written
                 if m.type == 'note_on']
        panic_controls = [m for m in port.written
                          if m.type == 'control_change']
        assert len(panic_controls) == 16
        assert all(m.control == 120 for m in panic_controls)

        # Everything already queued was dropped; only the first
        # message (possibly in flight) survives.
        assert len(notes) <= 1
        assert port.dropped_count >= 99
        port.close()

    def test_panic_is_not_dropped(self):
        port = SlowPort(delay=0.02, queue_size=2)
        for i in range(50):
            port.send(note(i))

        port.panic()
        port.flush()

        controls = [m.control for m in port.written
                    if m.type == 'control_change']
        assert controls == [120] * 16
        port.close()

    def test_reset_is_prioritised_but_keeps_normal_messages(self):
        port = SlowPort(delay=0.02, queue_size=100)
        for i in range(5):
            port.send(note(i))

        port.reset()
        port.flush()

        types = [(m.type, getattr(m, 'control', None))
                 for m in port.written]
        # The first note may already be in flight; after that the
        # reset messages come before the remaining notes.
        first_reset = next(i for i, (t, _) in enumerate(types)
                           if t == 'control_change')
        trailing_controls = [t for t in types[first_reset:]
                             if t[0] == 'control_change']
        remaining_notes = [i for i, (t, _) in enumerate(types)
                           if t == 'note_on' and i > first_reset]

        assert len(trailing_controls) == 32
        # Reset messages are all ahead of the notes still queued...
        assert types[first_reset:first_reset + 32] == trailing_controls
        # ...but the notes are kept, not dropped.
        assert len(remaining_notes) + first_reset == 5
        assert port.dropped_count == 0
        port.close()

    def test_panic_discards_pending_reset(self):
        """panic() is an emergency stop: it drops queued reset too."""
        port = SlowPort(delay=0.02, queue_size=100)
        for i in range(5):
            port.send(note(i))

        port.reset()
        port.panic()
        port.flush()

        controls = [m.control for m in port.written
                    if m.type == 'control_change']
        assert controls == [120] * 16
        port.close()

    def test_reset_after_panic_survives(self):
        port = SlowPort(delay=0.02, queue_size=100)
        for i in range(5):
            port.send(note(i))

        port.panic()
        port.reset()
        port.flush()

        controls = [m.control for m in port.written
                    if m.type == 'control_change']
        assert controls == [120] * 16 + [123, 121] * 16
        port.close()

    def test_panic_aborts_blocked_normal_send(self):
        port = SlowPort(delay=5.0, queue_size=1)
        port.send(note(0))
        assert wait_until(lambda: port._worker_busy())
        port.send(note(1))  # fills the queue

        result = {}

        def sender():
            try:
                port.send(note(2))  # blocks on the full queue
            except QueueAborted as error:
                result['error'] = error

        thread = threading.Thread(target=sender)
        thread.start()
        time.sleep(0.05)
        assert thread.is_alive()

        port.panic()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert isinstance(result.get('error'), QueueAborted)
        port.close(timeout=0)

    def test_panic_aborts_blocked_rendezvous_send(self):
        port = SlowPort(delay=5.0, queue_size=0)
        state = port._output_state

        first = threading.Thread(target=port.send, args=(note(0),))
        first.start()
        # Let the worker take the first message and start its slow
        # write. The handoff slot is then free again.
        assert wait_until(lambda: port._worker_busy())

        result = {}

        def sender():
            try:
                port.send(note(1))
            except QueueAborted as error:
                result['error'] = error

        second = threading.Thread(target=sender)
        second.start()
        assert wait_until(lambda: state.handoff is not None)

        port.panic()
        second.join(timeout=2)

        assert not second.is_alive()
        assert isinstance(result.get('error'), QueueAborted)
        port.close(timeout=0)
        first.join(timeout=1)


class TestFlush:
    def test_flush_without_queue_is_noop(self):
        with RecordingPort() as port:
            port.flush()

    def test_flush_waits_for_writes(self):
        port = SlowPort(delay=0.01, queue_size=10)
        for i in range(10):
            port.send(note(i))

        port.flush()
        assert len(port.written) == 10
        assert port.pending == 0
        port.close()

    def test_flush_timeout(self):
        port = SlowPort(delay=5.0, queue_size=4)
        port.send(note(1))
        time.sleep(0.02)

        with pytest.raises(TimeoutError):
            port.flush(timeout=0.05)

        port.close(timeout=0)

    def test_flush_propagates_background_error(self):
        class FailingPort(RecordingPort):
            def _send(self, message):
                if message.note == 5:
                    raise OSError('device gone')
                super()._send(message)

        port = FailingPort(queue_size=8)
        for i in range(10):
            try:
                port.send(note(i))
            except OSError:
                # Later send() calls surface the same failure; the
                # test only needs the queue primed.
                break

        with pytest.raises(OSError, match='device gone'):
            port.flush()

        assert isinstance(port.last_error, OSError)
        port.close()


class TestErrorPropagation:
    def test_send_after_failure_raises(self):
        class FailingPort(RecordingPort):
            def _send(self, message):
                raise OSError('broken')

        port = FailingPort(queue_size=4)
        port.send(note(1))

        assert wait_until(lambda: port.last_error is not None)

        with pytest.raises(OSError, match='broken'):
            port.send(note(2))

        port.close()

    def test_pending_messages_are_dropped_on_failure(self):
        class FailingPort(SlowPort):
            def _send(self, message):
                if message.note == 0:
                    raise OSError('boom')
                super()._send(message)

        port = FailingPort(delay=0.01, queue_size=10)
        for i in range(10):
            port.send(note(i))

        assert wait_until(lambda: port.last_error is not None)
        # Nine messages were still queued and are counted as dropped.
        assert port.dropped_count == 9
        assert port.pending == 0
        port.close()


class TestClose:
    def test_close_drains_by_default(self):
        port = SlowPort(delay=0.005, queue_size=16)
        for i in range(10):
            port.send(note(i))

        port.close()

        assert port.closed
        assert [m.note for m in port.written] == list(range(10))
        assert port.close_calls == 1

    def test_close_timeout_discards_and_closes(self):
        port = SlowPort(delay=5.0, queue_size=8)
        port.send(note(0))
        time.sleep(0.02)
        for i in range(1, 5):
            port.send(note(i))

        start = time.monotonic()
        port.close(timeout=0.05)
        elapsed = time.monotonic() - start

        assert port.closed
        assert elapsed < 0.5
        # The worker was inside _send(); it closes the backend when it
        # returns rather than closing under the write.
        assert port.close_calls == 0

        # Clean up the slow worker without waiting five seconds.
        port.delay = 0
        assert wait_until(lambda: port.close_calls == 1, timeout=6.0)

    def test_close_is_idempotent(self):
        port = SlowPort(queue_size=4)
        port.send(note(1))
        port.close()
        port.close()
        assert port.close_calls == 1

    def test_close_does_not_deadlock_while_send_blocks(self):
        port = SlowPort(delay=0.01, queue_size=2)

        def sender():
            for i in range(40):
                try:
                    port.send(note(i))
                except ValueError:
                    return

        thread = threading.Thread(target=sender)
        thread.start()
        time.sleep(0.01)
        port.close(timeout=0.5)
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert port.closed

    def test_send_from_close_does_not_deadlock(self):
        class ClosingPort(RecordingPort):
            def _send(self, message):
                super()._send(message)
                if message.note == 0:
                    self.close()

        port = ClosingPort(queue_size=4)
        port.send(note(0))
        assert wait_until(lambda: port.closed)
        assert port.written

    def test_autoreset_failure_does_not_prevent_close(self):
        class FailingResetPort(SlowPort):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.reset_raises = True

            def reset(self):
                if getattr(self, 'reset_raises', False):
                    raise OSError('reset failed')
                return super().reset()

        port = FailingResetPort(queue_size=4, autoreset=True)
        port.send(note(1))
        port.close()

        assert port.closed
        assert port.close_calls == 1

    def test_autoreset_is_queued_before_close(self):
        port = SlowPort(queue_size=8, autoreset=True)
        port.send(note(1))
        port.close()

        controls = [m.control for m in port.written
                    if m.type == 'control_change']
        assert controls == [123, 121] * 16
        assert [m.note for m in port.written
                if m.type == 'note_on'] == [1]


class TestReentrancy:
    def test_send_from_inside_send_does_not_deadlock(self):
        class NestedPort(RecordingPort):
            def _send(self, message):
                super()._send(message)
                if message.note < 2:
                    self.send(note(message.note + 1))

        port = NestedPort(queue_size=1)
        port.send(note(0))
        assert wait_until(lambda: len(port.written) == 3)
        assert [n for n, _ in port.written] == [0, 1, 2]
        port.close()

    def test_flush_from_inside_send_raises(self):
        errors = []

        class FlushingPort(RecordingPort):
            def _send(self, message):
                super()._send(message)
                try:
                    self.flush()
                except RuntimeError as error:
                    errors.append(error)

        port = FlushingPort(queue_size=2)
        port.send(note(0))
        assert wait_until(lambda: len(port.written) == 1)
        assert errors
        port.close()


class TestSlowPortThroughput:
    def test_async_send_decouples_caller_from_device(self):
        """A slow device no longer blocks each send() call."""
        delay = 0.02
        count = 10
        port = SlowPort(delay=delay, queue_size=count)

        start = time.monotonic()
        for i in range(count):
            port.send(note(i))
        enqueue_time = time.monotonic() - start

        # Enqueueing everything must be far faster than writing it.
        assert enqueue_time < delay * count / 2
        assert port.pending >= 1

        port.flush()
        total = time.monotonic() - start

        assert len(port.written) == count
        # One slow write per message is the floor.
        assert total >= delay * (count - 1)
        port.close()

    def test_queue_peak_is_bounded(self):
        capacity = 4
        port = SlowPort(delay=0.005, queue_size=capacity)
        for i in range(200):
            port.send(note(i % 128))

        port.flush()
        assert port.peak_pending <= capacity
        port.close()

    def test_sustained_slow_port_rejects_do_not_grow_memory(self):
        capacity = 3
        port = SlowPort(delay=0.05, queue_size=capacity,
                        queue_policy='reject')
        accepted = 0
        rejected = 0
        for i in range(1000):
            try:
                port.send(note(i % 128))
                accepted += 1
            except queue.Full:
                rejected += 1

        assert rejected > 0
        assert accepted + rejected == 1000
        assert port.peak_pending <= capacity
        assert port.pending <= capacity
        port.close(timeout=0)

    def test_zero_capacity_never_buffers(self):
        port = SlowPort(delay=0.001, queue_size=0)
        for i in range(20):
            port.send(note(i))
            # A blocking send only returns after the handoff was taken.
            assert port.pending <= 1

        port.flush()
        assert [m.note for m in port.written] == list(range(20))
        port.close()

    def test_measured_throughput_and_queue_peak(self):
        """Quantify the decoupling with a deterministic slow device."""
        delay = 0.01
        capacity = 40
        count = 80
        port = SlowPort(delay=delay, queue_size=capacity)

        start = time.monotonic()
        for i in range(count):
            port.send(note(i % 128))
        enqueue = time.monotonic() - start

        port.flush()
        total = time.monotonic() - start

        device_time = delay * count
        # The caller is not charged the device time per message: the
        # enqueue loop only waits when the queue is full.
        assert enqueue < device_time / 2
        # Total time is still bounded by the device, as it must be.
        assert total >= delay * (count - 1)
        # Queue occupancy never exceeds the configured capacity.
        assert port.peak_pending <= capacity
        assert port.sent_count == count
        # Effective device throughput is close to 1/delay.
        assert count / total <= 1 / delay * 1.5

        port.close()

    def test_repeated_reset_does_not_grow_queue(self):
        port = SlowPort(delay=0.05, queue_size=100)
        port.send(note(0))
        port.reset()
        after_first = port.pending

        for _ in range(50):
            port.reset()

        # Only the most recent reset request is kept; pending must not
        # grow with the number of reset() calls.
        assert after_first <= 2 + 32  # normal message + one reset batch
        assert port.pending <= after_first
        port.close(timeout=0)


class TestGarbageCollection:
    def test_dropping_port_stops_worker(self):
        import gc
        import weakref

        port = SlowPort(delay=0.01, queue_size=4)
        for i in range(4):
            port.send(note(i))

        ref = weakref.ref(port)
        del port
        gc.collect()

        assert wait_until(lambda: ref() is None, timeout=2.0)
