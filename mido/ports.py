# SPDX-FileCopyrightText: 2013 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Useful tools for working with ports
"""
import queue
import random
import threading
import time
import weakref
from collections import deque

from .messages import Message
from .parser import Parser

# How many seconds to sleep before polling again.
_DEFAULT_SLEEP_TIME = 0.001
_sleep_time = _DEFAULT_SLEEP_TIME

# Default amount of time close() waits for queued output to drain.
_DEFAULT_CLOSE_TIMEOUT = 1.0

# Keyword arguments consumed by BaseOutput instead of being passed on to
# the backend's _open().
_QUEUE_OPTION_NAMES = ('queue_size', 'max_queue_size', 'queue_policy',
                       'queue_timeout', 'close_timeout')

_QUEUE_POLICIES = {
    'block': 'block',
    'blocking': 'block',
    'wait': 'block',
    'timeout': 'timeout',
    'reject': 'reject',
    'fail': 'reject',
}

_QUEUE_POLICY_NAMES = tuple(sorted(set(_QUEUE_POLICIES.values())))


class QueueAborted(RuntimeError):
    """Raised when :meth:`BaseOutput.panic` aborts a blocked send.

    A thread that is waiting for room in a full output queue when
    ``panic()`` is called has not had its message queued yet.  Rather
    than allow it to enqueue a stale lighting message after the panic,
    the send is aborted with this exception.
    """


def extract_queue_options(kwargs):
    """Remove and return output queue options from a keyword dict.

    This is used by the I/O port wrapper so that queue options are only
    given to the output side of the port.  The dictionary is modified
    in place (the caller normally owns it).
    """
    options = {}
    for name in _QUEUE_OPTION_NAMES:
        if name in kwargs:
            options[name] = kwargs.pop(name)
    return options


# TODO: document this more.
def sleep():
    """Sleep for N seconds.

    This is used in ports when polling and waiting for messages. N can
    be set with set_sleep_time()."""
    time.sleep(_sleep_time)


def set_sleep_time(seconds=_DEFAULT_SLEEP_TIME):
    """Set the number of seconds sleep() will sleep."""
    global _sleep_time
    _sleep_time = seconds


def get_sleep_time():
    """Get number of seconds sleep() will sleep."""
    return _sleep_time


def reset_messages():
    """Yield "All Notes Off" and "Reset All Controllers" for all channels"""
    ALL_NOTES_OFF = 123
    RESET_ALL_CONTROLLERS = 121
    for channel in range(16):
        for control in [ALL_NOTES_OFF, RESET_ALL_CONTROLLERS]:
            yield Message('control_change', channel=channel, control=control)


def panic_messages():
    """Yield "All Sounds Off" for all channels.

    This will mute all sounding notes regardless of
    envelopes. Useful when notes are hanging and nothing else
    helps.
    """
    ALL_SOUNDS_OFF = 120
    for channel in range(16):
        yield Message('control_change',
                      channel=channel, control=ALL_SOUNDS_OFF)


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class BasePort:
    """
    Abstract base class for Input and Output ports.
    """
    is_input = False
    is_output = False
    _locking = True

    def __init__(self, name=None, **kwargs):
        if hasattr(self, 'closed'):
            # __init__() called twice (from BaseInput and BaseOutput).
            # This stops _open() from being called twice.
            return

        self.name = name
        self._close_timeout = _DEFAULT_CLOSE_TIMEOUT
        self.autoreset = False
        # Output queue options are consumed by BaseOutput. Strip them
        # here so they are never passed on to a backend's _open().
        # Input ports do not support a queue, so reject the options
        # instead of silently ignoring them.
        self._queue_options = extract_queue_options(kwargs)
        if self._locking:
            self._lock = threading.RLock()
        else:
            self._lock = DummyLock()
        self.closed = True
        if self._queue_options and not self.is_output:
            raise ValueError(
                'output queue options are not supported by input ports')
        self._open(**kwargs)
        self.closed = False

    def _open(self, **kwargs):
        pass

    def _close(self):
        pass

    def close(self):
        """Close the port.

        If the port is already closed, nothing will happen.  The port
        is automatically closed when the object goes out of scope or
        is garbage collected.
        """
        with self._lock:
            if not self.closed:
                if hasattr(self, 'autoreset') and self.autoreset:
                    try:
                        self.reset()
                    except OSError:
                        pass

                self._close()
                self.closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        return False

    def __repr__(self):
        if self.closed:
            state = 'closed'
        else:
            state = 'open'

        capabilities = self.is_input, self.is_output
        port_type = {(True, False): 'input',
                     (False, True): 'output',
                     (True, True): 'I/O port',
                     (False, False): 'mute port',
                     }[capabilities]

        name = self.name or ''

        try:
            device_type = self._device_type
        except AttributeError:
            device_type = self.__class__.__name__

        return '<{} {} {!r} ({})>'.format(
            state, port_type, name, device_type)


class BaseInput(BasePort):
    """Base class for input port.

    Subclass and override _receive() to create a new input port type.
    (See portmidi.py for an example of how to do this.)
    """
    is_input = True

    def __init__(self, name='', **kwargs):
        """Create an input port.

        name is the port name, as returned by input_names(). If
        name is not passed, the default input is used instead.
        """
        BasePort.__init__(self, name, **kwargs)
        self._parser = Parser()
        self._messages = self._parser.messages  # Shortcut.

    def _check_callback(self):
        if hasattr(self, 'callback') and self.callback is not None:
            raise ValueError('a callback is set for this port')

    def _receive(self, block=True):
        pass

    def iter_pending(self):
        """Iterate through pending messages."""
        while True:
            msg = self.poll()
            if msg is None:
                return
            else:
                yield msg

    def receive(self, block=True):
        """Return the next message.

        This will block until a message arrives.

        If you pass block=False it will not block and instead return
        None if there is no available message.

        If the port is closed and there are no pending messages IOError
        will be raised. If the port closes while waiting inside receive(),
        IOError will be raised. TODO: this seems a bit inconsistent. Should
        different errors be raised? What's most useful here?
        """
        if not self.is_input:
            raise ValueError('Not an input port')

        self._check_callback()

        # If there is a message pending, return it right away.
        with self._lock:
            if self._messages:
                return self._messages.popleft()

        if self.closed:
            if block:
                raise ValueError('receive() called on closed port')
            else:
                return None

        while True:
            with self._lock:
                msg = self._receive(block=block)
                if msg:
                    return msg

                if self._messages:
                    return self._messages.popleft()
                elif not block:
                    return None
                elif self.closed:
                    raise OSError('port closed during receive()')

            sleep()

    def poll(self):
        """Receive the next pending message or None

        This is the same as calling `receive(block=False)`."""
        return self.receive(block=False)

    def __iter__(self):
        """Iterate through messages until the port closes."""
        # This could have simply called receive() in a loop, but that
        # could result in a "port closed during receive()" error which
        # is hard to catch here.
        self._check_callback()
        while True:
            try:
                yield self.receive()
            except OSError:
                if self.closed:
                    # The port closed before or inside receive().
                    # (This makes the assumption that this is the reason,
                    # at the risk of masking other errors.)
                    return
                else:
                    raise


class _QueuedMessage:
    """One message waiting to be written by the output worker thread."""
    __slots__ = ('message', 'seq', 'accepted', 'aborted')

    def __init__(self, message, seq):
        self.message = message
        self.seq = seq
        # Set by the worker once it has taken the message. Only used
        # for zero-capacity (rendezvous) queues.
        self.accepted = False
        # Set when panic()/close() discards a handoff that a sender is
        # still waiting on.
        self.aborted = False


class _OutputQueueState:
    """Shared state for an output queue and its worker thread.

    This is kept separate from the port so that the worker thread does
    not hold a strong reference to the port (which would prevent the
    port from being garbage collected and closed).  The worker looks the
    port up through a weak reference only while it is actually writing.
    """

    def __init__(self, capacity, policy, queue_timeout):
        self.cond = threading.Condition()
        self.capacity = capacity
        self.policy = policy
        self.queue_timeout = queue_timeout
        self.queue = deque()
        self.reset_queue = deque()
        self.panic_queue = deque()
        self.handoff = None
        self.pending = 0
        self.in_flight = 0
        self.accepting = True
        self.stop_requested = False
        self.closing = False
        self.worker = None
        self.worker_ident = None
        self.worker_done = False
        self.backend_close_deferred = False
        self.backend_close_done = False
        self.deferred_port = None
        self.send_error = None
        self.close_error = None
        self.seq = 0
        self.sent_count = 0
        self.peak_pending = 0
        self.dropped_count = 0
        # Incremented by panic(). A sender blocked waiting for room
        # that notices a change aborts instead of enqueuing a stale
        # message after the emergency stop.
        self.abort_generation = 0

    def count_pending(self):
        self.pending += 1
        if self.pending > self.peak_pending:
            self.peak_pending = self.pending

    def has_work(self):
        return bool(self.panic_queue or self.reset_queue or
                    self.handoff is not None or self.queue)

    def take_next(self):
        """Take the next message. Must hold the cond lock."""
        if self.panic_queue:
            item = self.panic_queue.popleft()
        elif self.reset_queue:
            item = self.reset_queue.popleft()
        elif self.handoff is not None:
            item = self.handoff
            self.handoff = None
            item.accepted = True
        elif self.queue:
            item = self.queue.popleft()
        else:
            return None

        self.pending -= 1
        self.in_flight = 1
        self.cond.notify_all()
        return item

    def discard_pending(self):
        """Discard waiting messages. Must hold the cond lock.

        Returns the number of discarded messages.
        """
        dropped = (len(self.queue) + len(self.reset_queue) +
                   len(self.panic_queue))
        self.queue.clear()
        self.reset_queue.clear()
        self.panic_queue.clear()

        if self.handoff is not None:
            self.handoff.aborted = True
            self.handoff = None
            dropped += 1

        self.pending = 0
        self.dropped_count += dropped
        self.cond.notify_all()
        return dropped


def _output_worker_entry(port_ref, state):
    """Body of the background output thread.

    port_ref is a weak reference to the owning port; state holds the
    queues.  The worker never keeps the port alive on its own.
    """
    cond = state.cond
    state.worker_ident = threading.get_ident()
    deferred_close = False

    try:
        while True:
            with cond:
                while not state.has_work() and not state.stop_requested:
                    cond.wait()

                item = state.take_next()
                if item is None:
                    # Stop was requested and there is nothing left to
                    # drain.
                    break

            port = port_ref()
            if port is None:
                # The port was garbage collected. Anything queued in an
                # in-flight send was already taken by the worker, so
                # only waiting messages are dropped.
                with cond:
                    state.in_flight = 0
                    state.dropped_count += 1  # the item just taken
                    state.discard_pending()
                break

            try:
                port._send(item.message)
            except BaseException as error:
                # Drop the strong reference before waiting again so a
                # discarded port can be garbage collected.
                port = None
                with cond:
                    state.send_error = error
                    state.in_flight = 0
                    state.discard_pending()
                    state.accepting = False
                    cond.notify_all()
                break

            # Do not keep a strong reference to the port while idle,
            # otherwise the worker thread would prevent it from being
            # collected after the user drops it.
            port = None

            with cond:
                state.sent_count += 1
                state.in_flight = 0
                cond.notify_all()

                if state.stop_requested and not state.has_work():
                    break
    finally:
        with cond:
            state.worker_done = True
            state.worker_ident = None
            deferred_close = (state.backend_close_deferred and
                              not state.backend_close_done)
            if deferred_close:
                state.backend_close_done = True
            cond.notify_all()

        if deferred_close:
            # Prefer the port handed over by close() so it cannot be
            # garbage collected in the gap before this lookup.
            port = state.deferred_port
            state.deferred_port = None
            if port is None:
                port = port_ref()
            if port is not None:
                try:
                    port._close()
                except BaseException as error:
                    with cond:
                        state.close_error = error


class BaseOutput(BasePort):
    """
    Base class for output port.

    Subclass and override _send() to create a new port type.  (See
    portmidi.py for how to do this.)

    An output port is synchronous by default: ``send()`` calls
    ``_send()`` and does not return until the backend has written the
    message.  Passing ``queue_size`` (or ``max_queue_size``) to the
    constructor enables an optional bounded output queue and a
    background worker thread::

        port = mido.open_output('Integra-7', queue_size=256)

    In queued mode ``send()`` validates and copies the message and
    enqueues it.  The worker writes messages to the underlying port in
    FIFO order.  If several threads call ``send()``, the order is the
    order in which they acquired the queue lock; ``send()`` returns
    that message's sequence number (an increasing integer) so callers
    can reconstruct it.  In synchronous mode ``send()`` returns None,
    as before.

    Queue options:

    ``queue_size`` / ``max_queue_size``
        Maximum number of normal messages waiting to be written.  Must
        be a non-negative integer.  ``0`` means a rendezvous port: no
        message is buffered; a blocking send returns only once the
        worker has taken the message.  ``None`` (the default) disables
        queueing and keeps the original synchronous behaviour.

    ``queue_policy``
        What ``send()`` does when the queue is full:

        ``'block'`` (default)
            Wait until there is room, the port is closed, or the
            worker fails.  Blocking sends are also aborted by
            :meth:`panic` (raising :class:`QueueAborted`).
        ``'timeout'``
            Wait up to ``queue_timeout`` seconds, then raise
            :class:`queue.Full`.
        ``'reject'``
            Raise :class:`queue.Full` immediately.

    ``queue_timeout``
        Seconds to wait with ``queue_policy='timeout'``.  Defaults to
        ``0.0``.  Not used by the other policies.

    ``close_timeout``
        Seconds :meth:`close` waits for queued messages to drain.
        ``None`` (the default) waits for the queue to drain.  If the
        timeout expires, pending messages are discarded; if the worker
        is still inside a backend ``_send()`` the underlying port is
        closed by the worker as soon as that call returns, so the
        device is never used concurrently.

    ``reset()`` and ``panic()`` are always enqueued with priority and
    never wait for room:

    * ``panic()`` discards pending normal and reset messages and puts
      its own messages at the very front.  A message already inside
      ``_send()`` cannot be preempted.
    * ``reset()`` puts its messages ahead of normal messages but keeps
      them.  Panic messages take precedence over reset messages.

    A failed background ``_send()`` is recorded.  :meth:`flush` raises
    it, :attr:`last_error` returns it, and later ``send()`` calls also
    raise it.  Messages still in the queue when the failure happens are
    discarded and counted in :attr:`dropped_count`.
    """
    is_output = True

    # Class-level defaults so close()/send() work on partially
    # constructed objects (for example when _open() raised).
    _queue_enabled = False
    _output_state = None

    def __init__(self, name='', autoreset=False, **kwargs):
        """Create an output port

        name is the port name, as returned by output_names(). If
        name is not passed, the default output is used instead.
        """
        BasePort.__init__(self, name, **kwargs)
        self._setup_output_queue(self._queue_options)
        self.autoreset = autoreset

    def _setup_output_queue(self, kwargs):
        """Read queue options and initialize the internal state.

        The options have already been removed from the backend's
        keyword arguments by BasePort.
        """
        queue_size = kwargs.pop('queue_size', None)
        max_queue_size = kwargs.pop('max_queue_size', None)

        if queue_size is None:
            queue_size = max_queue_size
        elif max_queue_size is not None and max_queue_size != queue_size:
            raise ValueError(
                'queue_size and max_queue_size must be the same')

        policy = kwargs.pop('queue_policy', 'block')
        if not isinstance(policy, str):
            raise ValueError('queue_policy must be a string')

        try:
            policy = _QUEUE_POLICIES[policy.lower()]
        except KeyError:
            raise ValueError(
                f'unknown queue_policy {policy!r} '
                f'(expected one of {list(_QUEUE_POLICY_NAMES)})')

        queue_timeout = kwargs.pop('queue_timeout', None)
        if queue_timeout is None:
            queue_timeout = 0.0
        if not isinstance(queue_timeout, (int, float)):
            raise ValueError('queue_timeout must be a number')
        if queue_timeout < 0:
            raise ValueError('queue_timeout must not be negative')

        close_timeout = kwargs.pop('close_timeout', _DEFAULT_CLOSE_TIMEOUT)
        if close_timeout is not None:
            if not isinstance(close_timeout, (int, float)):
                raise ValueError('close_timeout must be a number or None')
            if close_timeout < 0:
                raise ValueError('close_timeout must not be negative')

        self._close_timeout = (None if close_timeout is None
                               else float(close_timeout))

        self._queue_enabled = False
        self._output_state = None

        if queue_size is None:
            return

        if isinstance(queue_size, bool) or not isinstance(queue_size, int):
            raise ValueError('queue_size must be a non-negative integer')
        if queue_size < 0:
            raise ValueError('queue_size must be a non-negative integer')

        self._queue_enabled = True
        self._output_state = _OutputQueueState(
            capacity=queue_size,
            policy=policy,
            queue_timeout=float(queue_timeout))

    def _send(self, msg):
        pass

    def is_queued(self):
        """Return True if this port uses a background output queue."""
        return self._queue_enabled

    @property
    def queue_size(self):
        """The configured queue capacity, or None if not queued."""
        if not self._queue_enabled:
            return None

        return self._output_state.capacity

    @property
    def pending(self):
        """Number of messages waiting to be written."""
        if not self._queue_enabled:
            return 0

        with self._output_state.cond:
            return self._output_state.pending

    @property
    def peak_pending(self):
        """Highest value of :attr:`pending` seen so far."""
        if not self._queue_enabled:
            return 0

        with self._output_state.cond:
            return self._output_state.peak_pending

    @property
    def sent_count(self):
        """Number of messages written by the worker so far."""
        if not self._queue_enabled:
            return 0

        with self._output_state.cond:
            return self._output_state.sent_count

    @property
    def dropped_count(self):
        """Messages discarded because of panic/close/send failure."""
        if not self._queue_enabled:
            return 0

        with self._output_state.cond:
            return self._output_state.dropped_count

    @property
    def last_error(self):
        """The exception raised by the background worker, if any."""
        if not self._queue_enabled:
            return None

        with self._output_state.cond:
            return self._output_state.send_error

    def send(self, msg):
        """Send a message on the port.

        A copy of the message will be sent, so you can safely modify
        the original message without any unexpected consequences.

        In queued mode this returns the message's enqueue sequence
        number; otherwise it returns None.
        """
        if not self.is_output:
            raise ValueError('Not an output port')
        elif not isinstance(msg, Message):
            raise TypeError('argument to send() must be a Message')
        elif self.closed:
            raise ValueError('send() called on closed port')

        if not self._queue_enabled:
            with self._lock:
                self._send(msg.copy())

            return None

        return self._enqueue(msg.copy())

    def _enqueue(self, message, priority=None):
        """Enqueue a copied message and return its sequence number.

        priority is None for normal messages, 'reset' or 'panic'.
        """
        state = self._output_state
        cond = state.cond

        with cond:
            if state.send_error is not None:
                raise state.send_error

            if not state.accepting or state.stop_requested or state.closing:
                raise ValueError('send() called on closed port')

            self._ensure_worker()

            state.seq += 1
            item = _QueuedMessage(message, state.seq)

            if priority == 'panic':
                self._push_priority(state, item, state.panic_queue)
            elif priority == 'reset':
                self._push_priority(state, item, state.reset_queue)
            else:
                self._push_normal(state, item)

            return item.seq

    def _push_priority(self, state, item, target):
        """Append an item to a priority queue. Must hold the lock."""
        target.append(item)
        state.count_pending()
        state.cond.notify_all()

    def _push_normal(self, state, item):
        if state.capacity == 0:
            self._push_rendezvous(state, item)
            return

        cond = state.cond
        reentrant = threading.get_ident() == state.worker_ident
        policy = state.policy
        capacity = state.capacity
        generation = state.abort_generation

        if policy == 'timeout':
            deadline = time.monotonic() + state.queue_timeout

        while len(state.queue) >= capacity:
            if state.send_error is not None:
                raise state.send_error
            if not state.accepting or state.stop_requested:
                raise ValueError('send() called on closed port')
            if state.abort_generation != generation:
                raise QueueAborted('send() aborted by panic')
            if reentrant:
                # Waiting for the worker to make room would wait on
                # the worker itself, which is this thread.
                raise queue.Full('output queue is full')
            if policy == 'reject':
                raise queue.Full('output queue is full')
            elif policy == 'timeout':
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Full('output queue is full')
                cond.wait(remaining)
            else:
                cond.wait()

        if state.send_error is not None:
            raise state.send_error
        if not state.accepting or state.stop_requested:
            raise ValueError('send() called on closed port')
        if state.abort_generation != generation:
            # panic() ran while we waited. Do not enqueue a message
            # that the user meant to stop.
            raise QueueAborted('send() aborted by panic')

        state.queue.append(item)
        state.count_pending()
        cond.notify_all()

    def _push_rendezvous(self, state, item):
        """Handle queue_size=0: one-slot handoff to the worker."""
        cond = state.cond
        policy = state.policy
        reentrant = threading.get_ident() == state.worker_ident
        generation = state.abort_generation

        # Wait for the handoff slot to become free.
        if policy == 'reject' or reentrant:
            if state.handoff is not None:
                raise queue.Full('output queue is full')
        else:
            deadline = None
            if policy == 'timeout':
                deadline = time.monotonic() + state.queue_timeout

            while state.handoff is not None:
                if state.send_error is not None:
                    raise state.send_error
                if not state.accepting or state.stop_requested:
                    raise ValueError('send() called on closed port')
                if state.abort_generation != generation:
                    raise QueueAborted('send() aborted by panic')

                if deadline is None:
                    cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Full('output queue is full')
                    cond.wait(remaining)

        state.handoff = item
        state.count_pending()
        cond.notify_all()

        # 'reject' and reentrant sends hand the message off without
        # waiting for the worker to take it. A blocking or timed send
        # waits for the worker to accept the message, which is what
        # makes a zero-capacity queue a rendezvous.
        if policy == 'reject' or reentrant:
            return

        deadline = None
        if policy == 'timeout':
            deadline = time.monotonic() + state.queue_timeout

        while not item.accepted:
            if item.aborted or state.abort_generation != generation:
                cond.notify_all()
                if state.send_error is not None:
                    raise state.send_error
                raise QueueAborted('send() aborted by panic')

            if state.handoff is item and (
                    state.send_error is not None or
                    not state.accepting or state.stop_requested):
                state.handoff = None
                state.pending -= 1
                cond.notify_all()
                if state.send_error is not None:
                    raise state.send_error
                raise ValueError('send() called on closed port')

            if deadline is None:
                cond.wait()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if state.handoff is item:
                        state.handoff = None
                        state.pending -= 1
                        cond.notify_all()
                    raise queue.Full('output queue is full')
                cond.wait(remaining)

    def _ensure_worker(self):
        """Start the worker thread if needed. Must hold the lock."""
        state = self._output_state

        if state.worker is None and not state.worker_done:
            worker = threading.Thread(
                target=_output_worker_entry,
                args=(weakref.ref(self), state),
                name=f'mido-output-{self.name or "port"}',
                daemon=True)
            state.worker = worker
            worker.start()

    def flush(self, timeout=None):
        """Wait until all queued messages have been written.

        Raises the exception raised by the background worker, if any.
        With ``timeout``, raises :class:`TimeoutError` if the queue
        does not drain in time.  Calling flush() from inside ``_send()``
        on the worker thread raises :class:`RuntimeError` instead of
        waiting for itself.
        """
        if not self._queue_enabled:
            return

        state = self._output_state
        cond = state.cond

        if threading.get_ident() == state.worker_ident:
            raise RuntimeError(
                'flush() called from the output worker thread')

        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        with cond:
            while state.pending or state.in_flight:
                if deadline is None:
                    cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('output queue did not flush')
                    cond.wait(remaining)

            error = state.send_error

        if error is not None:
            raise error

    def close(self, timeout=None):
        """Close the port.

        In queued mode this first waits for the queue to drain (see
        ``close_timeout``), then stops the worker and closes the
        underlying port.  If the worker is stuck in a slow ``_send()``
        it is allowed to finish and closes the port itself, so
        ``close()`` never touches the device concurrently with a
        write.
        """
        if not self._queue_enabled:
            return super().close()

        if timeout is None:
            timeout = self._close_timeout

        state = self._output_state

        # Do not hold self._lock while waiting for the worker: the
        # worker may be inside _send() and call close() itself, which
        # would deadlock against a lock held by the thread waiting for
        # it. The state lock serializes close() instead.
        with state.cond:
            if self.closed or state.closing:
                return
            state.closing = True

        try:
            if self.autoreset:
                try:
                    self.reset()
                except OSError:
                    pass

            with state.cond:
                state.accepting = False
                state.stop_requested = True
                state.cond.notify_all()

            self._shutdown_queue(timeout)
        finally:
            self.closed = True

    def _shutdown_queue(self, timeout):
        """Drain/stop the worker and close the underlying port."""
        state = self._output_state
        cond = state.cond

        # close() called from _send(): do not wait for ourselves.
        if threading.get_ident() == state.worker_ident:
            with cond:
                state.backend_close_deferred = True
            return

        backend_close = False

        with cond:
            if state.worker is None or state.worker_done:
                state.discard_pending()
                backend_close = True
            else:
                if timeout is None:
                    deadline = None
                else:
                    deadline = time.monotonic() + timeout

                # Let the worker drain. The stop flag is only acted
                # upon once the queues are empty.
                while state.pending or state.in_flight:
                    if deadline is None:
                        cond.wait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        cond.wait(remaining)

                if state.pending or state.in_flight:
                    state.discard_pending()
                    if state.in_flight:
                        # The worker is inside _send(). It will see the
                        # stop flag when the backend returns and close
                        # the port itself. Hand it a strong reference
                        # so a concurrent garbage collection cannot
                        # skip the backend close.
                        state.deferred_port = self
                        state.backend_close_deferred = True
                        cond.notify_all()
                        return

                # Nothing is in flight, so the worker will observe the
                # stop flag and exit without touching the device.
                while not state.worker_done:
                    cond.wait()

                backend_close = True

            state.worker = None

            if backend_close:
                state.backend_close_done = True

        if backend_close:
            self._close()

    def reset(self):
        """Send "All Notes Off" and "Reset All Controllers" on all channels"""
        if self.closed:
            return

        if not self._queue_enabled:
            for msg in reset_messages():
                self.send(msg)
            return

        state = self._output_state

        with state.cond:
            if not state.accepting or state.stop_requested:
                return
            if state.send_error is not None:
                raise state.send_error

            self._ensure_worker()
            # Repeated reset() calls should not grow the queue without
            # bound: keep only the most recent request.
            for item in state.reset_queue:
                item.aborted = True
            state.dropped_count += len(state.reset_queue)
            state.pending -= len(state.reset_queue)
            state.reset_queue.clear()

            for msg in reset_messages():
                state.seq += 1
                self._push_priority(
                    state, _QueuedMessage(msg.copy(), state.seq),
                    state.reset_queue)

    def panic(self):
        """Send "All Sounds Off" on all channels.

        This will mute all sounding notes regardless of
        envelopes. Useful when notes are hanging and nothing else
        helps.

        In queued mode panic messages are placed ahead of everything
        else and pending normal and reset messages are discarded. The
        messages are never dropped, and the call does not wait for the
        queue to drain.
        """
        if self.closed:
            return

        if not self._queue_enabled:
            for msg in panic_messages():
                self.send(msg)
            return

        state = self._output_state

        with state.cond:
            if not state.accepting or state.stop_requested:
                return
            if state.send_error is not None:
                raise state.send_error

            self._ensure_worker()
            # Abort senders waiting for room and drop queued messages.
            state.abort_generation += 1
            state.discard_pending()
            for msg in panic_messages():
                state.seq += 1
                self._push_priority(
                    state, _QueuedMessage(msg.copy(), state.seq),
                    state.panic_queue)


class BaseIOPort(BaseInput, BaseOutput):
    def __init__(self, name='', **kwargs):
        """Create an IO port.

        name is the port name, as returned by ioport_names().
        """
        BaseInput.__init__(self, name, **kwargs)
        BaseOutput.__init__(self, name, **kwargs)

class IOPort(BaseIOPort):
    """Input / output port.

    This is a convenient wrapper around an input port and an output
    port which provides the functionality of both. Every method call
    is forwarded to the appropriate port.
    """

    _locking = False

    def __init__(self, input, output):
        self.input = input
        self.output = output

        # We use str() here in case name is None.
        self.name = f'{str(input.name)} + {str(output.name)}'
        self._messages = self.input._messages
        self.closed = False
        self._lock = DummyLock()

    def _close(self):
        self.input.close()
        self.output.close()

    def flush(self, timeout=None):
        """Wait for the wrapped output port to drain its queue."""
        if hasattr(self.output, 'flush'):
            return self.output.flush(timeout=timeout)

    def panic(self):
        """Forward panic() to the wrapped output port."""
        return self.output.panic()

    def reset(self):
        """Forward reset() to the wrapped output port."""
        return self.output.reset()

    def _send(self, message):
        self.output.send(message)

    def _receive(self, block=True):
        return self.input.receive(block=block)


class EchoPort(BaseIOPort):
    def _send(self, message):
        self._messages.append(message)

    __iter__ = BaseIOPort.iter_pending


class MultiPort(BaseIOPort):
    def __init__(self, ports, yield_ports=False):
        BaseIOPort.__init__(self, 'multi')
        self.ports = list(ports)
        self.yield_ports = yield_ports

    def _send(self, message):
        for port in self.ports:
            if not port.closed:
                # TODO: what if a SocketPort connection closes in-between here?
                port.send(message)

    def _receive(self, block=True):
        self._messages.extend(multi_receive(self.ports,
                                            yield_ports=self.yield_ports,
                                            block=block))


def multi_receive(ports, yield_ports=False, block=True):
    """Receive messages from multiple ports.

    Generates messages from ever input port. The ports are polled in
    random order for fairness, and all messages from each port are
    yielded before moving on to the next port.

    If yield_ports=True, (port, message) is yielded instead of just
    the message.

    If block=False only pending messages will be yielded.
    """
    ports = list(ports)
    while True:
        # Make a shuffled copy of the port list.
        random.shuffle(ports)

        for port in ports:
            if not port.closed:
                for message in port.iter_pending():
                    if yield_ports:
                        yield port, message
                    else:
                        yield message

        if block:
            sleep()
        else:
            break


def multi_iter_pending(ports, yield_ports=False):
    """Iterate through all pending messages in ports.

    This is the same as calling multi_receive(ports, block=False).
    The function is kept around for backwards compatability.
    """
    return multi_receive(ports, yield_ports=yield_ports, block=False)


def multi_send(ports, msg):
    """Send message on all ports."""
    for port in ports:
        port.send(msg)
