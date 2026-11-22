# SPDX-FileCopyrightText: 2013 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
MIDI Parser

There is no need to use this module directly. All you need is
available in the top level module.
"""
from collections import deque
from numbers import Integral

from .messages import Message
from .tokenizer import Tokenizer, TokenizerOverflow


class ParserOverflow(Exception):
    """Raised when a bounded parser can no longer accept data.

    Two independent limits can be hit, see ``kind``:

    * ``'messages'``: the queue of completed but not yet consumed
      messages already contains ``max_messages`` messages. The
      completed messages are kept intact; drain them with
      :meth:`Parser.drain_messages` or :meth:`Parser.consume` and feed
      the remaining bytes again.
    * ``'bytes'``: the unfinished message (in practice a SysEx that has
      not seen its end byte yet) reached ``max_message_bytes``. The
      unfinished bytes are kept at the limit; either continue once the
      end byte is expected, or call :meth:`Parser.reset` to discard the
      partial message.

    Attributes:
        kind: ``'messages'`` or ``'bytes'``.
        limit: the configured limit that was reached.
        size: the current size against that limit (number of queued
            messages, or length in bytes the unfinished message would
            have had).
        byte_index: index inside the data passed to :meth:`feed` of
            the byte that triggered the error (0 when
            :meth:`feed_byte` was used).
        resume_at: index in the data passed to :meth:`feed` from which
            feeding should continue after recovery. For ``'bytes'``
            the offending byte was rejected and is still pending, so
            this equals ``byte_index`` (after :meth:`Parser.reset`,
            or once the message can be completed). For ``'messages'``
            the offending byte already completed a message, which is
            retained and delivered once room exists, so this equals
            ``byte_index + 1``.
    """

    def __init__(self, kind, limit, size, byte_index):
        self.kind = kind
        self.limit = limit
        self.size = size
        self.byte_index = byte_index
        self.resume_at = byte_index + (1 if kind == 'messages' else 0)
        super().__init__(
            f'{kind} limit ({limit}) reached at byte {byte_index}; '
            f'current size is {size}'
        )


def _check_limit(name, value):
    if value is not None:
        if not isinstance(value, Integral):
            raise TypeError(f'{name} must be an integer')
        if value < 0:
            raise ValueError(f'{name} must be >= 0')
    return value


class Parser:
    """
    MIDI byte stream parser

    Parses a stream of MIDI bytes and produces messages.

    Data can be put into the parser in the form of
    integers, byte arrays or byte strings.

    Optional bounds make memory usage deterministic when the consumer
    falls behind:

    * ``max_messages`` limits the number of completed messages waiting
      to be taken.
    * ``max_message_bytes`` limits the size of a single unfinished
      message (typically a SysEx split across feeds).

    Both default to None, which keeps the historical unbounded
    behavior. With a bound set, :meth:`feed` and :meth:`feed_byte`
    raise :class:`ParserOverflow` instead of growing without limit.
    Completed messages are never lost: they stay in the queue until
    taken via :meth:`get_message`, iteration, :meth:`drain_messages`
    or :meth:`consume`. The tokenizer's treatment of realtime messages
    inserted inside SysEx, stray data bytes and SysEx continued across
    feeds is identical to the unbounded parser.
    """
    def __init__(self, data=None, max_messages=None,
                 max_message_bytes=None):
        # For historical reasons self.messages is public and must be a
        # deque(). (It is referenced directly inside ports.)
        self.messages = deque()
        self._max_messages = _check_limit('max_messages', max_messages)
        self._tok = Tokenizer(max_message_bytes=max_message_bytes)
        # Index of the byte currently being processed inside feed().
        # Used to build a locatable ParserOverflow.
        self._feed_pos = None
        if data:
            self.feed(data)

    @property
    def max_messages(self):
        """Maximum number of completed messages that may queue up.

        None means unbounded (the historical behavior).
        """
        return self._max_messages

    @max_messages.setter
    def max_messages(self, value):
        self._max_messages = _check_limit('max_messages', value)

    @property
    def max_message_bytes(self):
        """Maximum size in bytes of a single unfinished message.

        None means unbounded (the historical behavior).
        """
        return self._tok.max_message_bytes

    @max_message_bytes.setter
    def max_message_bytes(self, value):
        self._tok.max_message_bytes = value

    def _overflow(self, kind, limit, size):
        return ParserOverflow(kind, limit, size,
                              0 if self._feed_pos is None
                              else self._feed_pos)

    def _pump(self):
        """Move completed tokens out of the tokenizer into the queue.

        Raises ParserOverflow('messages') when the message bound is
        reached; the offending token stays inside the tokenizer, so no
        completed message is lost.
        """
        while True:
            token = self._tok.drain_one()
            if token is None:
                return
            if (self._max_messages is not None
                    and len(self.messages) >= self._max_messages):
                # Put the token back so nothing is dropped. appendleft
                # preserves ordering relative to any later tokens.
                self._tok._messages.appendleft(token)
                raise self._overflow('messages', self._max_messages,
                                     len(self.messages))
            self.messages.append(Message.from_bytes(token))

    def _decode(self):
        self._pump()

    def feed(self, data):
        """Feed MIDI data to the parser.

        Accepts any object that produces a sequence of integers in
        range 0..255, such as:

            [0, 1, 2]
            (0, 1, 2)
            [for i in range(256)]
            (for i in range(256)]
            bytearray()

        When bounds are configured and reached, ParserOverflow is
        raised. Its ``byte_index`` attribute gives the position of the
        offending byte in ``data``. After making room (draining the
        queue for a ``'messages'`` overflow, or resetting / completing
        the partial message for a ``'bytes'`` overflow), feed the
        remaining bytes starting at the exception's ``resume_at``
        index to continue without losing or duplicating anything.
        """
        if self._max_messages is None and self._tok.max_message_bytes is None:
            # Historical fast path: identical behavior to the
            # unbounded parser.
            self._tok.feed(data)
            self._decode()
            return

        self._feed_pos = 0
        try:
            for byte in data:
                self.feed_byte(byte)
                self._feed_pos += 1
            # Even with no (or only rejected) bytes, an earlier
            # overflow may have left a completed token in the
            # tokenizer waiting for room in the queue.
            self._pump()
        finally:
            self._feed_pos = None

    def feed_byte(self, byte):
        """Feed one MIDI byte into the parser.

        The byte must be an integer in range 0..255.

        Raises ParserOverflow instead of growing past a configured
        bound. On a ``'bytes'`` overflow the offending byte is rejected
        and held back by the tokenizer; on a ``'messages'`` overflow
        the byte is consumed but the completed message it produced is
        retained in the tokenizer until the queue has room.
        """
        try:
            self._tok.feed_byte(byte)
        except TokenizerOverflow as err:
            raise self._overflow('bytes', err.limit, err.length) from err
        self._pump()

    def get_message(self):
        """Get the first parsed message.

        Returns None if there is no message yet. If you don't want to
        deal with None, you can use pending() to see how many messages
        you can get before you get None, or just iterate over the
        parser.
        """
        for msg in self:
            return msg
        else:
            return None

    def pending(self):
        """Return the number of pending messages."""
        return len(self.messages)

    __len__ = pending

    def pending_bytes(self):
        """Return the size in bytes of the unfinished message.

        This is partial data for a message not yet complete (for
        example a SysEx still waiting for its end byte). It is the
        amount of input data the parser has to remember between
        feeds in addition to the queued messages.
        """
        return self._tok.pending_bytes()

    def drain_messages(self, limit=None):
        """Remove and return up to ``limit`` pending messages as a list.

        With ``limit=None`` every pending message is returned. With
        ``limit=0`` or an empty queue an empty list is returned.

        This is the batch form of incremental consumption: feed,
        drain a batch, feed again. Messages are returned in the same
        order they appeared in the byte stream.
        """
        if limit is None:
            return list(self)
        if limit < 0:
            raise ValueError('limit must be >= 0')
        drained = []
        while len(drained) < limit and self.messages:
            drained.append(self.messages.popleft())
        return drained

    def consume(self, callback, limit=None):
        """Pass pending messages one by one to ``callback``.

        At most ``limit`` messages are delivered (all of them when
        ``limit`` is None). Returns the number of messages that were
        delivered to the callback.

        If ``callback`` raises, the message being processed is put
        back at the head of the queue and the exception propagates, so
        no message is lost and retrying delivers the same message
        again. No further messages are delivered after the failure.
        """
        if limit is not None and limit < 0:
            raise ValueError('limit must be >= 0')
        delivered = 0
        while limit is None or delivered < limit:
            if not self.messages:
                break
            msg = self.messages.popleft()
            try:
                callback(msg)
            except BaseException:
                self.messages.appendleft(msg)
                raise
            delivered += 1
        return delivered

    def reset(self):
        """Drop all pending messages and the unfinished message.

        The parser returns to the state of a freshly constructed one,
        including the tokenizer's cleared status byte, while keeping
        its configured limits. This is the deterministic way to
        recover from a ``max_message_bytes`` overflow when the partial
        message cannot be completed, or to reuse a parser after an
        error.
        """
        self._tok.reset()
        self.messages.clear()

    def __iter__(self):
        """Yield messages that have been parsed so far."""
        while len(self.messages) > 0:
            yield self.messages.popleft()


def parse_all(data):
    """Parse MIDI data and return a list of all messages found.

    This is typically used to parse a little bit of data with a few
    messages in it. It's best to use a Parser object for larger
    amounts of data. Also, tt's often easier to use parse() if you
    know there is only one message in the data.
    """
    return list(Parser(data))


def parse(data):
    """ Parse MIDI data and return the first message found.

    Data after the first message is ignored. Use parse_all()
    to parse more than one message.
    """
    return Parser(data).get_message()
