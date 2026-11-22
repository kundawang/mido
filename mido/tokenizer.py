# SPDX-FileCopyrightText: 2017 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

from collections import deque
from numbers import Integral
from .messages.specs import SYSEX_START, SYSEX_END, SPEC_BY_STATUS


class TokenizerOverflow(Exception):
    """Raised when a single unfinished message exceeds the byte budget.

    Attributes:
        limit: the configured maximum number of bytes per unfinished
            message.
        length: the length the message would have had if the offending
            byte had been accepted (always ``limit + 1``).
        state: the current status byte of the unfinished message (for
            example ``0xf0`` for a SysEx).
    """

    def __init__(self, limit, length, state):
        self.limit = limit
        self.length = length
        self.state = state
        super().__init__(
            f'unfinished message would be {length} bytes, '
            f'which exceeds max_message_bytes={limit} '
            f'(state byte 0x{state:02x})'
        )


class Tokenizer:
    """
    Splits a MIDI byte stream into messages.
    """
    def __init__(self, data=None, max_message_bytes=None):
        """Create a new decoder.

        If ``max_message_bytes`` is not None it limits the number of
        bytes a single unfinished message (in practice a SysEx that has
        not seen its end byte yet) may grow to. Completed messages are
        never truncated or dropped. When a new data or end byte would
        make the unfinished message longer than the limit a
        :class:`TokenizerOverflow` is raised. The offending byte is not
        stored, so the decoder state stays exactly at the limit and
        feeding can be retried (for example after :meth:`reset`).
        """

        self._status = 0
        self._bytes = []
        self._messages = deque()
        self._datalen = 0
        self.max_message_bytes = max_message_bytes

        if data is not None:
            self.feed(data)

    @property
    def max_message_bytes(self):
        """Maximum size in bytes of a single unfinished message.

        None means unbounded (the historical behavior).
        """
        return self._max_message_bytes

    @max_message_bytes.setter
    def max_message_bytes(self, value):
        if value is not None:
            if not isinstance(value, Integral):
                raise TypeError('max_message_bytes must be an integer')
            if value < 0:
                raise ValueError('max_message_bytes must be >= 0')
        self._max_message_bytes = value

    def _check_byte_budget(self, byte):
        """Append byte to the unfinished message with bound checking.

        Raises TokenizerOverflow without storing the byte if the
        configured limit would be exceeded.
        """
        limit = self._max_message_bytes
        if limit is not None and len(self._bytes) >= limit:
            raise TokenizerOverflow(limit, len(self._bytes) + 1,
                                    self._status)
        self._bytes.append(byte)

    def _feed_status_byte(self, byte):
        if byte == SYSEX_END:
            if self._status == SYSEX_START:
                self._check_byte_budget(SYSEX_END)
                self._messages.append(self._bytes)

            self._status = 0

        elif 0xf8 <= byte <= 0xff:
            if self._status != SYSEX_START:
                # Realtime messages are only allowed inside sysex
                # messages. Reset parser.
                self._status = 0

            if byte in SPEC_BY_STATUS:
                self._messages.append([byte])

        elif byte in SPEC_BY_STATUS:
            # New message.
            spec = SPEC_BY_STATUS[byte]

            if spec['length'] == 1:
                self._messages.append([byte])
                self._status = 0
            else:
                self._status = byte
                self._bytes = [byte]
                self._len = spec['length']
        else:
            # Undefined message. Reset parser.
            # (Undefined realtime messages are handled above.)
            # self._status = 0
            pass

    def _feed_data_byte(self, byte):
        if self._status:
            self._check_byte_budget(byte)
            if len(self._bytes) == self._len:
                # Complete message.
                self._messages.append(self._bytes)
                self._status = 0
        else:
            # Ignore stray data byte.
            pass

    def feed_byte(self, byte):
        """Feed MIDI byte to the decoder.

        Takes an int in range [0..255].
        """
        if not isinstance(byte, Integral):
            raise TypeError('message byte must be integer')

        if 0 <= byte <= 255:
            if byte <= 127:
                return self._feed_data_byte(byte)
            else:
                return self._feed_status_byte(byte)
        else:
            raise ValueError(f'invalid byte value {byte!r}')

    def feed(self, data):
        """Feed MIDI bytes to the decoder.

        Takes an iterable of ints in in range [0..255].
        """
        for byte in data:
            self.feed_byte(byte)

    def pending_bytes(self):
        """Return the size in bytes of the unfinished message.

        This is the partial data that has been received for a message
        that is not complete yet (for example a SysEx still waiting for
        its end byte), and therefore the lower bound on memory the
        decoder holds between feeds.
        """
        return len(self._bytes) if self._status else 0

    def drain_one(self):
        """Remove and return the next completed token, or None.

        Unlike iterating over the tokenizer this pops only one token.
        """
        if self._messages:
            return self._messages.popleft()
        return None

    def reset(self):
        """Discard every completed token and the unfinished message."""
        self._status = 0
        self._bytes = []
        self._datalen = 0
        self._messages.clear()

    def __len__(self):
        return len(self._messages)

    def __iter__(self):
        """Yield messages that have been parsed so far."""
        while len(self._messages):
            yield self._messages.popleft()
