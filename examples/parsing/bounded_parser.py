#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2024
#
# SPDX-License-Identifier: MIT

"""
Bounded parsing with backpressure.

A live device keeps feeding MIDI bytes while the consumer occasionally
falls behind. An unbounded Parser would then grow its message queue and
its unfinished SysEx buffer without limit. The optional bounds make
memory usage deterministic:

* ``max_messages`` caps the completed messages waiting to be taken.
* ``max_message_bytes`` caps one unfinished message (a SysEx split over
  several feeds).

When a bound is reached ``feed()`` raises ``ParserOverflow``. The
completed messages are kept; drain a batch (or hand them to a callback)
and resume feeding at the exception's ``resume_at`` index.
"""

import mido
from mido.parser import ParserOverflow

# A device stream standing in for port.iter_bytes() / poll() data.
device_stream = bytes([
    0x90, 60, 100,   # note_on
    0x80, 60, 0,     # note_off
    0xf8,            # clock
]) * 100


def batch_demo():
    parser = mido.Parser(max_messages=16, max_message_bytes=1024)
    incoming = device_stream
    processed = 0

    while incoming:
        try:
            parser.feed(incoming)
            incoming = b''
        except ParserOverflow as error:
            if error.kind == 'messages':
                # Consumer takes a bounded batch, then the producer
                # resumes exactly where it was told to.
                for message in parser.drain_messages(limit=8):
                    process(message)
                    processed += 1
                incoming = incoming[error.resume_at:]
            else:
                # A SysEx exceeds the byte budget; drop the partial
                # message deterministically and skip its rejected byte.
                parser.reset()
                incoming = incoming[error.resume_at:]

    # Flush whatever completed after the last chunk.
    for message in parser.drain_messages():
        process(message)
        processed += 1

    print(f'batch demo processed {processed} messages')


def callback_demo():
    parser = mido.Parser(max_messages=16, max_message_bytes=1024)
    incoming = device_stream
    processed = 0

    def handle(message):
        nonlocal processed
        process(message)
        processed += 1

    while incoming:
        try:
            parser.feed(incoming)
            incoming = b''
        except ParserOverflow as error:
            if error.kind == 'messages':
                # Callback consumption: if handle() raises, the message
                # is requeued and nothing is lost.
                parser.consume(handle, limit=8)
                incoming = incoming[error.resume_at:]
            else:
                parser.reset()
                incoming = incoming[error.resume_at:]

    parser.consume(handle)
    print(f'callback demo processed {processed} messages')


def process(message):
    """Stand-in for the real consumer work."""
    assert message is not None


if __name__ == '__main__':
    batch_demo()
    callback_demo()
