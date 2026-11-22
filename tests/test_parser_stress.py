# SPDX-FileCopyrightText: 2024
#
# SPDX-License-Identifier: MIT

"""Large-stream stress test for bounded, incremental parsing.

A million well-formed MIDI messages are generated as a raw byte stream
and fed through a bounded parser in random-sized chunks. The test
checks:

* exact ordering and content against the stock unbounded parser,
* that the completed-message queue and the unfinished-message buffer
  never exceed their configured bounds (structural peak-memory
  guarantee),
* that ParserOverflow fires at exactly the rejection boundary and that
  resuming via ``resume_at`` neither loses nor duplicates a message.

Run explicitly::

    pytest tests/test_parser_stress.py -s
"""

import random
import tracemalloc

import pytest

from mido.parser import Parser, ParserOverflow

N_MESSAGES = 1_000_000
MAX_MESSAGES = 64
MAX_MESSAGE_BYTES = 256


def generate_stream(n, seed=0xC0FFEE, max_sysex_wire=18):
    """Return the raw bytes of n well-formed messages.

    Every message's ordinal influences its data bytes so that any
    reordering shows up as a mismatch when comparing outputs.
    """
    rng = random.Random(seed)
    raw = bytearray()
    for ordinal in range(n):
        pick = rng.random()
        if pick < 0.10:
            # Single-byte realtime message.
            raw.append(0xF8)
        elif pick < 0.12:
            # Small SysEx, kept strictly under every byte budget used
            # in this module so a 'bytes' overflow never fires here;
            # that boundary has its own dedicated test.
            size = rng.randrange(0, max_sysex_wire - 2 + 1)
            raw.append(0xF0)
            raw.extend(rng.randrange(128) for _ in range(size))
            raw.append(0xF7)
        else:
            channel = ordinal % 16
            encoded = ordinal % 128
            kind = rng.randrange(3)
            if kind == 0:
                raw.extend((0x80 | channel, encoded, rng.randrange(128)))
            elif kind == 1:
                # program_change is a two-byte message.
                raw.extend((0xC0 | channel, encoded))
            else:
                raw.extend((0x90 | channel, encoded,
                            rng.randrange(1, 128)))
    return bytes(raw)


def drive(data, *, max_messages, max_message_bytes, seed):
    """Drive a bounded parser over ``data`` in random chunks and check
    it against the stock unbounded parser.

    Both parsers receive the whole logical stream (the reference in
    whole chunks, the bounded one with backpressure and resume). Every
    message emitted by the bounded parser is checked against the
    reference parser's *next* message: the reference acts as a
    read-ahead oracle, so its longer queue never forces alignment at
    chunk boundaries.

    Returns (messages_emitted, overflows, peak_queue, peak_partial).
    """
    rng = random.Random(seed)
    bounded = Parser(max_messages=max_messages,
                     max_message_bytes=max_message_bytes)
    reference = Parser()

    overflows = 0
    peak_queue = 0
    peak_partial = 0
    emitted = 0
    pos = 0

    def match(b_msg):
        """Match one bounded message against the oracle's next one."""
        nonlocal emitted
        ref_msg = reference.get_message()
        assert ref_msg is not None, \
            f'bounded parser emitted an unexpected message {b_msg!r}'
        assert ref_msg.bytes() == b_msg.bytes(), \
            f'order/content mismatch: {ref_msg!r} != {b_msg!r}'
        emitted += 1

    while pos < len(data):
        step = rng.randrange(1, 256)
        chunk = data[pos:pos + step]
        reference.feed(chunk)  # oracle reads ahead freely

        cursor = 0
        while cursor < len(chunk):
            try:
                bounded.feed(chunk[cursor:])
                break
            except ParserOverflow as err:
                overflows += 1
                if err.kind == 'messages':
                    # Hard guarantee: the queue never exceeds the bound.
                    assert err.size == max_messages
                    assert bounded.pending() == max_messages
                else:
                    assert bounded.pending_bytes() <= max_message_bytes
                # Make room (random batch size, at least one) and match
                # each drained message against the read-ahead oracle.
                drained = bounded.drain_messages(
                    rng.randrange(1, max_messages + 1))
                assert 1 <= len(drained) <= max_messages
                for msg in drained:
                    match(msg)
                cursor += err.resume_at
        peak_queue = max(peak_queue, bounded.pending())
        peak_partial = max(peak_partial, bounded.pending_bytes())
        for msg in bounded.drain_messages():
            match(msg)
        pos += step

    # Any token retained in the tokenizer after the last byte reaches
    # the queue once room exists; drain fully and reconcile with the
    # oracle, which must then be exhausted too.
    assert bounded.pending_bytes() == 0
    for msg in bounded.drain_messages():
        match(msg)
    assert bounded.pending() == 0
    assert reference.get_message() is None
    return emitted, overflows, peak_queue, peak_partial


def test_million_message_stream_ordering_bounds_and_resume():
    data = generate_stream(N_MESSAGES)
    assert len(data) > N_MESSAGES  # multi-byte messages present

    emitted, overflows, peak_queue, peak_partial = drive(
        data,
        max_messages=MAX_MESSAGES,
        max_message_bytes=MAX_MESSAGE_BYTES,
        seed=1234567,
    )

    assert emitted == N_MESSAGES
    # The queue bound must really have been exercised.
    assert overflows > 0
    # Structural peak-memory guarantees.
    assert peak_queue <= MAX_MESSAGES
    assert peak_partial <= MAX_MESSAGE_BYTES


def _run_bounded(n, seed):
    data = generate_stream(n, seed=seed)
    parser = Parser(max_messages=32, max_message_bytes=128)
    rng = random.Random(seed)
    pos = 0
    while pos < len(data):
        step = rng.randrange(1, 512)
        chunk = data[pos:pos + step]
        cursor = 0
        while cursor < len(chunk):
            try:
                parser.feed(chunk[cursor:])
                break
            except ParserOverflow as err:
                parser.drain_messages()
                cursor += err.resume_at
        parser.drain_messages()
        pos += step
    parser.drain_messages()
    return parser.pending(), parser.pending_bytes()


def test_peak_memory_is_independent_of_stream_length():
    """Retained memory of a drained bounded parser must not grow with
    the number of messages processed."""
    tracemalloc.start()
    _run_bounded(10_000, seed=1)
    _, peak_small = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _run_bounded(100_000, seed=1)
    _, peak_large = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ten times more history must not multiply retained memory. The
    # generous constant absorbs allocator and tracemalloc noise.
    assert peak_large < peak_small * 3 + 1_000_000


def test_rejection_boundary_is_exact_for_messages():
    parser = Parser(max_messages=3)
    for _ in range(3):
        parser.feed(b'\xf8')  # single-byte clocks fill the queue
    assert parser.pending() == 3
    with pytest.raises(ParserOverflow) as info:
        parser.feed(b'\xf8')
    assert info.value.kind == 'messages'
    assert info.value.size == 3
    assert info.value.limit == 3
    assert info.value.resume_at == info.value.byte_index + 1

    # Drain one: the retained clock is pumped on the next feed and the
    # queue is saturated again, still never over the bound.
    parser.drain_messages(1)
    parser.feed(b'')
    assert parser.pending() == 3
    with pytest.raises(ParserOverflow):
        parser.feed(b'\xf8')


def test_rejection_boundary_is_exact_for_bytes():
    parser = Parser(max_message_bytes=4)
    parser.feed(b'\xf0\x01\x02\x03')  # exactly at the limit
    assert parser.pending_bytes() == 4
    with pytest.raises(ParserOverflow) as info:
        parser.feed(b'\x04')
    assert info.value.kind == 'bytes'
    assert info.value.limit == 4
    assert info.value.size == 5
    assert info.value.resume_at == info.value.byte_index
    # The end byte at the same length is rejected too: the limit is a
    # hard ceiling, nothing is silently truncated.
    with pytest.raises(ParserOverflow):
        parser.feed(b'\xf7')


def test_random_chunking_never_exceeds_bounds():
    """Aggressive tiny chunks and tiny bounds over many iterations."""
    data = generate_stream(20_000, seed=7)
    emitted, overflows, peak_queue, peak_partial = drive(
        data, max_messages=4, max_message_bytes=32, seed=7)
    assert emitted == 20_000
    assert overflows > 0
    assert peak_queue <= 4
    assert peak_partial <= 32
