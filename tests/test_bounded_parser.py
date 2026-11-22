# SPDX-FileCopyrightText: 2024
#
# SPDX-License-Identifier: MIT

"""Tests for bounded, incremental Parser consumption."""

import random

import pytest

from mido.messages import Message
from mido.parser import Parser, ParserOverflow
from mido.tokenizer import Tokenizer, TokenizerOverflow


# --------------------------------------------------------------------------
# Construction and validation
# --------------------------------------------------------------------------

def test_defaults_are_unbounded():
    parser = Parser()
    assert parser.max_messages is None
    assert parser.max_message_bytes is None
    assert parser.pending() == 0
    assert parser.pending_bytes() == 0


@pytest.mark.parametrize('kwargs', [
    {'max_messages': -1},
    {'max_message_bytes': -1},
])
def test_negative_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        Parser(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'max_messages': 1.5},
    {'max_messages': '4'},
    {'max_message_bytes': 1.5},
    {'max_message_bytes': None},
])
def test_non_integer_limits_rejected(kwargs):
    if kwargs.get('max_message_bytes') is None and \
            'max_message_bytes' in kwargs:
        Parser(**kwargs)  # None is explicitly allowed
        return
    with pytest.raises(TypeError):
        Parser(**kwargs)


def test_tokenizer_validates_its_own_limit():
    with pytest.raises(ValueError):
        Tokenizer(max_message_bytes=-1)
    with pytest.raises(TypeError):
        Tokenizer(max_message_bytes=1.5)
    assert Tokenizer(max_message_bytes=None).max_message_bytes is None


def test_limit_setters_validate_and_update():
    parser = Parser()
    parser.max_messages = 2
    parser.max_message_bytes = 10
    assert parser.max_messages == 2
    assert parser.max_message_bytes == 10
    with pytest.raises(ValueError):
        parser.max_messages = -1
    with pytest.raises(TypeError):
        parser.max_message_bytes = object()
    # Failed validation must not change the configured value.
    assert parser.max_messages == 2
    assert parser.max_message_bytes == 10
    parser.max_messages = None
    parser.max_message_bytes = None
    assert parser.max_messages is None
    assert parser.max_message_bytes is None


# --------------------------------------------------------------------------
# max_messages: backpressure, locatable errors, no message loss
# --------------------------------------------------------------------------

def test_message_limit_accepts_up_to_boundary():
    parser = Parser(max_messages=2)
    parser.feed(b'\x90\x01\x02')  # 1 message
    parser.feed(b'\x90\x03\x04')  # 2 messages, exactly at limit
    assert parser.pending() == 2

    with pytest.raises(ParserOverflow) as info:
        parser.feed(b'\x90\x05\x06')
    err = info.value
    assert err.kind == 'messages'
    assert err.limit == 2
    assert err.size == 2
    # The third note_on completes on its third byte, relative index 2.
    assert err.byte_index == 2
    assert err.resume_at == 3


def test_message_overflow_keeps_every_completed_message():
    data = b'\x90\x01\x02' * 5
    bounded = Parser(max_messages=2)
    with pytest.raises(ParserOverflow) as first_error:
        bounded.feed(data)
    assert first_error.value.byte_index == 8
    assert first_error.value.resume_at == 9

    # Both slots are full; the third message is retained in the
    # tokenizer and will be delivered on the next feed. Drain, then
    # resume at the reported index to avoid re-feeding its last byte.
    first_batch = bounded.drain_messages()
    assert [m.bytes() for m in first_batch] == \
        [m.bytes() for m in Parser(data[:6])]

    resumed = []
    pos = first_error.value.resume_at
    while True:
        try:
            bounded.feed(data[pos:])
            break
        except ParserOverflow as err:
            assert err.kind == 'messages'
            # The offending byte completed a retained message, so
            # resume strictly after it to avoid a duplicate.
            assert err.resume_at == err.byte_index + 1
            resumed.extend(bounded.drain_messages())
            pos += err.resume_at
    resumed.extend(bounded.drain_messages())

    assert [m.bytes() for m in first_batch + resumed] == \
        [m.bytes() for m in Parser(data)]


def test_zero_message_capacity_signals_backpressure_immediately():
    parser = Parser(max_messages=0)
    with pytest.raises(ParserOverflow) as info:
        parser.feed(b'\xf8')  # single-byte realtime message
    assert info.value.kind == 'messages'
    assert info.value.byte_index == 0
    assert info.value.resume_at == 1
    assert parser.pending() == 0

    # Opening room lets the retained token flow into the queue.
    parser.max_messages = 2
    parser.feed(b'')  # retained token is pumped without new input
    assert parser.pending() == 1
    parser.feed(b'\xf8')
    assert parser.pending() == 2
    assert [m.type for m in parser] == ['clock', 'clock']


def test_byte_index_locates_offending_byte_within_chunk():
    parser = Parser(max_messages=1)
    parser.feed(b'\xf8')  # fills the single slot
    chunk = b'\x00\x00\xf8\x00\xf8'
    with pytest.raises(ParserOverflow) as info:
        parser.feed(chunk)
    # First clock is at index 2; stray data bytes before it produce
    # no message.
    assert info.value.byte_index == 2
    resume_at = info.value.resume_at
    assert resume_at == 3
    parser.drain_messages()
    # chunk[3:] = b'\x00\xf8': the stray data byte (index 0) starts no
    # message, the clock at relative index 1 fills the slot again.
    with pytest.raises(ParserOverflow) as info:
        parser.feed(chunk[resume_at:])
    assert info.value.byte_index == 1
    assert info.value.resume_at == 2


def test_feed_byte_overflow_attributes():
    parser = Parser(max_messages=1)
    parser.feed_byte(0xf8)
    with pytest.raises(ParserOverflow) as info:
        parser.feed_byte(0xf8)
    assert info.value.kind == 'messages'
    assert info.value.byte_index == 0
    assert info.value.resume_at == 1


# --------------------------------------------------------------------------
# max_message_bytes: unfinished message bound
# --------------------------------------------------------------------------

def test_sysex_byte_limit_boundary():
    # Limit 5: F0 + three data bytes + F7 is exactly 5 bytes, accepted.
    parser = Parser(max_message_bytes=5)
    parser.feed(b'\xf0\x01\x02\x03')
    assert parser.pending_bytes() == 4
    parser.feed(b'\xf7')
    msg = parser.get_message()
    assert msg.type == 'sysex'
    assert msg.bin() == b'\xf0\x01\x02\x03\xf7'


def test_sysex_byte_limit_overflow_is_locatable_and_partial_kept():
    parser = Parser(max_message_bytes=4)
    with pytest.raises(ParserOverflow) as info:
        # F0 + 4 data bytes would reach length 5.
        parser.feed(b'\xf0\x01\x02\x03\x04\xf7')
    err = info.value
    assert err.kind == 'bytes'
    assert err.limit == 4
    assert err.size == 5
    assert err.byte_index == 4  # fifth byte of the chunk
    # The offending byte was rejected, so resume points back at it.
    assert err.resume_at == 4
    # Offending byte not stored; parser sits exactly at the limit.
    assert parser.pending_bytes() == 4
    assert parser.pending() == 0


def test_tokenizer_overflow_carries_state():
    tok = Tokenizer(max_message_bytes=2)
    tok.feed(b'\xf0\x01')
    with pytest.raises(TokenizerOverflow) as info:
        tok.feed_byte(0x02)
    assert info.value.limit == 2
    assert info.value.length == 3
    assert info.value.state == 0xf0
    assert tok.pending_bytes() == 2


def test_zero_byte_capacity_rejects_every_multibyte_message():
    parser = Parser(max_message_bytes=0)
    # Single-byte realtime messages are not buffered and still pass.
    parser.feed(b'\xf8')
    assert parser.pending() == 1
    with pytest.raises(ParserOverflow):
        parser.feed(b'\xf0\xf7')  # even an empty sysex needs 2 bytes
    assert parser.pending_bytes() == 1  # F0 retained, F7 rejected
    parser.reset()
    with pytest.raises(ParserOverflow):
        parser.feed(b'\x90\x01\x02')  # channel message needs 3 bytes


def test_channel_message_respects_byte_limit():
    parser = Parser(max_message_bytes=2)
    # A three-byte channel message cannot fit a two-byte budget.
    with pytest.raises(ParserOverflow) as info:
        parser.feed(b'\x90\x01\x02')
    assert info.value.kind == 'bytes'
    assert info.value.byte_index == 2  # third byte rejected
    assert parser.pending_bytes() == 2
    parser.reset()
    assert parser.pending_bytes() == 0
    parser.max_message_bytes = 3
    parser.feed(b'\x90\x01\x02')
    assert parser.get_message().type == 'note_on'


def test_realtime_inside_sysex_does_not_count_toward_byte_limit():
    parser = Parser(max_message_bytes=4)
    # F0, data, clock (realtime, delivered separately), data, F7.
    # The sysex itself is 4 wire bytes (F0 + 2 data + F7); the
    # interleaved realtime message is one byte long and not buffered.
    parser.feed(b'\xf0\x01')
    parser.feed(b'\xf8')
    realtime = parser.get_message()
    assert realtime.type == 'clock'
    parser.feed(b'\x02\xf7')
    sysex = parser.get_message()
    assert sysex.type == 'sysex'
    assert sysex.bin() == b'\xf0\x01\x02\xf7'
    assert parser.pending_bytes() == 0


def test_byte_limit_applies_across_feeds():
    parser = Parser(max_message_bytes=3)
    parser.feed(b'\xf0\x01')
    parser.feed(b'\x02')
    assert parser.pending_bytes() == 3
    with pytest.raises(ParserOverflow) as info:
        parser.feed(b'\x03')
    assert info.value.byte_index == 0
    assert info.value.size == 4  # would-be length after accepting 0x03
    assert info.value.resume_at == 0
    # The end byte is also a byte: at the limit it is rejected, so a
    # message that never fits stays explicit rather than silently
    # truncating.
    with pytest.raises(ParserOverflow):
        parser.feed(b'\xf7')
    parser.reset()
    assert parser.pending_bytes() == 0


# --------------------------------------------------------------------------
# Incremental consumption APIs
# --------------------------------------------------------------------------

def test_drain_messages_batches():
    parser = Parser()
    parser.feed(b'\xf8' * 7)
    first = parser.drain_messages(3)
    second = parser.drain_messages(3)
    assert len(first) == 3 and len(second) == 3
    # One left.
    assert len(parser.drain_messages(10)) == 1
    assert parser.drain_messages() == []
    assert parser.drain_messages(0) == []
    with pytest.raises(ValueError):
        parser.drain_messages(-1)


def test_drain_works_under_bounds():
    parser = Parser(max_messages=3)
    for _ in range(3):
        parser.feed(b'\xf8')
    assert len(parser.drain_messages(2)) == 2
    # One message remains queued; draining creates room for two more.
    parser.feed(b'\xf8\xf8')
    assert parser.pending() == 3
    assert len(parser.drain_messages()) == 3
    assert parser.pending() == 0


def test_consume_callback_delivers_in_order():
    parser = Parser()
    parser.feed(b'\x90\x00\x01\x90\x01\x02\x90\x02\x03')
    seen = []
    delivered = parser.consume(seen.append, limit=2)
    assert delivered == 2
    assert [m.note for m in seen] == [0, 1]
    assert parser.consume(seen.append) == 1
    assert [m.note for m in seen] == [0, 1, 2]


def test_consume_callback_exception_requeues_message():
    parser = Parser()
    parser.feed(b'\x90\x00\x01\x90\x01\x02')

    class Boom(Exception):
        pass

    def callback(msg):
        raise Boom()

    with pytest.raises(Boom):
        parser.consume(callback)
    # Nothing lost: the failed message is back at the head, the second
    # one is still queued behind it.
    assert parser.pending() == 2
    head = parser.get_message()
    assert head.note == 0
    assert parser.get_message().note == 1


def test_consume_rejects_negative_limit():
    parser = Parser()
    with pytest.raises(ValueError):
        parser.consume(lambda msg: None, limit=-1)


# --------------------------------------------------------------------------
# Deterministic states: empty input, reuse, reset, parser semantics
# --------------------------------------------------------------------------

def test_empty_feed_is_deterministic():
    for kwargs in ({}, {'max_messages': 0}, {'max_message_bytes': 0},
                   {'max_messages': 0, 'max_message_bytes': 0}):
        parser = Parser(**kwargs)
        parser.feed(b'')
        parser.feed([])
        parser.feed_byte  # sanity: method exists
        assert parser.pending() == 0
        assert parser.pending_bytes() == 0
        assert parser.get_message() is None
        assert parser.drain_messages() == []
        assert parser.consume(lambda msg: None) == 0


def test_reset_returns_to_fresh_state_but_keeps_limits():
    parser = Parser(max_messages=1, max_message_bytes=4)
    parser.feed(b'\xf0\x01\x02')
    parser.feed(b'\xf8')
    parser.reset()
    assert parser.pending() == 0
    assert parser.pending_bytes() == 0
    assert parser.max_messages == 1
    assert parser.max_message_bytes == 4

    # The tokenizer's status byte was cleared: stray data bytes are
    # ignored again instead of extending a half-finished message.
    parser.feed(b'\x03\x04')
    assert parser.pending() == 0

    # Limits still enforced after reset.
    with pytest.raises(ParserOverflow):
        parser.feed(b'\xf0\x01\x02\x03\x04')


def test_parser_reuse_after_message_overflow():
    parser = Parser(max_messages=1)
    with pytest.raises(ParserOverflow):
        parser.feed(b'\xf8\xf8')
    parser.drain_messages()
    # The retained token pumps on the next feed without new bytes.
    parser.feed(b'')
    assert parser.pending() == 1
    parser.drain_messages()
    parser.feed(b'\xf8')
    assert parser.pending() == 1
    parser.reset()
    parser.feed(b'\x90\x40\x50')
    assert parser.get_message() == \
        Message('note_on', note=0x40, velocity=0x50)


def test_message_split_across_feeds_matches_unbounded_parser():
    # The stream tokenizer does not implement MIDI running status
    # (only the MIDI file reader does); a bounded parser must keep
    # exactly the same split-message semantics as the unbounded one.
    bounded = Parser(max_messages=10)
    bounded.feed(b'\x90\x01')
    bounded.feed(b'\x02')  # final data byte completes the note_on
    reference = Parser(b'\x90\x01\x02')
    assert [m.bytes() for m in bounded] == \
        [m.bytes() for m in reference]


def test_stray_data_bytes_treated_identically_to_unbounded_parser():
    # Statusless data bytes after a completed message are stray and
    # ignored, in both bounded and unbounded parsers.
    data = b'\x90\x01\x02\x03\x04'
    bounded = Parser(max_messages=10)
    bounded.feed(data)
    assert [m.bytes() for m in bounded] == \
        [m.bytes() for m in Parser(data)]


def test_realtime_insertion_ordering_preserved_when_bounded():
    data = b'\xf0\x00\x00\xfb\x00\x00\xf7'
    bounded = Parser(max_messages=10, max_message_bytes=10)
    # Split at arbitrary points, including right around the realtime.
    for i in range(len(data)):
        parser = Parser(max_messages=10, max_message_bytes=10)
        parser.feed(data[:i])
        parser.feed(data[i:])
        assert [m.type for m in parser] == ['continue', 'sysex']


# --------------------------------------------------------------------------
# Randomized parity: bounded incremental parsing == unbounded parser
# --------------------------------------------------------------------------

def _make_stream(rng, n_messages, max_sysex_len=64):
    """Build a raw MIDI stream and the messages an unbounded parser
    produces from it. Only well-formed constructs are emitted, mixed
    with occasional realtime bytes. Every channel message carries an
    explicit status byte, since the stream tokenizer does not apply
    MIDI running status."""
    raw = bytearray()
    for i in range(n_messages):
        kind = rng.randrange(6)
        if kind == 0:  # sysex, small enough for the byte bound used
            length = rng.randrange(0, max_sysex_len)
            raw.append(0xf0)
            for _ in range(length):
                if rng.random() < 0.1:
                    raw.append(rng.choice([0xf8, 0xfa, 0xfb, 0xfc]))
                raw.append(rng.randrange(0, 128))
            raw.append(0xf7)
        elif kind == 1:  # realtime
            raw.append(rng.choice([0xf8, 0xfa, 0xfb, 0xfc, 0xfe]))
        elif kind == 2:  # one-byte system common
            raw.append(rng.choice([0xf6]))
        else:  # channel voice, always with an explicit status byte
            status = rng.choice([0x80, 0x90, 0xb0, 0xc0, 0xe0])
            raw.append(status)
            raw.append(rng.randrange(0, 128))
            if status != 0xc0:
                raw.append(rng.randrange(0, 128))
    expected = [m.bytes() for m in Parser(bytes(raw))]
    return bytes(raw), expected


def test_random_chunking_matches_unbounded_parser():
    rng = random.Random(20240917)
    data, expected = _make_stream(rng, 5000)
    bound = rng.randrange(1, 32)
    parser = Parser(max_messages=bound, max_message_bytes=128)

    got = []
    pos = 0
    overflows = 0
    while pos < len(data):
        step = rng.randrange(1, 50)
        try:
            parser.feed(data[pos:pos + step])
            pos += step
        except ParserOverflow as err:
            assert err.kind == 'messages'
            overflows += 1
            got.extend(m.bytes() for m in
                       parser.drain_messages(rng.randrange(1, bound + 1)))
            pos += err.resume_at
    got.extend(m.bytes() for m in parser.drain_messages())

    assert got == expected
    assert overflows > 0  # the bound was genuinely exercised
    assert parser.pending() == 0
    assert parser.pending_bytes() == 0


def test_random_feed_byte_matches_unbounded_parser():
    rng = random.Random(987654321)
    data, expected = _make_stream(rng, 2000)
    parser = Parser(max_messages=8, max_message_bytes=128)
    got = []
    for byte in data:
        try:
            parser.feed_byte(byte)
        except ParserOverflow as err:
            assert err.kind == 'messages'
            # For feed_byte the offending byte was already consumed
            # (resume_at == 1); simply make room and continue with the
            # next byte, never re-feeding the current one.
            assert err.resume_at == 1
            got.extend(m.bytes() for m in parser.drain_messages(4))
    got.extend(m.bytes() for m in parser.drain_messages())
    assert got == expected


def test_random_byte_limit_overflow_never_corrupts_completed_messages():
    rng = random.Random(424242)
    byte_bound = rng.randrange(2, 16)
    parser = Parser(max_message_bytes=byte_bound, max_messages=64)
    delivered = []
    pos = 0
    resets = 0
    data, _ = _make_stream(rng, 3000, max_sysex_len=40)
    while pos < len(data):
        end = min(len(data), pos + rng.randrange(1, 20))
        try:
            parser.feed(data[pos:end])
            pos = end
        except ParserOverflow as err:
            if err.kind == 'bytes':
                # Every delivered sysex must fit the bound; the
                # unfinished one is discarded deliberately. The
                # rejected byte was never consumed, so resume there.
                assert err.resume_at == err.byte_index
                parser.reset()
                resets += 1
                pos += err.resume_at
            else:
                delivered.extend(m.bytes() for m in
                                 parser.drain_messages())
                pos += err.resume_at
    delivered.extend(m.bytes() for m in parser.drain_messages())

    reference = [m.bytes() for m in Parser(data)]
    # Delivered messages must be an in-order subsequence of what the
    # unbounded parser finds, and every sysex in it within the bound.
    matched = 0
    for token in delivered:
        while matched < len(reference) and reference[matched] != token:
            matched += 1
        assert matched < len(reference), f'unknown message {token!r}'
        matched += 1
        if token[0] == 0xf0:
            assert len(token) <= byte_bound
    assert resets > 0
    assert parser.pending_bytes() <= byte_bound
