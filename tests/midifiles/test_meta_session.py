# SPDX-License-Identifier: MIT

"""End-to-end isolation tests through MidiFile read/write sessions."""
import io

from mido.messages import Message
from mido.midifiles.meta import (
    MetaMessage,
    MetaSpec,
    MetaSpecRegistry,
    UnknownMetaMessage,
)
from mido.midifiles.midifiles import MidiFile, MidiTrack

PRIVATE_BYTE = 0x0A


class MetaSpec_rgb(MetaSpec):
    type_byte = PRIVATE_BYTE
    attributes = ['r', 'g', 'b']  # noqa: RUF012 - spec API uses list class attrs
    defaults = [0, 0, 0]  # noqa: RUF012 - spec API uses list class attrs

    def decode(self, message, data):
        message.r, message.g, message.b = data

    def encode(self, message):
        return [message.r, message.g, message.b]


class MetaSpec_packed(MetaSpec):
    type_byte = PRIVATE_BYTE
    attributes = ['packed']  # noqa: RUF012 - spec API uses list class attrs
    defaults = [()]  # noqa: RUF012 - spec API uses list class attrs

    def decode(self, message, data):
        message.packed = tuple(data)

    def encode(self, message):
        return list(message.packed)


def make_file_with_private_meta():
    """Build a MIDI file whose private 0x0A payload is [1, 2, 3]."""
    track = MidiTrack([
        MetaMessage('track_name', name='Session', time=0),
        UnknownMetaMessage(PRIVATE_BYTE, [1, 2, 3], time=0),
        Message('note_on', note=64, velocity=64, time=5),
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(type=1, ticks_per_beat=480, tracks=[track])
    buf = io.BytesIO()
    mid.save(file=buf)
    buf.seek(0)
    return buf


def test_default_session_degrades_to_unknown():
    mid = MidiFile(file=make_file_with_private_meta())
    private = [m for m in mid.tracks[0]
               if isinstance(m, UnknownMetaMessage)
               and m.type_byte == PRIVATE_BYTE]
    assert len(private) == 1
    assert private[0].data == (1, 2, 3)


def test_session_registry_decodes_private_type():
    registry = MetaSpecRegistry.builtin()
    registry.add(MetaSpec_rgb)
    mid = MidiFile(file=make_file_with_private_meta(),
                   meta_registry=registry)

    private = [m for m in mid.tracks[0] if m.type == 'rgb']
    assert len(private) == 1
    assert (private[0].r, private[0].g, private[0].b) == (1, 2, 3)


def test_two_sessions_decode_same_bytes_differently():
    reg_rgb = MetaSpecRegistry.builtin()
    reg_rgb.add(MetaSpec_rgb)
    reg_packed = MetaSpecRegistry.builtin()
    reg_packed.add(MetaSpec_packed)

    mid_rgb = MidiFile(file=make_file_with_private_meta(),
                       meta_registry=reg_rgb)
    mid_packed = MidiFile(file=make_file_with_private_meta(),
                          meta_registry=reg_packed)

    rgb_msg = next(m for m in mid_rgb.tracks[0] if m.type == 'rgb')
    packed_msg = next(m for m in mid_packed.tracks[0]
                      if m.type == 'packed')

    assert hasattr(rgb_msg, 'r') and not hasattr(rgb_msg, 'packed')
    assert hasattr(packed_msg, 'packed') and not hasattr(packed_msg, 'r')
    # The wire encoding of identical input is still identical.
    assert rgb_msg.bytes() == packed_msg.bytes()


def test_session_write_back_round_trips_private_payload():
    registry = MetaSpecRegistry.builtin()
    registry.add(MetaSpec_rgb)
    mid = MidiFile(file=make_file_with_private_meta(),
                   meta_registry=registry)

    rgb_msg = next(m for m in mid.tracks[0] if m.type == 'rgb')
    rgb_msg.g = 200

    out = io.BytesIO()
    mid.save(file=out)
    out.seek(0)

    reread = MidiFile(file=out, meta_registry=registry)
    again = next(m for m in reread.tracks[0] if m.type == 'rgb')
    assert (again.r, again.g, again.b) == (1, 200, 3)

    # A default-registry reader sees an opaque unknown message with the
    # mutated raw payload.
    out.seek(0)
    default_view = MidiFile(file=out)
    unknown = next(m for m in default_view.tracks[0]
                   if isinstance(m, UnknownMetaMessage)
                   and m.type_byte == PRIVATE_BYTE)
    assert unknown.data == (1, 200, 3)


def test_session_registry_does_not_touch_default_registry():
    from mido.midifiles.meta import _REGISTRY

    registry = MetaSpecRegistry.builtin()
    registry.add(MetaSpec_rgb)
    try:
        assert _REGISTRY.spec_for(PRIVATE_BYTE) is None
        assert registry.spec_for(PRIVATE_BYTE) is not None
    finally:
        registry.remove(MetaSpec_rgb)
    assert registry.spec_for(PRIVATE_BYTE) is None
    # Default registry remains untouched throughout.
    assert _REGISTRY.spec_for(PRIVATE_BYTE) is None


def test_write_back_is_byte_stable_for_unknown_messages():
    # The unknown-meta degradation path must preserve the exact bytes
    # of an unmodified file across a default-registry round trip.
    original = make_file_with_private_meta().read()

    mid = MidiFile(file=io.BytesIO(original))
    out = io.BytesIO()
    mid.save(file=out)
    assert out.getvalue() == original
