# SPDX-License-Identifier: MIT

"""Tests for the instance-based meta spec registry and its isolation."""
import threading

import pytest

from mido.frozen import FrozenMetaMessage, freeze_message
from mido.midifiles.meta import (
    _REGISTRY,
    MetaMessage,
    MetaSpec,
    MetaSpecRegistry,
    UnknownMetaMessage,
    add_meta_spec,
    build_meta_message,
    remove_meta_spec,
)

PRIVATE_BYTE = 0x0A


class MetaSpec_color(MetaSpec):
    type_byte = PRIVATE_BYTE
    attributes = ['r', 'g', 'b']  # noqa: RUF012 - spec API uses list class attrs
    defaults = [0, 0, 0]  # noqa: RUF012 - spec API uses list class attrs

    def decode(self, message, data):
        message.r, message.g, message.b = data

    def encode(self, message):
        return [message.r, message.g, message.b]

    def check(self, name, value):
        if not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError(f'{name} must be a byte')


class MetaSpec_alt_color(MetaSpec):
    type_byte = PRIVATE_BYTE
    attributes = ['packed']  # noqa: RUF012 - spec API uses list class attrs
    defaults = [0]  # noqa: RUF012 - spec API uses list class attrs

    def decode(self, message, data):
        message.packed = tuple(data)

    def encode(self, message):
        return list(message.packed)


@pytest.fixture
def color_registry():
    registry = MetaSpecRegistry.builtin()
    registry.add(MetaSpec_color)
    return registry


# ---------------------------------------------------------- builtin reuse

def test_builtin_registry_contains_all_builtin_specs():
    registry = MetaSpecRegistry.builtin()
    # A representative sample across the spec range.
    for name in ('track_name', 'set_tempo', 'end_of_track',
                 'key_signature', 'sequencer_specific'):
        assert registry.spec_for_type(name) is not None
    assert registry.spec_for(0x03).type == 'track_name'


def test_registries_get_independent_spec_instances():
    first = MetaSpecRegistry.builtin()
    second = MetaSpecRegistry.builtin()
    spec_a = first.spec_for_type('track_name')
    spec_b = second.spec_for_type('track_name')
    assert spec_a is not spec_b
    assert type(spec_a) is type(spec_b)


def test_registry_copy_is_independent():
    registry = MetaSpecRegistry.builtin()
    clone = registry.copy()
    clone.add(MetaSpec_color)

    assert 'color' not in registry
    assert 'color' in clone
    assert clone.spec_for_type('track_name') is not \
        registry.spec_for_type('track_name')


# ------------------------------------------------------------- isolation

def test_custom_spec_is_scoped_to_its_registry(color_registry):
    # The global/default view must not see the private registration.
    assert color_registry.spec_for(PRIVATE_BYTE) is not None
    assert _REGISTRY.spec_for(PRIVATE_BYTE) is None

    msg = color_registry.build_meta_message(PRIVATE_BYTE, [1, 2, 3])
    assert msg.type == 'color'
    assert (msg.r, msg.g, msg.b) == (1, 2, 3)

    default = build_meta_message(PRIVATE_BYTE, [1, 2, 3])
    assert isinstance(default, UnknownMetaMessage)
    assert default.data == (1, 2, 3)


def test_build_meta_message_accepts_type_name(color_registry):
    # Legacy behavior: a registered type name resolves too.
    msg = color_registry.build_meta_message('color', [1, 2, 3])
    assert msg.type == 'color'
    assert (msg.r, msg.g, msg.b) == (1, 2, 3)


def test_two_conflicting_plugins_get_distinct_views():
    reg_a = MetaSpecRegistry.builtin()
    reg_a.add(MetaSpec_color)
    reg_b = MetaSpecRegistry.builtin()
    reg_b.add(MetaSpec_alt_color)

    msg_a = reg_a.build_meta_message(PRIVATE_BYTE, [1, 2, 3])
    msg_b = reg_b.build_meta_message(PRIVATE_BYTE, [1, 2, 3])

    assert hasattr(msg_a, 'r') and not hasattr(msg_a, 'packed')
    assert hasattr(msg_b, 'packed') and not hasattr(msg_b, 'r')
    # Identical input still serializes to identical bytes.
    assert msg_a.bytes() == msg_b.bytes()
    # Yet the attribute models exposed to the plugins differ.
    assert msg_a.dict() != msg_b.dict()


def test_messages_keep_working_after_registry_changes(color_registry):
    msg = color_registry.build_meta_message(PRIVATE_BYTE, [1, 2, 3])
    color_registry.remove(MetaSpec_color)

    # The already-decoded message stays fully functional on its bound spec.
    assert msg.bytes() == [0xFF, PRIVATE_BYTE, 3, 1, 2, 3]
    msg.b = 200
    assert msg.bytes() == [0xFF, PRIVATE_BYTE, 3, 1, 2, 200]
    assert msg.copy().bytes() == msg.bytes()

    # New decodes through that registry fall back to unknown.
    fallback = color_registry.build_meta_message(PRIVATE_BYTE, [1, 2, 3])
    assert isinstance(fallback, UnknownMetaMessage)


# ------------------------------------------------- duplicate registration

def test_readding_same_class_is_idempotent(color_registry):
    before = color_registry.spec_for(PRIVATE_BYTE)
    returned = color_registry.add(MetaSpec_color)
    after = color_registry.spec_for(PRIVATE_BYTE)
    assert before is after is returned


def test_duplicate_type_name_is_rejected():
    registry = MetaSpecRegistry.builtin()
    registry.add(MetaSpec_color)
    with pytest.raises(ValueError, match='already registered'):
        registry.add(MetaSpec_alt_color)
    # The failed registration must leave the original mapping intact.
    assert isinstance(registry.spec_for(PRIVATE_BYTE), MetaSpec_color)


class MetaSpec_name_clash(MetaSpec):
    # Different byte, but a type name already taken by a built-in.
    type_byte = 0x0B
    attributes = []  # noqa: RUF012 - spec API uses list class attrs
    defaults = []  # noqa: RUF012 - spec API uses list class attrs

    def decode(self, message, data):
        pass

    def encode(self, message):
        return []


# Set the name explicitly to collide with the built-in track_name spec
# while using an unclaimed type byte, so the *name* conflict is what is
# detected (name lookup happens before byte lookup).
MetaSpec_name_clash.type = 'track_name'


def test_duplicate_type_name_is_rejected_on_builtin():
    registry = MetaSpecRegistry.builtin()
    with pytest.raises(ValueError, match="named 'track_name'"):
        registry.add(MetaSpec_name_clash)
    assert registry.spec_for(0x0B) is None
    assert registry.spec_for_type('track_name').type_byte == 0x03


class MetaSpec_byte_clash(MetaSpec):
    # Unique name but the same byte as the color spec.
    type = 'shade'
    type_byte = PRIVATE_BYTE
    attributes = ['level']  # noqa: RUF012 - spec API uses list class attrs
    defaults = [0]  # noqa: RUF012 - spec API uses list class attrs

    def decode(self, message, data):
        message.level = data[0]

    def encode(self, message):
        return [message.level]


def test_duplicate_type_byte_with_unique_name_is_rejected(color_registry):
    with pytest.raises(ValueError, match='type byte'):
        color_registry.add(MetaSpec_byte_clash)
    # Original spec remains the owner of the byte.
    assert isinstance(color_registry.spec_for(PRIVATE_BYTE), MetaSpec_color)
    assert color_registry.spec_for_type('shade') is None


def test_remove_unknown_class_raises():
    registry = MetaSpecRegistry.builtin()
    with pytest.raises(KeyError):
        registry.remove(MetaSpec_color)


# ------------------------------------------------------------- concurrency

def test_concurrent_registration_is_race_free():
    errors = []

    def worker(registry):
        try:
            for _ in range(200):
                registry.add(MetaSpec_color)
                assert registry.spec_for(PRIVATE_BYTE) is not None
        except Exception as exc:  # noqa: BLE001 - surface any thread error
            errors.append(exc)

    registry = MetaSpecRegistry.builtin()
    threads = [threading.Thread(target=worker, args=(registry,))
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert isinstance(registry.spec_for(PRIVATE_BYTE), MetaSpec_color)


def test_concurrent_conflicting_registrations_have_one_winner():
    outcomes = []
    lock = threading.Lock()
    registry = MetaSpecRegistry.builtin()

    def register(spec_class):
        try:
            registry.add(spec_class)
        except ValueError as exc:
            with lock:
                outcomes.append(('rejected', str(exc)))
        else:
            with lock:
                outcomes.append(('accepted', spec_class))

    threads = [
        threading.Thread(target=register, args=(MetaSpec_color,)),
        threading.Thread(target=register, args=(MetaSpec_alt_color,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(status for status, _ in outcomes) == ['accepted', 'rejected']


# ------------------------------------------------------- legacy global API

def test_legacy_add_meta_spec_uses_default_registry():
    add_meta_spec(MetaSpec_color)
    try:
        assert isinstance(_REGISTRY.spec_for(PRIVATE_BYTE), MetaSpec_color)
        # Global construction now resolves the custom type.
        msg = MetaMessage('color', r=1, g=2, b=3)
        assert msg.bytes() == [0xFF, PRIVATE_BYTE, 3, 1, 2, 3]
    finally:
        remove_meta_spec(MetaSpec_color)
    assert _REGISTRY.spec_for(PRIVATE_BYTE) is None


def test_unknown_meta_fallback_path_unchanged():
    msg = build_meta_message(PRIVATE_BYTE, [9, 8, 7], delta=4)
    assert isinstance(msg, UnknownMetaMessage)
    assert msg.type_byte == PRIVATE_BYTE
    assert msg.data == (9, 8, 7)
    assert msg.time == 4
    # Round-trips the raw payload verbatim.
    assert msg.bytes() == [0xFF, PRIVATE_BYTE, 3, 9, 8, 7]


def test_unknown_meta_message_copy_supports_overrides():
    # UnknownMetaMessage has no bound spec; copy() must not assume one.
    original = UnknownMetaMessage(99, [1, 2, 3], time=10)
    plain = original.copy()
    assert plain == original

    overridden = original.copy(time=5)
    assert isinstance(overridden, UnknownMetaMessage)
    assert overridden.time == 5
    assert overridden.type_byte == 99
    assert overridden.data == (1, 2, 3)


# --------------------------------------------- message contract stability

def test_public_serialization_hides_internal_spec():
    msg = MetaMessage('track_name', name='Test', time=2)
    assert '_spec' not in msg.dict()
    assert vars(msg)['_spec'] is not None  # bound internally
    assert eval(repr(msg)) == msg  # noqa: S307


def test_equality_ignores_registry_identity(color_registry):
    via_global = MetaMessage('track_name', name='Test')
    via_session = color_registry.build_meta_message(0x03, list(b'Test'))
    assert via_global == via_session


def test_frozen_meta_hash_is_stable_across_registries(color_registry):
    via_global = freeze_message(MetaMessage('track_name', name='Test'))
    via_session = freeze_message(
        color_registry.build_meta_message(0x03, list(b'Test')))
    assert isinstance(via_global, FrozenMetaMessage)
    assert hash(via_global) == hash(via_session)
    assert via_global == via_session


def test_copy_preserves_bound_spec(color_registry):
    msg = color_registry.build_meta_message(PRIVATE_BYTE, [1, 2, 3])
    plain_copy = msg.copy()
    assert plain_copy.bytes() == msg.bytes()

    overridden = msg.copy(g=99)
    assert overridden.g == 99
    assert overridden.bytes() == [0xFF, PRIVATE_BYTE, 3, 1, 99, 3]


def test_from_bytes_accepts_registry(color_registry):
    raw = [0xFF, PRIVATE_BYTE, 3, 1, 2, 3]
    session_msg = MetaMessage.from_bytes(raw, registry=color_registry)
    assert session_msg.type == 'color'
    assert (session_msg.r, session_msg.g, session_msg.b) == (1, 2, 3)

    default_msg = MetaMessage.from_bytes(raw)
    assert isinstance(default_msg, UnknownMetaMessage)


def test_builtin_constructor_still_resolves_through_default_registry():
    msg = MetaMessage('set_tempo', tempo=123)
    assert msg.bytes() == [0xFF, 0x51, 3, 0, 0, 123]
    # The bound spec is the default registry instance.
    assert vars(msg)['_spec'] is _REGISTRY.spec_for_type('set_tempo')
