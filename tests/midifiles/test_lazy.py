# SPDX-FileCopyrightText: 2026 Mido contributors
#
# SPDX-License-Identifier: MIT

"""Tests for on-demand (lazy) track loading."""

import io
import os
import struct
import tempfile

from pytest import raises

from mido.messages import Message
from mido.midifiles.meta import MetaMessage, UnknownMetaMessage
from mido.midifiles.midifiles import LazyTrackList, MidiFile, MidiTrack


def make_track(track_index, messages=20):
    track = MidiTrack()
    track.append(MetaMessage('track_name', name=f'Track {track_index}'))
    track.append(MetaMessage('set_tempo', tempo=500000 + track_index))
    for i in range(messages):
        track.append(Message('note_on', note=60 + i % 60,
                             velocity=64, time=1 + track_index))
        track.append(Message('note_off', note=60 + i % 60,
                             velocity=64, time=1))
    return track


def make_midifile(num_tracks=6, messages=20, type=1):
    return MidiFile(type=type, ticks_per_beat=480, tracks=[
        make_track(i, messages) for i in range(num_tracks)
    ])


def to_bytes(midifile):
    file = io.BytesIO()
    midifile.save(file=file)
    return file.getvalue()


def eager(data):
    return MidiFile(file=io.BytesIO(data))


def lazy(data, **kwargs):
    return MidiFile(file=io.BytesIO(data), lazy=True, **kwargs)


def tracks_as_lists(midifile):
    return [list(track) for track in midifile.tracks]


class CountingFile:
    """Seekable stream that counts the bytes actually read."""

    def __init__(self, data):
        self._file = io.BytesIO(data)
        self.bytes_read = 0
        self.read_calls = 0
        self.closed = False

    def read(self, size=-1):
        data = self._file.read(size)
        self.bytes_read += len(data)
        self.read_calls += 1
        return data

    def tell(self):
        return self._file.tell()

    def seek(self, offset, whence=0):
        return self._file.seek(offset, whence)

    def seekable(self):
        return True

    def close(self):
        self.closed = True
        self._file.close()


class ForwardOnlyStream:
    """Read-only stream with no tell() or seek()."""

    def __init__(self, data):
        self._buffer = io.BytesIO(data)

    def read(self, size=-1):
        return self._buffer.read(size)


def test_lazy_tracks_equal_eager_tracks():
    data = to_bytes(make_midifile())
    eager_mid = eager(data)
    lazy_mid = lazy(data)

    assert len(lazy_mid.tracks) == len(eager_mid.tracks)
    assert tracks_as_lists(lazy_mid) == tracks_as_lists(eager_mid)
    assert lazy_mid.type == eager_mid.type
    assert lazy_mid.ticks_per_beat == eager_mid.ticks_per_beat


def test_initial_scan_does_not_decode_tracks():
    data = to_bytes(make_midifile(num_tracks=8, messages=50))
    file = CountingFile(data)

    mid = MidiFile(file=file, lazy=True)

    # Only the header (14 bytes) and one 8-byte chunk header per track
    # should have been read.
    expected = 14 + 8 * 8
    assert file.bytes_read == expected
    assert file.bytes_read < len(data) / 10
    assert all(not mid.tracks.is_loaded(i) for i in range(8))
    assert isinstance(mid.tracks, LazyTrackList)


def test_accessing_one_track_only_decodes_that_track():
    data = to_bytes(make_midifile(num_tracks=8, messages=50))
    file = CountingFile(data)
    mid = MidiFile(file=file, lazy=True)
    scanned = file.bytes_read

    track = mid.tracks[3]
    first_read = file.bytes_read - scanned

    # 2 meta messages + 50 note_on/note_off pairs + end_of_track.
    assert len(track) == 2 + 2 * 50 + 1
    assert mid.tracks.is_loaded(3)
    assert not mid.tracks.is_loaded(0)
    assert not mid.tracks.is_loaded(7)

    # Re-accessing uses the cache and reads nothing.
    assert mid.tracks[3] is track
    assert file.bytes_read - scanned == first_read

    # A track is at most about 1/8th of the payload, so this must be
    # far below the full file size.
    assert first_read < len(data) / 4


def test_negative_index_and_slice_resolve_tracks():
    data = to_bytes(make_midifile(num_tracks=4))
    mid = lazy(data)

    assert mid.tracks[-1].name == 'Track 3'
    assert [track.name for track in mid.tracks[1:3]] == [
        'Track 1', 'Track 2']


def test_track_order_is_preserved():
    data = to_bytes(make_midifile(num_tracks=6))
    mid = lazy(data)

    # Decode in a scrambled order; iteration must still be in file order.
    for index in [5, 0, 3, 1, 4, 2]:
        assert mid.tracks[index].name == f'Track {index}'
    assert [track.name for track in mid.tracks] == [
        f'Track {i}' for i in range(6)]


def test_running_status_and_unknown_meta_match_eager():
    # Track with running status (note_on then note_off reusing status)
    # and a raw unknown meta message byte (0x7e is not a built-in type).
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='running'))
    track.append(UnknownMetaMessage(type_byte=0x7e, data=[1, 2, 3]))
    track.append(Message('note_on', note=60, velocity=64, time=0))
    track.append(Message('note_on', note=62, velocity=64, time=10))
    track.append(Message('note_off', note=62, velocity=0, time=10))
    track.append(Message('note_off', note=60, velocity=0, time=10))

    data = to_bytes(MidiFile(type=1, tracks=[track]))
    # Sanity check: the saved bytes really use running status.
    assert data.count(bytes([0x90])) == 1

    eager_mid = eager(data)
    lazy_mid = lazy(data)
    assert tracks_as_lists(lazy_mid) == tracks_as_lists(eager_mid)
    assert isinstance(lazy_mid.tracks[0][1], UnknownMetaMessage)


def test_truncated_track_is_rejected_at_open_time():
    data = bytearray(to_bytes(make_midifile(num_tracks=2)))
    # Extend the first track's declared chunk size beyond the end of the
    # file.
    first_track_header = data.index(b'MTrk')
    data[first_track_header + 4:first_track_header + 8] = struct.pack(
        '>L', len(data))

    with raises(EOFError):
        lazy(bytes(data))


def test_missing_track_header_is_rejected_at_open_time():
    data = bytearray(to_bytes(make_midifile(num_tracks=2)))
    track_header = data.index(b'MTrk')
    data[track_header:track_header + 4] = b'XXXX'

    with raises(OSError):
        lazy(bytes(data))


def test_save_read_back_roundtrip():
    data = to_bytes(make_midifile(num_tracks=5))
    mid = lazy(data)

    # Force a partial cache before saving.
    mid.tracks[2]

    out = io.BytesIO()
    mid.save(file=out)

    assert out.getvalue() == data


def make_note_only_midifile(num_tracks=4, messages=20):
    tracks = []
    for k in range(num_tracks):
        track = MidiTrack()
        for i in range(messages):
            track.append(Message('note_on', note=60 + i,
                                 velocity=64, time=1 + k))
            track.append(Message('note_off', note=60 + i,
                                 velocity=64, time=1))
        tracks.append(track)
    return MidiFile(type=1, ticks_per_beat=480, tracks=tracks)


def test_note_only_tracks_match_eager():
    data = to_bytes(make_note_only_midifile(num_tracks=4))
    assert tracks_as_lists(lazy(data)) == tracks_as_lists(eager(data))


def test_lazy_filename_does_not_leak_file_descriptors():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'song.mid')
        with open(path, 'wb') as file:
            file.write(to_bytes(make_midifile(num_tracks=5)))

        before = len(os.listdir('/proc/self/fd'))
        mid = MidiFile(path, lazy=True)
        after_open = len(os.listdir('/proc/self/fd'))

        assert after_open == before  # no descriptor kept by lazy scan

        mid.tracks[4]
        assert len(os.listdir('/proc/self/fd')) == before

        assert len(mid.tracks) == 5
        assert len(os.listdir('/proc/self/fd')) == before


def test_lazy_filename_close_keeps_decoded_tracks():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'song.mid')
        with open(path, 'wb') as file:
            file.write(to_bytes(make_midifile(num_tracks=4)))

        mid = MidiFile(path, lazy=True)
        decoded = mid.tracks[0]
        mid.close()

        assert mid.tracks[0] is decoded
        with raises(ValueError):
            mid.tracks[1]


def test_lazy_file_object_keeps_position_and_does_not_close():
    data = to_bytes(make_midifile(num_tracks=4))
    file = io.BytesIO(data)
    position = file.tell()

    mid = MidiFile(file=file, lazy=True)

    assert file.tell() == position
    assert not file.closed

    mid.tracks[2]
    assert file.tell() == position

    mid.close()
    assert not file.closed
    assert len(mid.tracks) == 4


def test_closed_file_object_raises_for_uncached_tracks():
    data = to_bytes(make_midifile(num_tracks=4))
    file = io.BytesIO(data)
    mid = MidiFile(file=file, lazy=True)
    cached = mid.tracks[0]
    file.close()

    assert mid.tracks[0] is cached
    with raises(ValueError):
        mid.tracks[1]


def test_forward_only_stream_is_supported_by_spooling():
    data = to_bytes(make_midifile(num_tracks=4))
    mid = MidiFile(file=ForwardOnlyStream(data), lazy=True)

    assert tracks_as_lists(mid) == tracks_as_lists(eager(data))
    mid.close()


def test_lazy_track_list_mutation():
    data = to_bytes(make_midifile(num_tracks=3))
    mid = lazy(data)

    extra = MidiTrack([Message('note_on', note=1, time=0)])
    mid.tracks.append(extra)
    assert len(mid.tracks) == 4
    assert mid.tracks[3] is extra

    del mid.tracks[0]
    assert [track.name for track in mid.tracks] == ['Track 1', 'Track 2', '']

    mid.tracks[1] = extra
    assert mid.tracks[1] is extra
    assert mid.tracks[0].name == 'Track 1'


def test_eager_mode_uses_plain_list():
    data = to_bytes(make_midifile(num_tracks=2))
    mid = MidiFile(file=io.BytesIO(data))
    assert type(mid.tracks) is list


def test_concurrent_track_access_decodes_once():
    from concurrent.futures import ThreadPoolExecutor

    data = to_bytes(make_midifile(num_tracks=8, messages=30))
    mid = lazy(data)

    # Count decodes by wrapping the loader.
    loader = mid.tracks._loader
    calls = []

    class CountingLoader:
        def __call__(self, entry):
            calls.append(entry.offset)
            return loader(entry)

    mid.tracks._loader = CountingLoader()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: mid.tracks[i % 8], range(64)))

    assert len(calls) == 8
    assert all(results[i] is results[i % 8] for i in range(64))
    assert len(mid.tracks) == 8


def test_decode_error_is_raised_lazily():
    data = bytearray(to_bytes(make_midifile(num_tracks=2)))
    # Corrupt a data byte in the first track so decoding fails, while the
    # chunk directory itself is still valid.
    start = data.index(b'MTrk')
    note_on = data.index(bytes([0x90]), start)
    data[note_on + 2] = 0xFF

    mid = lazy(bytes(data))  # scan must not decode the bad track
    assert not mid.tracks.is_loaded(0)

    with raises(OSError):
        mid.tracks[0]

    # Other tracks remain readable and the failed track was not cached
    # as a broken value.
    assert mid.tracks[1].name == 'Track 1'
    assert not mid.tracks.is_loaded(0)


def test_lazy_type_0_and_empty_tracks():
    empty = MidiFile(type=0, ticks_per_beat=96, tracks=[MidiTrack()])
    data = to_bytes(empty)
    mid = lazy(data)
    assert mid.type == 0
    assert mid.ticks_per_beat == 96
    assert len(mid.tracks) == 1
    assert list(mid.tracks[0]) == [
        MetaMessage('end_of_track', time=0)]


def test_lazy_no_tracks():
    data = to_bytes(MidiFile(type=1, tracks=[]))
    mid = lazy(data)
    assert len(mid.tracks) == 0
    assert list(mid.tracks) == []
    assert mid.tracks == []


def test_save_after_partial_cache_matches_eager():
    data = to_bytes(make_midifile(num_tracks=5))
    mid = lazy(data)
    with raises(IndexError):
        mid.tracks[5]

    out = io.BytesIO()
    mid.save(file=out)
    assert out.getvalue() == data


def test_save_after_close_with_uncached_tracks_raises():
    data = to_bytes(make_midifile(num_tracks=5))
    mid = lazy(data)
    mid.tracks[0]
    mid.close()

    with raises(ValueError):
        mid.save(file=io.BytesIO())


def test_close_then_load_all_raises():
    data = to_bytes(make_midifile(num_tracks=3))
    mid = lazy(data)
    mid.close()
    with raises(ValueError):
        mid.tracks.load_all()


def test_lazy_tracks_equality_with_list():
    data = to_bytes(make_midifile(num_tracks=3))
    mid = lazy(data)
    assert mid.tracks == [mid.tracks[0], mid.tracks[1], mid.tracks[2]]
    assert mid.tracks != []
    assert list(mid.tracks) == list(eager(data).tracks)


def test_lazy_tracks_repr_does_not_decode_tracks():
    data = to_bytes(make_midifile(num_tracks=2, messages=1))
    mid = lazy(data)
    text = repr(mid.tracks)
    assert 'not loaded' in text
    assert not mid.tracks.is_loaded(0)

    mid.tracks[0]
    assert not mid.tracks.is_loaded(1)
    assert 'note_on' in repr(mid.tracks)


def test_lazy_tracks_repr_after_close_does_not_raise():
    data = to_bytes(make_midifile(num_tracks=2, messages=1))
    mid = lazy(data)
    mid.close()
    assert 'not loaded' in repr(mid.tracks)


def test_midifile_repr_does_not_decode_lazy_tracks():
    data = to_bytes(make_midifile(num_tracks=3, messages=1))
    mid = lazy(data)
    assert 'not loaded' in repr(mid)
    assert not mid.tracks.is_loaded(0)


def test_lazy_path_source_reopens_file():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'song.mid')
        with open(path, 'wb') as file:
            file.write(to_bytes(make_midifile(num_tracks=3)))

        mid = MidiFile(path, lazy=True)
        assert mid.tracks[0].name == 'Track 0'

        # A fresh descriptor is opened for the next track, so replacing
        # the file on disk affects uncached tracks only.
        os.remove(path)
        with raises(OSError):
            mid.tracks[1]
        assert mid.tracks[0].name == 'Track 0'


def test_lazy_track_list_iteration_is_in_file_order_after_mutation():
    data = to_bytes(make_midifile(num_tracks=4))
    mid = lazy(data)
    mid.tracks[1]
    mid.tracks.append(MidiTrack([Message('note_on', note=1, time=0)]))

    names = [track.name for track in mid.tracks]
    assert names == ['Track 0', 'Track 1', 'Track 2', 'Track 3', '']


def test_lazy_debug_mode_matches_eager(capsys):
    data = to_bytes(make_midifile(num_tracks=2, messages=2))
    eager_mid = MidiFile(file=io.BytesIO(data), debug=True)
    eager_output = capsys.readouterr().out

    lazy_mid = lazy(data, debug=True)
    lazy_mid.tracks.load_all()
    lazy_output = capsys.readouterr().out

    assert tracks_as_lists(lazy_mid) == tracks_as_lists(eager_mid)
    # Debug mode is noisy but must not change the decoded result; check
    # that it printed the same number of message lines.
    assert lazy_output.count('Message:') == eager_output.count('Message:')


def test_stream_positioned_mid_file():
    data = to_bytes(make_midifile(num_tracks=3))
    prefix = b'junk-prefix\x00\x01\x02'
    file = io.BytesIO(prefix + data)
    file.seek(len(prefix))

    eager_mid = MidiFile(file=file)
    file.seek(len(prefix))
    lazy_mid = MidiFile(file=file, lazy=True)

    assert tracks_as_lists(lazy_mid) == tracks_as_lists(eager_mid)
    assert file.tell() == len(prefix)


def test_initial_peak_memory_is_lower_than_eager():
    import tracemalloc

    data = to_bytes(make_midifile(num_tracks=200, messages=100))
    assert len(data) > 100_000

    tracemalloc.start()
    eager_mid = eager(data)
    _, eager_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    lazy_mid = lazy(data)
    _, lazy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The lazy construction keeps only a directory of offsets; it must
    # use far less memory than decoding every track. The eager object is
    # still alive during the lazy measurement, so this is conservative.
    assert lazy_peak < eager_peak / 4
    assert len(lazy_mid.tracks) == len(eager_mid.tracks) == 200
    assert not any(lazy_mid.tracks.is_loaded(i) for i in range(200))


