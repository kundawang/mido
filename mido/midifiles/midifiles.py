# SPDX-FileCopyrightText: 2016 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
MIDI file reading and playback.

References:

http://home.roadrunner.com/~jgglatt/
http://home.roadrunner.com/~jgglatt/tech/miditech.htm
http://home.roadrunner.com/~jgglatt/tech/midifile.htm

http://www.sonicspot.com/guide/midifiles.html
http://www.ccarh.org/courses/253/assignment/midifile/
https://code.google.com/p/binasc/wiki/mainpage
http://stackoverflow.com/questions/2984608/midi-delta-time
http://www.recordingblogs.com/sa/tabid/82/EntryId/44/MIDI-Part-XIII-Delta-time-a
http://www.sonicspot.com/guide/midifiles.html
"""

import operator
import shutil
import string
import struct
import tempfile
import threading
import time
from collections.abc import MutableSequence
from contextlib import contextmanager
from numbers import Integral

from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second

# The default tempo is 120 BPM.
# (500000 microseconds per beat (quarter note).)
DEFAULT_TEMPO = 500000
DEFAULT_TICKS_PER_BEAT = 480

# Maximum message length to attempt to read.
MAX_MESSAGE_LENGTH = 1000000


def print_byte(byte, pos=0):
    char = chr(byte)
    if char.isspace() or char not in string.printable:
        char = '.'

    print(f'  {pos:06x}: {byte:02x}  {char}')


class DebugFileWrapper:
    def __init__(self, file):
        self.file = file

    def read(self, size):
        data = self.file.read(size)

        for byte in data:
            print_byte(byte, self.file.tell())

        return data

    def tell(self):
        return self.file.tell()

    def seek(self, offset, whence=0):
        return self.file.seek(offset, whence)


def read_byte(self):
    byte = self.read(1)
    if byte == b'':
        raise EOFError
    else:
        return ord(byte)


def read_bytes(infile, size):
    if size > MAX_MESSAGE_LENGTH:
        raise OSError('Message length {} exceeds maximum length {}'.format(
            size, MAX_MESSAGE_LENGTH))
    return [read_byte(infile) for _ in range(size)]


def _dbg(text=''):
    print(text)


# We can't use the chunk module for two reasons:
#
# 1. we may have mixed big and little endian chunk sizes. (RIFF is
# little endian while MTrk is big endian.)
#
# 2. the chunk module assumes that chunks are padded to the nearest
# multiple of 2. This is not true of MIDI files.

def read_chunk_header(infile):
    header = infile.read(8)
    if len(header) < 8:
        raise EOFError

    # TODO: check for b'RIFF' and switch endian?

    return struct.unpack('>4sL', header)


def read_file_header(infile):
    name, size = read_chunk_header(infile)

    if name != b'MThd':
        raise OSError('MThd not found. Probably not a MIDI file')
    else:
        data = infile.read(size)

        if len(data) < 6:
            raise EOFError

        return struct.unpack('>hhh', data[:6])


def read_message(infile, status_byte, peek_data, delta, clip=False):
    try:
        spec = SPEC_BY_STATUS[status_byte]
    except LookupError:
        raise OSError(f'undefined status byte 0x{status_byte:02x}')

    # Subtract 1 for status byte.
    size = spec['length'] - 1 - len(peek_data)
    data_bytes = peek_data + read_bytes(infile, size)

    if clip:
        data_bytes = [byte if byte < 127 else 127 for byte in data_bytes]
    else:
        for byte in data_bytes:
            if byte > 127:
                raise OSError('data byte must be in range 0..127')

    return Message.from_bytes([status_byte] + data_bytes, time=delta)


def read_sysex(infile, delta, clip=False):
    length = read_variable_int(infile)
    data = read_bytes(infile, length)

    # Strip start and end bytes.
    # TODO: is this necessary?
    if data and data[0] == 0xf0:
        data = data[1:]
    if data and data[-1] == 0xf7:
        data = data[:-1]

    if clip:
        data = [byte if byte < 127 else 127 for byte in data]

    return Message('sysex', data=data, time=delta)


def read_variable_int(infile):
    delta = 0

    while True:
        byte = read_byte(infile)
        delta = (delta << 7) | (byte & 0x7f)
        if byte < 0x80:
            return delta


def read_meta_message(infile, delta):
    meta_type = read_byte(infile)
    length = read_variable_int(infile)
    data = read_bytes(infile, length)
    return build_meta_message(meta_type, data, delta)


def read_track(infile, debug=False, clip=False):
    track = MidiTrack()

    name, size = read_chunk_header(infile)

    if name != b'MTrk':
        raise OSError('no MTrk header at start of track')

    if debug:
        _dbg(f'-> size={size}')
        _dbg()

    start = infile.tell()
    last_status = None

    while True:
        # End of track reached.
        if infile.tell() - start == size:
            break

        if debug:
            _dbg('Message:')

        delta = read_variable_int(infile)

        if debug:
            _dbg(f'-> delta={delta}')

        status_byte = read_byte(infile)

        if status_byte < 0x80:
            if last_status is None:
                raise OSError('running status without last_status')
            peek_data = [status_byte]
            status_byte = last_status
        else:
            if status_byte != 0xff:
                # Meta messages don't set running status.
                last_status = status_byte
            peek_data = []

        if status_byte == 0xff:
            msg = read_meta_message(infile, delta)
        elif status_byte in [0xf0, 0xf7]:
            # TODO: I'm not quite clear on the difference between
            # f0 and f7 events.
            msg = read_sysex(infile, delta, clip)
        else:
            msg = read_message(infile, status_byte, peek_data, delta, clip)

        track.append(msg)

        if debug:
            _dbg(f'-> {msg!r}')
            _dbg()

    return track


class _TrackEntry:
    """Location of an as-yet undecoded track in the source file.

    ``offset`` points at the first byte of the ``MTrk`` chunk header and
    ``size`` is the declared chunk size (that is, the number of bytes
    following the header).
    """

    __slots__ = ('offset', 'size')

    def __init__(self, offset, size):
        self.offset = offset
        self.size = size

    def __repr__(self):
        return f'_TrackEntry(offset={self.offset}, size={self.size})'


class _RestoringFileView:
    """Temporary seek view over a caller-owned file.

    The position is saved on creation and restored on ``restore()``, so
    reading a lazy track does not move the caller's file position.
    """

    def __init__(self, file):
        self.file = file
        self._position = file.tell()

    def read(self, size):
        return self.file.read(size)

    def tell(self):
        return self.file.tell()

    def seek(self, offset, whence=0):
        return self.file.seek(offset, whence)

    def restore(self):
        try:
            self.file.seek(self._position)
        except (AttributeError, OSError, ValueError):
            pass


class _TrackSource:
    """Base class for lazily readable track sources."""

    #: Human-readable description used in error messages.
    kind = 'source'

    def __init__(self):
        self.closed = False
        self._lock = threading.RLock()

    @contextmanager
    def open(self):
        """Yield a seekable file-like object positioned at the start."""
        if self.closed:
            raise ValueError(
                f'cannot read track: {self.kind} is closed')
        with self._open() as infile:
            yield infile

    @contextmanager
    def _open(self):
        raise NotImplementedError

    def scan(self, debug=False):
        """Read the header and track directory without decoding tracks."""
        with self.open() as infile:
            if debug:
                infile = DebugFileWrapper(infile)
                _dbg('Header:')
            (file_type,
             num_tracks,
             ticks_per_beat) = read_file_header(infile)
            if debug:
                _dbg('-> type={}, tracks={}, ticks_per_beat={}'.format(
                    file_type, num_tracks, ticks_per_beat))
                _dbg()
            entries = _scan_track_directory(infile, num_tracks)
        return file_type, ticks_per_beat, entries

    def decode_track(self, entry, charset, debug=False, clip=False):
        """Decode one track from the source."""
        with self._lock:
            with self.open() as infile:
                infile.seek(entry.offset)
                if debug:
                    infile = DebugFileWrapper(infile)
                    _dbg(f'-> size={entry.size}')
                    _dbg()
                with meta_charset(charset):
                    return read_track(infile, debug=debug, clip=clip)

    def close(self):
        self.closed = True


class _TrackLoader:
    """Callable that decodes one ``_TrackEntry`` from a source.

    Kept separate from :class:`MidiFile` so the loader does not keep a
    reference cycle alive through the track list.
    """

    __slots__ = ('source', 'charset', 'debug', 'clip')

    def __init__(self, source, charset, debug, clip):
        self.source = source
        self.charset = charset
        self.debug = debug
        self.clip = clip

    def __call__(self, entry):
        return self.source.decode_track(
            entry,
            charset=self.charset,
            debug=self.debug,
            clip=self.clip,
        )


class _PathSource(_TrackSource):
    """Read tracks by re-opening a file name.

    No file handle is kept between accesses, so a lazily opened
    ``MidiFile`` never holds a descriptor open. The file is re-opened
    for each uncached track, so deleting or replacing it after the
    initial scan affects later access.
    """

    kind = 'source file'

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    @contextmanager
    def _open(self):
        with open(self.filename, 'rb') as infile:
            yield infile


class _FileSource(_TrackSource):
    """Read tracks from a caller-owned seekable file object.

    The caller keeps ownership: the file is never closed here, and its
    position is restored after every scan or decode. If the caller
    closes the file, cached tracks remain readable while uncached track
    access raises a descriptive ``ValueError``.
    """

    kind = 'source file'

    def __init__(self, file, owns_file=False):
        super().__init__()
        self.file = file
        self._owns_file = owns_file

    @contextmanager
    def _open(self):
        if getattr(self.file, 'closed', False):
            raise ValueError(
                'cannot read track: the source file has been closed')
        # Do not rely on ``closed`` alone: some file-like objects (for
        # example an object wrapping a closed descriptor) only fail on
        # read. Probe with tell() so the error is descriptive.
        try:
            view = _RestoringFileView(self.file)
        except (AttributeError, OSError, ValueError) as exc:
            raise ValueError(
                'cannot read track: the source file is not seekable'
                ' or has been closed') from exc
        try:
            yield view
        finally:
            view.restore()

    def close(self):
        super().close()
        if self._owns_file:
            try:
                self.file.close()
            except Exception:
                pass


class LazyTrackList(MutableSequence):
    """A ``tracks`` sequence that decodes tracks on first access.

    It behaves like a list of :class:`MidiTrack` objects, but entries
    that have not been accessed yet are stored as file offsets. The
    source is read for a track only when that track is indexed,
    iterated over or otherwise accessed, and decoded tracks are cached
    in the sequence.

    :meth:`is_loaded` reports whether an entry has been decoded and
    :meth:`load_all` decodes the rest eagerly.
    """

    def __init__(self, entries, loader):
        self._entries = list(entries)
        self._loader = loader
        self._lock = threading.RLock()

    # -- helpers

    def _normalize_index(self, index):
        index = operator.index(index)
        length = len(self._entries)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError('list index out of range')
        return index

    def _resolve(self, index):
        entry = self._entries[index]
        if isinstance(entry, _TrackEntry):
            with self._lock:
                # Re-check inside the lock so concurrent readers decode
                # an entry only once.
                entry = self._entries[index]
                if isinstance(entry, _TrackEntry):
                    entry = self._loader(entry)
                    self._entries[index] = entry
        return entry

    def is_loaded(self, index):
        """Return True if the track at *index* has been decoded."""
        return not isinstance(
            self._entries[self._normalize_index(index)], _TrackEntry)

    def load_all(self):
        """Decode every remaining track eagerly and return ``self``."""
        with self._lock:
            for index in range(len(self._entries)):
                self._resolve(index)
        return self

    # -- MutableSequence interface

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._resolve(i)
                    for i in range(*index.indices(len(self._entries)))]
        return self._resolve(self._normalize_index(index))

    def __setitem__(self, index, value):
        with self._lock:
            if isinstance(index, slice):
                self._entries[index] = list(value)
            else:
                self._entries[self._normalize_index(index)] = value

    def __delitem__(self, index):
        with self._lock:
            del self._entries[index]

    def insert(self, index, value):
        with self._lock:
            self._entries.insert(index, value)

    def __iter__(self):
        for index in range(len(self._entries)):
            yield self._resolve(index)

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __add__(self, values):
        return list(self) + list(values)

    def __radd__(self, values):
        return list(values) + list(self)

    def __mul__(self, count):
        return list(self) * count

    __rmul__ = __mul__

    def __reversed__(self):
        for index in reversed(range(len(self._entries))):
            yield self._resolve(index)

    def copy(self):
        """Return a shallow copy as a plain list."""
        return list(self)

    def sort(self, *, key=None, reverse=False):
        with self._lock:
            self.load_all()
            self._entries.sort(key=key, reverse=reverse)

    def __eq__(self, other):
        if isinstance(other, LazyTrackList):
            other = list(other)
        elif not isinstance(other, (list, tuple)):
            return NotImplemented
        return list(self) == list(other)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __repr__(self):
        entries = []
        for index, entry in enumerate(self._entries):
            entries.append(self._repr_entry(index))
        return '[{}]'.format(',\n '.join(entries))

    def _repr_entry(self, index):
        entry = self._entries[index]
        if isinstance(entry, _TrackEntry):
            return f'<track {index} not loaded>'
        return repr(entry)


def _scan_track_directory(infile, num_tracks):
    """Scan the chunk directory without decoding any track data.

    Returns a list of ``_TrackEntry`` values. Raises the same errors
    eager reading raises for a bad chunk name, a short chunk header or
    a chunk that extends past the end of the file. The returned entries
    point at each ``MTrk`` chunk header.
    """
    # Seeking past EOF is allowed, so the real file size is needed to
    # validate the declared chunk sizes.
    position = infile.tell()
    file_size = infile.seek(0, 2)
    infile.seek(position)

    entries = []
    for _ in range(num_tracks):
        header_offset = infile.tell()
        header = infile.read(8)
        if len(header) < 8:
            raise EOFError
        name, size = struct.unpack('>4sL', header)
        if name != b'MTrk':
            raise OSError('no MTrk header at start of track')

        chunk_end = header_offset + 8 + size
        if chunk_end > file_size:
            raise EOFError
        infile.seek(chunk_end)
        entries.append(_TrackEntry(header_offset, size))

    return entries


def _is_seekable(infile):
    """Return True if *infile* supports seek() and tell()."""
    try:
        seekable = getattr(infile, 'seekable', None)
        if seekable is not None:
            if callable(seekable):
                if not seekable():
                    return False
            elif not seekable:
                return False
        position = infile.tell()
        infile.seek(position)
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    return True


def write_chunk(outfile, name, data):
    """Write an IFF chunk to the file.

    `name` must be a bytestring."""
    outfile.write(name)
    outfile.write(struct.pack('>L', len(data)))
    outfile.write(data)


def write_track(outfile, track):
    data = bytearray()

    running_status_byte = None
    for msg in fix_end_of_track(track):
        if not isinstance(msg.time, Integral):
            raise ValueError('message time must be int in MIDI file')
        if msg.time < 0:
            raise ValueError('message time must be non-negative in MIDI file')

        if msg.is_realtime:
            raise ValueError('realtime messages are not allowed in MIDI files')

        data.extend(encode_variable_int(msg.time))

        if msg.is_meta:
            data.extend(msg.bytes())
            running_status_byte = None
        elif msg.type == 'sysex':
            data.append(0xf0)
            # length (+ 1 for end byte (0xf7))
            data.extend(encode_variable_int(len(msg.data) + 1))
            data.extend(msg.data)
            data.append(0xf7)
            running_status_byte = None
        else:
            msg_bytes = msg.bytes()
            status_byte = msg_bytes[0]

            if status_byte == running_status_byte:
                data.extend(msg_bytes[1:])
            else:
                data.extend(msg_bytes)

            if status_byte < 0xf0:
                running_status_byte = status_byte
            else:
                running_status_byte = None

    write_chunk(outfile, b'MTrk', data)


def get_seconds_per_tick(tempo, ticks_per_beat):
    # Tempo is given in microseconds per beat (default 500000).
    # At this tempo there are (500000 / 1000000) == 0.5 seconds
    # per beat. At the default resolution of 480 ticks per beat
    # this is:
    #
    #    (500000 / 1000000) / 480 == 0.5 / 480 == 0.0010417
    #
    return (tempo / 1000000.0) / ticks_per_beat


class MidiFile:
    """A Standard MIDI File.

    Pass ``lazy=True`` to scan the file header and track directory up
    front and decode each track only when it is first accessed. This is
    useful for large files (for example, hundreds of tracks where only
    a few are needed): the initial read stays proportional to the
    number of tracks instead of the amount of message data.

    With ``lazy=True`` the ``tracks`` attribute is a
    :class:`LazyTrackList` (a mutable sequence that otherwise behaves
    like a list). Decoded tracks are cached, so repeated access does not
    touch the source again.

    Lazy mode has these source requirements:

    * ``filename``: the file is re-opened for each uncached track, so no
      descriptor stays open between accesses.
    * ``file``: a seekable file object is read in place. The caller
      keeps ownership; the position is restored after every access, and
      the object is never closed by ``MidiFile``.
    * ``file`` that is not seekable (a pipe or socket): the stream is
      spooled to an anonymous temporary file during construction, using
      disk instead of memory. The spool is closed by :meth:`close` or
      when the ``MidiFile`` is garbage collected.

    Call :meth:`close` to drop the source explicitly; already-decoded
    tracks stay accessible, while accessing an undecoded track then
    raises ``ValueError``.
    """

    def __init__(self, filename=None, file=None,
                 type=1, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
                 charset='latin1',
                 debug=False,
                 clip=False,
                 tracks=None,
                 lazy=False,
                 ):

        self.filename = filename
        self.type = type
        self.ticks_per_beat = ticks_per_beat
        self.charset = charset
        self.debug = debug
        self.clip = clip
        self.lazy = bool(lazy)

        self.tracks = []
        self._merged_track = None
        self._source = None

        if type not in range(3):
            raise ValueError(
                f'invalid format {format} (must be 0, 1 or 2)')

        if tracks is not None:
            self.tracks = tracks
        elif file is not None:
            if self.lazy:
                self._load_lazy(self._open_lazy_source(file))
            else:
                self._load(file)
        elif self.filename is not None:
            if self.lazy:
                self._load_lazy(_PathSource(self.filename))
            else:
                with open(filename, 'rb') as file:
                    self._load(file)

    @property
    def merged_track(self):
        # The tracks of type 2 files are not in sync, so they can
        # not be played back like this.
        if self.type == 2:
            raise TypeError("can't merge tracks in type 2 (asynchronous) file")

        if self._merged_track is None:
            self._merged_track = merge_tracks(self.tracks)
        return self._merged_track

    @merged_track.deleter
    def merged_track(self):
        self._merged_track = None

    def add_track(self, name=None):
        """Add a new track to the file.

        This will create a new MidiTrack object and append it to the
        track list.
        """
        track = MidiTrack()
        if name is not None:
            track.name = name
        self.tracks.append(track)
        del self.merged_track  # uncache merged track
        return track

    def _load(self, infile):
        if self.debug:
            infile = DebugFileWrapper(infile)

        with meta_charset(self.charset):
            if self.debug:
                _dbg('Header:')

            (self.type,
             num_tracks,
             self.ticks_per_beat) = read_file_header(infile)

            if self.debug:
                _dbg('-> type={}, tracks={}, ticks_per_beat={}'.format(
                    self.type, num_tracks, self.ticks_per_beat))
                _dbg()

            for i in range(num_tracks):
                if self.debug:
                    _dbg(f'Track {i}:')

                self.tracks.append(read_track(infile,
                                              debug=self.debug,
                                              clip=self.clip))
                # TODO: used to ignore EOFError. I hope things still work.

    def _load_lazy(self, source):
        """Scan *source* and expose its tracks without decoding them."""
        try:
            file_type, ticks_per_beat, entries = source.scan(
                debug=self.debug)
        except BaseException:
            # Release a spooled temporary file when the scan fails.
            source.close()
            raise
        self.type = file_type
        self.ticks_per_beat = ticks_per_beat
        self._source = source
        self.tracks = LazyTrackList(
            entries,
            loader=_TrackLoader(source, self.charset, self.debug, self.clip),
        )

    def _open_lazy_source(self, infile):
        """Build a lazily readable source for a caller-supplied stream."""
        if _is_seekable(infile):
            return _FileSource(infile)

        # A forward-only stream cannot be seeked back to a track
        # boundary. Spool it to an anonymous temporary file so lazy
        # access still works; memory use stays constant while the
        # spooled bytes live on disk.
        spool = tempfile.TemporaryFile()
        try:
            shutil.copyfileobj(infile, spool)
            spool.seek(0)
            source = _FileSource(spool, owns_file=True)
        except BaseException:
            spool.close()
            raise
        return source

    def __del__(self):
        # Closing an anonymous TemporaryFile also unlinks it. Cached
        # tracks do not need the source, so this is safe to do here.
        source = getattr(self, '_source', None)
        if source is not None:
            try:
                source.close()
            except Exception:
                pass

    def close(self):
        """Release the source used for on-demand track loading.

        Already-decoded tracks remain accessible. Accessing a track
        that has not been decoded yet raises ``ValueError`` afterwards.
        A file object passed by the caller is never closed by this
        method; a spooled forward-only stream is.
        """
        source = self._source
        if source is not None:
            source.close()
            self._source = None

    @property
    def length(self):
        """Playback time in seconds.

        This will be computed by going through every message in every
        track and adding up delta times.
        """
        if self.type == 2:
            raise ValueError('impossible to compute length'
                             ' for type 2 (asynchronous) file')

        return sum(msg.time for msg in self)

    def __iter__(self):
        tempo = DEFAULT_TEMPO
        for msg in self.merged_track:
            # Convert message time from absolute time
            # in ticks to relative time in seconds.
            if msg.time > 0:
                delta = tick2second(msg.time, self.ticks_per_beat, tempo)
            else:
                delta = 0

            yield msg.copy(time=delta)

            if msg.type == 'set_tempo':
                tempo = msg.tempo

    def play(self, meta_messages=False, now=time.time):
        """Play back all tracks.

        The generator will sleep between each message by
        default. Messages are yielded with correct timing. The time
        attribute is set to the number of seconds slept since the
        previous message.

        By default you will only get normal MIDI messages. Pass
        meta_messages=True if you also want meta messages.

        You will receive copies of the original messages, so you can
        safely modify them without ruining the tracks.

        By default the system clock is used for the timing of yielded
        MIDI events. To use a different clock (e.g. to synchronize to
        an audio stream), pass now=time_fn where time_fn is a zero
        argument function that yields the current time in seconds.
        """
        start_time = now()
        input_time = 0.0

        for msg in self:
            input_time += msg.time

            playback_time = now() - start_time
            duration_to_next_event = input_time - playback_time

            if duration_to_next_event > 0.0:
                time.sleep(duration_to_next_event)

            if isinstance(msg, MetaMessage) and not meta_messages:
                continue
            else:
                yield msg

    def save(self, filename=None, file=None):
        """Save to a file.

        If file is passed the data will be saved to that file. This is
        typically an in-memory file or and already open file like sys.stdout.

        If filename is passed the data will be saved to that file.

        Raises ValueError if both file and filename are None,
        or if a type 0 file has != one track.
        """
        if self.type == 0 and len(self.tracks) != 1:
            raise ValueError('type 0 file must have exactly 1 track')

        if file is not None:
            self._save(file)
        elif filename is not None:
            with open(filename, 'wb') as file:
                self._save(file)
        else:
            raise ValueError('requires filename or file')

    def _save(self, outfile):
        with meta_charset(self.charset):
            header = struct.pack('>hhh', self.type,
                                 len(self.tracks),
                                 self.ticks_per_beat)

            write_chunk(outfile, b'MThd', header)

            for track in self.tracks:
                write_track(outfile, track)

    def print_tracks(self, meta_only=False):
        """Prints out all messages in a .midi file.

        May take argument meta_only to show only meta messages.

        Use:
        print_tracks() -> will print all messages
        print_tracks(meta_only=True) -> will print only MetaMessages
        """
        for i, track in enumerate(self.tracks):
            print(f'=== Track {i}')
            for msg in track:
                if not isinstance(msg, MetaMessage) and meta_only:
                    pass
                else:
                    print(f'{msg!r}')

    def __repr__(self):
        if self.tracks:
            if isinstance(self.tracks, LazyTrackList):
                tracks_str = ',\n'.join(
                    self.tracks._repr_entry(index)
                    for index in range(len(self.tracks)))
            else:
                tracks_str = ',\n'.join(repr(track) for track in self.tracks)
            tracks_str = '  ' + tracks_str.replace('\n', '\n  ')
            tracks_str = f', tracks=[\n{tracks_str}\n]'
        else:
            tracks_str = ''

        return '{}(type={}, ticks_per_beat={}{})'.format(
            self.__class__.__name__,
            self.type,
            self.ticks_per_beat,
            tracks_str,
        )

    # The context manager has no purpose but is kept around since it was
    # used in examples in the past.
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Release a spooled source when leaving a ``with`` block, but
        # keep the documented rule that a caller-supplied file object is
        # never closed by MidiFile.
        source = self._source
        if source is not None and getattr(source, '_owns_file', False):
            self.close()
        return False
