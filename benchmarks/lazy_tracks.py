# SPDX-FileCopyrightText: 2026 Mido contributors
#
# SPDX-License-Identifier: MIT

"""Benchmark eager vs. lazy MidiFile track loading.

Builds a large multi-track file (default: 300 tracks, about 20 MB), then
compares:

* initial peak memory (``tracemalloc``),
* bytes read from the source while opening (and while reading one
  track),
* latency of opening and of accessing a single track.

Run with::

    python3 benchmarks/lazy_tracks.py [--tracks N] [--messages N]

This script is not part of the test suite.
"""

import argparse
import io
import os
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mido import Message, MidiFile, MidiTrack  # noqa: E402


class CountingReader(io.RawIOBase):
    def __init__(self, data):
        self._file = io.BytesIO(data)
        self.bytes_read = 0

    def readable(self):
        return True

    def read(self, size=-1):
        data = self._file.read(size)
        self.bytes_read += len(data)
        return data

    def tell(self):
        return self._file.tell()

    def seek(self, offset, whence=0):
        return self._file.seek(offset, whence)

    def seekable(self):
        return True


def build(tracks, messages):
    mid = MidiFile(type=1, ticks_per_beat=480)
    for track_index in range(tracks):
        track = MidiTrack()
        for i in range(messages):
            note = 36 + (i % 60)
            track.append(Message('note_on', note=note, velocity=64,
                                 time=1 + track_index % 4))
            track.append(Message('note_off', note=note, velocity=64,
                                 time=1))
        mid.tracks.append(track)

    file = io.BytesIO()
    mid.save(file=file)
    return file.getvalue()


def measure_peak(function):
    tracemalloc.start()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def measure_bytes_and_time(function):
    reader = CountingReader(DATA)
    start = time.perf_counter()
    result = function(reader)
    elapsed = time.perf_counter() - start
    return result, reader.bytes_read, elapsed


def main():
    global DATA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=int, default=300)
    parser.add_argument('--messages', type=int, default=10000,
                        help='note pairs per track')
    options = parser.parse_args()

    start = time.perf_counter()
    DATA = build(options.tracks, options.messages)
    print(f'file: {len(DATA) / 1e6:.1f} MB, '
          f'{options.tracks} tracks x {options.messages} note pairs '
          f'(built in {time.perf_counter() - start:.2f}s)')

    # --- initial peak memory -------------------------------------------
    eager_file, eager_peak = measure_peak(
        lambda: MidiFile(file=io.BytesIO(DATA)))
    lazy_file, lazy_peak = measure_peak(
        lambda: MidiFile(file=io.BytesIO(DATA), lazy=True))

    print('\ninitial peak memory (tracemalloc)')
    print(f'  eager: {eager_peak / 1e6:8.2f} MB')
    print(f'  lazy:  {lazy_peak / 1e6:8.2f} MB')
    print(f'  ratio: {lazy_peak / eager_peak:.1%} of eager')

    # --- bytes read while opening (and reading one track) --------------
    eager_bytes_open = CountingReader(DATA)
    start = time.perf_counter()
    eager_mid = MidiFile(file=eager_bytes_open)
    eager_open_time = time.perf_counter() - start
    eager_bytes_total = eager_bytes_open.bytes_read

    lazy_bytes = CountingReader(DATA)
    start = time.perf_counter()
    lazy_mid = MidiFile(file=lazy_bytes, lazy=True)
    lazy_open_time = time.perf_counter() - start
    lazy_scan_bytes = lazy_bytes.bytes_read

    start = time.perf_counter()
    lazy_mid.tracks[42]
    lazy_track_open_time = time.perf_counter() - start
    lazy_one_track_bytes = lazy_bytes.bytes_read

    print('\nbytes read from source')
    print(f'  eager open:              {eager_bytes_total:>10,} bytes')
    print(f'  lazy open (scan only):   {lazy_scan_bytes:>10,} bytes')
    print(f'  lazy open + one track:   {lazy_one_track_bytes:>10,} bytes')
    print(f'  lazy / eager for scan:   '
          f'{lazy_scan_bytes / eager_bytes_total:.3%}')

    # --- latency --------------------------------------------------------
    lazy_latency = CountingReader(DATA)
    lazy_latency_mid = MidiFile(file=lazy_latency, lazy=True)
    start = time.perf_counter()
    lazy_latency_mid.tracks[42]
    lazy_access = time.perf_counter() - start
    start = time.perf_counter()
    lazy_latency_mid.tracks[42]
    cached_access = time.perf_counter() - start

    print('\nlatency')
    print(f'  eager open:              {eager_open_time * 1e3:9.2f} ms')
    print(f'  lazy open (scan only):   {lazy_open_time * 1e3:9.2f} ms')
    print(f'  lazy first track access: {lazy_track_open_time * 1e3:9.2f} ms')
    print(f'  lazy access (fresh):     {lazy_access * 1e3:9.2f} ms')
    print(f'  lazy access (cached):    {cached_access * 1e6:9.2f} us')

    # --- correctness ----------------------------------------------------
    assert [list(t) for t in eager_file.tracks] == \
        [list(t) for t in lazy_file.tracks]
    assert list(eager_mid.tracks[42]) == list(lazy_mid.tracks[42])

    # --- filename mode: no descriptor leaked ---------------------------
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'large.mid')
        with open(path, 'wb') as file:
            file.write(DATA)

        before = len(os.listdir('/proc/self/fd'))
        path_mid = MidiFile(path, lazy=True)
        during = len(os.listdir('/proc/self/fd'))
        path_mid.tracks[42]
        after = len(os.listdir('/proc/self/fd'))
        print('\nfile descriptors (filename mode)')
        print(f'  before={before} after-open={during} after-track={after}')
        assert before == during == after


if __name__ == '__main__':
    main()
