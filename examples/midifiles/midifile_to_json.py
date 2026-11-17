# SPDX-FileCopyrightText: 2016 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import json
import sys

import mido


def midifile_to_dict(mid):
    tracks = []
    for track in mid.tracks:
        tracks.append([vars(msg).copy() for msg in track])

    result = {
        'ticks_per_beat': mid.ticks_per_beat,
        'tracks': tracks,
    }
    if mid.is_smpte:
        # SMPTE files use frame-based timing instead of ticks per beat.
        result['fps'] = mid.fps
        result['ticks_per_frame'] = mid.ticks_per_frame
    return result


mid = mido.MidiFile(sys.argv[1])

print(json.dumps(midifile_to_dict(mid), indent=2))
