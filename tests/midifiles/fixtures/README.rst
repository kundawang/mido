.. SPDX-FileCopyrightText: 2026
..
.. SPDX-License-Identifier: CC-BY-4.0

SMF byte-fidelity fixtures
==========================

Provenance
----------

The three ``.mid`` files in this directory are **fixed-byte Standard MIDI
File fixtures**, generated deterministically by :mod:`build_fixtures` from
first principles (chunk layout, status bytes, VLQ encoding) rather than
captured from a third-party source. The byte layout of every event is
annotated in ``build_fixtures.py`` and pinned by the assertions in
``tests/midifiles/test_smf_byte_fidelity.py``.

The encoding follows the *Standard MIDI Files 1.0* specification
published by the MIDI Manufacturers Association (https://midi.org/standard-midi-files):

* chunk structure: four-byte magic (``MThd``/``MTrk``), 32-bit big-endian
  length, payload;
* delta times and meta/SysEx lengths are variable-length quantities
  (31-bit, big-endian, continuation bit 7 set on every byte but the last);
* meta events are ``FF <type> <length> <data>``, system exclusive events
  are ``F0/F7 <length> <data>``;
* running status lets a channel message omit a repeated status byte and is
  reset by any meta, SysEx or system-common event.

Coverage matrix
---------------

====================  ======  ======  ======  ======  ======  =====
Fixture               format  running  VLQ     SysEx   unknown  EOT
                              status   bounds          meta
====================  ======  ======  ======  ======  ======  =====
``type0.mid``         0       yes     1..4    F0+F7   yes      yes
``type1.mid``         1       yes     1..3    F0      yes      yes
``type2.mid``         2       yes     1..4    -       -        yes
====================  ======  ======  ======  ======  ======  =====

The VLQ boundary values exercised are 127/128 (one/two-byte),
16383/16384 (two/three-byte) and 2097151/2097152 (three/four-byte), plus
a 127/128 boundary on a meta payload length.

Regeneration
------------

::

    python3 tests/midifiles/fixtures/build_fixtures.py

The test ``test_fixtures_exist_and_match_builder`` fails if a checked-in
binary ever drifts from what the builder emits.

Mutation harness
----------------

``mutation_demo.py`` injects six classes of encoding defects (truncation,
inflated chunk length, undefined status byte, running status without a
prior status, semantic status flip, unterminated VLQ) into each fixture
and verifies that every defect is either rejected with a positioned
error or changes the decoded semantics::

    python3 tests/midifiles/fixtures/mutation_demo.py

Test-suite baseline
-------------------

Recorded against mido checkout ``0592984`` ("Properly decode SMPTE
hours") before these tests were added:

* original suite: **121 passed**
* after adding the SMF byte-fidelity work: **164 passed**
  (43 new tests, no regressions)

With the library fixes reverted locally, the new tests fail on the F7
escape regression and on every error-positioning assertion, demonstrating
that they exercise real behaviour rather than tautologies. The mutation
harness independently demonstrates detection of six classes of injected
encoding defects across all three fixtures.
