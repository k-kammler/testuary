"""The exact reduction's bundled CBC never stalls across repeated solves.

With ``-threads 1``, CBC 2.10.3 loses a worker wakeup in a few percent of
solves and then waits for the worker's fixed 10 s timeout. The serial search
has no worker. Real solver and real subprocesses, so this test is slow.
"""
from __future__ import annotations

import time

import pytest

from testuary.reduction import exact_cover

pytestmark = pytest.mark.slow

# A small cover with several optimal selections, the shape on which the
# stall occurs.
TOKENS = {
    "t0": frozenset({"a", "b"}), "t1": frozenset({"b", "c", "d"}),
    "t2": frozenset({"d", "e"}), "t3": frozenset({"a", "e", "f"}),
    "t4": frozenset({"f", "g"}), "t5": frozenset({"g", "h", "a"}),
    "t6": frozenset({"h"}), "t7": frozenset({"c", "g"}),
}
COSTS = {"t0": 3.0, "t1": 5.0, "t2": 2.0, "t3": 4.0,
         "t4": 1.0, "t5": 6.0, "t6": 1.0, "t7": 2.0}


def test_repeated_exact_solves_never_stall():
    selections = set()
    slowest = 0.0
    for _ in range(150):
        time.sleep(0.02)
        start = time.monotonic()
        result = exact_cover(TOKENS, COSTS, time_limit_s=60.0)
        slowest = max(slowest, time.monotonic() - start)
        selections.add(result.keep)
    assert slowest < 5.0, f"a solve took {slowest:.2f} s"
    # Serial is repeatable: one selection across all solves.
    assert len(selections) == 1
