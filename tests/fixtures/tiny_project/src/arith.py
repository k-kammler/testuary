"""Tiny mutation-friendly module used by the integration tests.

Three functions; each has an arithmetic / comparison / boolean
operation that mutmut and cosmic-ray will mutate. Tests for these are
in ``../tests/test_arith.py``; they intentionally exercise every
mutable operator so the mutation kill rate is high enough to validate
the integration plumbing (the score isn't what we're testing here,
the *measurement pipeline* is).
"""
from __future__ import annotations


def add(a: int, b: int) -> int:
    """``a + b``. The + is mutated to - by AOR; tests pin a+b=5 for 2+3
    and 0+0=0 to kill both."""
    return a + b


def is_positive(n: int) -> bool:
    """``n > 0``. ROR mutates > to >=, <, <=, ==, != ; the four boundary
    tests around n=0 catch every variant."""
    return n > 0


def both_truthy(a: object, b: object) -> bool:
    """``bool(a) and bool(b)``. LCR mutates ``and`` to ``or``; the test
    that asserts ``both_truthy(True, False)`` is False kills that
    mutant."""
    return bool(a) and bool(b)
