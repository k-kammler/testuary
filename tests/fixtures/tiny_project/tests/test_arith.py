"""Fixture suite for ``arith.py``: covers every line and branch and pins each
mutable operator to a distinct value. The integration tests run it. The main
suite does not collect it, since the testpaths in ``pyproject.toml`` leave out
``tests/fixtures``.
"""
from __future__ import annotations

from arith import add, both_truthy, is_positive


def test_add_two_positives_returns_sum():
    """Pins ``+`` to 5. It kills ``a - b``, which gives -1, and ``a * b``,
    which gives 6."""
    assert add(2, 3) == 5


def test_add_with_zero_is_identity():
    """Boundary case at zero, which kills ``a + 1``-style off-by-one mutants."""
    assert add(0, 0) == 0
    assert add(7, 0) == 7


def test_is_positive_true_for_positive_int():
    assert is_positive(5) is True


def test_is_positive_false_at_zero():
    """Kills the ``>= 0`` mutant of ``> 0``."""
    assert is_positive(0) is False


def test_is_positive_false_for_negative_int():
    assert is_positive(-3) is False


def test_both_truthy_two_true():
    assert both_truthy(True, True) is True


def test_both_truthy_with_one_false_returns_false():
    """Kills the ``or`` mutant of ``and``: for True and False, ``and``
    returns False and ``or`` returns True."""
    assert both_truthy(True, False) is False
    assert both_truthy(False, True) is False


def test_both_truthy_two_false():
    assert both_truthy(False, False) is False
