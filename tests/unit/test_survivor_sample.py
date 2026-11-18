"""The survivor sample shown to the model spreads over the target.

When there are more surviving mutants than the sample shows, the changes
shown are spread over the target's lines and over the mutation operators,
deterministically, rather than the first ones by line. Every survivor is
still listed by id and line.
"""
from __future__ import annotations

import re

from testuary.languages.mutation import render_survivors
from testuary.languages import mutation as _kill_matrix


def _survivor(mid: str, line: int, operator: str) -> dict:
    return {"id": mid, "line": line, "operator": operator,
            "diff": f"@@ -{line} +{line} @@\n-a\n+b {mid}"}


def _shown_ids(text: str) -> list:
    return re.findall(r"^--- mutant (\S+) \(line", text, re.M)


def test_the_sample_spreads_over_lines():
    survivors = ([_survivor(f"a{i}", 1, "Op") for i in range(5)]
                 + [_survivor("b", 5, "Op"), _survivor("c", 9, "Op"), _survivor("d", 13, "Op")])
    text = render_survivors(survivors, max_mutants=4, max_chars=None)
    shown = _shown_ids(text)
    assert len(shown) == 4
    assert {s["line"] for s in survivors if s["id"] in shown} == {1, 5, 9, 13}
    assert "All 8 surviving mutants" in text


def test_the_sample_spreads_over_operators_on_a_line():
    survivors = [_survivor("1", 3, "A"), _survivor("2", 3, "A"), _survivor("3", 3, "A"),
                 _survivor("4", 3, "B"), _survivor("5", 3, "C")]
    shown = _shown_ids(render_survivors(survivors, max_mutants=3, max_chars=None))
    assert sorted(s["operator"] for s in survivors if s["id"] in shown) == ["A", "B", "C"]


def test_more_lines_than_places_are_evenly_spaced_and_deterministic():
    survivors = [_survivor(str(n), n, "Op") for n in range(10, 110, 10)]
    first = render_survivors(survivors, max_mutants=3, max_chars=None)
    assert _shown_ids(first) == ["20", "60", "90"]
    assert render_survivors(list(reversed(survivors)), max_mutants=3, max_chars=None) == first


def test_the_stored_matrix_sample_spreads_over_operators():
    matrix = {"mutants": [
        {"id": i, "line": 7, "operator": op, "status": "survived", "diff": f"-a\n+b{i}"}
        for i, op in enumerate(["A", "A", "A", "B"])]}
    text = _kill_matrix.render_survivors_from_matrix(matrix, max_mutants=2, max_chars=None)
    assert _shown_ids(text) == ["0", "3"]
