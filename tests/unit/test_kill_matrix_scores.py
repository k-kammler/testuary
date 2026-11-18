"""Kill-matrix scores, selective catalogues and the survivors a stored matrix
holds, read without a mutation run."""
from __future__ import annotations

from testuary.languages import mutation as km


def _matrix(statuses: dict) -> dict:
    """Build a kill matrix from {mutant_id: status}."""
    return {
        "mutants": [{"id": i, "line": None, "status": s} for i, s in statuses.items()],
        "summary": {},
    }


def test_an_empty_matrix_counts_no_score() -> None:
    assert km.matrix_counts(_matrix({}))["mutation_score"] is None


# Matrices with operator labels: the rendered survivors and the counts below.


def _matrix_ops(rows: dict) -> dict:
    """Build a kill matrix from {mutant_id: (status, operator)}."""
    return {
        "mutants": [
            {"id": i, "line": None, "status": s, "operator": op}
            for i, (s, op) in rows.items()
        ],
        "summary": {},
    }


def test_render_survivors_from_matrix_formats_diffs() -> None:
    """The rendered baseline mirrors the engine's in-loop survivor block:
    a per-mutant diff with an id and line tag, derived from the stored matrix."""
    m = {"mutants": [
        {"id": "m1", "line": 10, "status": "survived", "operator": "OpA",
         "diff": "- a\n+ b"},
        {"id": "m2", "line": 20, "status": "killed", "operator": "OpA",
         "diff": "killed"},
        {"id": "m3", "line": 30, "status": "survived", "operator": "OpB",
         "diff": "- x\n+ y"},
    ]}
    rendered = km.render_survivors_from_matrix(m)
    assert "--- mutant m1 (line 10) ---" in rendered
    assert "- a\n+ b" in rendered
    assert "--- mutant m3 (line 30) ---" in rendered
    assert "m2" not in rendered  # killed, not a survivor
    # A selective catalogue renders only that view's survivors.
    selective = km.render_survivors_from_matrix(m, operators={"OpA"})
    assert "m1" in selective
    assert "m3" not in selective
    assert km.render_survivors_from_matrix(None) == ""


def test_render_survivors_from_matrix_over_cap_lists_all() -> None:
    """Over the diff cap, the render leads with the complete id@line list so the
    model still sees every survivor."""
    m = {"mutants": [
        {"id": f"m{i}", "line": i, "status": "survived", "diff": f"d{i}"}
        for i in range(5)
    ]}
    rendered = km.render_survivors_from_matrix(m, max_mutants=2)
    assert "All 5 surviving mutants" in rendered
    assert "m4@L4" in rendered
    assert "[... 3 further mutant changes omitted" in rendered


def test_stored_survivors_render_like_the_live_sample() -> None:
    """The stored matrix's survivors are ordered by source line and their
    changes compacted, as on the live path (not in matrix order with the
    full stored diff)."""
    long_change = "@@ -9,1 +9,1 @@\n-" + "a" * 300 + "\n+" + "b" * 300
    m = {"mutants": [
        {"id": "late", "line": 9, "status": "survived", "diff": long_change},
        {"id": "early", "line": 3, "status": "survived",
         "diff": "--- a/x.py\n+++ b/x.py\n@@ -3,1 +3,1 @@\n- a\n+ b"},
    ]}
    rendered = km.render_survivors_from_matrix(m, max_mutants=None, max_chars=240)
    assert rendered.index("--- mutant early (line 3)") < rendered.index("--- mutant late (line 9)")
    assert "--- a/x.py" not in rendered
    assert "chars of the change omitted ...]" in rendered


# One kill matrix's score and counts, without a mutation run.
MATRIX = _matrix_ops({
    1: ("killed", "OpA"), 2: ("timeout", "OpA"), 3: ("survived", "OpB"),
    4: ("survived", "OpA"), 5: ("untested", "OpB"),
})


def test_the_counts_count_an_untested_mutant_as_not_killed():
    assert km.matrix_counts(MATRIX) == {
        "mutation_score": 40.0, "mutation_score_lower_bound": 40.0,
        "killed": 2, "survived": 2, "untested": 1, "total": 5}


def test_an_operator_subset_counts_its_mutants_only():
    assert km.matrix_counts(MATRIX, operators={"OpA"}) == {
        "mutation_score": 100.0 * 2 / 3, "mutation_score_lower_bound": 100.0 * 2 / 3,
        "killed": 2, "survived": 1, "untested": 0, "total": 3}


def test_a_matrix_stamped_incomplete_has_only_its_lower_bound():
    stamped = {**MATRIX, "incomplete": {"reason": "untested_mutants_in_catalogue"}}
    counts = km.matrix_counts(stamped)
    assert counts["mutation_score"] is None and counts["mutation_score_lower_bound"] == 40.0


def test_no_matrix_has_no_score():
    assert km.matrix_counts(None) == {
        "mutation_score": None, "mutation_score_lower_bound": None,
        "killed": 0, "survived": 0, "untested": 0, "total": 0}
