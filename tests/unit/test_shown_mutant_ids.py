"""The model sees, and the records name, each mutant by its stable id.

LangAdapter gives every kill-matrix row a ``mutant_id`` that is the same
for the same mutant in every run (Cosmic Ray's engine id is minted per
session). The survivor list, the stored-matrix rendering and the
derive-vs-measure records show that id, with the row's ``id`` for a row
written without one. The result schema asks for the ids as shown, and its
own text counts as a placeholder.
"""
from __future__ import annotations

import re

from langadapter import mutation as native
from langadapter.mutation import Classification, MutationClass
from langadapter.mutation import Mutant, MutantKey, MutationRun

from testuary.languages import mutation
from testuary.languages import mutation as _kill_matrix
from testuary import consolidation as consolidate
from testuary.agent import mutation_agent as tm
from testuary import starter_selection

STABLE = "0f" * 16


def _shown_ids(text: str) -> list:
    return re.findall(r"^--- mutant (\S+) \(line", text, re.M)


def _survivor(job: str, line: int, row: dict) -> Mutant:
    return Mutant(
        key=MutantKey("calc.py", "core/Op", 0, line, 0, None),
        engine_id=job, test_outcome=None, worker_outcome=None,
        classification=Classification("survived", MutationClass.UNDETECTED,
                                      native.SURVIVED),
        killers=(), diff="@@ -1 +1 @@\n-a\n+b", row={"id": job, "status": "survived", **row})


def test_the_live_survivor_list_shows_the_stable_id():
    run = MutationRun(target_file="calc.py", requested_catalogue="full",
                      requested_operators=(), summary={},
                      mutants=(_survivor("job-7", 3, {"mutant_id": STABLE}),
                               _survivor("job-8", 5, {})))
    text = mutation.MutationMeasurement.of(run).survivors_text(max_mutants=None)
    assert _shown_ids(text) == [STABLE, "job-8"]
    assert "job-7" not in text


def test_the_stored_matrix_rendering_shows_the_stable_id():
    matrix = {"mutants": [
        {"id": 1, "mutant_id": STABLE, "line": 2, "status": "survived", "diff": "-a\n+b"},
        {"id": 2, "line": 4, "status": "survived", "diff": "-c\n+d"}]}
    text = _kill_matrix.render_survivors_from_matrix(matrix, max_mutants=None)
    assert _shown_ids(text) == [STABLE, "2"]


def test_the_derive_versus_measure_records_name_the_stable_id():
    derived = {"mutants": [{"id": 1, "mutant_id": STABLE, "operator": "op", "line": 2,
                            "diff": "-a\n+b", "status": "killed"}]}
    measured = {"mutants": [{"id": 9, "mutant_id": STABLE, "operator": "op", "line": 2,
                             "diff": "-a\n+b", "status": "survived"}]}
    logged = []
    records = consolidate._log_catalogue_diff(derived, measured, logged.append)
    assert [r["mutant_id"] for r in records] == [STABLE]
    found = starter_selection.differences(None, derived, measured, killed=set(), lines={2})
    assert [r["mutant_id"] for r in found] == [STABLE]
    assert "measured_mutant_id" not in found[0]


def test_the_schema_asks_for_the_shown_ids():
    description = tm.MutationTestResult.model_fields["targeted_mutants"].description
    assert "file:line:operator" not in description
    assert "--- mutant" in description
    assert tm._looks_placeholder_mutant(description)
    assert not tm._looks_placeholder_mutant(STABLE)
