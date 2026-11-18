"""A resume checks a file's own tests.csv rows, not the whole shared table, so
a multi-file resume skips a complete file instead of measuring it again."""
from __future__ import annotations

import csv

from testuary import phase_state


def _write(path, target_numbers):
    fields = ["target_number", "run", *phase_state._RESUME_REQUIRED_CSV_FIELDS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(fields)))
        writer.writeheader()
        for target_number in target_numbers:
            writer.writerow({field: "1" for field in fields} | {"target_number": target_number, "run": 0})


def test_each_file_of_a_shared_table_is_complete_for_its_own_target(tmp_path):
    results = tmp_path / "results"
    (results / "module").mkdir(parents=True)
    _write(results / "tests.csv", [1, 2])
    assert phase_state.resume_csv_complete(results / "module", 1, target_number=1)
    assert phase_state.resume_csv_complete(results / "module", 1, target_number=2)
    assert not phase_state.resume_csv_complete(results / "module", 1, target_number=3)
    # Without ``rank`` the whole table must hold exactly the expected rows.
    assert not phase_state.resume_csv_complete(results / "module", 1)


def _row(path, **cells):
    fields = ["target_number", "run", *phase_state._RESUME_REQUIRED_CSV_FIELDS,
              "finishing_mutation_measurement", "finishing_total"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(fields)))
        writer.writeheader()
        writer.writerow({field: "1" for field in fields} | {"target_number": 1, "run": 0} | cells)


def test_a_cell_the_record_holds_no_value_for_is_complete_as_it_is(tmp_path):
    """A target without branches: finishing recorded no branch
    coverage, so the empty cell is the value and a resume keeps the row
    instead of measuring again. A 0 never stands for it."""

    results = tmp_path / "results"
    target_dir = results / "module"
    target_dir.mkdir(parents=True)
    _row(results / "tests.csv", branch_coverage="")
    assert not phase_state.resume_csv_complete(target_dir, 1, target_number=1, run=0)
    phase_state.save(target_dir, {"finishing": {"0": {"metrics": {
        "final_line_coverage": 90.0, "final_branch_coverage": None}}}})
    assert phase_state.resume_csv_complete(target_dir, 1, target_number=1, run=0)
    phase_state.save(target_dir, {"finishing": {"0": {"metrics": {
        "final_line_coverage": 90.0, "final_branch_coverage": 75.0}}}})
    assert not phase_state.resume_csv_complete(target_dir, 1, target_number=1, run=0)


def test_a_mutation_score_without_a_valid_mutant_is_complete_as_it_is(tmp_path):
    results = tmp_path / "results"
    target_dir = results / "module"
    target_dir.mkdir(parents=True)
    _row(results / "tests.csv", mutation_score="", finishing_mutation_measurement="complete",
         finishing_total="0")
    assert phase_state.resume_csv_complete(target_dir, 1, target_number=1, run=0)
    # A measurement that left mutants untested has no score either, and is missing.
    _row(results / "tests.csv", mutation_score="", finishing_mutation_measurement="incomplete",
         finishing_total="7")
    assert not phase_state.resume_csv_complete(target_dir, 1, target_number=1, run=0)

