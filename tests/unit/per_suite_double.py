"""A test double for a mutation runner's per-suite measurement.

``per_suite_double(measure)`` turns a stub ``measure(test_dir)`` (one kill
matrix per staged directory) into a runner callable with ``per_suite``. The
union holds each mutant's verdict per suite, keyed by operator, line and
diff, with the union status rules of the backend's per-suite contract.
Each killed row and verdict names its killing tests, read from the stub's
pytest node ids as LangAdapter records them.

``measure`` returns None for a suite that is not green (it never enters the
request). ``measured_nothing(outcome)`` is a suite whose run measured
nothing: an outcome, no matrix, no mutant result. A matrix with an
``_outcome`` key measured but did not complete, as when cut by the wall clock.
"""
from __future__ import annotations

_KILLING = ("killed", "collection_error")


def measured_nothing(outcome="infrastructure_failure", named=()):
    """A suite whose run measured nothing, with the test id objects the run's
    check before the first mutant ``named``."""
    return {"_measured_nothing": outcome, **({"_named": tuple(named)} if named else {})}


def _verdict(row):
    status = row.get("status")
    if status == "killed":
        return "killed" if (row.get("killers") or row.get("killer")) else "collection_error"
    return status if status in ("timeout", "non_viable", "survived") else "untested"


def _killers(row):
    return list(row.get("killers") or ([row["killer"]] if row.get("killer") else []))


def killing_tests(killers):
    """The tests LangAdapter records for pytest node ids (``file::test``)."""
    return [{"file": file, "qualname": test.split("[", 1)[0].replace("::", "."),
             "params": None, "text": killer}
            for killer in killers for file, _, test in [str(killer).partition("::")] if test]


def _killing_tests(row):
    return list(row["killing_tests"]) if "killing_tests" in row else killing_tests(_killers(row))


def placed(matrix):
    """``matrix`` with each killed row's killing tests."""
    if not matrix or "mutants" not in matrix:
        return matrix
    return {**matrix, "mutants": [
        {**row, "killing_tests": _killing_tests(row)} if _killers(row) else row
        for row in matrix["mutants"]]}


def _union_status(verdicts):
    if any(v in _KILLING for v in verdicts):
        return "killed"
    if "untested" in verdicts:
        return "untested"
    if "timeout" in verdicts:
        return "timeout"
    if "non_viable" in verdicts:
        return "non_viable"
    return "survived"


def per_suite_double(measure, *, implementation="test_double"):
    def run(test_dir, *, attribution=None):
        # Record the attribution the runner passes on.
        run.attributions.append(attribution)
        matrix = placed(measure(test_dir))
        # The matrix records the attribution asked for, as LangAdapter's
        # runs do, unless the measurement fell back to another.
        if matrix is not None and attribution is not None:
            matrix.setdefault("attribution", attribution)
        return matrix

    run.attributions = []

    def per_suite(suites):
        per_suite.calls += 1
        own, outcomes, unmeasured = {}, {}, []
        for name, test_dir in suites.items():
            matrix = placed(measure(test_dir))
            run.named[str(test_dir)] = (matrix or {}).get("_named", ())
            if matrix is None:
                unmeasured.append(name)
            elif "_measured_nothing" in matrix:
                outcomes[name] = matrix["_measured_nothing"]
                unmeasured.append(name)
            else:
                outcomes[name] = matrix.get("_outcome", "completed")
                own[name] = matrix
        if not outcomes:
            return None
        rows = {}
        for name, matrix in own.items():
            for row in matrix.get("mutants") or []:
                key = (row.get("operator"), row.get("line"), row.get("diff"), row.get("id")
                       if row.get("operator") is None else None)
                union = rows.setdefault(key, {"id": row.get("id"), "operator": row.get("operator"),
                                              "line": row.get("line"), "diff": row.get("diff"),
                                              "suite_verdicts": {}})
                union["suite_verdicts"][name] = {"verdict": _verdict(row), "killers": _killers(row)}
                if _killers(row):
                    union["suite_verdicts"][name]["killing_tests"] = _killing_tests(row)
        for union in rows.values():
            for name in outcomes:
                union["suite_verdicts"].setdefault(name, {"verdict": "untested", "killers": []})
            verdicts = [v["verdict"] for v in union["suite_verdicts"].values()]
            union["status"] = _union_status(verdicts)
            union["killers"] = [k for v in union["suite_verdicts"].values() for k in v["killers"]]
            if union["killers"]:
                union["killer"] = union["killers"][0]
        return {"union": {"mutants": list(rows.values())}, "suites": own, "outcomes": outcomes,
                "implementation": implementation, "unmeasured": sorted(unmeasured)}

    per_suite.calls = 0
    run.per_suite = per_suite
    # The tests each measured directory's checks named, as the runner's
    # ``named_of`` gives them.
    run.named = {}
    run.named_of = lambda test_dir: run.named.get(str(test_dir), ())
    return run
