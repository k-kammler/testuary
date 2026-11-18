"""``collect_eviction_data`` on a real instrumented run of
``tests/fixtures/tiny_project``, without subprocess mocks.

The suite runs under ``coverage run`` with a per-test dynamic context and
JUnit XML, and the tests check the per-test token and timing table the
suite-time reducer consumes. Unlike the unit tests, this exercises
coverage.py's dynamic contexts and rcfile, pytest's JUnit id format and its
reconciliation with the coverage contexts, and the PYTHONPATH and working
directory of the instrumented run.

Marked ``slow``, a few seconds.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pytest

from testuary.languages import GenericBackend


# Slow, and the code under test reads the run's settings: each test runs in one.
pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("default_run")]


def _stage(tmp_path: Path, fixtures_dir: Path) -> Tuple[str, str, str]:
    """Copy the tiny fixture project into ``tmp_path``. Return
    (source_dir, target_file, test_file)."""
    work = tmp_path / "project"
    shutil.copytree(fixtures_dir / "tiny_project", work)
    target = str(work / "src" / "arith.py")
    test_file = str(work / "tests" / "test_arith.py")
    return str(work), target, test_file


def test_collect_eviction_data_real_per_test_tokens_and_times(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    work, target, test_file = _stage(tmp_path, fixtures_dir)
    data = GenericBackend("python", (".py",)).collect_eviction_data(
        [test_file], work, measure_coverage_of=target, timeout=120, warmup=False,
    )
    assert data is not None
    assert data.includes_kills is False

    def _find(ids, suffix: str):
        for i in ids:
            if i == suffix or i.endswith("." + suffix):
                return i
        return None

    # JUnit lists every test, so all eight fixture tests appear, under their
    # JUnit ids (path.module.qualname).
    assert len(data.per_test_tokens) == 8
    assert all("test_arith" in tid for tid in data.per_test_tokens)

    add_id = _find(
        data.per_test_tokens, "test_arith.test_add_two_positives_returns_sum"
    )
    assert add_id is not None
    assert any(t.startswith("L:arith.py:") for t in data.per_test_tokens[add_id])

    all_tokens = set().union(*data.per_test_tokens.values())
    assert any(t.startswith("L:arith.py:") for t in all_tokens)
    assert any(t.startswith("B:arith.py:") for t in all_tokens)

    assert add_id in data.per_test_time_ms
    assert data.per_test_time_ms[add_id] >= 0.0
    assert data.suite_time_ms > 0.0


def test_collect_eviction_data_real_returns_none_on_red_suite(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    work, target, _ = _stage(tmp_path, fixtures_dir)
    bad = Path(work) / "tests" / "test_red.py"
    bad.write_text("def test_fails():\n    assert False\n", encoding="utf-8")
    data = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(bad)], work, measure_coverage_of=target, timeout=120, warmup=False,
    )
    assert data is None


def test_collect_eviction_data_includes_kills_with_a_killer_matrix(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """A kill matrix whose mutant records a killer node id sets
    includes_kills and adds a K:<id> token to the matching test."""
    import json

    work, target, test_file = _stage(tmp_path, fixtures_dir)
    # Coverage-only baseline: no kills.
    base = GenericBackend("python", (".py",)).collect_eviction_data(
        [test_file], work, measure_coverage_of=target, timeout=120, warmup=False,
    )
    assert base is not None and base.includes_kills is False

    km = Path(work) / "tests" / "kill_matrix.json"
    km.write_text(
        json.dumps({"mutants": [
            {"id": "m0", "status": "killed",
             "killer": "test_arith.py::test_add_two_positives_returns_sum",
             "killers": ["test_arith.py::test_add_two_positives_returns_sum"],
             "killing_tests": [{"file": "test_arith.py", "params": None, "text": "",
                                "qualname": "test_add_two_positives_returns_sum"}]},
        ]}),
        encoding="utf-8",
    )
    data = GenericBackend("python", (".py",)).collect_eviction_data(
        [test_file], work, measure_coverage_of=target, timeout=120, warmup=False,
        kill_matrix_path=str(km),
    )
    assert data is not None and data.includes_kills is True
    killer = next(
        t for t in data.per_test_tokens
        if t.endswith("test_arith.test_add_two_positives_returns_sum")
    )
    assert "K:m0" in data.per_test_tokens[killer]


# Tests that replace a built-in function with ``monkeypatch``, in shapes that
# generated tests take. coverage.py flushes a test's data when the test function
# returns, before fixture teardown restores the built-in, so without the
# protection each of these crashes the instrumented run although every test passes.
_REPLACING = {
    "len_negative_for_lists": (
        "test_replaces_len",
        "def test_replaces_len(monkeypatch):\n"
        "    original_len = builtins.len\n"
        "\n"
        "    def fake_len(obj):\n"
        "        if isinstance(obj, list):\n"
        "            return -1\n"
        "        return original_len(obj)\n"
        "\n"
        "    monkeypatch.setattr(builtins, 'len', fake_len)\n"
        "    assert is_positive(1) is True\n",
    ),
    "range_with_one_argument": (
        "test_replaces_range",
        "def test_replaces_range(monkeypatch):\n"
        "    def fake_range(n):\n"
        "        assert n == 2\n"
        "        return iter([-1, 0])\n"
        "\n"
        "    monkeypatch.setattr(builtins, 'range', fake_range)\n"
        "    assert is_positive(1) is True\n",
    ),
    "set_with_one_argument": (
        "test_replaces_set",
        "def test_replaces_set(monkeypatch):\n"
        "    real_set = builtins.set\n"
        "\n"
        "    class CustomSet:\n"
        "        def __init__(self, data):\n"
        "            self.data = real_set(data)\n"
        "\n"
        "    def fake_set(iterable):\n"
        "        return CustomSet(iterable)\n"
        "\n"
        "    monkeypatch.setattr(builtins, 'set', fake_set)\n"
        "    assert is_positive(1) is True\n",
    ),
}


def _replacing_suite(work: str, body: str) -> str:
    path = Path(work) / "tests" / "test_replacing.py"
    path.write_text(
        "import builtins\n\nfrom arith import add, is_positive\n\n\n"
        "def test_before():\n    assert add(2, 3) == 5\n\n\n"
        + body
        + "\n\ndef test_after():\n    assert is_positive(-1) is False\n",
        encoding="utf-8",
    )
    return str(path)


def _short(test_id: str) -> str:
    return test_id.rsplit(".", 1)[-1]


@pytest.mark.parametrize("shape", sorted(_REPLACING))
def test_eviction_records_every_test_when_a_test_replaces_a_builtin(
    tmp_path: Path, fixtures_dir: Path, shape: str,
) -> None:
    work, target, _ = _stage(tmp_path, fixtures_dir)
    name, body = _REPLACING[shape]
    suite = _replacing_suite(work, body)
    data = GenericBackend("python", (".py",)).collect_eviction_data(
        [suite], work, measure_coverage_of=target, timeout=120, warmup=False,
    )
    assert data is not None
    assert sorted(map(_short, data.per_test_tokens)) == sorted(
        ["test_before", name, "test_after"])
    for test_id, tokens in data.per_test_tokens.items():
        assert any(t.startswith("L:arith.py:") for t in tokens), test_id
        assert test_id in data.per_test_time_ms


def test_the_code_under_test_still_sees_a_replaced_builtin(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """Only coverage.py is bound to the real functions: a test asserting how
    the code under test behaves under a replacement passes in the
    instrumented run as in a plain run. The replacement also returns -1 for
    lists, which crashes coverage.py's bookkeeping, so one run shows both the
    protection and the code under test's view. Another test runs first, as in
    a real suite: pytest's JUnit writer compiles a regular expression for the
    first report, and the compiler calls ``len`` on lists, which the
    protection leaves alone."""
    work, _, _ = _stage(tmp_path, fixtures_dir)
    counting = Path(work) / "src" / "counting.py"
    counting.write_text("def size(items):\n    return len(items)\n",
                        encoding="utf-8")
    suite = Path(work) / "tests" / "test_counting.py"
    suite.write_text(
        "import builtins\n\nfrom counting import size\n\n\n"
        "class Marker:\n"
        "    def __len__(self):\n"
        "        return 1\n"
        "\n\n"
        "def test_before():\n"
        "    assert size([1, 2]) == 2\n"
        "\n\n"
        "def test_size_sees_the_replacement(monkeypatch):\n"
        "    original_len = builtins.len\n"
        "\n"
        "    def fake_len(obj):\n"
        "        if isinstance(obj, Marker):\n"
        "            return 42\n"
        "        if isinstance(obj, list):\n"
        "            return -1\n"
        "        return original_len(obj)\n"
        "\n"
        "    monkeypatch.setattr(builtins, 'len', fake_len)\n"
        "    assert size(Marker()) == 42\n",
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), "-q", "-p",
         "no:cacheprovider"],
        cwd=work, capture_output=True, text=True, timeout=120,
    )
    assert plain.returncode == 0, plain.stdout[-500:]
    data = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(suite)], work, measure_coverage_of=str(counting), timeout=120,
        warmup=False,
    )
    assert data is not None
    assert sorted(map(_short, data.per_test_tokens)) == [
        "test_before", "test_size_sees_the_replacement"]
    for tokens in data.per_test_tokens.values():
        assert any(t.startswith("L:counting.py:") for t in tokens)


def test_a_replacement_that_breaks_pytest_itself_still_fails_the_run(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """The documented limit of the protection: when the first test of the
    run replaces ``len``, pytest's JUnit writer compiles its first regular
    expression under the replacement and pytest stops with an internal
    error. The protection leaves pytest alone, so collecting the per-test
    evidence fails and the phase end falls back. The plain run passes."""
    work, target, _ = _stage(tmp_path, fixtures_dir)
    name, body = _REPLACING["len_negative_for_lists"]
    suite = Path(work) / "tests" / "test_first_replaces.py"
    suite.write_text(
        "import builtins\n\nfrom arith import add, is_positive\n\n\n"
        + body
        + "\n\ndef test_after():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), "-q", "-p",
         "no:cacheprovider"],
        cwd=work, capture_output=True, text=True, timeout=120,
    )
    assert plain.returncode == 0, plain.stdout[-500:]
    failures = []
    data = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(suite)], work, measure_coverage_of=target, timeout=120,
        warmup=False,
        failure_fn=lambda kind, detail: failures.append((kind, detail)),
    )
    assert data is None
    assert failures and failures[-1][0] == "infrastructure"
    assert "__len__() should return >= 0" in failures[-1][1]


def test_every_coverage_module_is_bound_before_the_last_test_runs(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """The protection binds once at configure time. A coverage.py module
    first imported later would stay unbound; the last test of the run checks
    that none exists after every earlier test's data was flushed."""
    work, target, _ = _stage(tmp_path, fixtures_dir)
    suite = Path(work) / "tests" / "test_bound.py"
    suite.write_text(
        "import sys\n\nfrom arith import add\n\n\n"
        "def test_first():\n    assert add(1, 1) == 2\n\n\n"
        "def test_second():\n    assert add(2, 2) == 4\n\n\n"
        "def test_zz_every_coverage_module_is_bound():\n"
        "    unbound = sorted(\n"
        "        name for name, module in list(sys.modules.items())\n"
        "        if module is not None\n"
        "        and (name == 'coverage' or name.startswith('coverage.'))\n"
        "        and 'len' not in vars(module)\n"
        "    )\n"
        "    assert unbound == []\n",
        encoding="utf-8",
    )
    data = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(suite)], work, measure_coverage_of=target, timeout=120,
        warmup=False,
    )
    assert data is not None
    assert len(data.per_test_tokens) == 3
