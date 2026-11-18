"""The Python backend's tree-sitter analysis: ``check_structure``,
``test_metrics``, ``skeleton``, ``function_fingerprints`` and ``imports``.
"""
from __future__ import annotations

import pytest

from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend

from testuary.languages import get_backend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


PY = get_backend("python")


# check_structure

@pytest.mark.parametrize("backend,src", [
    (PY, "def f():\n    return 1\n"),
])
def test_check_structure_accepts_valid(backend, src):
    r = backend.check_structure(src)
    assert r.ok is True
    assert r.message == ""
    assert r.line == 0 and r.col == 0


@pytest.mark.parametrize("backend,src", [
    (PY, "def f(:\n    return\n"),
])
def test_check_structure_rejects_invalid_with_location(backend, src):
    r = backend.check_structure(src)
    assert r.ok is False
    assert "line" in r.message       # human-readable message
    assert r.line >= 1 and r.col >= 1  # structured location for the snippet


def test_check_structure_fails_loudly_when_structural_parser_is_unavailable(
    monkeypatch,
):
    """A missing grammar is not mislabeled as valid target source: the
    backend's failure reaches the caller."""
    def unavailable(_backend, _source):
        raise ModuleNotFoundError("tree_sitter_python")

    monkeypatch.setattr(ProjectPythonBackend, "check_structure", unavailable)
    with pytest.raises(RuntimeError, match="missing_setup.*tree_sitter_python"):
        PY.check_structure("def test_a():\n    assert True\n")


# test_metrics: pytest semantics, matching the AST analyzer

def test_metrics_counts_tests_and_assertions():
    """``def test_*`` counts as a test, ``assert`` as an assertion, and a
    ``with pytest.raises(...)`` item as one assertion."""
    src = (
        "def test_one():\n"
        "    assert add(1, 2) == 3\n"
        "    assert add(0, 0) == 0\n"
        "def test_two():\n"
        "    with pytest.raises(ValueError):\n"
        "        boom()\n"
    )
    m = PY.test_metrics(src)
    assert m["test_count"] == 2
    assert m["assertion_count"] == 3   # 2 asserts and 1 pytest.raises


def test_metrics_counts_fixtures_qualified_and_imported():
    """``@pytest.fixture``, ``@fixture``, ``@pytest.fixture(scope=...)``
    each count once per function."""
    src = (
        "import pytest\n"
        "from pytest import fixture\n"
        "@pytest.fixture\n"
        "def a():\n    return 1\n"
        "@fixture\n"
        "def b():\n    return 2\n"
        "@pytest.fixture(scope='session')\n"
        "def c():\n    return 3\n"
    )
    assert PY.test_metrics(src)["fixture_count"] == 3


def test_metrics_counts_parametrize_cases_literal_only():
    """A literal ``@pytest.mark.parametrize("n", [..])`` contributes
    len(case_list), also when a module-level name binds the literal list,
    as pytest runs its cases. Non-literal case lists are under-counted (0)."""
    src = (
        "@pytest.mark.parametrize('x', [1, 2, 3])\n"
        "def test_one(x):\n    assert x\n"
        "@pytest.mark.parametrize('y', [(1, 2), (3, 4)])\n"
        "def test_two(y):\n    assert y\n"
        "CASES = [1, 2]\n"
        "@pytest.mark.parametrize('z', CASES)\n"
        "def test_three(z):\n    assert z\n"
        "@pytest.mark.parametrize('w', list(range(4)))\n"
        "def test_four(w):\n    assert w\n"
    )
    assert PY.test_metrics(src)["parametrize_case_count"] == 7   # 3 + 2 + 2 + 0


def test_metrics_syntax_error_keeps_counts_unavailable():
    src = "def test_x(:\n    syntax error\n"
    m = PY.test_metrics(src)
    assert m == dict.fromkeys(("loc", "test_count", "assertion_count",
                             "fixture_count", "parametrize_case_count"))
    observation = PY.test_metrics_observation(src)
    assert not observation.completed and observation.value is None
    assert observation.reason and observation.parser


def test_metrics_loc_counts_code_lines():
    """loc is code lines: blank lines and pure comment lines do not
    count, and a trailing comment leaves a code line a code line.
    Delivered suites keep the comment blocks of tests the set cover removed,
    so a raw line count would measure what removal left behind."""
    assert PY.test_metrics("a\nb\n")["loc"] == 2
    assert PY.test_metrics("a\nb")["loc"] == 2
    assert PY.test_metrics("")["loc"] == 0
    assert PY.test_metrics("# banner\n\na = 1  # trailing\n")["loc"] == 1


def test_metrics_match_the_ast_analyzer_on_a_real_suite():
    """Compare tree-sitter test counts with the independent Python AST analyzer."""
    import tempfile
    from pathlib import Path
    from testuary import measurement

    src = (
        "import pytest\n"
        "from pytest import fixture\n"
        "@fixture\n"
        "def data():\n    return [0] * 10\n"
        "@pytest.mark.parametrize('n', [1, 2, 3, 4])\n"
        "def test_scale(n, data):\n"
        "    assert len(data) == 10\n"
        "    with pytest.raises(KeyError):\n"
        "        data[n]\n"
        "def test_plain():\n"
        "    assert True\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test_x.py"
        p.write_text(src, encoding="utf-8")
        ast_m = measurement._analyze_test_file(p)
    ts_m = PY.test_metrics(src)
    assert ts_m["test_count"] == ast_m.test_count
    assert ts_m["assertion_count"] == ast_m.assertion_count
    assert ts_m["fixture_count"] == ast_m.fixture_count
    assert ts_m["parametrize_case_count"] == ast_m.parametrize_case_count
    assert ts_m["loc"] == ast_m.loc


# skeleton: syntax only

_SKEL = {
    "python": (
        "class T:\n    def test_a(self):\n        assert f(1) == 2\n"
        "    def test_b(self):\n        pass\n"
        "def helper():\n    return 9\n",
        ("T", "T.test_a", "T.test_b", "helper"),
    ),
}


@pytest.mark.parametrize("lang", ["python"])
def test_skeleton_is_qualified_and_ordered(lang):
    src, expected = _SKEL[lang]
    assert get_backend(lang).skeleton(src) == expected


@pytest.mark.parametrize("lang", ["python"])
def test_skeleton_empty_on_parse_error(lang):
    assert get_backend(lang).skeleton("@@@ not valid @@@") == ()


# function_fingerprints: blind to formatting and comments, not to code

def test_fingerprints_ignore_formatting_and_comments():
    a = "def f():\n    x = [0]*10\n    return g(x)\n"
    b = "# header\ndef f():\n    x = [0] * 10   # inline\n    return g(x)\n"
    fa, fb = PY.function_fingerprints(a), PY.function_fingerprints(b)
    touched = sum(1 for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    assert touched == 0


def test_fingerprints_detect_a_real_body_change():
    a = "def f():\n    return 9\ndef g():\n    return 1\n"
    b = "def f():\n    return 10\ndef g():\n    return 1\n"
    fa, fb = PY.function_fingerprints(a), PY.function_fingerprints(b)
    touched = sum(1 for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    assert touched == 1   # only f changed


def test_fingerprints_qualified_keys_avoid_same_name_collision():
    """Keep same-named methods in different classes under distinct fingerprint keys."""
    src = (
        "class A:\n    def setUp(self):\n        self.x = 1\n"
        "class B:\n    def setUp(self):\n        self.y = 2\n"
    )
    fps = PY.function_fingerprints(src)
    assert set(fps) == {"A.setUp", "B.setUp"}


@pytest.mark.parametrize("lang", ["python"])
def test_fingerprints_available_for_every_language(lang):
    src = _SKEL[lang][0]
    fps = get_backend(lang).function_fingerprints(src)
    assert isinstance(fps, dict) and len(fps) >= 1


# The imports function


def test_imports_python_covers_all_documented_shapes():
    """Every documented import shape: ``import x``, ``import x, y``, dotted
    ``import a.b.c``, ``from x.y import z`` (leaf dropped on purpose) and
    aliased imports, in order and without duplicates."""
    src = (
        "import functools\n"
        "import os, sys\n"
        "import sympy.core.expr\n"
        "from sympy.core import expr\n"
        "from sympy.core.expr import Expr\n"
        "import functools as ft\n"  # an alias adds no duplicate
    )
    out = PY.imports(src)
    assert out == (
        "functools", "os", "sys",
        "sympy.core.expr", "sympy.core",
    )


def test_imports_python_relative_imports_kept_as_dot_prefixed():
    """Relative imports cannot be resolved without the importer's
    package context. ``imports`` emits them verbatim so the graph
    builder visibly skips them."""
    src = "from . import sibling\nfrom ..pkg import x\nfrom .submod import y\n"
    out = PY.imports(src)
    assert "." in out
    assert "..pkg" in out
    assert ".submod" in out


def test_imports_parse_error_returns_empty_tuple():
    """Unparseable source gives ``()``, as ``test_metrics`` degrades too."""
    assert PY.imports("def f(:\n") == ()
