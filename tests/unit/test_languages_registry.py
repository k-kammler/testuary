"""The language-backend registry (``get_backend``, ``detect_language``,
``available_language_names``) and ``GenericBackend`` against small fixtures in
a temporary directory, without subprocess mocks where the code does real I/O."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from langadapter import placement

import pytest

from testuary.languages import (
    LanguageBackend,
    GenericBackend,
    available_language_names,
    detect_language,
    get_backend,
)
import os
import re

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

# ``languages.TestRunResult`` is used through the module: a top-level
# ``TestRunResult`` in a test module trips pytest's ``Test*`` class collection.

_TINY_PROJECT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "tiny_project"
)


# Registry surface

def test_get_backend_python_returns_python_backend():
    backend = get_backend("python")
    assert isinstance(backend, GenericBackend)
    assert backend.name == "python"
    assert isinstance(backend, LanguageBackend)


def test_get_backend_unknown_returns_none():
    """An unknown language gives None, which callers report as unsupported."""
    assert get_backend("haskell") is None


def test_available_language_names_sorted_and_complete():
    """``--languages`` validates against this list of every registered
    backend, in sorted order: the built-in ones and any installed (C's)."""
    from langadapter import names

    found = available_language_names()
    assert found == sorted(names()) and {"java", "python"} <= set(found)


# detect_language

@pytest.mark.parametrize("path,expected", [
    ("foo.py", "python"),
    ("/abs/dir/foo.py", "python"),
    ("FOO.PY", "python"),                  # case-insensitive
    ("src/main/java/demo/Calc.java", "java"),
    ("README.md", "unknown"),
    ("no_extension", "unknown"),
])
def test_detect_language_known_extensions(path: str, expected: str) -> None:
    assert detect_language(path) == expected


# is_test_file: per-language classification

@pytest.mark.parametrize("name,expected", [
    ("foo.py", False),
    ("test_foo.py", True),
    ("foo_test.py", True),
    ("conftest.py", True),
    ("tests.py", True),
    ("FooTest.py", False),          # a Java-style name is no Python test
    ("test_foo.txt", False),        # wrong extension
])
def test_python_backend_is_test_file(name: str, expected: bool) -> None:
    assert GenericBackend("python", (".py",)).is_test_file(Path(name)) is expected


# target_test_path: the write path, symmetric with find_target_test_file

def test_target_test_path_python_knows_only_the_delivered_file(tmp_path):
    p = GenericBackend("python", (".py",)).target_test_path(
        str(tmp_path), "/abs/proj/foo.py",
    )
    assert p == tmp_path / "test_foo.py"


# scratch_test_path: a language-valid name for a one-off measurement file

def test_scratch_test_path_delivered_stages_under_the_delivered_name(tmp_path):
    """``stage_as_delivered=True`` stages under the delivered name for that
    target, so a measurement runs the suite under the module name it ships
    with."""
    backend = GenericBackend("python", (".py",))
    source = "def test_a():\n    pass\n"
    staged = backend.scratch_test_path(
        str(tmp_path), source, "/abs/proj/foo.py", stage_as_delivered=True)
    delivered = backend.target_test_path(
        str(tmp_path), "/abs/proj/foo.py")
    assert staged == tmp_path / "test_foo.py"
    assert staged.name == delivered.name


def test_scratch_test_path_fixed_stem_is_target_independent(tmp_path):
    """``stage_as_delivered=False`` stages under one fixed stem whatever the
    target is, so the per-test ids and killer strings a measurement records
    read the same across every target."""
    backend = GenericBackend("python", (".py",))
    source = "def test_a():\n    pass\n"
    first = backend.scratch_test_path(
        str(tmp_path), source, "/abs/proj/foo.py", stage_as_delivered=False)
    second = backend.scratch_test_path(
        str(tmp_path), source, "/abs/proj/other.py", stage_as_delivered=False)
    assert first == second == tmp_path / "test_candidate.py"


def test_scratch_test_path_requires_the_staging_choice(tmp_path):
    """The name is the caller's decision, so the keyword has no default."""
    with pytest.raises(TypeError):
        GenericBackend("python", (".py",)).scratch_test_path(
            str(tmp_path), "def test_a():\n    pass\n", "/abs/proj/foo.py")


def test_scratch_test_path_override_keeps_a_class_matching_name(tmp_path):
    """A language whose test filename must match a declaration in the source
    overrides the default, and the override is what the staging callers
    receive."""
    class ClassNamedBackend(GenericBackend):
        def scratch_test_path(self, directory, source, target_file_path,
                              *, stage_as_delivered):
            declared = source.split("public class ", 1)[1].split()[0]
            return Path(directory) / f"{declared}.java"

    staged = ClassNamedBackend("java", (".java",)).scratch_test_path(
        str(tmp_path), "public class WidgetTest { }", "/abs/proj/Widget.java",
        stage_as_delivered=True)
    assert staged == tmp_path / "WidgetTest.java"


# classify_test_kind: collectable suite roles

@pytest.mark.parametrize("filename,expected", [
    ("test_foo.py", "coverage"),
    ("conftest.py", "unknown"),
    ("not_a_test.py", "unknown"),
    ("test_bar.txt", "unknown"),
])
def test_classify_test_kind_python(filename, expected):
    assert GenericBackend("python", (".py",)).classify_test_kind(filename) == expected


def test_target_test_path_matches_find_target_test_file_python(tmp_path):
    """Symmetry: writing at ``target_test_path`` is then found by
    ``find_target_test_file`` (the convention is the same on both sides)."""
    target = "/abs/proj/widget.py"
    write_path = GenericBackend("python", (".py",)).target_test_path(str(tmp_path), target)
    write_path.write_text("def test_one(): pass\n", encoding="utf-8")
    found = GenericBackend("python", (".py",)).find_target_test_file(str(tmp_path), target)
    assert found == write_path


# GenericBackend.find_target_test_file

def test_find_target_test_file_picks_canonical_name(tmp_path):
    """``test_<stem>.py`` wins over other files: targets share the tests
    directory, so an alphabetical pick would take another target's file."""
    (tmp_path / "test_foo.py").write_text("# rank")
    (tmp_path / "test_bar.py").write_text("# other")
    out = GenericBackend("python", (".py",)).find_target_test_file(str(tmp_path), "/abs/foo.py")
    assert out == tmp_path / "test_foo.py"


def test_find_target_test_file_falls_back_to_sole_sibling(tmp_path):
    """Without the conventional name the single test_*.py file is picked. The
    test_failed.py marker is no candidate."""
    (tmp_path / "test_helpers.py").write_text("# alt")
    (tmp_path / "test_failed.py").write_text("# excluded")
    out = GenericBackend("python", (".py",)).find_target_test_file(str(tmp_path), "/abs/foo.py")
    assert out == tmp_path / "test_helpers.py"


def test_find_target_test_file_none_on_ambiguity(tmp_path):
    """Several unrelated files and no conventional name give None."""
    (tmp_path / "test_a.py").write_text("")
    (tmp_path / "test_b.py").write_text("")
    out = GenericBackend("python", (".py",)).find_target_test_file(str(tmp_path), "/abs/foo.py")
    assert out is None


# GenericBackend.setup_test_env

def test_setup_test_env_writes_conftest_with_self_relative_path(tmp_path):
    """conftest.py is written into test_output_dir and resolves
    source_dir self-relatively (counting dirname() levels) so the
    file works after the test dir is renamed or cloned to a worker."""
    source_dir = tmp_path / "project"
    source_dir.mkdir()
    test_dir = source_dir / "tests"
    test_dir.mkdir()

    GenericBackend("python", (".py",)).setup_test_env(str(test_dir), str(source_dir))

    conftest = test_dir / "conftest.py"
    assert conftest.exists()
    body = conftest.read_text(encoding="utf-8")
    # Self-relative: an os.path.dirname() walk instead of an absolute path.
    assert "os.path.dirname" in body
    assert "sys.path.insert" in body
    # A module is imported once per test process: no re-exec or sys.modules
    # swap, since two live copies break identity (isinstance, recursive_repr
    # guards). Process-level coverage tracks imports without a re-exec.
    assert "LLM" "IOS_TARGET_FILE" not in body
    assert "exec_module" not in body
    # The mutation source-root override stays: a worker imports the
    # source from its own clone.
    assert "LANGADAPTER_MUTATION_SOURCE_ROOT" in body
    # The gated in-process fd watcher is present (off unless LANGADAPTER_FDLIMIT_DEBUG),
    # so a descriptor leak can be captured from inside the affected pytest.
    assert "LANGADAPTER_FDLIMIT_DEBUG" in body and "_fd_watch" in body


def test_setup_test_env_resolves_source_dir_relative_for_sibling_test_dir(
    tmp_path,
):
    """The setup file resolves a sibling project from its own location, so
    worker clones import their mutated source. No absolute project path is
    embedded."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    source_dir = work_dir / "sympy"
    source_dir.mkdir()
    test_dir = work_dir / "tests"
    test_dir.mkdir()

    GenericBackend("python", (".py",)).setup_test_env(str(test_dir), str(source_dir))
    body = (test_dir / "conftest.py").read_text(encoding="utf-8")

    # The path is computed at runtime, and the absolute path does not appear.
    assert "os.path.dirname" in body
    assert str(source_dir) not in body, (
        "conftest baked in the absolute source_dir path. This "
        "breaks cosmic-ray worker clones (they import the unmutated "
        "original instead of the mutated clone). Use a runtime-relative "
        "expression based on __file__."
    )
    # The project directory is named relative to the work directory.
    assert "sympy" in body


def test_setup_test_env_conftest_subpath_uses_posix_separators(tmp_path):
    """The conftest's _PROJECT_ROOT sub-path is embedded with forward slashes,
    so a conftest generated on a Windows host but run inside a Linux container
    resolves the import root correctly. Backslashes would be read in the
    container as one literal-backslash filename and the import root would not
    exist (it is then masked only when the package is also installed)."""
    work_dir = tmp_path / "work"
    source_dir = work_dir / "a" / "b" / "sympy"
    source_dir.mkdir(parents=True)
    test_dir = work_dir / "tests"
    test_dir.mkdir(parents=True)

    GenericBackend("python", (".py",)).setup_test_env(str(test_dir), str(source_dir))
    body = (test_dir / "conftest.py").read_text(encoding="utf-8")

    # The sub-path below the common work directory uses "/" on every host.
    assert "a/b/sympy" in body


# collect_eviction_data: per-test reduction inputs

def test_eviction_data_dataclass_shape():
    """``EvictionData`` carries per-test tokens, per-test times, the suite
    total, and the ``includes_kills`` flag."""
    from testuary.languages.generic import EvictionData

    ed = EvictionData(
        per_test_tokens={"t": frozenset({"L:m.py:1"})},
        per_test_time_ms={"t": 1.5},
        suite_time_ms=1.5,
        includes_kills=False,
    )
    assert ed.per_test_tokens["t"] == frozenset({"L:m.py:1"})
    assert ed.per_test_time_ms["t"] == 1.5
    assert ed.suite_time_ms == 1.5
    assert ed.includes_kills is False


def test_prune_tests_removes_matched_defs_and_methods():
    """prune_tests removes a matched top-level function (with its
    decorators), a class whose every member is a matched test, and an
    individual matched method from a partially reduced class
    while leaving its surviving siblings in place. Module-level code is
    never touched. The result parses."""
    backend = GenericBackend("python", (".py",))
    src = (
        "import pytest\n"
        "from mod import f\n\n"
        "DATA = [1, 2]\n\n"
        "def test_keep():\n    assert f(1) == 1\n\n"
        '@pytest.mark.parametrize("v", [1, 2])\n'
        "def test_param(v):\n    assert f(v) == v\n\n"
        "def test_drop():\n    assert f(2) == 2\n\n"
        "class TestAllGone:\n"
        "    def test_a(self):\n        assert True\n\n"
        "    def test_b(self):\n        assert True\n\n"
        "class TestPartial:\n"
        "    def test_c(self):\n        assert True\n\n"
        "    def test_d(self):\n        assert True\n"
    )
    remove = [
        "tests.test_x.test_param",
        "tests.test_x.test_drop",
        "tests.test_x.TestAllGone.test_a",
        "tests.test_x.TestAllGone.test_b",
        "tests.test_x.TestPartial.test_c",  # individual method, test_d kept
    ]
    out = backend.prune_tests(src, remove)
    assert backend.check_structure(out).ok
    assert "def test_keep" in out
    assert "def test_param" not in out
    assert "parametrize" not in out  # the decorator goes with the function
    assert "def test_drop" not in out
    assert "class TestAllGone" not in out
    assert "class TestPartial" in out  # the class survives via test_d
    assert "def test_c" not in out     # the matched method is gone
    assert "def test_d" in out         # its sibling stays
    assert "import pytest" in out and "DATA = [1, 2]" in out


def test_prune_tests_drops_class_left_with_only_a_comment():
    """A class whose every statement is a removed test is removed whole even
    when it holds a comment. A comment is a named child but not a statement, so
    keeping it alone would leave ``class X:`` with an empty body (a syntax
    error). The pruned source always parses. A class that keeps a real
    statement member (attribute, docstring) is reduced, not removed."""
    backend = GenericBackend("python", (".py",))
    src = (
        "class TestCommentOnly:\n"
        "    def test_a(self):\n        assert True\n\n"
        "    # an inline comment between methods\n"
        "    def test_b(self):\n        assert True\n\n"
        "class TestHasAttr:\n"
        "    marker = 1\n"
        "    def test_c(self):\n        assert True\n"
    )
    out = backend.prune_tests(src, [
        "t.TestCommentOnly.test_a",
        "t.TestCommentOnly.test_b",
        "t.TestHasAttr.test_c",
    ])
    assert backend.check_structure(out).ok  # an empty class body would not parse
    assert "class TestCommentOnly" not in out
    assert "an inline comment" not in out  # the comment goes with the class
    # The attribute-bearing class survives (a real statement remains).
    assert "class TestHasAttr" in out and "marker = 1" in out
    assert "def test_c" not in out


def test_prune_tests_method_id_does_not_remove_same_named_top_level_def():
    """A class-method remove id (``mod.TestFoo.test_x``) removes only the
    method, never a same-named top-level ``def test_x``. Both names are
    ``.``-bounded suffixes of the id, so the id resolves to the longest
    one (the method). The bare top-level function the cover kept survives.
    This is the shape of an xarray test file: module-level ``test_*``
    functions beside ``Test*`` classes whose methods share leaf names."""
    backend = GenericBackend("python", (".py",))
    src = (
        "import target\n\n"
        "def test_x():\n    assert target.f1() == 1\n\n"
        "def test_keep():\n    assert target.f3() == 3\n\n"
        "class TestFoo:\n"
        "    def test_x(self):\n        assert target.f2() == 2\n"
    )
    # The cover keeps both top-level functions and removes the class method.
    out = backend.prune_tests(src, ["m.TestFoo.test_x"])
    assert backend.check_structure(out).ok
    assert "def test_x():" in out          # the kept top-level def survives
    assert "def test_keep():" in out
    assert "class TestFoo" not in out      # its only method gone -> class gone

    # The reverse id removes only the top-level def, leaving the method.
    out2 = backend.prune_tests(src, ["m.test_x"])
    assert backend.check_structure(out2).ok
    assert "def test_x():" not in out2
    assert "class TestFoo" in out2
    assert "def test_x(self)" in out2

    # Both ids present -> both the top-level def and the method are removed.
    out3 = backend.prune_tests(src, ["m.test_x", "m.TestFoo.test_x"])
    assert "def test_x():" not in out3
    assert "class TestFoo" not in out3
    assert "def test_keep():" in out3


def test_prune_tests_keeps_class_setup_and_helpers():
    """Removing a class's only matched test leaves its non-test members
    (setup_method, helpers, attributes) intact: those are never test ids,
    so the class is not emptied and stays."""
    backend = GenericBackend("python", (".py",))
    src = (
        "class TestThing:\n"
        "    DATA = [1]\n\n"
        "    def setup_method(self):\n        self.x = 1\n\n"
        "    def _helper(self):\n        return self.x\n\n"
        "    def test_a(self):\n        assert self._helper() == 1\n\n"
        "    def test_b(self):\n        assert self.x == 1\n"
    )
    out = backend.prune_tests(src, ["t.TestThing.test_a"])
    assert backend.check_structure(out).ok
    assert "class TestThing" in out
    assert "def setup_method" in out
    assert "def _helper" in out
    assert "DATA = [1]" in out
    assert "def test_a" not in out
    assert "def test_b" in out


def test_prune_tests_no_match_returns_source_unchanged():
    backend = GenericBackend("python", (".py",))
    src = "def test_a():\n    assert True\n"
    assert backend.prune_tests(src, ["nomatch.test_z"]) == src


def test_prune_tests_parse_error_returns_source_unchanged():
    backend = GenericBackend("python", (".py",))
    broken = "def test_a(:\n    assert True\n"
    assert backend.prune_tests(broken, ["x.test_a"]) == broken


def test_setup_test_env_conftest_resolves_in_a_cloned_dir(tmp_path):
    """A conftest copied with its work directory to a new place, as in a
    cosmic-ray worker clone, computes ``_PROJECT_ROOT`` inside the clone, not
    the original."""
    import shutil
    orig = tmp_path / "orig"
    orig.mkdir()
    (orig / "sympy").mkdir()
    (orig / "tests").mkdir()
    GenericBackend("python", (".py",)).setup_test_env(
        str(orig / "tests"), str(orig / "sympy"),
    )

    # A cosmic-ray worker clone.
    clone = tmp_path / "clone_42"
    shutil.copytree(orig, clone)

    # Run the conftest's path code with the clone's __file__.
    body = (clone / "tests" / "conftest.py").read_text(encoding="utf-8")
    # Only the path-resolution prologue runs, not the fresh-load hook.
    prologue_lines = []
    for line in body.splitlines():
        prologue_lines.append(line)
        if "_PROJECT_ROOT" in line and "sys.path.insert" in line:
            break
        if "sys.path.insert" in line and "_PROJECT_ROOT" in prologue_lines[-2] if len(prologue_lines) >= 2 else False:
            break
    # The prologue ends at the first "Fresh-load" comment.
    cut = body.find("# Fresh-load")
    if cut == -1:
        cut = len(body)
    prologue = body[:cut]

    sandbox: dict = {
        "__file__": str(clone / "tests" / "conftest.py"),
        "__name__": "__main__",
    }
    exec(prologue, sandbox)
    root_in_conftest = sandbox.get("_PROJECT_ROOT")
    assert root_in_conftest is not None
    assert str(clone) in root_in_conftest, (
        f"_PROJECT_ROOT resolved to {root_in_conftest!r}, "
        f"which is NOT inside the clone {str(clone)!r}. cosmic-ray "
        f"workers will import the unmutated original."
    )
    assert str(orig) not in root_in_conftest


def _conftest_prologue(text: str) -> str:
    """The path-resolution prologue of a generated conftest, up to (but
    not including) the coverage fresh-load hook."""
    cut = text.find("# Fresh-load")
    return text[:cut] if cut != -1 else text


def test_setup_test_env_conftest_honors_mutation_source_root_override(
    tmp_path, monkeypatch,
):
    """A cosmic-ray HTTP worker runs the conftest from the live shared tests
    directory (the sibling layout copies no tests into the clone), so the
    ``__file__``-relative path names the unmutated source.
    ``LANGADAPTER_MUTATION_SOURCE_ROOT``, set by each worker to its clone,
    wins so the worker imports the mutated copy. Without it every mutant
    survives. The slow cosmic-ray integration test covers this end to end.
    """
    work = tmp_path / "work"
    (work / "proj").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    GenericBackend("python", (".py",)).setup_test_env(str(work / "tests"), str(work / "proj"))
    prologue = _conftest_prologue(
        (work / "tests" / "conftest.py").read_text(encoding="utf-8")
    )
    live_conftest = str(work / "tests" / "conftest.py")
    # A worker clone's source root: a distinct path from the live source.
    clone_src = str(tmp_path / "cr_worker_0_xyz")

    saved_path = list(sys.path)
    try:
        # Override set -> _PROJECT_ROOT is the clone, not the live source.
        monkeypatch.setenv("LANGADAPTER_MUTATION_SOURCE_ROOT", clone_src)
        sb: dict = {"__file__": live_conftest, "__name__": "x"}
        exec(prologue, sb)
        assert sb["_PROJECT_ROOT"] == clone_src
        assert str(work / "proj") not in sb["_PROJECT_ROOT"]

        # Override absent -> falls back to the live source, the behavior
        # every non-mutation pytest run depends on.
        monkeypatch.delenv("LANGADAPTER_MUTATION_SOURCE_ROOT", raising=False)
        sb2: dict = {"__file__": live_conftest, "__name__": "x"}
        exec(prologue, sb2)
        assert str(work / "proj") in sb2["_PROJECT_ROOT"]
    finally:
        sys.path[:] = saved_path


# GenericBackend.benchmark_tests

def test_benchmark_tests_warmup_failure_returns_error(monkeypatch, tmp_path, timed_run):
    """A failing warmup -> early ``{ok: False, error, returncode}``."""
    (tmp_path / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        placement, "run_timed",
        timed_run(SimpleNamespace(
            returncode=1, stdout="FAILED test_x", stderr="",
        )),
    )
    out = GenericBackend("python", (".py",)).benchmark_tests(
        source_dir=str(tmp_path),
        test_file_paths=[str(tmp_path / "test_foo.py")],
        warmup_runs=1, benchmark_runs=5, timeout=300,
    )
    assert out["ok"] is False
    assert "FAILED" in out["error"] or "failed" in out["error"].lower()


def test_benchmark_tests_all_green_returns_stats(monkeypatch, tmp_path, timed_run):
    """Warmup and every measured repetition succeed: the stats hold median,
    mean, min, max and times_ms."""
    (tmp_path / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        placement, "run_timed",
        timed_run(SimpleNamespace(
            returncode=0, stdout="1 passed in 0.05s", stderr="",
        )),
    )
    out = GenericBackend("python", (".py",)).benchmark_tests(
        source_dir=str(tmp_path),
        test_file_paths=[str(tmp_path / "test_foo.py")],
        warmup_runs=1, benchmark_runs=3, timeout=300,
    )
    assert out["ok"] is True
    assert len(out["times_ms"]) == 3
    assert out["median_ms"] is not None
    assert out["timing_run_count"] == 3


def test_benchmark_tests_measured_rep_failure_returns_error(
    monkeypatch, tmp_path, timed_run,
):
    """A measured repetition failing after a green warmup aborts with an
    error, which catches flaky tests."""
    (tmp_path / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")
    call = {"n": 0}

    def fake_run(*a, **kw):
        call["n"] += 1
        if call["n"] == 1:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=1, stdout="FAILED flaky_test", stderr="",
        )
    monkeypatch.setattr(placement, "run_timed", timed_run(fake_run))
    out = GenericBackend("python", (".py",)).benchmark_tests(
        source_dir=str(tmp_path),
        test_file_paths=[str(tmp_path / "test_foo.py")],
        warmup_runs=1, benchmark_runs=5, timeout=300,
    )
    assert out["ok"] is False and "flaky_test" in out["error"]


def test_benchmark_tests_timeout_returns_timed_out(monkeypatch, tmp_path):
    """A rep that exceeds the timeout is normalized to
    ``{ok: False, timed_out: True}`` rather than raising, so the caller
    stays language-agnostic and never sees a subprocess exception."""
    (tmp_path / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")
    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=300)
    monkeypatch.setattr(placement, "run_timed", raise_timeout)
    out = GenericBackend("python", (".py",)).benchmark_tests(
        source_dir=str(tmp_path),
        test_file_paths=[str(tmp_path / "test_foo.py")],
        warmup_runs=1, benchmark_runs=5, timeout=300,
    )
    assert out["ok"] is False and out["timed_out"] is True


# GenericBackend.run_test_files

def _copy_tiny_project(tmp_path: Path) -> Path:
    dst = tmp_path / "proj"
    shutil.copytree(_TINY_PROJECT, dst)
    return dst


@pytest.mark.skipif(
    not _TINY_PROJECT.is_dir(), reason="tiny_project fixture missing",
)
def test_run_test_files_real_green_suite(tmp_path):
    """A real subprocess runs the green fixture suite, which catches pytest
    flag, PYTHONPATH and conftest errors a mocked test cannot."""
    proj = _copy_tiny_project(tmp_path)
    r = GenericBackend("python", (".py",)).run_test_files(
        [str(proj / "tests" / "test_arith.py")], str(proj), timeout=120,
    )
    assert r.ok is True, r.output
    assert r.collected == 8
    assert r.passed == 8 and r.failed == 0 and r.errors == 0
    assert r.line_coverage is None  # not requested


@pytest.mark.skipif(
    not _TINY_PROJECT.is_dir(), reason="tiny_project fixture missing",
)
def test_run_test_files_real_coverage(tmp_path):
    """A real coverage run: the fixture suite runs every line of arith.py, so
    line coverage is 100%. arith.py has no branches, so branch coverage is
    None ("not measurable", never 0 or 100)."""
    proj = _copy_tiny_project(tmp_path)
    r = GenericBackend("python", (".py",)).run_test_files(
        [str(proj / "tests" / "test_arith.py")],
        str(proj),
        timeout=120,
        measure_coverage_of=str(proj / "src" / "arith.py"),
    )
    assert r.ok is True, r.output
    assert r.line_coverage == 100.0
    assert r.branch_coverage is None


def test_run_test_files_coverage_line_is_pure_not_branch_blended(tmp_path):
    """line_coverage is pure line coverage (covered statements / total), not
    coverage.py's --branch-blended percent_covered. The module's every line
    runs but one branch is untaken, so pure line is 100 while the blended
    figure would be lower (~85.7)."""
    (tmp_path / "m.py").write_text(
        "def f(x):\n    y = 0\n    if x:\n        y = 1\n    return y\n",
        encoding="utf-8")
    (tmp_path / "test_m.py").write_text(
        "import m\n\n\ndef test_f():\n    assert m.f(1) == 1\n",
        encoding="utf-8")
    r = GenericBackend("python", (".py",)).run_test_files(
        [str(tmp_path / "test_m.py")], str(tmp_path), timeout=120,
        measure_coverage_of=str(tmp_path / "m.py"))
    assert r.ok is True, r.output
    assert r.line_coverage == 100.0   # every statement ran
    assert r.branch_coverage == 50.0  # 1 of 2 branches taken
    # The untaken arc is captured for branch-level feedback: only f(1) ran, so
    # the x-false branch (line 3 -> 5) was never taken.
    assert r.missing_branches == ("3->5",)


# shared_setup_files and validate_test_source

def test_python_shared_setup_files_finds_conftest(tmp_path):
    """Python's setup convention is conftest.py. Only an existing one is
    returned."""
    b = get_backend("python")
    assert b.shared_setup_files(str(tmp_path)) == ()
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")
    out = b.shared_setup_files(str(tmp_path))
    assert len(out) == 1 and out[0].endswith("conftest.py")


def test_python_validate_test_source_uses_target_interpreter(tmp_path):
    """Python target validation returns a source-bound structured outcome."""
    b = get_backend("python")
    valid = b.validate_test_source(
        "def t():\n    assert True\n", str(tmp_path)
    )
    assert valid.ok is True
    assert valid.status == "valid"
    assert "Python" in valid.validator_identity
    assert len(valid.source_sha256) == 64
    invalid = b.validate_test_source("def t(:\n    pass\n", str(tmp_path))
    assert invalid.ok is False
    assert invalid.status == "syntax_error"
    assert invalid.line == 1


# Tests for execution-context identity and cache invalidation.
def _generated_conftest(tmp_path: Path) -> str:
    source_dir = tmp_path / "project"
    (source_dir / "tests").mkdir(parents=True)
    GenericBackend("python", (".py",)).setup_test_env(
        str(source_dir / "tests"), str(source_dir),
        target_file_path=str(source_dir / "mod.py"),
    )
    return (source_dir / "tests" / "conftest.py").read_text(
        encoding="utf-8")


def test_conftest_has_no_module_swap_constructs(tmp_path):
    body = _generated_conftest(tmp_path)
    for construct in ("exec_module", "module_from_spec",
                      "spec_from_file_location"):
        assert construct not in body
    # sys.modules may be read, never assigned.
    assert not re.search(r"sys\.modules\[[^]]*\]\s*=", body)


def test_conftest_env_conditionals_are_exactly_the_documented_set(tmp_path):
    """Any new ``os.environ`` conditional in the conftest is a new
    context-dependent trigger and must be added here deliberately."""
    body = _generated_conftest(tmp_path)
    reads = set(re.findall(r"os\.environ\.get\(\s*['\"]([A-Z_]+)['\"]", body))
    assert reads == {
        # A worker imports the source from its own clone.
        "LANGADAPTER_MUTATION_SOURCE_ROOT",
        # fd-watch diagnostics (off unless armed).
        "LANGADAPTER_FDLIMIT_DEBUG",
        "LANGADAPTER_FDLEAK_THRESHOLD",
        "LANGADAPTER_FDLEAK_DIR",
        "LANGADAPTER_PROJECT_WORKZONE",
    }, f"conftest env conditionals changed: {sorted(reads)}"


def test_setup_test_env_leaves_process_environment_untouched(tmp_path):
    before = dict(os.environ)
    _generated_conftest(tmp_path)
    assert dict(os.environ) == before


def test_run_command_is_env_insensitive(tmp_path, monkeypatch):
    """The pytest command does not vary with ambient environment variables."""
    source_dir = tmp_path / "p"
    source_dir.mkdir()
    tf = tmp_path / "test_x.py"
    tf.write_text("def test_a():\n    pass\n", encoding="utf-8")

    captured = []

    def fake_run(executor, cmd, **kw):
        captured.append(list(cmd))
        return subprocess.CompletedProcess(list(cmd), 0, "1 passed", "")

    import subprocess
    from langadapter.execution import LocalExecutor
    # Suite runs launch through the public executor seam, not subprocess.run.
    monkeypatch.setattr(LocalExecutor, "run", fake_run)
    be = GenericBackend("python", (".py",))

    def pytest_cmds():
        """The pytest runs, not the interpreter probe, with each run's own
        scratch directory (its report's parent) replaced."""
        commands = []
        for command in captured:
            args = [str(a) for a in command]
            if not any(args[i:i + 2] == ["-m", "pytest"]
                       for i in range(len(args) - 1)):
                continue
            scratch = [str(a)[len("--junit-xml="):] for a in command
                       if str(a).startswith("--junit-xml=")]
            run_dir = str(Path(scratch[0]).parent) if scratch else None
            # Each run measures a fresh prepared copy, so its name is masked.
            copy = re.search(r"langadapter_project_[A-Za-z0-9_]+", " ".join(args))
            commands.append([(str(a).replace(run_dir, "<run>") if run_dir else str(a))
                             .replace(copy.group(0) if copy else "\0", "<copy>") for a in command])
        return commands

    be.run_test_files([str(tf)], str(source_dir), timeout=30)
    first = list(pytest_cmds())
    captured.clear()
    # An earlier target variable, its name spelled in two parts.
    monkeypatch.setenv("LLM" "IOS_TARGET_FILE", "/x/y.py")
    monkeypatch.setenv("LANGADAPTER_MUTATION_SOURCE_ROOT", "/z")
    be.run_test_files([str(tf)], str(source_dir), timeout=30)
    assert first and first == pytest_cmds()
