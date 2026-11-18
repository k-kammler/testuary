"""The load repair after each automatic transformation: a suite that does not
load gets the backend's recoveries, is kept only once it loads, and records
what was removed. Python cases run pytest, Java cases a scripted backend."""
from __future__ import annotations

import pytest

from testuary import languages
from testuary import loadable

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

TARGET = "def add(a, b):\n    return a + b\n"
UNLOADABLE = (
    "import nonexistent_module_for_the_load_repair\n"
    "from calc import add\n\n"
    "def test_add():\n    assert add(1, 1) == 2\n\n"
    "def test_more():\n    assert add(2, 2) == 4\n"
)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    return root


class _Ok:
    ok = True


class _JavaBackend:
    """A scripted Java backend: javac rejects the class until the member it
    names is removed."""

    def __init__(self, broken_member: str, *, test: bool):
        self.member = broken_member
        self.test = test

    def prune_unresolvable_imports(self, source, source_dir):
        return source, []

    def prune_collection_breakers(self, source, source_dir, *, timeout):
        if self.member not in source:
            return source, []
        kept = "\n".join(line for line in source.splitlines() if self.member not in line)
        return kept + "\n", [f"CalcTest.{self.member}()"]

    def finalize_test_source(self, source):
        return source

    def test_metrics(self, source):
        return {"test_count": source.count("@Test")}


JAVA = ("class CalcTest {\n"
        "  @Test void adds() { assertEquals(2, Calc.add(1, 1)); }\n"
        "  @Test void broken() { undefined(); }\n"
        "  int helper() { return missing(); }\n"
        "}\n")


def _java_probe(member):
    def probe(text):
        return (None, False, "collection_failed") if member in text else ([], False, "completed")
    return probe


# ensure_loadable itself

def test_a_python_suite_that_does_not_collect_is_recovered(project):
    backend = languages.get_backend("python")
    probe = loadable.scratch_probe(backend, str(project), str(project / "calc.py"), timeout=120)
    result = loadable.ensure_loadable(
        backend, UNLOADABLE, str(project), timeout=120, probe=probe,
        validate=lambda text: _Ok(), log=lambda _m: None, label="test")
    assert result.loads and result.rejected is None
    assert "nonexistent_module_for_the_load_repair" not in result.source
    assert list(result.removed) == ["unresolvable_imports"]


def test_a_java_class_that_does_not_compile_is_recovered():
    backend = _JavaBackend("helper", test=False)
    result = loadable.ensure_loadable(
        backend, JAVA, "/p", timeout=120, probe=_java_probe("helper"),
        validate=lambda text: _Ok(), log=lambda _m: None, label="test")
    assert result.loads
    assert result.removed == {"collection_breakers": ["CalcTest.helper()"]}
    assert not loadable.fewer_tests(backend, JAVA, result.source)


def test_a_loading_suite_passes_through_untouched():
    backend = _JavaBackend("helper", test=False)
    result = loadable.ensure_loadable(
        backend, "class Fine {}\n", "/p", timeout=120, probe=lambda _t: ([], False, "completed"),
        validate=lambda text: _Ok(), log=lambda _m: None, label="test")
    assert result.loads and result.removed == {} and result.source == "class Fine {}\n"


def test_a_suite_the_recoveries_cannot_make_load_is_kept_as_it_was():
    backend = _JavaBackend("helper", test=False)
    result = loadable.ensure_loadable(
        backend, JAVA, "/p", timeout=120, probe=lambda _t: (None, False, "collection_failed"),
        validate=lambda text: _Ok(), log=lambda _m: None, label="test")
    assert not result.loads


# Its call sites

def test_a_reduced_suite_that_does_not_load_is_recovered_when_no_test_is_lost(project):
    """The recovery of an unresolvable import removes no test, so it is kept."""
    backend = languages.get_backend("python")
    source, removed = loadable.keep_if_loadable(
        backend, UNLOADABLE, str(project), timeout=120,
        target_file_path=str(project / "calc.py"), validate=lambda text: _Ok(),
        log=lambda _m: None, label="consolidate")
    assert removed and "nonexistent_module_for_the_load_repair" not in source
    assert not loadable.fewer_tests(backend, UNLOADABLE, source)


def test_a_reduced_java_class_that_loads_only_without_a_test_is_refused():
    """javac names a kept test. The recovery would lose it, so the reduction
    treats the reduced suite as not loading."""
    backend = _JavaBackend("broken", test=True)
    result = loadable.ensure_loadable(
        backend, JAVA, "/p", timeout=120, probe=_java_probe("broken"),
        validate=lambda text: _Ok(), log=lambda _m: None, label="consolidate")
    assert result.loads and loadable.fewer_tests(backend, JAVA, result.source)


def test_the_starter_repair_passes_through_the_helper(project, tmp_path, monkeypatch):
    """The starter cleanup recovers collection through ensure_loadable."""
    from testuary import starter_salvage

    calls = []
    real = loadable.ensure_loadable
    monkeypatch.setattr(loadable, "ensure_loadable",
                        lambda *a, **k: calls.append(k["label"]) or real(*a, **k))
    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        languages.get_backend("python"), UNLOADABLE, str(project / "calc.py"), str(project),
        tmp_path / "scratch", timeout=120, log=lambda _m: None, cleanup_out=cleanup)
    assert calls == ["starter"]
    assert classification == "all_passing" and green is not None
    assert list(cleanup["collection_recovery_removed"]) == ["unresolvable_imports"]


def test_an_unknown_test_count_cannot_show_that_no_test_was_removed():
    """A language without a static test count gives no evidence that a
    recovery kept every test, so the recovery counts as removing one and is
    not kept."""
    from types import SimpleNamespace

    backend = SimpleNamespace(test_metrics=lambda source: {})
    assert loadable.fewer_tests(backend, UNLOADABLE, UNLOADABLE)
