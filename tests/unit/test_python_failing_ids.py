"""``GenericBackend.failing_test_ids`` and ``prune_unresolvable_imports``,
which the starter phase's green handling relies on, and the backend's
salvage steps. The failing-id tests run real pytest subprocesses,
not parsed console text."""
from __future__ import annotations

from pathlib import Path

import pytest

from testuary.languages import get_backend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _backend():
    return get_backend("python")


def test_collection_under_a_strict_parent_config_still_runs(tmp_path):
    # A parent dir ships a pyproject that turns warnings into errors. A suite
    # whose import emits a DeprecationWarning must still collect and run.
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\nfilterwarnings = [\"error\"]\n",
        encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    suite = scratch / "test_warn.py"
    suite.write_text(
        "import warnings\n"
        "def test_a():\n"
        "    warnings.warn('legacy', DeprecationWarning)\n"
        "    assert 1\n",
        encoding="utf-8")
    ids = _backend().failing_test_ids(
        [str(suite)], str(scratch), timeout=120)
    assert ids == []


def _write(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


def test_all_green_suite_returns_empty_list(tmp_path):
    suite = _write(tmp_path, "test_green.py",
                   "def test_a():\n    assert 1\n\n"
                   "def test_b():\n    assert 2\n")
    ids = _backend().failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120)
    assert ids == []


def test_failing_function_ids_in_prune_form(tmp_path):
    suite = _write(tmp_path, "test_mix.py",
                   "def test_ok():\n    assert 1\n\n"
                   "def test_bad():\n    assert 0\n\n"
                   "def test_err():\n    raise RuntimeError('x')\n")
    ids = _backend().failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120)
    assert ids is not None
    names = {i.split(".")[-1] for i in ids}
    assert names == {"test_bad", "test_err"}


def test_failing_class_method_id_carries_the_class(tmp_path):
    suite = _write(tmp_path, "test_cls.py",
                   "class TestFoo:\n"
                   "    def test_ok(self):\n        assert 1\n"
                   "    def test_bad(self):\n        assert 0\n")
    ids = _backend().failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120)
    assert ids is not None and len(ids) == 1
    assert ids[0].endswith("TestFoo.test_bad")


def test_parametrized_failures_name_each_failing_case(tmp_path):
    """Each failing case of a parametrized test is listed by its id, so its
    removal can take out that case alone."""
    suite = _write(tmp_path, "test_param.py",
                   "import pytest\n\n"
                   "@pytest.mark.parametrize('v', [1, 2, 3])\n"
                   "def test_p(v):\n    assert v == 1\n")
    ids = _backend().failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120)
    assert ids is not None
    assert [test_id.rsplit(".", 1)[-1] for test_id in ids] == ["test_p[2]", "test_p[3]"]


def test_collection_error_returns_none(tmp_path):
    suite = _write(tmp_path, "test_broken.py",
                   "import does_not_exist_anywhere\n\n"
                   "def test_a():\n    assert 1\n")
    ids = _backend().failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120)
    assert ids is None


def test_syntax_error_returns_none(tmp_path):
    suite = _write(tmp_path, "test_syntax.py", "def broken(:\n")
    ids = _backend().failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120)
    assert ids is None


def test_failing_ids_feed_prune_tests_round_trip(tmp_path):
    """Ids from failing_test_ids remove the failing tests via prune_tests,
    and the remainder is green."""
    backend = _backend()
    src = ("def test_ok():\n    assert 1\n\n"
           "def test_bad():\n    assert 0\n")
    suite = _write(tmp_path, "test_rt.py", src)
    ids = backend.failing_test_ids([str(suite)], str(tmp_path), timeout=120)
    assert ids
    pruned = backend.prune_tests(src, ids)
    assert "def test_bad" not in pruned
    assert "def test_ok" in pruned
    suite.write_text(pruned, encoding="utf-8")
    assert backend.failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120) == []


def test_unresolvable_import_dropped_resolvable_kept(tmp_path):
    src = ("import json\n"
           "import surely_not_a_module_xyz\n"
           "from another_missing_mod import thing\n\n"
           "def test_a():\n    assert json.dumps({}) == '{}'\n")
    out, dropped = _backend().prune_unresolvable_imports(src, str(tmp_path))
    assert set(dropped) == {
        "surely_not_a_module_xyz", "another_missing_mod.thing",
    }
    assert "import json" in out
    assert "surely_not_a_module_xyz" not in out
    assert "another_missing_mod" not in out


def test_all_imports_resolvable_is_a_no_op(tmp_path):
    src = "import json\nimport os\n\ndef test_a():\n    assert 1\n"
    out, dropped = _backend().prune_unresolvable_imports(src, str(tmp_path))
    assert dropped == []
    assert out == src


def test_hallucinated_name_pruned_from_real_module(tmp_path):
    """A from-import naming things the module does not provide keeps the
    surviving names and loses only the dead ones."""
    src = ("from json import loads, not_a_json_thing, dumps\n\n"
           "def test_a():\n    assert loads(dumps([1])) == [1]\n")
    out, dropped = _backend().prune_unresolvable_imports(src, str(tmp_path))
    assert dropped == ["json.not_a_json_thing"]
    assert "from json import loads, dumps" in out
    assert "not_a_json_thing" not in out


def test_from_import_dropped_whole_when_no_name_survives(tmp_path):
    src = ("from json import nope_one, nope_two\n"
           "import os\n\ndef test_a():\n    assert os.sep\n")
    out, dropped = _backend().prune_unresolvable_imports(src, str(tmp_path))
    assert set(dropped) == {"json.nope_one", "json.nope_two"}
    assert "from json import" not in out
    assert "import os" in out


def test_submodule_name_in_from_import_survives(tmp_path):
    """``from package import submodule`` is valid even when the package
    object has no such attribute before import. The spec probe keeps it."""
    src = ("from xml import etree\nfrom os import path\n\n"
           "def test_a():\n    assert etree and path\n")
    out, dropped = _backend().prune_unresolvable_imports(src, str(tmp_path))
    assert dropped == []
    assert out == src


def test_alias_in_from_import_preserved_on_rewrite(tmp_path):
    src = ("from json import loads as jl, fake_name, dumps as jd\n\n"
           "def test_a():\n    assert jl(jd([])) == []\n")
    out, dropped = _backend().prune_unresolvable_imports(src, str(tmp_path))
    assert dropped == ["json.fake_name"]
    assert "from json import loads as jl, dumps as jd" in out


def test_multiline_from_import_rewritten_to_survivors(tmp_path):
    src = ("from json import (\n"
           "    loads,\n"
           "    ghost_symbol,\n"
           "    dumps,\n"
           ")\n\n"
           "def test_a():\n    assert loads(dumps(1)) == 1\n")
    out, dropped = _backend().prune_unresolvable_imports(src, str(tmp_path))
    assert dropped == ["json.ghost_symbol"]
    assert "from json import loads, dumps" in out
    assert "ghost_symbol" not in out
    import ast
    ast.parse(out)


def test_name_prune_salvages_hallucinated_import_suite(tmp_path):
    """One hallucinated name imported from a real module breaks collection
    of the whole suite. The removal keeps every real name and the suite
    collects again."""
    backend = _backend()
    src = ("from json import loads, fantasy_loader\n\n"
           "def test_uses_ghost():\n"
           "    assert fantasy_loader('1') == 1\n\n"
           "def test_fine():\n    assert loads('1') == 1\n")
    suite = _write(tmp_path, "test_ghost.py", src)
    assert backend.failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120) is None
    pruned, dropped = backend.prune_unresolvable_imports(src, str(tmp_path))
    assert dropped == ["json.fantasy_loader"]
    suite.write_text(pruned, encoding="utf-8")
    ids = backend.failing_test_ids([str(suite)], str(tmp_path), timeout=120)
    assert ids is not None and len(ids) == 1
    assert ids[0].endswith("test_uses_ghost")
    suite.write_text(backend.prune_tests(pruned, ids), encoding="utf-8")
    assert backend.failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120) == []


def test_multiline_import_statement_dropped_whole(tmp_path):
    src = ("from missing_pkg import (\n"
           "    alpha,\n"
           "    beta,\n"
           ")\n\n"
           "def test_a():\n    assert 1\n")
    out, dropped = _backend().prune_unresolvable_imports(src, str(tmp_path))
    assert dropped == ["missing_pkg.alpha", "missing_pkg.beta"]
    assert "alpha" not in out
    assert "def test_a" in out


def test_pruning_imports_then_tests_salvages_a_collection_broken_suite(tmp_path):
    """The salvage chain on a real run: the broken import makes
    collection fail. Removing it turns the affected test into a NameError at
    run time. Removing that test leaves a green suite."""
    backend = _backend()
    src = ("import json\n"
           "import surely_not_a_module_xyz\n\n"
           "def test_uses_missing():\n"
           "    assert surely_not_a_module_xyz.f() == 1\n\n"
           "def test_fine():\n    assert json.loads('1') == 1\n")
    suite = _write(tmp_path, "test_salvage.py", src)
    assert backend.failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120) is None
    pruned, dropped = backend.prune_unresolvable_imports(src, str(tmp_path))
    assert dropped == ["surely_not_a_module_xyz"]
    suite.write_text(pruned, encoding="utf-8")
    ids = backend.failing_test_ids([str(suite)], str(tmp_path), timeout=120)
    assert ids is not None and len(ids) == 1
    assert ids[0].endswith("test_uses_missing")
    suite.write_text(backend.prune_tests(pruned, ids), encoding="utf-8")
    assert backend.failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120) == []


def test_salvage_parseable_source_passes_through_unchanged():
    src = "def test_a():\n    assert 1\n"
    assert _backend().salvage_syntax(src, ".") == src


def test_salvage_inserts_missing_block_body():
    src = ("def helper():\n\n"
           "def test_a():\n    assert 1\n")
    out = _backend().salvage_syntax(src, ".")
    assert out is not None
    import ast
    ast.parse(out)
    assert "def test_a" in out


def test_salvage_blanks_a_glitched_line():
    src = ("def test_a():\n    assert 1\n\n"
           "x = ((1, 2)))\n\n"
           "def test_b():\n    assert 2\n")
    out = _backend().salvage_syntax(src, ".")
    assert out is not None
    import ast
    ast.parse(out)
    assert "def test_a" in out and "def test_b" in out


def test_salvage_mid_file_glitch_keeps_both_sides():
    """A single defect in the middle of a long suite costs the glitched
    region only. Everything before and after survives."""
    head = "\n".join(f"def test_h{i}():\n    assert {i} == {i}\n"
                     for i in range(10))
    tail = "\n".join(f"def test_t{i}():\n    assert {i} >= 0\n"
                     for i in range(10))
    src = head + "\nclass TestGlitch:\n    def broken(self:\n\n" + tail
    out = _backend().salvage_syntax(src, ".")
    assert out is not None
    import ast
    ast.parse(out)
    for i in range(10):
        assert f"def test_h{i}" in out
        assert f"def test_t{i}" in out


def test_salvage_cpython_is_the_authority_over_treesitter():
    """A block-less def parses under tree-sitter but not CPython. The
    salvage gate repairs it instead of accepting it."""
    backend = _backend()
    src = ("def empty_body():\n\n"
           "def test_a():\n    assert 1\n")
    assert backend.check_structure(src).ok     # tree-sitter tolerates it
    out = backend.salvage_syntax(src, ".")    # target Python repairs
    assert out is not None and out != src
    import ast
    ast.parse(out)


def test_salvage_gives_up_on_garbage():
    assert _backend().salvage_syntax("}\n" * 60, ".") is None


def test_collection_breaker_module_statement_dropped(tmp_path):
    """A module-scope statement raising at load time is removed and the
    suite loads. The tests survive."""
    src = ("import json\n"
           "BAD = json.loads('not json at all')\n\n"
           "def test_a():\n    assert 1\n\n"
           "def test_b():\n    assert 2\n")
    out, dropped = _backend().prune_collection_breakers(
        src, str(tmp_path), timeout=120)
    assert dropped
    assert "def test_a" in out and "def test_b" in out
    assert "not json at all" not in out


def test_collection_breaker_chain_of_two(tmp_path):
    """Each round removes one breaker. Two sequential breakers need two
    rounds and both go."""
    src = ("x = 1 / 0\n\n"
           "def test_a():\n    assert 1\n\n"
           "y = [][5]\n\n"
           "def test_b():\n    assert 2\n")
    out, dropped = _backend().prune_collection_breakers(
        src, str(tmp_path), timeout=120)
    assert len(dropped) == 2
    assert "def test_a" in out and "def test_b" in out


def test_collection_breaker_class_scope_statement(tmp_path):
    """Class-body code executes at class creation. The offending member
    is removed, the class and its tests stay."""
    src = ("class TestThing:\n"
           "    BAD = 1 / 0\n\n"
           "    def test_a(self):\n        assert 1\n")
    out, dropped = _backend().prune_collection_breakers(
        src, str(tmp_path), timeout=120)
    assert dropped
    assert "def test_a" in out
    assert "1 / 0" not in out


def test_collection_breaker_healthy_suite_untouched(tmp_path):
    src = "def test_a():\n    assert 1\n"
    out, dropped = _backend().prune_collection_breakers(
        src, str(tmp_path), timeout=120)
    assert out == src and dropped == []


def test_collection_breaker_salvage_chain_to_green(tmp_path):
    """The production order on a fixture-misuse breaker: collection
    fails, the breaker removal takes out the module-scope misuse, the suite
    loads, dependents fail at run time, and the test removal takes out exactly
    those."""
    backend = _backend()
    src = ("import pytest\n\n"
           "@pytest.fixture\n"
           "def fix():\n    return 2\n\n"
           "HALF = 1 / fix\n\n"
           "def test_uses_half():\n    assert HALF == 0.5\n\n"
           "def test_fine():\n    assert 1 + 1 == 2\n")
    suite = _write(tmp_path, "test_breaker.py", src)
    assert backend.failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120) is None
    fixed, dropped = backend.prune_collection_breakers(
        src, str(tmp_path), timeout=120)
    assert dropped
    suite.write_text(fixed, encoding="utf-8")
    ids = backend.failing_test_ids([str(suite)], str(tmp_path), timeout=120)
    assert ids is not None and len(ids) == 1
    assert ids[0].endswith("test_uses_half")
    suite.write_text(backend.prune_tests(fixed, ids), encoding="utf-8")
    assert backend.failing_test_ids(
        [str(suite)], str(tmp_path), timeout=120) == []


def test_every_pytest_invocation_pins_rootdir_to_the_suite_dir(
        tmp_path, monkeypatch):
    """Suites run from work directories outside the project tree.
    Every pytest command the backend builds carries --rootdir pinned to
    the suites' own directory: junit classnames stay module-relative
    (the shape the per-test eviction ids join on) and collection does not
    depend on the rootdir ancestor walk from the suite's location."""
    from langadapter.execution import LocalExecutor

    backend = _backend()
    captured = []
    real_run = LocalExecutor.run

    def spy(self, cmd, **kw):
        # Suite runs and collection launch through the backend's executor.
        if "pytest" in [str(item) for item in cmd]:
            captured.append([str(item) for item in cmd])
        return real_run(self, cmd, **kw)

    monkeypatch.setattr(LocalExecutor, "run", spy)
    proj = tmp_path / "proj"
    proj.mkdir()
    suite_dir = tmp_path / "rank1" / "tests"
    suite_dir.mkdir(parents=True)
    suite = _write(suite_dir, "test_ok.py",
                   "def test_ok():\n    assert True\n")
    backend.run_test_files([str(suite)], str(proj), timeout=120)
    backend.failing_test_ids([str(suite)], str(proj), timeout=120)
    backend.prune_collection_breakers(
        "import nonexistent_module_xyz\n", str(proj), timeout=60)
    assert captured, "no pytest commands were captured"
    suite_pin = f"--rootdir={suite_dir.resolve()}"
    for cmd in captured:
        flags = [a for a in cmd if a.startswith("--rootdir=")]
        assert flags, f"pytest command without rootdir pin: {cmd}"
        if str(suite) in cmd:
            assert flags == [suite_pin], f"wrong rootdir target: {flags}"


def test_failing_ids_keep_module_shape_for_out_of_root_suites(tmp_path):
    """A suite outside the source directory keeps module-qualified junit
    classnames, so failing ids carry the ``module.test`` shape the
    eviction data and prune_tests use."""
    backend = _backend()
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "mod.py").write_text("def f(x):\n    return x + 1\n",
                                 encoding="utf-8")
    suite_dir = tmp_path / "rank1" / "tests"
    suite_dir.mkdir(parents=True)
    suite = _write(suite_dir, "test_mod.py",
                   "from mod import f\n\n"
                   "def test_ok():\n    assert f(1) == 2\n\n"
                   "def test_bad():\n    assert f(1) == 99\n")
    ids = backend.failing_test_ids([str(suite)], str(proj), timeout=120)
    assert ids == ["test_mod.test_bad"]


def test_prune_collection_breakers_salvages_a_module_level_error(tmp_path):
    """End to end: a module-scope statement that raises at collection is
    blanked so the suite collects again."""
    backend = _backend()
    src = ("import os\n"
           "BROKEN = undefined_name_xyz()\n\n"
           "def test_a():\n    assert os.sep\n")
    out, removed = backend.prune_collection_breakers(
        src, str(tmp_path), timeout=120)
    assert removed
    assert "undefined_name_xyz" not in out
    assert "def test_a" in out
    import ast
    ast.parse(out)
