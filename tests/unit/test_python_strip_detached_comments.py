"""``GenericBackend.finalize_test_source``, the normalization the merge's
delivery step applies to shipped suites, and its delivery wiring. The
transform is pure and idempotent, keeps attached rationale and every
semantic construct, and removes only detached full-line comment blocks.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from testuary.languages.generic import GenericBackend
from testuary import measurement as tm
from per_suite_double import placed

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _b() -> GenericBackend:
    return GenericBackend("python", (".py",))


def test_attached_rationale_is_kept() -> None:
    src = (
        "# checks the boundary case\n"
        "def test_a():\n"
        "    assert True\n"
    )
    assert _b().finalize_test_source(src) == src


def test_leading_module_header_is_preserved() -> None:
    src = (
        "# === section banner ===\n"
        "\n"
        "def test_a():\n"
        "    assert True\n"
    )
    assert _b().finalize_test_source(src) == src


def test_contiguous_banner_and_rationale_stay_together() -> None:
    src = (
        "# === Tests for f ===\n"
        "# covers the empty input\n"
        "def test_a():\n"
        "    assert True\n"
    )
    assert _b().finalize_test_source(src) == src


def test_block_at_end_of_file_is_removed() -> None:
    src = (
        "def test_a():\n"
        "    assert True\n"
        "\n"
        "# test_b was here\n"
        "# it checked things\n"
    )
    assert _b().finalize_test_source(src) == (
        "def test_a():\n"
        "    assert True\n"
    )


def test_trailing_comment_on_code_line_is_untouched() -> None:
    src = (
        "def test_a():\n"
        "    assert f(1) == 2  # noqa: E501\n"
    )
    assert _b().finalize_test_source(src) == src


def test_shebang_and_coding_cookie_survive() -> None:
    src = (
        "#!/usr/bin/env python\n"
        "# -*- coding: utf-8 -*-\n"
        "\n"
        "def test_a():\n"
        "    assert True\n"
    )
    assert _b().finalize_test_source(src) == src


def test_indented_class_body_comments_follow_the_same_rule() -> None:
    src = (
        "class TestFoo:\n"
        "    # attached to the method below\n"
        "    def test_a(self):\n"
        "        assert True\n"
        "\n"
        "    # orphaned rationale\n"
        "\n"
        "    def test_b(self):\n"
        "        assert True\n"
    )
    out = _b().finalize_test_source(src)
    assert "# attached to the method below" in out
    assert "# orphaned rationale" not in out


def test_unrelated_blank_runs_are_not_reformatted() -> None:
    src = (
        "\n"
        "\n"
        "def test_a():\n"
        "    assert True\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "def test_b():\n"
        "    assert True\n"
        "\n"
        "\n"
    )
    assert _b().finalize_test_source(src) == src


def test_pruning_consumes_only_the_removed_tests_attached_comment() -> None:
    src = (
        "# module header\n"
        "\n"
        "# rationale for removed\n"
        "def test_remove():\n"
        "    assert False\n"
        "\n"
        "# rationale for retained\n"
        "def test_keep():\n"
        "    assert True\n"
    )
    out = _b().prune_tests(src, ["test_remove"])
    assert "# module header" in out
    assert "rationale for removed" not in out
    assert "rationale for retained" in out
    assert "test_keep" in out


def test_docstrings_and_string_hashes_are_untouched() -> None:
    src = (
        'MARK = "# not a comment"\n'
        "\n"
        "def test_a():\n"
        '    """# looks like a comment\n'
        "\n"
        "    but is a docstring\"\"\"\n"
        "    assert MARK\n"
    )
    assert _b().finalize_test_source(src) == src


def test_hash_line_inside_a_string_is_not_treated_as_a_comment() -> None:
    """The reduction reads lines, so a ``#``-leading line inside a
    triple-quoted string satisfies its rule. Dropping it would change a
    string's value, so the reduction is declined and the source stands."""
    src = (
        "def test_sql():\n"
        '    q = """\n'
        "# leading hash inside a string\n"
        "\n"
        'SELECT 1"""\n'
        "    assert q\n"
    )
    out = _b().finalize_test_source(src)
    assert out == src
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))


def test_blank_run_inside_a_string_is_not_collapsed() -> None:
    """Blank-run collapsing is a line rule too, and a blank run inside a
    string is part of its value."""
    src = (
        "def test_doc():\n"
        '    q = """a\n'
        "\n"
        "\n"
        "\n"
        "\n"
        'b"""\n'
        "    assert q\n"
    )
    assert _b().finalize_test_source(src) == src


def test_unparseable_source_is_returned_unchanged() -> None:
    """A source the parser rejects cannot be shown safe, so it stands."""
    src = "# detached\n\ndef test_a(:\n    pass\n"
    assert _b().finalize_test_source(src) == src


def test_transform_is_idempotent_ast_identical_and_loc_invariant() -> None:
    src = (
        "# ======================\n"
        "# Tests for concat over\n"
        "# ======================\n"
        "\n"
        "# test_gone: pruned by the cover\n"
        "\n"
        "# kept rationale\n"
        "def test_a():\n"
        "    assert True  # trailing survives\n"
        "\n"
        "# dangling at eof\n"
    )
    b = _b()
    once = b.finalize_test_source(src)
    assert b.finalize_test_source(once) == once
    assert ast.dump(ast.parse(once)) == ast.dump(ast.parse(src))
    # Stored line counts come from the backend's structural observation.
    before, after = b.test_metrics_observation(src), b.test_metrics_observation(once)
    assert before.completed and after.completed
    assert after.value.code_lines == before.value.code_lines


def test_delivery_uses_the_finalized_source_that_was_measured(
        tmp_path: Path, monkeypatch) -> None:
    """The merged suite is finalized before validation and measurement, while the
    immutable provenance candidates retain their original input bytes."""
    from testuary import config
    monkeypatch.setattr(config, "consolidation_mutation_method",
                        lambda: "derive")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "mod.py").write_text(
        "def f(x):\n    return x + 1\n", encoding="utf-8")
    (proj / "pyproject.toml").write_text("", encoding="utf-8")
    test_dir = tmp_path / "rank1" / "tests"
    test_dir.mkdir(parents=True)
    cov_src = (
        "# === detached banner of a pruned section ===\n"
        "\n"
        "from mod import f\n"
        "\n"
        "# attached rationale\n"
        "def test_weak():\n"
        "    assert f(1) is not None\n"
    )
    (test_dir / "test_mod.py").write_text(cov_src, encoding="utf-8")
    (test_dir / "test_mod_mutation.py").write_text(
        "from mod import f\n\n"
        "def test_strong():\n"
        "    assert f(1) == 2\n",
        encoding="utf-8",
    )

    def run_mutation_on(d):
        pyfile = next(Path(d).glob("test_*.py"))
        return placed({"mutants": [
            {"id": "m1", "status": "killed",
             "killer": f"{pyfile.name}::test_strong"},
        ]})

    identity = {
        "target_source_sha256": hashlib.sha256(
            (proj / "mod.py").read_bytes()
        ).hexdigest(),
        "backend": "GenericBackend",
        "catalogue": "full",
        "execution_name": "test_mod.py",
    }

    res = tm.consolidate_test_suite(
        test_output_dir=str(test_dir),
        coverage_file_name="test_mod.py",
        candidate1_source=(test_dir / "test_mod.py").read_text(encoding="utf-8"),
        candidate2_source=(test_dir / "test_mod_mutation.py").read_text(encoding="utf-8"),
        reduction=config.solver_settings(config.get_agent_config("test_coverage")),
        language="python",
        target_file_path=str(proj / "mod.py"),
        source_dir=str(proj),
        run_mutation_on=run_mutation_on,
        candidate1_km={"mutants": [
            {"id": "m1", "status": "survived", "killer": None},
        ]},
        candidate1_coverage={
            "covered_statement_lines": [1, 2],
            "covered_branch_arcs": [],
        },
        candidate1_measurement_identity=identity,
        measurement_identity=identity,
        timeout=120,
    )
    assert res is not None
    shipped = (test_dir / "test_mod.py").read_text(encoding="utf-8")
    assert "detached banner" not in shipped
    assert "test_strong" in shipped
    assert _b().finalize_test_source(shipped) == shipped
    pre = test_dir.parent / "_pre_merge"
    assert (pre / "candidate_inherited.py").read_text(
        encoding="utf-8"
    ) == cov_src
    assert "test_strong" in (pre / "candidate_agent.py").read_text(
        encoding="utf-8"
    )
