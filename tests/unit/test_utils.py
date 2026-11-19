"""Testuary's helpers: line-number stripping, project and output paths, CSV
fields, runtime flags, complexity, hints validation, per-mutant timeouts and
pytest configuration. Settings and prompt loading are covered in
test_config_loader.py and test_prompts_loader.py.
"""
from __future__ import annotations

import functools
from pathlib import Path, PurePosixPath
import sys

import pytest

from testuary import results as csv_writer
from testuary import measurement as _source_metrics
from testuary import config as config_module
from testuary import hints as hints_module
from testuary import project as project_module
from testuary.agent import _helpers as agent_helpers
from testuary.agent import _edit_tool as edit_tool

# The helpers read the run's settings, as inside a phase.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def _the_test_dir_is_the_user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The user directory holds config.ini. Here it is the test's own directory,
    so a config.ini a test writes into ``tmp_cwd`` is the user's."""
    from testuary import settings

    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)


# strip_line_number_prefixes

def test_strip_line_number_prefixes_basic() -> None:
    """The read and edit views number lines ``N<TAB>code``. A block pasted
    from them loses its numbers, and its final newline stays."""
    text = "1\timport os\n2\timport sys\n"
    assert edit_tool.strip_line_number_prefixes(text) == "import os\nimport sys\n"
    assert agent_helpers.number_line(7, "x = 1") == "7\tx = 1"


def test_strip_line_number_prefixes_needs_consecutive_numbers() -> None:
    assert edit_tool.strip_line_number_prefixes("1\ta\n3\tb") == "1\ta\n3\tb"
    assert edit_tool.strip_line_number_prefixes("4\ta\n4\tb") == "4\ta\n4\tb"
    # A pasted empty line that lost its tab takes the next number.
    assert edit_tool.strip_line_number_prefixes("1\ta\n\n3\tb") == "a\n\nb"


def test_strip_line_number_prefixes_takes_a_bare_expected_number_as_an_empty_line() -> None:
    """An empty source line shows as "12<TAB>". A paste that drops only the
    tab leaves "12", which is that empty line, not text."""
    assert edit_tool.strip_line_number_prefixes("11\tx = 1\n12\n13\ty = 2\n") == "x = 1\n\ny = 2\n"
    # Only the expected number: another number, or one on the first line, is text.
    assert edit_tool.strip_line_number_prefixes("11\tx = 1\n7\n13\ty = 2") == "11\tx = 1\n7\n13\ty = 2"
    assert edit_tool.strip_line_number_prefixes("12\n13\ty = 2") == "12\n13\ty = 2"


def test_strip_line_number_prefixes_keeps_every_case_that_is_content() -> None:
    """The cases that must stay as written: a dict literal with integer keys,
    the colon form, non-consecutive numbers and numbers after leading spaces."""
    for text in ("    1: 'one',\n    2: 'two',\n", "1: x\n2: y", "1\ta\n3\tb", "  1\ta\n  2\tb"):
        assert edit_tool.strip_line_number_prefixes(text) == text


# resolve_source_dir: container mode keeps the POSIX path

def test_resolve_source_dir_container_uses_repo_dir_verbatim(monkeypatch, in_run) -> None:
    """In container mode ``file_path`` is an in-container POSIX path. pathlib on
    a Windows host would rewrite it (/home/... -> C:\\home\\...) and the marker
    walk cannot see the container filesystem, so the resolver returns the
    source directory's variable unchanged."""
    monkeypatch.setattr(config_module, "get_project_config", lambda: {})
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    monkeypatch.setenv("TESTUARY_PROJECT_SOURCE_DIR", "/home/user/sympy")
    in_run()
    root = project_module.resolve_source_dir(
        "/home/user/sympy/sympy/codegen/pyutils.py")
    assert root == "/home/user/sympy"
    assert "\\" not in root  # no Windows separators


def test_a_guest_source_dir_falls_back_to_its_variable(monkeypatch, in_run) -> None:
    """project_repo_dir falls back to the source directory's variable as given."""
    monkeypatch.setenv("TESTUARY_PROJECT_SOURCE_DIR", "/work/repo")
    in_run()
    assert project_module.project_repo_dir() == "/work/repo"


@pytest.mark.parametrize("container", [False, True])
def test_the_project_dir_is_the_root_and_the_python_import_root_lies_below(
    container, tmp_path, monkeypatch, in_run,
) -> None:
    """Every target's root is the project directory, where its tests run with
    the project's configuration and a C extension's build files lie
    (setup.py beside src). ``[project] python_import_root`` is the directory
    Python imports the project from: a Python target's import root and the
    directory its prompt path is relative to."""
    project = tmp_path / "project"
    (project / "src" / "pkg").mkdir(parents=True)
    (project / "setup.py").write_text("")
    project = project.resolve()
    monkeypatch.setattr(config_module, "get_project_config", lambda: {"python_import_root": "src"})
    if container:
        project = PurePosixPath("/home/user/project")
        monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
        monkeypatch.setenv("TESTUARY_PROJECT_SOURCE_DIR", str(project))
    in_run()
    package = project / "src" / "pkg"
    for name in ("mod.py", "_speedups.c", "Main.java"):
        assert project_module.resolve_source_dir(str(package / name)) == str(project)
    assert project_module._below(str(project), "src") == str(project / "src")
    assert project_module.display_path(str(package / "mod.py"), str(project)) == "pkg/mod.py"
    for other in ("_speedups.c", "Main.java"):
        assert project_module.display_path(str(package / other), str(project)) == f"src/pkg/{other}"


def test_strip_line_number_prefixes_preserves_non_prefixed_lines() -> None:
    """The helper only strips when every non-empty line is numbered. Mixed
    input passes through unchanged."""
    text = "import os\n1\timport sys\nprint('hi')"
    cleaned = edit_tool.strip_line_number_prefixes(text)
    assert cleaned == text


def test_strip_line_number_prefixes_no_op_when_no_prefixes() -> None:
    text = "def f():\n    return 42\n"
    assert edit_tool.strip_line_number_prefixes(text) == text


def test_strip_line_number_prefixes_empty() -> None:
    assert edit_tool.strip_line_number_prefixes("") == ""


# tests_added_since: the shared timeout-suspect computation

class _WordSkeletonBackend:
    """A stand-in backend whose skeleton is the set of whitespace tokens."""

    def skeleton(self, content):
        return tuple(content.split())


def test_tests_added_since_returns_new_definition_names() -> None:
    b = _WordSkeletonBackend()
    assert agent_helpers.tests_added_since(
        b, "test_a test_b test_c", "test_a test_b") == ["test_c"]


def test_tests_added_since_empty_when_nothing_new() -> None:
    b = _WordSkeletonBackend()
    assert agent_helpers.tests_added_since(b, "test_a", "test_a test_b") == []


def test_tests_added_since_without_fingerprints_keeps_new_tests() -> None:
    """A backend with a skeleton but no per-function fingerprints still
    names every added test, and a definition nested in an added one is
    left out as its helper."""
    b = _WordSkeletonBackend()
    assert agent_helpers.tests_added_since(
        b, "test_a test_c test_c.fn TestD TestD.test_e", "test_a") == [
            "TestD", "test_c"]


def test_tests_added_since_swallows_backend_error() -> None:
    class _Boom:
        def skeleton(self, content):
            raise RuntimeError("parse boom")
    assert agent_helpers.tests_added_since(_Boom(), "x", "y") == []


# parse_csv_field

def test_parse_csv_field_basic() -> None:
    assert csv_writer.parse_csv_field("a,b,c") == ["a", "b", "c"]


def test_parse_csv_field_strips_whitespace() -> None:
    assert csv_writer.parse_csv_field(" a , b , c ") == ["a", "b", "c"]


def test_parse_csv_field_drops_empty_entries() -> None:
    """Trailing and consecutive commas give no empty strings."""
    assert csv_writer.parse_csv_field("a,,b,") == ["a", "b"]


def test_parse_csv_field_empty_input() -> None:
    assert csv_writer.parse_csv_field("") == []
    assert csv_writer.parse_csv_field("   ") == []


# The frozen-module strategy reaches Python's run context

@pytest.mark.parametrize("strategy,flags", [("unfreeze", ("-X", "frozen_modules=off")),
                                            ("skip", ())])
def test_the_frozen_module_strategy_sets_pythons_interpreter_flags(strategy, flags, tmp_path,
                                                                   in_run):
    """Under ``unfreeze`` (the default) every Python run gets ``-X
    frozen_modules=off``, so source edits to frozen stdlib modules take
    effect, and ``skip`` emits no flag."""
    from testuary.languages import language_project_context

    in_run(config_files=[], environment={},
           overrides={"project": {"frozen_module_strategy": strategy,
                                  "python_executable": sys.executable}})
    assert language_project_context("python", str(tmp_path)).interpreter_flags == flags


# compute_file_complexity: cyclomatic complexity of any language via lizard

def test_compute_file_complexity_python(tmp_path: Path) -> None:
    """Python sources are McCabe-scored. Branches increase the CC."""
    src = tmp_path / "foo.py"
    src.write_text(
        "def f(x):\n"
        "    if x > 0:\n"
        "        for i in range(x):\n"
        "            if i % 2 == 0 and i > 5:\n"
        "                pass\n"
        "    return x\n",
        encoding="utf-8",
    )
    cc = _source_metrics.compute_file_complexity(src)
    assert isinstance(cc, int)
    assert cc >= 4  # 1 (base) + 2 if + 1 for + 1 and


def test_compute_file_complexity_java(tmp_path: Path) -> None:
    """Java sources are scored by the same helper, without per-language
    tooling."""
    src = tmp_path / "Foo.java"
    src.write_text(
        "public class Foo {\n"
        "    int f(int x) {\n"
        "        if (x > 0) {\n"
        "            for (int i = 0; i < x; i++) {\n"
        "                if (i % 2 == 0 && i > 5) { return i; }\n"
        "            }\n"
        "        }\n"
        "        return x;\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    cc = _source_metrics.compute_file_complexity(src)
    assert isinstance(cc, int)
    assert cc >= 4


def test_compute_file_complexity_c(tmp_path: Path) -> None:
    """C sources route through the same helper."""
    src = tmp_path / "foo.c"
    src.write_text(
        "int f(int x) {\n"
        "    if (x > 0) {\n"
        "        for (int i = 0; i < x; i++) {\n"
        "            if (i & 1) return i;\n"
        "        }\n"
        "    }\n"
        "    return x;\n"
        "}\n",
        encoding="utf-8",
    )
    cc = _source_metrics.compute_file_complexity(src)
    assert isinstance(cc, int)
    assert cc >= 3


def test_compute_file_complexity_missing_is_unavailable(tmp_path: Path) -> None:
    """Missing evidence is unavailable, not a measured complexity of zero."""
    missing = tmp_path / "does_not_exist.py"
    assert _source_metrics.compute_file_complexity(missing) is None


def test_compute_file_complexity_no_functions_returns_zero(tmp_path: Path) -> None:
    """A file with only module-level statements has no McCabe functions
    to score, so the helper returns 0."""
    src = tmp_path / "config.py"
    src.write_text("X = 1\nY = 2\n", encoding="utf-8")
    assert _source_metrics.compute_file_complexity(src) == 0


def test_compute_file_complexity_stats_sum_and_max(tmp_path: Path) -> None:
    """The stats variant returns ``(sum, max)`` of the per-function
    complexity. With two functions of different complexity the sum exceeds
    the maximum."""
    src = tmp_path / "two.py"
    src.write_text(
        "def simple():\n"
        "    return 1\n"
        "def branchy(x):\n"
        "    if x > 0:\n"
        "        if x > 10:\n"
        "            return 2\n"
        "    return 0\n",
        encoding="utf-8",
    )
    total, mx = _source_metrics.compute_file_complexity_stats(src)
    assert isinstance(total, int) and isinstance(mx, int)
    assert total > mx  # sum across both functions exceeds the single max
    assert mx >= 3     # branchy: 1 base + 2 if


def test_compute_file_complexity_stats_unmeasured_is_none_none(
    tmp_path: Path,
) -> None:
    """Missing, unreadable or unparseable source gives ``(None, None)``, so the
    metrics CSV leaves the cell blank instead of recording 0. test_metrics
    overwrites the dataclass default only when the value is not None."""
    missing = tmp_path / "nope.py"
    assert _source_metrics.compute_file_complexity_stats(missing) == (None, None)


def test_compute_file_complexity_stats_no_functions_is_zero_zero(
    tmp_path: Path,
) -> None:
    """Parsed source without functions gives ``(0, 0)``: a measured zero,
    distinct from the unmeasured ``(None, None)``."""
    src = tmp_path / "consts.py"
    src.write_text("A = 1\nB = 2\n", encoding="utf-8")
    assert _source_metrics.compute_file_complexity_stats(src) == (0, 0)


# relativize_path

def test_relativize_path_passthrough_without_source_dir(tmp_path: Path) -> None:
    """Without a configured source directory ``relativize_path`` returns its input
    unchanged or a relative form, and keeps the file name."""
    abs_path = tmp_path / "src" / "foo.py"
    abs_path.parent.mkdir(parents=True)
    abs_path.write_text("# test")
    rel = project_module.relativize_path(str(abs_path))
    assert isinstance(rel, str)
    assert "foo.py" in rel


def test_relativize_path_empty_input() -> None:
    assert project_module.relativize_path("") == ""


def test_relativize_path_already_relative_input(tmp_cwd: Path) -> None:
    assert project_module.relativize_path("src/foo.py") == "src/foo.py"


# resolve_work_dir_namespace

def test_resolve_work_dir_namespace_auto_unset(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """Auto mode without an execution identity leaves the namespace empty."""
    for var in ("TESTUARY_EXECUTION_ID", "TESTUARY_MAIN_WORK_DIR_NAMESPACE"):
        monkeypatch.delenv(var, raising=False)
    (tmp_cwd / "config.ini").write_text(
        "[main]\nwork_dir_job_namespace = auto\n", encoding="utf-8",
    )
    in_run()
    assert project_module.resolve_work_dir_namespace() == ""


def test_resolve_work_dir_namespace_auto_with_execution_id(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """Auto mode uses the supplied execution identity."""
    monkeypatch.setenv("TESTUARY_EXECUTION_ID", "12345")
    (tmp_cwd / "config.ini").write_text(
        "[main]\nwork_dir_job_namespace = auto\n", encoding="utf-8",
    )
    in_run()
    ns = project_module.resolve_work_dir_namespace()
    assert "12345" in ns


def test_resolve_work_dir_namespace_never(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """``work_dir_job_namespace = never`` wins over the environment."""
    monkeypatch.setenv("TESTUARY_EXECUTION_ID", "12345")
    (tmp_cwd / "config.ini").write_text(
        "[main]\nwork_dir_job_namespace = never\n", encoding="utf-8",
    )
    in_run()
    assert project_module.resolve_work_dir_namespace() == ""


def test_resolve_work_dir_namespace_always_without_env(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """Always mode names the process without a label."""
    import os as _os
    for var in ("TESTUARY_EXECUTION_ID", "TESTUARY_MAIN_WORK_DIR_NAMESPACE"):
        monkeypatch.delenv(var, raising=False)
    (tmp_cwd / "config.ini").write_text(
        "[main]\nwork_dir_job_namespace = always\n", encoding="utf-8",
    )
    in_run()
    ns = project_module.resolve_work_dir_namespace()
    assert ns.startswith(f"job_local_{_os.getpid()}_")
    assert "/" not in ns and "\\" not in ns


def test_resolve_work_dir_namespace_execution_value_is_prefixed(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """Execution identities keep the shared job_ work-directory prefix."""
    monkeypatch.setenv("TESTUARY_EXECUTION_ID", "999")
    (tmp_cwd / "config.ini").write_text(
        "[main]\nwork_dir_job_namespace = auto\n", encoding="utf-8",
    )
    in_run()
    assert project_module.resolve_work_dir_namespace() == "job_999"


# sanitize_path_component and output_dir templating

@pytest.mark.parametrize("raw,expected", [
    ("simple", "simple"),
    ("with_under", "with_under"),
    ("with-dash", "with-dash"),
    ("Python-3.12.13", "Python-3_12_13"),         # dots collapse
    ("Qwen3 235B VL", "Qwen3_235B_VL"),           # spaces collapse
    ("ollama_qwen3_coder_next", "ollama_qwen3_coder_next"),
    ("foo/bar:baz", "foo_bar_baz"),               # slashes and colons
    ("", "_"),                                    # empty -> literal _
])
def test_sanitize_path_component_matches_namespace_rule(raw, expected) -> None:
    """The sanitizer keeps ``[A-Za-z0-9_-]`` and turns anything else into
    ``_``, the rule of ``resolve_work_dir_namespace``, so work-dir namespaces
    and run directories agree. An empty input gives ``_``, never an empty path
    segment."""
    assert project_module.sanitize_path_component(raw) == expected


def test_output_dir_default_is_the_models_dir_of_the_projects_results(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """The bundled empty ``output_dir`` is ``results/<model>`` in the
    project's Testuary directory, with the active model alias (sanitized)."""
    monkeypatch.setenv("TESTUARY_API_MODEL", "ollama_qwen3_coder_next")
    monkeypatch.setenv("TESTUARY_PROJECT_NAME", "commons-cli")
    in_run(base_dir=tmp_cwd)
    out = project_module.output_dir()
    assert out == (tmp_cwd / ".testuary" / "results" / "ollama_qwen3_coder_next").resolve()


def test_output_dir_template_substitutes_model_and_project(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """An ``output_dir`` of ``results/${model}/${project}`` is filled with
    the active model alias and project name (sanitized)."""
    monkeypatch.setenv("TESTUARY_API_MODEL", "ollama_qwen3_coder_next")
    monkeypatch.setenv("TESTUARY_PROJECT_NAME", "commons-cli")
    monkeypatch.setenv("TESTUARY_PROJECT_OUTPUT_DIR", "results/${model}/${project}")
    in_run(base_dir=tmp_cwd)
    out = project_module.output_dir()
    assert out == (tmp_cwd / "results" / "ollama_qwen3_coder_next" / "commons-cli").resolve()


def test_output_dir_template_sanitizes_unsafe_pieces(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """Dots and spaces in an alias become ``_``, as in the work-dir rule, so
    every path segment stays valid."""
    def output_dir():
        in_run(base_dir=tmp_cwd)
        return project_module.output_dir()

    monkeypatch.setenv("TESTUARY_API_MODEL", "Some Vendor 3.5")
    monkeypatch.setenv("TESTUARY_PROJECT_NAME", "Python-3.12.13")
    monkeypatch.setenv("TESTUARY_PROJECT_OUTPUT_DIR", "results/${model}/${project}")
    out = output_dir()
    assert out.name == "Python-3_12_13"
    assert out.parent.name == "Some_Vendor_3_5"
    monkeypatch.delenv("TESTUARY_PROJECT_OUTPUT_DIR")
    assert output_dir().name == "Some_Vendor_3_5"


def test_output_dir_explicit_option_skips_templating(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """An ``--output-dir`` without ``$`` placeholders is used as given:
    ``--output-dir custom`` gives ``./custom``, not
    ``./custom/<model>/<project>``."""
    monkeypatch.setenv("TESTUARY_PROJECT_OUTPUT_DIR", "custom_dir")
    in_run(base_dir=tmp_cwd)
    out = project_module.output_dir()
    assert out == (tmp_cwd / "custom_dir").resolve()


def test_output_dir_absolute_option_passes_through(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """An absolute ``--output-dir`` is used as given, not relativized to the
    working directory."""
    abs_path = (tmp_cwd / "abs_target").resolve()
    monkeypatch.setenv("TESTUARY_PROJECT_OUTPUT_DIR", str(abs_path))
    in_run(base_dir=tmp_cwd)
    out = project_module.output_dir()
    assert out == abs_path


# read_file_content

def test_read_file_content_returns_string(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("hello\nworld\n", encoding="utf-8")
    assert project_module.read_file_content(str(f)) == "hello\nworld\n"


def test_read_file_content_missing_file_raises(tmp_path: Path) -> None:
    """A missing file raises ``FileNotFoundError``. The helper does not
    swallow it."""
    with pytest.raises(FileNotFoundError):
        project_module.read_file_content(str(tmp_path / "missing.py"))


# validate_hints: unknown phase and cell keys

def test_validate_hints_clean_file_has_no_warnings():
    hints = {
        "coverage": {"general": ["x"], "per_language": {"python": ["y"]}},
        "mutation": {"per_profile": {"foo.py": ["z"]}},
        "initial": {"general": ["w"]},
    }
    assert hints_module.validate_hints(hints) == []


def test_validate_hints_flags_unknown_phase():
    warnings = hints_module.validate_hints({"covrage": {"general": ["typo"]}})
    assert len(warnings) == 1
    assert "covrage" in warnings[0] and "phase" in warnings[0]


def test_validate_hints_flags_unknown_cell():
    warnings = hints_module.validate_hints({"coverage": {"per_lang": {"python": ["x"]}}})
    assert len(warnings) == 1
    assert "per_lang" in warnings[0] and "coverage" in warnings[0]


def test_validate_hints_flags_non_object_phase_block():
    warnings = hints_module.validate_hints({"coverage": ["not", "an", "object"]})
    assert len(warnings) == 1
    assert "coverage" in warnings[0]


def test_validate_hints_empty_is_clean_but_non_object_is_refused():
    assert hints_module.validate_hints({}) == []
    assert hints_module.validate_hints(None) == [  # type: ignore[arg-type]
        "hints.json: hints file must contain a JSON object",
    ]


def test_per_mutant_timeout_auto_scaling_and_pin():
    """``auto`` scales as max(floor, ceil(multiplier x suite_time)), capped at
    cap_s. A positive value pins a fixed timeout whatever the suite time."""
    # Multiplier 1.5 and floor 10, TestGenEval's values.
    per_mutant_timeout = functools.partial(config_module.per_mutant_timeout, floor_s=10, multiplier=1.5)
    assert per_mutant_timeout("auto", 0.0) == 10     # 10 s floor
    assert per_mutant_timeout("auto", 6.0) == 10     # 1.5*6=9, below floor
    assert per_mutant_timeout("auto", 20.0) == 30    # 1.5*20 above floor
    assert per_mutant_timeout("auto", 20.04) == 31   # ceil(30.06)
    assert per_mutant_timeout("auto", None) == 10    # no suite time -> floor
    # The floor is configurable, and the larger of floor and scaled value wins.
    assert per_mutant_timeout("auto", 0.3, floor_s=30) == 30
    assert per_mutant_timeout("auto", 20.0, floor_s=120) == 120
    # The multiplier is configurable.
    assert per_mutant_timeout("auto", 6.0, multiplier=10.0) == 60
    assert per_mutant_timeout("auto", 0.3, multiplier=10.0, floor_s=30) == 30
    # cap_s bounds the auto value (and a cap below the floor still wins).
    assert per_mutant_timeout("auto", 50.0, cap_s=30) == 30    # 75 -> cap
    assert per_mutant_timeout("auto", 20.0, cap_s=300) == 30   # under cap
    assert per_mutant_timeout("auto", 0.0, cap_s=5) == 5       # cap < floor
    # The cap does not clamp a pinned value.
    assert per_mutant_timeout(180, 20.0) == 180      # pinned, ignores time
    assert per_mutant_timeout(600, 6.0, cap_s=300) == 600  # pinned, uncapped
    assert per_mutant_timeout("60", None) == 60      # string int, pinned
    # fixed_per_mutant_timeout: auto, non-positive or empty gives 0
    assert config_module.fixed_per_mutant_timeout("auto") == 0
    assert config_module.fixed_per_mutant_timeout(0) == 0
    assert config_module.fixed_per_mutant_timeout("") == 0
    assert config_module.fixed_per_mutant_timeout(180) == 180


def test_bundled_mutation_timeout_is_load_robust():
    """The bundled per-mutant timeout is 10x the suite's green timing, at least
    30 s, so a slow but green mutant is judged by its assertions, not by a
    load spike. With a tighter timeout a mutant could time out under load and be
    killed when quiet, so the score would depend on load. Every mutation
    measurement reads these keys."""
    cfg = config_module.get_agent_config("test_mutation")
    assert int(cfg["min_per_mutant_timeout_seconds"]) == 30
    assert float(cfg["per_mutant_timeout_factor"]) == 10.0
    # A 7 s suite gets 70 s, not the 11 s of a 1.5x factor with a 10 s floor.
    t = config_module.per_mutant_timeout(
        "auto", 7.0,
        cap_s=config_module.resolve_mutation_max_timeout(cfg),
        floor_s=int(cfg["min_per_mutant_timeout_seconds"]),
        multiplier=float(cfg["per_mutant_timeout_factor"]))
    assert t == 70


def test_level_tolerant_log_adapts_single_arg_callable():
    """The engine calls log_fn(msg, level=...). A single-argument log callable
    is wrapped so it does not raise, and one that accepts level is returned
    unchanged."""
    seen = []
    def one(m):
        seen.append(m)
    w = _source_metrics.level_tolerant_log(one)
    w("a", level=1)   # engine-style call
    w("b")            # metrics-style call
    assert seen == ["a", "b"]

    def two(msg, level=1):
        pass
    assert _source_metrics.level_tolerant_log(two) is two   # passthrough, no wrap

    def kw(msg, **kwargs):
        pass
    assert _source_metrics.level_tolerant_log(kw) is kw     # **kwargs accepts level
    assert _source_metrics.level_tolerant_log(None) is None
