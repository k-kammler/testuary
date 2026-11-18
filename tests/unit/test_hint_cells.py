"""Which hint cells reach a phase's prompt, and what the run reports.

``hints.json`` offers four cells per phase: ``general``, ``per_profile``,
``per_language`` and ``per_profile_per_language``. The profile is the
project's ``hint_profile``. The validator names the file in use and reports
what renders nothing, and the end-of-run summary lists the cells that apply
to the run's profile. The prompts are resolved, never sent.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from testuary import orchestrator as run_orchestrator
from testuary import config
from testuary import hints as hints_module
from testuary import prompts as prompts_module
from testuary import settings

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

PHASE = "coverage"
AGENT = "test_coverage"
HEADING = "## Additional guidance"

HINTS = {
    PHASE: {
        "general": "general guidance",
        "per_profile": {"speed": "profile guidance"},
        "per_language": {"python": "language guidance"},
        "per_profile_per_language": {"speed": {"python": "cross guidance"}},
    },
}


def _system_prompt(monkeypatch, hints, **kwargs):
    """Resolve a phase's prompts with a hints payload, through the real
    resolver. ``load_hints`` is the loader every caller reaches."""
    monkeypatch.setattr(hints_module, "load_hints", lambda: hints)
    prompts = prompts_module.resolve_agent_prompts(AGENT, "python", **kwargs)
    return prompts["system_prompt"]


def _guidance(system_prompt):
    """The rendered guidance block, or "" when no heading was appended."""
    _, _, tail = system_prompt.partition(f"{HEADING}\n\n")
    return tail


# ---------------------------------------------------------------------------
# What renders
# ---------------------------------------------------------------------------

def test_without_a_profile_only_the_profile_free_cells_render(monkeypatch):
    guidance = _guidance(_system_prompt(monkeypatch, HINTS))

    assert guidance == "general guidance\n\nlanguage guidance"


def test_with_a_profile_all_four_cells_render_in_order(monkeypatch):
    guidance = _guidance(_system_prompt(monkeypatch, HINTS,
                                        hint_profile="speed"))

    assert guidance == (
        "general guidance\n\n"
        "profile guidance\n\n"
        "language guidance\n\n"
        "cross guidance"
    )
    other = _guidance(_system_prompt(monkeypatch, HINTS, hint_profile="size"))
    assert other == "general guidance\n\nlanguage guidance"


def test_every_run_time_call_site_passes_the_projects_profile():
    """Every package call site that resolves prompts passes the project's
    profile: the three phases and the generation origins."""
    package_dir = Path(config.__file__).resolve().parent
    with_profile = []
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None))
            if name != "resolve_agent_prompts":
                continue
            site = path.relative_to(package_dir).as_posix()
            keywords = {kw.arg for kw in node.keywords}
            assert "hint_profile" in keywords, (site, node.lineno)
            profile = next(kw.value for kw in node.keywords if kw.arg == "hint_profile")
            assert isinstance(profile, ast.Call), (site, node.lineno)
            assert isinstance(profile.func, ast.Attribute), (site, node.lineno)
            assert profile.func.attr == "hint_profile", (site, node.lineno)
            assert isinstance(profile.func.value, ast.Name) and profile.func.value.id in ("hints", "hints_module"), (site, node.lineno)
            with_profile.append(site)

    assert sorted(with_profile) == [
        "agent/coverage_agent.py",
        "agent/mutation_agent.py",
        "evidence.py", "starters.py"]


def test_the_profile_comes_from_the_project_configuration(monkeypatch, in_run):
    in_run()
    assert hints_module.hint_profile() == ""
    monkeypatch.setenv("TESTUARY_PROJECT_HINT_PROFILE", "speed")
    in_run()
    assert hints_module.hint_profile() == "speed"


def test_an_empty_hints_file_appends_no_heading(monkeypatch):
    """A baseline run is byte-identical to one with an empty hints file."""
    without = _system_prompt(monkeypatch, {})
    empty_cells = _system_prompt(
        monkeypatch, {PHASE: {"general": "", "per_language": {}}},
        hint_profile="speed")

    assert HEADING not in without
    assert without == empty_cells


def test_an_initial_block_renders_and_the_validator_knows_it(monkeypatch):
    hints = {"initial": {"general": "seed guidance"}}
    monkeypatch.setattr(hints_module, "load_hints", lambda: hints)

    prompts = prompts_module.resolve_agent_prompts("test_initial", "python")

    assert _guidance(prompts["system_prompt"]) == "seed guidance"
    assert hints_module.validate_hints(hints) == []


# ---------------------------------------------------------------------------
# What the validator reports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [42, {"nested": "x"}, True, 3.5])
def test_a_cell_value_that_is_not_text_renders_nothing_and_is_reported(
    monkeypatch, value,
):
    hints = {PHASE: {"general": value,
                     "per_language": {"python": value},
                     "per_profile_per_language": {"speed": {"python": value}}}}

    guidance = _guidance(_system_prompt(monkeypatch, hints,
                                        hint_profile="speed"))

    assert guidance == ""
    warnings = hints_module.validate_hints(hints)
    assert len(warnings) == 3 and all("is not text" in w for w in warnings)
    assert run_orchestrator._list_active_hint_cells(hints, "speed") == []


def test_a_list_of_lines_is_accepted(monkeypatch):
    hints = {PHASE: {"general": ["first line", "second line"]}}

    assert _guidance(_system_prompt(monkeypatch, hints)) == (
        "first line\nsecond line")
    assert hints_module.validate_hints(hints) == []


def test_the_validator_names_the_file_in_use(monkeypatch, tmp_path, in_run):
    (tmp_path / "ablation_hints.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_MAIN_HINTS_FILE", str(tmp_path / "ablation_hints.json"))
    in_run()
    warnings = hints_module.validate_hints({"nosuchphase": {"general": "x"},
                                     "_about": ["a note"]})

    assert len(warnings) == 1
    assert warnings[0].startswith(str(tmp_path / "ablation_hints.json"))
    assert "unknown phase 'nosuchphase'" in warnings[0]


# ---------------------------------------------------------------------------
# What the end-of-run summary reports
# ---------------------------------------------------------------------------

def test_the_summary_lists_the_cells_that_apply_to_the_profile(monkeypatch):
    applied = run_orchestrator._list_active_hint_cells(HINTS, "speed")
    rendered = _guidance(_system_prompt(monkeypatch, HINTS,
                                        hint_profile="speed"))

    assert applied == [
        "coverage.general",
        "coverage.per_profile[speed]",
        "coverage.per_language[python]",
        "coverage.per_profile_per_language[speed][python]",
    ]
    for text in ("general", "profile", "language", "cross"):
        assert f"{text} guidance" in rendered
    for profile in ("", "size"):
        assert run_orchestrator._list_active_hint_cells(HINTS, profile) == [
            "coverage.general", "coverage.per_language[python]"]


def test_a_mistyped_phase_is_left_to_the_validator(monkeypatch):
    """A phase no resolver reads applies to nothing: the summary leaves it
    out and the warning at run start names it."""
    hints = {
        "coverage": {"general": "c"},
        "covrage": {"general": "typo"},
        "_about": ["documentation"],
    }

    assert run_orchestrator._list_active_hint_cells(hints) == [
        "coverage.general"]
    assert _guidance(_system_prompt(monkeypatch, hints)) == "c"
    assert [w for w in hints_module.validate_hints(hints)
            if "unknown phase 'covrage'" in w]


def test_the_bundled_hints_template_is_inert(monkeypatch):
    """The shipped template documents the format and fills no cell: no
    guidance renders, the summary lists nothing and the validator is silent."""
    template = settings._bundled_template_path("hints.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert run_orchestrator._list_active_hint_cells(payload, "speed") == []
    assert HEADING not in _system_prompt(monkeypatch, payload,
                                         hint_profile="speed")
    assert hints_module.validate_hints(payload) == []
