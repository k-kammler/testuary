"""Loading and resolving the shipped agent prompts: package data through
``importlib.resources``, language slots, hints, and the parity of the
coverage and mutation prompts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from testuary import config
from testuary import hints as hints_module
from testuary import prompts as prompts_module
from testuary.prompts import load_agent_prompts, resolve_agent_prompts, render_read_protocol, _READ_PROTOCOL, _join_lines
from testuary.languages import get_backend

from agent_prompts import c_prompts

# The prompts are resolved inside a run, as a phase resolves them.
pytestmark = pytest.mark.usefixtures("default_run")


# Every agent that ships a prompt file under ``testuary/prompts/``.
SHIPPED_AGENTS = ["test_initial", "test_coverage", "test_mutation"]


@pytest.mark.parametrize("agent_name", SHIPPED_AGENTS)
def test_each_shipped_agent_loads(agent_name: str) -> None:
    """Every bundled prompt JSON loads and has at least ``system_prompt``."""
    prompts = load_agent_prompts(agent_name)
    assert "system_prompt" in prompts, (
        f"{agent_name}: shipped prompt JSON is missing system_prompt"
    )
    assert isinstance(prompts["system_prompt"], str)
    assert len(prompts["system_prompt"]) > 0


@pytest.mark.parametrize("agent_name", SHIPPED_AGENTS)
def test_list_of_lines_joins_with_newline(agent_name: str) -> None:
    """JSON list-of-lines values are joined with ``\\n``. The non-empty
    character count checks that no key loses its content."""
    prompts = load_agent_prompts(agent_name)
    sp = prompts["system_prompt"]
    # Each shipped prompt is multi-line in its source.
    assert "\n" in sp, f"{agent_name} system_prompt looks single-line; check JSON"


def test_tool_messages_section_when_present_is_a_dict() -> None:
    """``tool_messages`` is an optional top-level key. When present it is a
    dict of strings, joined from list-of-lines by the loader."""
    # test_coverage has the largest tool_messages section.
    prompts = load_agent_prompts("test_coverage")
    if "tool_messages" not in prompts:
        pytest.skip("test_coverage prompt has no tool_messages section")
    tm = prompts["tool_messages"]
    assert isinstance(tm, dict)
    assert len(tm) > 0
    for key, value in tm.items():
        assert isinstance(key, str)
        assert isinstance(value, str), (
            f"tool_messages[{key!r}] must be str after loader join, "
            f"got {type(value).__name__}"
        )


def test_missing_agent_raises() -> None:
    """A typo in the agent name raises instead of giving an empty dict."""
    with pytest.raises((FileNotFoundError, Exception)):
        load_agent_prompts("does_not_exist")


def test_join_lines_string_passthrough() -> None:
    """``_join_lines`` accepts both strings (returned as-is) and lists
    (joined with ``\\n``)."""
    assert _join_lines("already a string", where="t") == "already a string"
    assert _join_lines(["one", "two", "three"], where="t") == "one\ntwo\nthree"


def test_join_lines_rejects_bad_types() -> None:
    """Values other than str or list raise instead of being coerced."""
    with pytest.raises(ValueError):
        _join_lines(42, where="t")
    with pytest.raises(ValueError):
        _join_lines(None, where="t")
    with pytest.raises(ValueError):
        _join_lines({"bad": "shape"}, where="t")


# prompt_fragments(): the only language-dependent input of the prompts

@pytest.fixture(autouse=True)
def _stable_runtime_version(monkeypatch):
    """The runtime-version fragment reports the execution interpreter. The
    byte-for-byte snapshot comparison needs it stable across interpreters,
    so every resolution in this module sees a fixed token."""
    from testuary import prompts as vocabulary
    monkeypatch.setattr(
        vocabulary, "runtime_line",
        lambda facts, name, **_: "Python (the execution interpreter)",
    )
    # The language level the adapter reports is the interpreter's version
    # too: the prompts name the fixed token in its place.
    monkeypatch.setattr(prompts_module, "_language_level", lambda backend, source_dir: None)


LANG_AGENTS = ["test_coverage", "test_mutation", "test_initial"]


@pytest.mark.parametrize("agent_name", ["test_initial", "test_coverage", "test_mutation"])
def test_the_validity_rule_is_the_languages_own(agent_name: str) -> None:
    """The rule on what a test may read of the target is the language's
    validity_rule, in its own terms: a Java prompt names no Python
    mechanism."""
    for language in ("python", "java"):
        rule = get_backend(language).prompt_fragments()["lang_validity_rule"]
        assert rule, language
        out = resolve_agent_prompts(agent_name, language)
        assert f"\n- {rule}" in out["system_prompt"]
        if agent_name != "test_initial":
            assert rule in out["tool_messages"]["edit_generated_test_validity"]
    java = resolve_agent_prompts(agent_name, "java")
    assert "inspect.getsource" not in json.dumps(java)
    assert "__dict__ and vars()" not in json.dumps(java)


def test_the_starter_prompt_states_the_test_file_form_after_the_single_file_rule() -> None:
    """A language's test_file_form entry, such as C's, is a requirement of
    its own after the single-file rule in the starter prompt."""
    assert ("or a ``main`` of your own.\n"
            "- A test file holds, in this order:\n"
            "  - includes of ``utest.h`` and the target's header\n"
            "  - tests as ``UTEST(suite, name) { ... }``\n"
            "  - a fixture test as ``UTEST_F(fixture, name)``, which needs\n"
            "    ``struct fixture``, ``UTEST_F_SETUP(fixture)`` and\n"
            "    ``UTEST_F_TEARDOWN(fixture)``, and reads its fixture through\n"
            "    ``utest_fixture``\n"
            "  - checks inside a test only: ``ASSERT_`` or ``EXPECT_`` with ``TRUE``,\n"
            "    ``FALSE``, ``EQ``, ``NE``, ``LT``, ``LE``, ``GT``, ``GE``, ``NEAR``,\n"
            "    ``STREQ``, ``STRNE``, ``STRNEQ``, ``STRNNE`` or ``MEMEQ``\n"
            "  - ``UTEST_MAIN()`` as the last line, which is utest.h's ``main``, so\n"
            "    the file has no ``main`` of its own\n"
            "- Return only the test file in one ```c code block.") in c_prompts(
        "test_initial")["system_prompt"]


@pytest.mark.parametrize("agent_name", LANG_AGENTS)
def test_a_language_without_a_backend_is_refused(agent_name: str) -> None:
    """A run refuses a file no language claims before any prompt is built,
    so prompts exist only for registered languages."""
    with pytest.raises(ValueError, match="unknown language 'unknown'"):
        resolve_agent_prompts(agent_name, "unknown")


@pytest.mark.parametrize("agent_name", ["test_coverage", "test_mutation"])
def test_the_unsound_async_message_exists_only_where_the_language_checks(
        agent_name: str) -> None:
    """The strict-green message on unsound asynchronous tests is offered
    only where the language's adapter reads such warnings (Python). Java's
    runs never reach it, so its prompts do not carry it, and no prompt says
    "traceback" where a stack trace is meant."""
    assert "tests_strict_green_unsound" in resolve_agent_prompts(
        agent_name, "python")["tool_messages"]
    out = resolve_agent_prompts(agent_name, "java")
    assert "tests_strict_green_unsound" not in out["tool_messages"]
    assert "asyncio" not in json.dumps(out)
    assert "traceback" not in resolve_agent_prompts(agent_name, "java")["system_prompt"]


def test_prompt_fragments_python_supplies_every_used_slot() -> None:
    """The Python backend supplies every ``$lang_*`` slot the shipped
    prompts reference. A slot a prompt uses but the backend omits would
    leave a literal ``$lang_...`` in a model-facing prompt. The timeout
    decorator is empty where the project's pytest has no pytest-timeout,
    and no prompt uses it. Python's coverage reports no counts it cannot
    read, so it names no cause and no fix for them. A Python test file needs
    no form beyond its tests."""
    py = get_backend("python").prompt_fragments()
    empty = {"lang_timeout_decorator", "lang_coverage_counts_inconsistent_cause",
             "lang_coverage_counts_inconsistent_fix", "lang_test_file_form"}
    assert all(isinstance(v, str) and (v or k in empty) for k, v in py.items())
    import re
    used: set = set()
    for agent in LANG_AGENTS:
        raw = load_agent_prompts(agent)
        blob = raw.get("system_prompt", "") + raw.get(
            "user_prompt_template", ""
        )
        for v in raw.get("tool_messages", {}).values():
            blob += v
        used |= set(re.findall(r"\$\{?(lang_[a-z_]+)\}?", blob))
    missing = used - set(py)
    assert not missing, f"prompts use slots the backend does not supply: {missing}"


# resolve_agent_prompts: slot filling and the optional language_extra block

def test_resolve_fills_lang_slots_and_keeps_runtime_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``$lang_*`` is filled from the backend. Unrelated ``$runtime``
    placeholders survive for the caller's later ``.substitute``."""
    monkeypatch.setattr(
        prompts_module, "load_agent_prompts",
        lambda name: {
            "system_prompt": "Write a ${lang_test_decl} in $file_path.",
            "user_prompt_template": "lang=$language code=$file_content",
            "tool_messages": {"k": "not ${lang_valid_source}"},
        },
    )
    out = resolve_agent_prompts("whatever", "python")
    assert out["system_prompt"] == (
        "Write a ``def test_*(...)`` in $file_path."
    )
    assert out["user_prompt_template"] == "lang=$language code=$file_content"
    assert out["tool_messages"]["k"] == "not valid Python"


def test_resolve_runtime_version_override_replaces_lang_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``runtime_version`` overrides ``lang_runtime_version``, so a caller
    that knows the interpreter the tests are measured under, such as a
    benchmark file pinned to another version than the host's, can name it
    to the model."""
    monkeypatch.setattr(
        prompts_module, "load_agent_prompts",
        lambda name: {
            "system_prompt": "",
            "user_prompt_template": "Runtime: ${lang_runtime_version}",
            "tool_messages": {},
        },
    )
    # default: the (stubbed) execution interpreter
    base = resolve_agent_prompts("test_initial", "python")
    assert base["user_prompt_template"] == (
        "Runtime: Python (the execution interpreter)")
    # override: the target's pinned version
    out = resolve_agent_prompts("test_initial", "python",
                                runtime_version="Python 3.10")
    assert out["user_prompt_template"] == "Runtime: Python 3.10"


def test_resolve_unrecognized_language_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A language other than 'unknown' without a backend raises. Only the
    explicit 'unknown' takes the generic path."""
    monkeypatch.setattr(
        prompts_module, "load_agent_prompts",
        lambda name: {"system_prompt": "x"},
    )
    with pytest.raises(ValueError):
        resolve_agent_prompts("whatever", "cobol")


def test_resolve_agent_prompts_strips_test_prefix_for_phase_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hint loader looks up ``coverage`` and ``mutation`` in hints.json
    for the agent modules ``test_coverage`` and ``test_mutation``: the JSON
    uses the phase names, not the module names."""
    raw = {"system_prompt": "BASE."}
    monkeypatch.setattr(prompts_module, "load_agent_prompts", lambda name: dict(raw))
    monkeypatch.setattr(
        hints_module, "load_hints",
        lambda: {"mutation": {"general": "Mutation-only hint."}},
    )
    # Agent module name is ``test_mutation``. Phase key is ``mutation``.
    out = resolve_agent_prompts("test_mutation", "python")
    assert "Mutation-only hint." in out["system_prompt"]


# hints.json: four-cell composition, the working directory's file and
# cells_active tracking


def test_resolve_hints_empty_input_returns_empty_string() -> None:
    """Absent or empty hints give an empty string and no
    ``## Additional guidance`` heading, so a run without hints keeps the
    prompt byte for byte."""
    assert hints_module.resolve_hints(
        "runtime", "python", phase="mutation", hints={},
    ) == ""
    assert hints_module.resolve_hints(
        "runtime", "python", phase="mutation", hints=None,
    ) is not None
    # Empty cells inside the phase block:
    assert hints_module.resolve_hints(
        "runtime", "python", phase="mutation",
        hints={"mutation": {
            "general": "", "per_profile": {}, "per_language": {},
            "per_profile_per_language": {},
        }},
    ) == ""


def test_resolve_hints_requires_phase_to_render_anything() -> None:
    """``phase=None`` returns an empty string even with populated cells:
    hints belong to a phase block, and the agent loader always knows its
    phase."""
    hints = {"mutation": {"general": "Should NOT appear."}}
    out = hints_module.resolve_hints(
        "runtime", "python", phase=None, hints=hints,
    )
    assert out == ""


def test_resolve_hints_four_cells_render_in_documented_order() -> None:
    """Render general, target, language and target-language hints in that order.

    Separate non-empty cells with blank lines in the system prompt."""
    hints = {
        "mutation": {
            "general": "G hint.",
            "per_profile": {"runtime": "T hint."},
            "per_language": {"python": "L hint."},
            "per_profile_per_language": {"runtime": {"python": "TL hint."}},
        },
    }
    out = hints_module.resolve_hints(
        "runtime", "python", phase="mutation", hints=hints,
    )
    assert out == "G hint.\n\nT hint.\n\nL hint.\n\nTL hint."


def test_resolve_hints_picks_only_matching_phase_profile_language() -> None:
    """A hints.json with entries across several phases and (target,
    language) pairs yields only the cells of the active phase block that
    match the active pair, here for a Memory/Java mutation case."""
    hints = {
        "mutation": {
            "general": "",
            "per_profile": {
                "runtime": "should NOT appear",
                "memory": "MEM target hint.",
            },
            "per_language": {
                "python": "should NOT appear",
                "java": "JAVA language hint.",
            },
            "per_profile_per_language": {
                "runtime": {"python": "should NOT appear"},
                "memory": {
                    "python": "should NOT appear",
                    "java": "MEM x JAVA cross hint.",
                },
            },
        },
        "coverage": {
            "general": "should NOT appear (wrong phase).",
            "per_profile_per_language": {
                "memory": {"java": "should NOT appear (wrong phase)."},
            },
        },
    }
    out = hints_module.resolve_hints(
        "memory", "java", phase="mutation", hints=hints,
    )
    assert "should NOT appear" not in out
    assert out == (
        "MEM target hint.\n\nJAVA language hint.\n\nMEM x JAVA cross hint."
    )


def test_resolve_hints_supports_list_and_string_cell_values() -> None:
    """A cell may be a single string or a list of lines, as multi-line
    prompts elsewhere. The resolver joins lines on newline and strips
    trailing whitespace."""
    out = hints_module.resolve_hints(
        "runtime", "python", phase="mutation",
        hints={"mutation": {"general": ["Line one.", "Line two."]}},
    )
    assert out == "Line one.\nLine two."
    out = hints_module.resolve_hints(
        "runtime", "python", phase="mutation",
        hints={"mutation": {"general": "Single string hint.   "}},
    )
    assert out == "Single string hint."


def test_load_hints_reads_cwd_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """A ``./hints.json`` in the working directory is read whole: only the
    phase blocks it carries are present, and no empty blocks are added for
    the phases it omits."""
    cwd_hints = {
        "mutation": {
            "general": "From CWD.",
            "per_profile": {"runtime": "CWD target hint."},
        },
    }
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hints.json").write_text(
        json.dumps(cwd_hints), encoding="utf-8",
    )
    in_run()
    loaded = hints_module.load_hints()
    assert loaded["mutation"]["general"] == "From CWD."
    # A phase the CWD file omits is simply absent (whole-file load).
    assert loaded.get("coverage", "absent") == "absent"


def test_hints_filename_default_is_hints_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an override the loader looks for ``./hints.json`` in the
    working directory."""
    monkeypatch.setattr(config, "load_config", lambda section: {})
    assert hints_module.hints_filename() == "hints.json"


def test_hints_filename_config_key_wins_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[main] hints_file`` in config.ini selects the file name for this
    run, so files such as ``hints_memory.json`` and ``hints_runtime.json``
    can sit side by side."""
    monkeypatch.setattr(
        config, "load_config",
        lambda section: {"hints_file": "hints_memory.json"}
        if section == "main" else {},
    )
    assert hints_module.hints_filename() == "hints_memory.json"


def test_load_hints_reads_configured_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """The loader opens ``./<hints_filename()>``, not a fixed
    ``hints.json``: a ``hints_memory.json`` in the working directory is loaded
    when the setting names it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hints_memory.json").write_text(
        json.dumps({"mutation": {"general": "Memory mode active."}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TESTUARY_MAIN_HINTS_FILE", "hints_memory.json")
    in_run()
    loaded = hints_module.load_hints()
    assert loaded["mutation"]["general"] == "Memory mode active."


def test_load_hints_refuses_when_selected_file_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """A selected hints file that is missing is a configuration error."""
    from testuary.settings import ConfigurationError

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TESTUARY_MAIN_HINTS_FILE", "hints_missing.json")
    with pytest.raises(ConfigurationError, match="selected hints file"):
        in_run()


def test_phase_key_names_the_starter_phase_initial_and_passes_others_through() -> None:
    """The hints and profile files name the starter phase ``initial``.
    Another phase keeps its name, so one that reads no hints finds no
    block instead of an error."""
    from testuary._phase_keys import phase_key

    assert [phase_key(phase) for phase in ("starter", "coverage", "mutation", "finishing")] == [
        "initial", "coverage", "mutation", "finishing"]


# Resolving the shipped prompts for "python" reproduces the Python snapshot
# byte for byte.

@pytest.mark.parametrize("agent_name", LANG_AGENTS)
def test_resolved_python_prompts_are_byte_identical_to_snapshot(
    agent_name: str, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, in_run,
) -> None:
    """With the guidance on replacing built-ins off, the prompts equal the
    snapshot byte for byte."""
    monkeypatch.setenv(
        "TESTUARY_COVERAGE_BUILTIN_REPLACEMENT_GUIDANCE", "false")
    in_run()
    snap = json.loads(
        (fixtures_dir / "prompt_python_snapshot.json").read_text("utf-8")
    )[agent_name]
    out = resolve_agent_prompts(agent_name, "python")
    assert out["system_prompt"] == snap["system_prompt"]
    assert out.get("user_prompt_template", "") == snap["user_prompt_template"]
    assert out.get("tool_messages", {}) == snap["tool_messages"]


_PORTABLE_TESTS_RULE_END = "  name or location. Derive it at runtime if needed."


@pytest.mark.parametrize("agent_name", LANG_AGENTS)
def test_the_built_in_guidance_is_one_added_hard_rule(
    agent_name: str, fixtures_dir: Path,
) -> None:
    """Under the package defaults the coverage and mutation prompts are the
    snapshot's with one hard rule added after the rule on deterministic,
    portable tests. The starter suites' prompt has no such rule and is
    unchanged."""
    snap = json.loads(
        (fixtures_dir / "prompt_python_snapshot.json").read_text("utf-8")
    )[agent_name]
    out = resolve_agent_prompts(agent_name, "python")
    rule = get_backend("python").prompt_fragments()[
        "lang_builtin_replacement_rule"]
    if agent_name == "test_initial":
        assert out["system_prompt"] == snap["system_prompt"]
        return
    assert snap["system_prompt"].count(_PORTABLE_TESTS_RULE_END) == 1
    assert out["system_prompt"] == snap["system_prompt"].replace(
        _PORTABLE_TESTS_RULE_END, _PORTABLE_TESTS_RULE_END + "\n" + rule)
    assert "monkeypatch.setattr(module, \"len\", fake, raising=False)" in rule
    assert "monkeypatch.setitem(sys.modules, \"name\", None)" in rule
    # The tool messages, the two stands-alone notes among them, are the
    # snapshot's.
    assert out.get("tool_messages", {}) == snap["tool_messages"]
    assert snap["tool_messages"]["tests_stands_alone_note"].startswith("[STANDS ALONE, NOTE]")


def test_test_quality_rules_are_consistent_across_agent_prompts() -> None:
    prompts = {
        name: " ".join(
            resolve_agent_prompts(name, "python")["system_prompt"].split()
        )
        for name in LANG_AGENTS
    }

    for name in ("test_initial", "test_coverage"):
        prompt = prompts[name]
        assert "explicit error paths" in prompt
        assert "representative values just below and above" in prompt
        assert "Use strong assertions that would fail" in prompt
        assert "Assert exact observable outcomes when the target defines them" in prompt
        assert "property, range, or relation defined by the target" in prompt

    assert "feasible branches" in prompts["test_initial"]
    assert "feasible branch outcomes" in prompts["test_coverage"]

    mutation = prompts["test_mutation"]
    assert "distinguish the original behavior from the mutated behavior" in mutation
    assert "check non-null results or successful execution" in mutation
    assert "survivor diffs" in mutation

    for prompt in prompts.values():
        assert "stable parts of the behavior under test" not in prompt
        assert "supported input domain" not in prompt
        assert "kills almost nothing" not in prompt


@pytest.mark.parametrize("agent_name", LANG_AGENTS)
def test_resolved_prompts_have_no_unfilled_lang_slot(
    agent_name: str,
) -> None:
    """After resolving for Python, no literal ``$lang_...`` remains in any
    model-facing text."""
    out = resolve_agent_prompts(agent_name, "python")
    blob = out.get("system_prompt", "") + out.get(
        "user_prompt_template", ""
    )
    for v in out.get("tool_messages", {}).values():
        blob += v
    assert "$lang_" not in blob and "${lang_" not in blob, (
        f"{agent_name}: unfilled lang slot remains"
    )


@pytest.mark.parametrize(
    "agent_name", ["test_coverage", "test_mutation"]
)
def test_resolved_prompts_fill_elided_sentinel_slot(agent_name: str) -> None:
    """The history-trim marker is defined once in ``_helpers`` and reaches
    the prompts only through the ``${elided_sentinel}`` slot. Resolving fills
    it with that marker, so the history-marker messages name the exact string
    the write validator detects."""
    from testuary.agent._helpers import ELIDED_SENTINEL
    out = resolve_agent_prompts(agent_name, "python")
    blob = "".join(out.get("tool_messages", {}).values())
    assert "elided_sentinel" not in blob, (
        f"{agent_name}: unfilled elided_sentinel slot remains"
    )
    assert ELIDED_SENTINEL in blob, (
        f"{agent_name}: canonical marker missing after resolve"
    )


# Read protocol: the single "what to read first" instruction


def test_render_read_protocol_per_agent() -> None:
    """Each agent's spec renders the one required first read with its reason.
    An unknown agent renders nothing."""
    for agent, (tool, reason) in _READ_PROTOCOL.items():
        out = render_read_protocol(agent)
        assert f"``{tool}``" in out
        assert reason in out
        assert "not in this prompt" in out
        assert "Do not request another tool in the same response" in out
        assert "wait for this read to return" in out
    assert render_read_protocol("not_an_agent") == ""


@pytest.mark.parametrize(
    "agent_name",
    ["test_coverage", "test_mutation"],
)
def test_read_protocol_slot_filled_in_resolved_prompts(agent_name: str) -> None:
    """The read-first instruction reaches the editing agents only through the
    ``$read_protocol`` slot: resolving fills it (no literal slot leaks) and the
    rendered sentence is present."""
    out = resolve_agent_prompts(agent_name, "python")
    blob = out["system_prompt"] + "\n" + out.get("user_prompt_template", "")
    assert "$read_protocol" not in blob
    assert render_read_protocol(agent_name) in blob


def test_editing_agents_always_begin_with_the_preceding_suite() -> None:
    """Coverage and Mutation have one entry protocol with no source fallback."""
    for agent_name in ("test_coverage", "test_mutation"):
        out = render_read_protocol(agent_name)
        assert "``read_tests``" in out
        assert "suite you are extending" in out
    from testuary.agent.coverage_agent import CoverageAgent
    from testuary.agent.mutation_agent import MutationAgent
    assert CoverageAgent.designated_read_name == "read_tests"
    assert MutationAgent.designated_read_name == "read_tests"


def test_generated_test_validity_rule_is_system_only_and_exactly_once() -> None:
    rule = (
        "Allowed: the target's runtime behaviour and documented metadata "
        "(attributes, __dict__ and vars(), __name__, __doc__, annotations, "
        "inspect.signature). Forbidden: reading or analysing its source text "
        "(inspect.getsource, its file) or bytecode (__code__, dis), executing "
        "copied source, or attributing synthetic code to its source locations."
    )
    for agent_name in LANG_AGENTS:
        prompt = resolve_agent_prompts(agent_name, "python")
        assert prompt["system_prompt"].count(rule) == 1
        assert rule not in prompt.get("user_prompt_template", "")


def test_the_validity_rejection_states_the_rule_of_the_prompts() -> None:
    """The rejection of an edit that reads the target's implementation
    states the rule as the system prompts do, in both loops."""
    for agent_name in ("test_coverage", "test_mutation"):
        prompt = resolve_agent_prompts(agent_name, "python")
        rejection = prompt["tool_messages"]["edit_generated_test_validity"]
        assert (
            "Allowed: the target's runtime behaviour and documented metadata "
            "(attributes, __dict__ and vars(), __name__, __doc__, annotations, "
            "inspect.signature). Forbidden: reading or analysing its source text "
            "(inspect.getsource, its file) or bytecode (__code__, dis), executing "
            "copied source, or attributing synthetic code to its source locations."
        ) in rejection


def test_phase_one_prompt_is_tool_free() -> None:
    """The starter suites come from a direct zero-shot generation request,
    not a tool loop."""
    prompt = resolve_agent_prompts("test_initial", "python")
    blob = "\n".join([
        prompt["system_prompt"], prompt["user_prompt_template"],
    ])
    assert prompt["tool_messages"] == {}
    for tool_name in (
        "read_source", "read_tests", "edit_tests", "run_tests",
        "measure_coverage", "measure_mutation",
    ):
        assert tool_name not in blob


@pytest.mark.parametrize("agent_name", ["test_coverage", "test_mutation"])
def test_editing_prompt_matches_the_initial_tool_gate(agent_name: str) -> None:
    """The static instructions invite no read past the initial tool gate in
    the first turn."""
    prompt = resolve_agent_prompts(agent_name, "python")
    system = prompt["system_prompt"]
    user = prompt["user_prompt_template"]
    assert render_read_protocol(agent_name) in system
    assert "After it, ``read_source`` becomes available." in system
    assert "first\nread the current tests (``read_tests`` is the only tool offered), then\nmeasure them" in system
    for stale in (
        "Load the counterpart artifact",
        "Reads are always available",
        "load the counterpart file",
    ):
        assert stale not in system
        assert stale not in user


def test_editing_read_source_tool_docs_match_the_gate() -> None:
    from testuary.agent.coverage_agent import CoverageAgent
    from testuary.agent.mutation_agent import MutationAgent

    for method in (CoverageAgent.read_source, MutationAgent.read_source):
        doc = method.__doc__ or ""
        assert "after the mandatory initial ``read_tests``" in doc
        assert "full source is in your initial user prompt" not in doc


@pytest.mark.parametrize("agent_name", ["test_coverage", "test_mutation"])
def test_reactive_feedback_prescribes_only_valid_next_progress(
    agent_name: str,
) -> None:
    """Small tool messages must agree with the state transition they cause."""
    messages = resolve_agent_prompts(agent_name, "python")["tool_messages"]
    measure = (
        "measure_coverage" if agent_name == "test_coverage"
        else "measure_mutation"
    )
    cached = messages[f"{measure}_cached_rejected"]
    assert "Repeating the unchanged\nmeasurement does not improve the suite" in cached
    assert "keep calling" not in cached
    assert "DID advance" not in cached

    for key in ("read_tests_no_file", "read_tests_read_error"):
        message = messages[key]
        assert "``edit_tests`` remains unavailable" in message
        assert "Retry ``read_tests`` once" in message

    timeout = messages["tests_timeout_restored"]
    assert "restored file no longer contains those additions" in timeout
    assert "do not reapply the same slow test" in timeout


# Structural parity of the coverage and mutation prompts: every shared block
# is word for word the same after substituting the per-phase nouns. Details
# may differ, style and structure may not.

_PARITY_NOUNS = {
    "test_coverage": {
        "measure": "measure_coverage",
        "result": "CoverageTestResult",
        "criterion": "coverage",
        "criterion_short": "coverage",
        "file_noun": "working file",
    },
    "test_mutation": {
        "measure": "measure_mutation",
        "result": "MutationTestResult",
        "criterion": "mutation",
        "criterion_short": "the score",
        "file_noun": "working file",
    },
}

#: The start and the cycle, in the same words in both system prompts.
_START_AND_CYCLE = (
    "Tool availability is phase-gated: each turn offers only the tools\n"
    "valid for the current step. The loop starts with two steps: first\n"
    "read the current tests (``read_tests`` is the only tool offered), then\n"
    "measure them (the measurement tool is the only tool offered, and its\n"
    "result comes from Testuary's record of that suite). Then the cycle\n"
    "starts in Revise, which offers ``read_source``, ``read_tests`` and\n"
    "``edit_tests``. An edit moves it to Validate, which adds ``run_tests``.\n"
    "A failing run returns to Revise; a passing run moves it to Assess,\n"
    "which offers the two reads and the measurement tool, and ``edit_tests``\n"
    "too after a run with a STANDS ALONE note. After the\n"
    "measurement the framework ends the phase or the loop returns to\n"
    "Revise. The tools offered on a turn are exactly the callable set; a\n"
    "tool missing from the offer is not available yet.")

_SHARED_BLOCKS = [
    # exit conditions
    ("- The framework ends the phase the moment ``{measure}``\n"
     "  reports a value at-or-above the {criterion} goal, and\n"
     "  otherwise after the configured number of consecutive measurements\n"
     "  without improvement. You do not need to do anything special to\n"
     "  finish."),
    ("- The output validator REJECTS a returned result unless the suite is\n"
     "  green AND ``{measure}`` produced at least one measurement.\n"
     "  NEVER return ``{result}`` while ANY test is failing."),
    # safety net
    ("**Safety net:** the framework keeps a best-green snapshot of the\n"
     "{file_noun}. An edit that regresses below it is auto-restored\n"
     "after several consecutive red runs, so already-green tests cannot be\n"
     "permanently lost."),
    # phase-gated availability
    _START_AND_CYCLE,
    # shared hard rules
    ("- Verify every edit with ``run_tests``.\n"
     "  Never assume an edit worked."),
    ("- If ``edit_tests`` is REJECTED with ``not found``, your view of\n"
     "  the file has drifted: call ``read_tests``, copy the EXACT\n"
     "  current block, and retry the surgical edit."),
    ("- When a test fails to load (${{lang_collection_error}}, import error,\n"
     "  syntax error), inspect the stack trace and ``read_tests`` first."),
    ("  The source under test is read-only in this phase."),
    ("- Keep tests deterministic and portable. Do not introduce network\n"
     "  access, machine-specific paths, unseeded randomness, or dependence\n"
     "  on the current time. Use temporary files and platform-independent\n"
     "  paths for file behavior. Do not hard-code the working test file's\n"
     "  name or location. Derive it at runtime if needed."),
    "- Keep the suite focused and fast.",
    # shared workflow steps (the gated cycle)
    "4. Verify with ``run_tests`` (it binds once edits are pending).",
    ("5. When the suite is green with unmeasured changes,\n"
     "   ``{measure}`` becomes available: take the measurement."),
    "6. Repeat 3-5 until the framework ends the phase.",
    # shared tool text
    ("- ``edit_tests(old_text, new_text, replace_all=False)``: the only\n"
     "  edit tool, content-based (no line numbers). Three operations from\n"
     "  one tool. INSERT a test: anchor ``old_text`` on a unique nearby\n"
     "  line and put that line plus the new ${{lang_test_decl}} bodies in\n"
     "  ``new_text``. REPLACE a block: ``old_text`` is the exact block,\n"
     "  ``new_text`` the replacement. DELETE: pass empty ``new_text``.\n"
     "  ``old_text`` must be the EXACT bytes from the file with enough\n"
     "  surrounding context to be unique; copy whitespace exactly (an exact\n"
     "  miss falls back to a whitespace-loose match). Line numbers copied\n"
     "  from the read and edit views (``42<TAB>``) are removed. The\n"
     "  tool refuses when old_text is absent or ambiguous.\n"
     "  Several changes in one reply: pass them as ``edits`` (applied in\n"
     "  order, all or nothing, checked once on the result; changes that\n"
     "  depend on each other need this form), or as several calls, which\n"
     "  apply in call order, each against the file after the previous ones."),
    ("- ``run_tests()``: run the suite. Pass/fail summary plus\n"
     "  first-failure stack traces; the cheapest signal during multi-edit\n"
     "  batches. Identical re-runs (no edit in between) return the cached\n"
     "  result."),
]

_SHARED_USER_BLOCKS = [
    ("- The framework ends the phase the moment ``{measure}``\n"
     "  reports a value at-or-above the goal; do not keep editing past"),
    ("- The validator rejects any result that is not green with at least\n"
     "  one ``{measure}`` measurement."),
    "## Target",
    "## Starting state",
    "## Goal & exit",
]


@pytest.mark.parametrize("agent", ["test_coverage", "test_mutation"])
def test_agent_prompts_carry_the_shared_blocks(agent):
    import json as _json
    from importlib.resources import files as _files

    raw = _json.loads(
        (_files("testuary.prompts") / f"{agent}.json").read_text("utf-8"))
    sys_text = "\n".join(raw["system_prompt"])
    user_text = "\n".join(raw["user_prompt_template"])
    nouns = _PARITY_NOUNS[agent]
    for block in _SHARED_BLOCKS:
        want = block.format(**nouns)
        assert want in sys_text, f"{agent} missing shared block: {want[:60]}"
    for block in _SHARED_USER_BLOCKS:
        want = block.format(**nouns)
        assert want in user_text, f"{agent} missing user block: {want[:60]}"


# Parity of tool_messages: each paired mutation value is the exact
# substitution image of its coverage counterpart, criterion slots first, then
# the noun rows. The key sets are the paired keys plus each side's allow-list.

_MESSAGE_PAIRS = [
    ('tests_stands_alone_note', 'tests_stands_alone_note', [], []),
    ('tests_stands_alone_order', 'tests_stands_alone_order', [], []),
    ('goal_waits_for_note', 'goal_waits_for_note', [], []),
    ('measure_coverage_cached_rejected', 'measure_mutation_cached_rejected', [('targeting uncovered lines', 'targeting surviving mutants')], [('coverage would emit', 'mutation would emit'), ('measure_coverage', 'measure_mutation')]),
    ('measure_coverage_cached_red', 'measure_mutation_cached_red', [('there are no numbers cached', 'there are no killed/survived counts cached')], [('measure_coverage', 'measure_mutation'), ('Coverage was never measured', 'Mutation was never measured'), ('red file', 'red working file')]),
    ('measure_coverage_refused_known_red', 'measure_mutation_refused_known_red', [('Coverage cannot be measured on a red\nsuite', "Mutation testing on a red suite\nis meaningless (every mutant 'kills' something already broken)")], [('measure_coverage', 'measure_mutation')]),
    ('measure_coverage_red_suite', 'measure_mutation_red_suite', [('Coverage on a red suite is misleading because\nfailing tests short-circuit before their assertions --\nlines behind a failing test report as uncovered even when\ntheir logic would have been exercised on a green suite.', "Mutation on a red suite is meaningless because\nevery mutant 'kills' something already broken, so the\nscore measures the existing failure, not your tests."), ('add tests for uncovered\nregions', 'target surviving mutants')], [('measure_coverage', 'measure_mutation'), ('coverage cannot be', 'mutation cannot be'), ('The coverage phase', 'The mutation phase'), ('the coverage goal', 'the mutation goal')]),
    ('read_tests_read_error', 'read_tests_read_error', [], []),
    ('edit_ok', 'edit_ok', [], []),
    ('edit_ok_list', 'edit_ok_list', [], []),
    ('edit_list_item_failed', 'edit_list_item_failed', [], []),
    ('edit_list_gate_failed', 'edit_list_gate_failed', [], []),
    ('edit_list_mixed', 'edit_list_mixed', [], []),
    ('edit_list_empty', 'edit_list_empty', [], []),
    ('edit_list_too_many', 'edit_list_too_many', [], []),
    ('edit_not_found_after_edit', 'edit_not_found_after_edit', [], []),
    ('edit_list_not_found', 'edit_list_not_found', [], []),
    ('edit_list_multi_match', 'edit_list_multi_match', [], []),
    ('edit_list_replace_all_note', 'edit_list_replace_all_note', [], []),
    ('edit_list_unchanged', 'edit_list_unchanged', [], []),
    ('edit_empty_after_cleaning', 'edit_empty_after_cleaning', [], []),
    ('edit_no_arguments', 'edit_no_arguments', [], []),
    ('edit_missing_new_text', 'edit_missing_new_text', [], []),
    ('edit_bad_characters', 'edit_bad_characters', [], []),
    ('tests_timeout_named', 'tests_timeout_named', [], []),
    ('tests_timeout_restored', 'tests_timeout_restored', [], []),
    ('tests_timeout_abort', 'tests_timeout_abort', [], [('the coverage phase', 'the mutation phase'), ('emit_coverage_test_result', 'emit_mutation_test_result')]),
    ('tests_strict_green_unsound', 'tests_strict_green_unsound', [('${sources_block}\nWhat to do', '${sources_block}\nA leaked-async test that falsely passes kills no mutants and\nwould seed a misleading mutation run.\nWhat to do')], []),
    ('revert_on_regression', 'revert_on_regression', [], [('the goal coverage', 'the mutation score')]),
    ('validator_not_green', 'validator_not_green', [], [('CoverageTestResult', 'MutationTestResult')]),
    ('validator_no_coverage_measurement', 'validator_no_mutation_measurement', [('(a red\nsuite)', '(a red\nsuite, or 0 mutants generated)'), ('produced a coverage %', 'produced a mutation score')], [('a coverage MEASUREMENT', 'a mutation MEASUREMENT'), ('measure_coverage', 'measure_mutation')]),
    ('validator_stagnation_not_exhausted', 'validator_stagnation_not_exhausted', [('Write tests for the\nuncovered code from your last measure_coverage, add\nboundary-condition cases, or try a different angle, then\nrun_tests and measure_coverage.', 'Add tests that kill the surviving mutants -- look at\nthe surviving-mutant diffs from your last measure_mutation,\nsharpen or add assertions, then\nrun_tests and measure_mutation.')], [('CoverageTestResult', 'MutationTestResult'), ('the coverage\ngoal', 'the mutation\ngoal'), ('measure_coverage', 'measure_mutation')]),
    ('validator_empty_targeted_functions', 'validator_empty_targeted_mutants', [("List the\nactual function or method names that your tests exercise\n(e.g. 'JSONDecoder.decode', 'scanstring').", "List the\nids of the mutants that your tests kill, as the survivor list\nshows them (the id after '--- mutant').")], [('CoverageTestResult.targeted_functions', 'MutationTestResult.targeted_mutants')]),
    ('validator_placeholder_targeted_functions', 'validator_placeholder_targeted_mutants', [('the actual function or method names that your tests\nexercise.', 'the ids of the mutants that your tests kill, as the\nsurvivor list shows them.')], [('CoverageTestResult.targeted_functions', 'MutationTestResult.targeted_mutants')]),
    ('run_tests_cached_rejected', 'run_tests_cached_rejected', [], []),
    ('tests_restore_note', 'tests_restore_note', [('(highest\ncoverage seen this run)', '(highest\nmutation score seen this run)')], []),
    ('validator_stale_coverage_measurement', 'validator_stale_mutation_measurement', [], [('coverage measurement', 'mutation measurement'), ('measure_coverage', 'measure_mutation')]),
    ('read_tests_no_file', 'read_tests_no_file', [], []),
    ('tests_timeout', 'tests_timeout', [], []),
    ('tests_collection_error', 'tests_collection_error', [], []),
    ('read_source_loop_breaker_prefix', 'read_source_loop_breaker_prefix', [], [('measure_coverage', 'measure_mutation')]),
    ('validator_placeholder_explanation', 'validator_placeholder_explanation', [('the cases they\ncover, and the achieved coverage', 'the mutants they\nkill, and the achieved mutation score')], [('CoverageTestResult', 'MutationTestResult')]),
    ('stall_hint_coverage_keep_iterating', 'stall_hint_mutation_keep_iterating', [('Coverage has not improved recently, but', 'The mutation score has not improved recently, but'), ("(target an uncovered branch you\nhaven't tried, add a boundary-condition test, split a\ncomplex test)", "(target a surviving mutant you\nhaven't tried, sharpen an existing assertion, split a\ncomplex test)")], []),
    ('read_source_cached', 'read_source_cached', [], []),
    ('edit_no_file', 'edit_no_file', [], []),
    ('edit_read_failed', 'edit_read_failed', [], []),
    ('edit_empty_old', 'edit_empty_old', [], []),
    ('edit_history_marker', 'edit_history_marker', [], []),
    ('edit_not_found', 'edit_not_found', [], []),
    ('edit_multi_match', 'edit_multi_match', [], []),
    ('edit_duplicate_test_name', 'edit_duplicate_test_name', [], []),
    ('edit_unchanged', 'edit_unchanged', [], []),
    ('edit_post_edit_syntax_error', 'edit_post_edit_syntax_error', [], []),
    ('edit_generated_test_validity', 'edit_generated_test_validity', [], []),
    ('tool_not_offered_initial_read', 'tool_not_offered_initial_read', [], []),
    ('tool_not_offered_initial_measure', 'tool_not_offered_initial_measure', [], []),
    ('tool_not_offered_measure_first', 'tool_not_offered_measure_first', [], []),
    ('tool_not_offered_run_tests', 'tool_not_offered_run_tests', [], []),
    ('tool_not_offered_measure_untested', 'tool_not_offered_measure_untested', [], []),
    ('tool_not_offered_measure_red', 'tool_not_offered_measure_red', [], []),
    ('tool_not_offered_measure_measured', 'tool_not_offered_measure_measured', [], []),
    ('tool_not_offered_measure_entry', 'tool_not_offered_measure_entry', [], []),
    ('tool_not_offered_measure_refused', 'tool_not_offered_measure_refused', [], []),
    ('measure_coverage_refused_known_refusal', 'measure_mutation_refused_known_refusal', [], [('measure_coverage', 'measure_mutation')]),
    ('run_tests_cached_refused', 'run_tests_cached_refused', [], []),
]


_COVERAGE_ONLY_KEYS = {
    "stall_hint_both_goals_met",
    "stall_hint_both_stalled_return",
    "measure_coverage_no_test_file",
    "measure_coverage_no_prior_result",
    "measure_coverage_red_when_measured",
    "measure_coverage_unmeasured",
    "measure_coverage_counts_inconsistent",
    "measure_coverage_timeout_when_measured",
    "measure_coverage_timeout_when_measured_named",
    "tests_no_test_file",
    "tests_unsupported_language",
}

_MUTATION_ONLY_KEYS = {
    "measure_mutation_engine_not_implemented",
    "measure_mutation_engine_reason",
    "measure_mutation_red_when_measured",
    "measure_mutation_red_under_coverage",
    "measure_mutation_unmeasured_under_coverage",
    "measure_mutation_counts_inconsistent",
    "measure_mutation_timeout_under_coverage",
    "measure_mutation_timeout_under_coverage_named",
    "measure_mutation_red_when_measured_named",
    "measure_mutation_time_limit",
    "measure_mutation_zero_mutants",
    "measure_mutation_inconclusive",
    "tests_timeout_inherited",
}


def _raw_tool_messages(agent):
    import json as _json
    from importlib.resources import files as _files

    raw = _json.loads(
        (_files("testuary.prompts") / f"{agent}.json").read_text("utf-8"))
    return raw["tool_messages"]


def _joined_value(value):
    return value if isinstance(value, str) else "\n".join(value)


def test_paired_tool_messages_are_substitution_images():
    cov = _raw_tool_messages("test_coverage")
    mut = _raw_tool_messages("test_mutation")
    for cov_key, mut_key, slots, nouns in _MESSAGE_PAIRS:
        text = _joined_value(cov[cov_key])
        for find, repl in slots + nouns:
            assert find in text, (
                f"{cov_key}: stale substitution row {find[:50]!r}"
            )
            text = text.replace(find, repl)
        assert text == _joined_value(mut[mut_key]), (
            f"{mut_key} is not the substitution image of {cov_key}"
        )


def test_tool_message_key_sets_are_an_exact_partition():
    cov = set(_raw_tool_messages("test_coverage"))
    mut = set(_raw_tool_messages("test_mutation"))
    paired_cov = {c for c, _m, _s, _n in _MESSAGE_PAIRS}
    paired_mut = {m for _c, m, _s, _n in _MESSAGE_PAIRS}
    assert cov == paired_cov | _COVERAGE_ONLY_KEYS
    assert mut == paired_mut | _MUTATION_ONLY_KEYS


def test_strict_green_text_quotes_without_escapes():
    """The STRICT GREEN text shows the decorator as it is written, without
    literal backslash escapes."""
    for agent in ("test_coverage", "test_mutation"):
        text = _joined_value(_raw_tool_messages(agent)["tests_strict_green_unsound"])
        assert '``@pytest.mark.filterwarnings("ignore::RuntimeWarning")``' in text
        assert "\\" not in text


@pytest.mark.parametrize("agent", ["test_coverage", "test_mutation"])
def test_the_texts_describe_the_gated_cycle(agent):
    """No text claims a measurement runs the suite first: the measurement
    is offered only after a passing run_tests. The restore note does not
    offer a measurement, which needs a passing run of the restored file."""
    messages = resolve_agent_prompts(agent, "python")["tool_messages"]
    system = resolve_agent_prompts(agent, "python")["system_prompt"]
    for key, text in [("system_prompt", system), *messages.items()]:
        assert "prelude" not in text and "auto-run" not in text, key
    assert "Writes bind" not in system
    assert "Re-measure" not in messages["tests_restore_note"]
    assert "or different edit, then run_tests" in messages["tests_timeout_restored"]


def test_strict_green_text_recommends_only_what_is_installed():
    """pytest-asyncio is not among the test libraries: the fix the text
    recommends runs the coroutine with asyncio.run in a plain test."""
    for agent in ("test_coverage", "test_mutation"):
        text = _joined_value(_raw_tool_messages(agent)["tests_strict_green_unsound"])
        assert "pytest.mark.asyncio" not in text
        assert "asyncio.run(" in text


def test_stands_alone_advice_covers_import_time_code():
    """Reaching an ``if TYPE_CHECKING:`` block or an ``except ImportError:``
    fallback needs a reload. The note's advice, Python's vocabulary entry,
    says how to reload and restore, so it can be followed for that code
    too."""
    for agent in ("test_coverage", "test_mutation"):
        messages = resolve_agent_prompts(agent, "python")["tool_messages"]
        text = " ".join(messages["tests_stands_alone_note"].split())
        assert "if TYPE_CHECKING:" in text and "except ImportError:" in text, agent
        assert "saves ``dict(module.__dict__)``" in text, agent
        assert "``module.__dict__.update(saved)``" in text, agent


@pytest.mark.parametrize("agent_name", ["test_coverage", "test_mutation"])
def test_the_note_advice_and_the_load_errors_are_the_languages_own(agent_name: str) -> None:
    """The note on tests that leave shared state changed, and a suite that
    does not load, get the language's own advice: Java's names JUnit's
    lifecycle and javac's messages, never pytest's or Python's. No message
    treats a passing run as red for how the file runs on its own."""
    for language in ("python", "java"):
        slots = get_backend(language).prompt_fragments()
        out = resolve_agent_prompts(agent_name, language)["tool_messages"]
        assert slots["lang_stands_alone_advice"] in out["tests_stands_alone_note"]
        assert slots["lang_load_error_patterns"] in out["tests_collection_error"]
        assert "tests_stands_alone_red" not in out
    java = resolve_agent_prompts(agent_name, "java")["tool_messages"]
    for key in ("tests_stands_alone_note", "tests_collection_error"):
        for term in ("monkeypatch", "__dict__", "ImportError", "IndentationError", "dask", "np."):
            assert term not in java[key], (key, term)


def test_shared_blocks_appear_in_the_same_order_in_both_agents():
    """The shared framework blocks sit at the same relative positions in
    both system prompts: one section order."""
    import json as _json
    from importlib.resources import files as _files

    orders = {}
    for agent in ("test_coverage", "test_mutation"):
        raw = _json.loads(
            (_files("testuary.prompts") / f"{agent}.json").read_text("utf-8"))
        sys_text = "\n".join(raw["system_prompt"])
        nouns = _PARITY_NOUNS[agent]
        positions = []
        for i, block in enumerate(_SHARED_BLOCKS):
            found = sys_text.find(block.format(**nouns))
            assert found >= 0, f"{agent}: missing shared block {i}"
            positions.append((found, i))
        orders[agent] = [i for _pos, i in sorted(positions)]
    assert orders["test_coverage"] == orders["test_mutation"]


_REAL_LANGUAGE_LEVEL = prompts_module._language_level


@pytest.mark.parametrize("level,expected", [("Java 8", "Java 8"), (None, None)])
def test_the_prompts_name_the_language_level_the_tests_are_written_in(
    level, expected, tmp_path, monkeypatch,
) -> None:
    """Java's tests compile for the project's release, which can be older
    than the JDK that runs them. A prompt that names only the JDK ("Java
    21") leads to var and Map.of in tests for a Java 8 target. Every prompt
    names the adapter's language level, and the runtime line where the
    adapter reports none."""
    from types import SimpleNamespace

    from langadapter.context import ProjectContext
    from langadapter.java.adapter import JavaAdapter

    from testuary import languages
    from testuary.languages.generic import GenericBackend

    monkeypatch.setattr(prompts_module, "_language_level", _REAL_LANGUAGE_LEVEL)
    facts = SimpleNamespace(test_framework="JUnit 5", timeout_marker="",
                            runtime_version="21.0.12.1", language_level=level)

    class FixedFacts(JavaAdapter):
        def language_facts(self):
            return facts

    project = FixedFacts(ProjectContext(str(tmp_path)))

    class FixedJava(GenericBackend):
        def project_backend(self, source_dir=None):
            return project

    java = FixedJava("java", (".java",))
    monkeypatch.setitem(languages._ADAPTERS, "java", java)
    expected = expected or java.prompt_fragments()["lang_runtime_version"]
    prompts = {agent: resolve_agent_prompts(agent, "java") for agent in LANG_AGENTS}
    starter = prompts["test_initial"]["user_prompt_template"].splitlines()
    assert f"Write the tests in: {expected}" in starter
    for agent in ("test_coverage", "test_mutation"):
        text = prompts[agent]["system_prompt"] + "\n" + prompts[agent].get("user_prompt_template", "")
        assert f"**Write the tests in:** {expected}" in text.splitlines()
    for prompt in prompts.values():
        assert "${language_level}" not in json.dumps(prompt)


def test_the_coverage_texts_do_not_ask_for_branches_a_target_may_lack():
    """The system prompt and the user template are shared by targets with
    and without branches, and the goal metric may be line, branch, min or
    max: they name coverage and leave the metric to the goal clause."""
    raw = _raw_tool_messages("test_coverage")
    prompts = resolve_agent_prompts("test_coverage", "python")
    assert "line AND branch" not in prompts["system_prompt"]
    assert "line and branch coverage reach" not in prompts["user_prompt_template"]
    assert "uncovered branches" not in _joined_value(raw["validator_stagnation_not_exhausted"])


def test_the_starter_prompt_introduces_the_import_hint_as_a_statement_of_fact():
    """The hint can be "none; the test declares the target's package ...",
    which "Use this import statement:" would contradict. The line names
    what follows instead."""
    template = resolve_agent_prompts("test_initial", "python")["user_prompt_template"]
    assert "Import of the target: $import_hint" in template.splitlines()
    assert "Use this import statement" not in template


@pytest.mark.parametrize("agent", ["test_coverage", "test_mutation"])
def test_both_loops_name_the_edited_file_the_working_file(agent):
    """One term for the file the loop edits, in both loops: no prompt text
    or tool message calls it the test file."""
    prompts = resolve_agent_prompts(agent, "python")
    texts = [prompts["system_prompt"], prompts["user_prompt_template"],
             *prompts["tool_messages"].values()]
    for text in texts:
        stripped = text.replace("working test file's", "")
        assert "test file" not in stripped and "Test file" not in stripped, text[:80]
    assert ("- ``read_tests()``: your working file: the suite you start from\n"
            "  plus everything you have changed.") in prompts["system_prompt"]


def test_both_loops_verify_with_the_same_step():
    """Step 4 asks both loops to iterate until green and to fix a failure
    introduced in a test the working file started with: both apply to both
    phases."""
    step = ("4. Verify with ``run_tests`` (it binds once edits are pending).\n"
            "   Iterate until the suite is green. A failure you introduced in a\n"
            "   pre-seeded test (a shadowed helper, a broken import) is a\n"
            "   regression; fix it.")
    for agent in ("test_coverage", "test_mutation"):
        assert step in resolve_agent_prompts(agent, "python")["system_prompt"], agent


def test_both_loops_open_with_the_same_goal_wording():
    """The opening names the edited file and the goal in the same words in
    both loops: the goal is the one the task states."""
    for agent, metric in (("test_coverage", "coverage"), ("test_mutation", "mutation score")):
        system = resolve_agent_prompts(agent, "python")["system_prompt"]
        assert ("one target source file is already on disk in your working file;\n"
                f"your goal is to raise its {metric} to the goal the task states by") in system, agent
        assert "configured goal" not in system.split("##")[0]


def test_both_loops_name_the_suite_a_timeout_restores_alike():
    """Both loops restore the best-green snapshot after a timeout (the
    best-scoring suite, which ran in time), and say so in the same words."""
    for agent in ("test_coverage", "test_mutation"):
        text = _joined_value(_raw_tool_messages(agent)["tests_timeout_restored"])
        assert "Reverted the working file to the best-scoring suite that\nran in time" in text


def test_both_openings_keep_the_inherited_suite_as_a_candidate():
    """Both phases start from a working file that already holds tests and are
    consolidated with the unchanged inherited suite: both openings say so in
    one wording."""
    sentence = ("The inherited suite remains a separate phase-end candidate, but "
                "edits to the working file are not assumed to preserve every prior "
                "test or measurement.")
    for agent in ("test_coverage", "test_mutation"):
        opening = resolve_agent_prompts(agent, "python")["system_prompt"].split("## ")[0]
        assert sentence in " ".join(opening.split()), agent
