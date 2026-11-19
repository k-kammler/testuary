"""The edit tool of the Java coverage and mutation agents on the real Java
backend, through each agent's graph with a scripted model, run in a process of
its own by ``test_java_edit_tool.py``. Every check of an edit runs for real:
the structure parse, the Java target parser, the duplicate-definition and the
validity checks. The report records each case's tool results, the file after
it, the writes and how often the target parser ran.

Usage: java_edit_tool.py <workspace dir> <JDK home> <mvn launcher>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WORK = Path(sys.argv[1]).resolve()
JDK, MAVEN = sys.argv[2], sys.argv[3]
MAIN = """package demo;

public final class Calc {
    private Calc() {
    }

    public static int sign(int x) {
        if (x > 0) {
            return 1;
        }
        if (x < 0) {
            return -1;
        }
        return 0;
    }
}
"""
TESTS = """package demo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CalcTest {
    @Test
    void positive() {
        assertEquals(1, Calc.sign(5));
    }

    @Test
    void negative() {
        assertEquals(-1, Calc.sign(-5));
    }
}
"""
NEG = "        assertEquals(-1, Calc.sign(-5));\n    }\n"
ZERO = "\n    @Test\n    void zero() {\n        assertEquals(0, Calc.sign(0));\n    }\n"

CASES = {
    "single": [{"old_text": NEG, "new_text": NEG + ZERO}],
    "list_new_version_then_delete": [{"edits": [
        {"old_text": NEG, "new_text": NEG + "\n    @Test\n    void positive() {\n"
                                          "        assertEquals(1, Calc.sign(7));\n    }\n"},
        {"old_text": "    @Test\n    void positive() {\n        assertEquals(1, Calc.sign(5));\n    }\n\n",
         "new_text": ""}]}],
    "list_header_then_body": [{"edits": [
        {"old_text": NEG, "new_text": NEG + "\n    @Test\n    void zero() {\n    }\n"},
        {"old_text": "    void zero() {\n", "new_text": "    void zero() {\n"
                                                    "        assertEquals(0, Calc.sign(0));\n"}]}],
    "list_rejects_a_broken_result": [{"edits": [
        {"old_text": NEG, "new_text": NEG + ZERO},
        {"old_text": "        assertEquals(1, Calc.sign(5));\n",
         "new_text": "        assertEquals(1, Calc.sign(5);\n"}]}],
    "list_rejects_a_missed_item": [{"edits": [
        {"old_text": NEG, "new_text": NEG + ZERO},
        {"old_text": "        assertEquals(7, Calc.sign(7));\n", "new_text": ""}]}],
    "single_rejects_a_duplicate_method": [{"old_text": NEG, "new_text": NEG +
        "\n    @Test\n    void negative() {\n        assertEquals(-1, Calc.sign(-9));\n    }\n"}],
    "two_calls_apply_in_order": [{"old_text": NEG, "new_text": NEG + ZERO},
                                 {"old_text": "        assertEquals(0, Calc.sign(0));\n",
                                  "new_text": "        assertEquals(0, Calc.sign(0));\n"
                                              "        assertEquals(0, Calc.sign(-0));\n"}],
}


def respond(agent, calls, lg, Fake):
    """One scripted model reply holding ``calls``: the tool results in order."""
    from langchain_core.messages import AIMessage, HumanMessage

    message = AIMessage(content="", tool_calls=[
        {"name": "edit_tests", "id": f"e{i}", "args": args} for i, args in enumerate(calls)])
    tracker = agent.session.usage_tracker_ref = lg.UsageTracker(request_limit=1)
    graph = agent.build_graph(model=Fake(responses=[message]), usage_tracker=tracker)
    results = []
    try:
        for update in graph.stream({"messages": [HumanMessage(content="x")]}, stream_mode="updates"):
            for node in update.values():
                results += [str(m.content) for m in (node or {}).get("messages", [])
                            if getattr(m, "type", "") == "tool"]
    except lg.UsageLimitExceeded:
        pass
    return results


def main() -> int:
    # The helpers the test files share are in tests/, as pytest finds them.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from java_fixtures import POM

    project = WORK / "projects" / "calc"
    (project / "src/main/java/demo").mkdir(parents=True, exist_ok=True)
    target = project / "src/main/java/demo/Calc.java"
    target.write_text(MAIN, encoding="utf-8")
    (project / "pom.xml").write_text(POM, encoding="utf-8")
    settings_file = WORK / "preset.ini"
    settings_file.write_text(f"[project]\njava_home = {JDK}\nmaven = {MAVEN}\n", encoding="utf-8")
    for name in list(os.environ):
        if name.startswith(("TESTUARY_", "LANGADAPTER_")):
            del os.environ[name]
    # The run's settings file: the run below resolves it.
    os.environ["TESTUARY_CONFIG_FILE"] = str(settings_file)
    os.chdir(WORK)

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    from testuary.agent import _langgraph as lg
    from testuary import config
    from testuary import languages
    from testuary import prompts
    from testuary.agent import coverage_agent as tc
    from testuary.agent import mutation_agent as tm
    from testuary.settings import resolve_settings

    class Fake(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

    backend = languages.get_backend("java")
    parses = []
    validate = backend.validate_test_source

    def counted(source, source_dir):
        parses.append(source)
        return validate(source, source_dir)

    backend.validate_test_source = counted

    def agent_for(kind, tests_dir):
        # As the phase entry points do: the backend names and prepares the file.
        path = backend.target_test_path(str(tests_dir), str(target))
        backend.setup_test_env(str(tests_dir), str(project), str(target))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TESTS, encoding="utf-8")
        # As target_phases does: the phase's log starts before its agent is built.
        phase = tc if kind == "coverage" else tm
        phase.init_logging("calc", str(WORK / "logs" / tests_dir.name), f"{kind}.jsonl")
        if kind == "coverage":
            deps = tc.CoverageTestDeps(
                target_file_path=str(target), target_file_content=MAIN, source_dir=str(project),
                test_output_dir=str(tests_dir), language="java", key_functions=["sign"])
            agent = tc.CoverageAgent(prompts=prompts.resolve_agent_prompts("test_coverage", "java"),
                                     deps=deps, session=tc.TestCoverageSession(),
                                     logger=phase._log.logger)
            agent.session.active_backend = backend
        else:
            deps = tm.MutationTestDeps(
                target_file_path=str(target), target_file_content=MAIN, source_dir=str(project),
                test_output_dir=str(tests_dir), language="java", mutation_test_file_name=path.name,
                coverage_baseline_pct=80.0)
            agent = tm.MutationAgent(prompts=prompts.resolve_agent_prompts("test_mutation", "java"),
                                     deps=deps, session=tm.TestMutationSession(),
                                     logger=phase._log.logger)
        agent.session.has_read = True
        agent.session.initial_measured = True  # past the start's measurement
        assert agent.target_path() == path, (agent.target_path(), path)
        return agent, path

    report = {"original": TESTS}
    # Configuration is read inside a run only: the backend's setup and the
    # agents read the settings file's JDK and Maven. A run selects a model,
    # as every run does. The scripted model answers and none is called.
    run = resolve_settings(base_dir=WORK, models={}, project_settings=False,
                           overrides={"api": {"provider": "ollama", "model": "test-model"}})
    with config.use_run_settings(run):
        for kind in ("coverage", "mutation"):
            for case, calls in CASES.items():
                agent, path = agent_for(kind, WORK / f"tests_{kind}_{case}")
                parses.clear()
                results = respond(agent, calls, lg, Fake)
                report[f"{kind}/{case}"] = {"results": results,
                                            "file": path.read_text(encoding="utf-8"),
                                            "writes": agent.session.writes_done,
                                            "target_parses": len(parses)}
    (WORK / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
