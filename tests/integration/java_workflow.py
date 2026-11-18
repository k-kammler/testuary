"""Testuary's pipeline on a small Maven project with a scripted model, run
in a process of its own by ``test_java_workflow.py``.

The starter suites and the agents' turns are fixed here. Every measurement
is real (Maven's resolution, javac, the JUnit Platform's console launcher,
JaCoCo, PIT). The ``full`` variant answers three different starter suites,
whose merge resolves a collision and keeps a setup method on its own tests.
``weak`` answers the same starter suite three times, so both agents run: the
coverage agent adds clamp tests, the mutation agent the case that kills the
surviving conditional boundaries.

Usage: java_workflow.py <workspace dir> <JDK home> <mvn launcher> [weak]
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

WORK = Path(sys.argv[1]).resolve()
JDK, MAVEN = sys.argv[2], sys.argv[3]
#: "weak": every starter suite is the first one, so the goals stay unmet and
#: both agents run.
VARIANT = sys.argv[4] if len(sys.argv) > 4 else "full"

STARTERS = [
    # Green tests and one wrong expectation the cleanup strips.
    """\
package demo;

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

    @Test
    void zeroIsPositive() {
        assertEquals(1, Calc.sign(0));
    }
}
""",
    # A setup method and the clamp. Collides with the first starter suite on
    # positive().
    """\
package demo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class CalcTest {
    private int high;

    @BeforeEach
    void setUp() {
        high = 10;
    }

    @Test
    void positive() {
        assertEquals(1, Calc.sign(42));
    }

    @Test
    void clampsHigh() {
        assertEquals(high, Calc.clamp(50, 0, high));
    }

    @Test
    void clampsLow() {
        assertEquals(0, Calc.clamp(-50, 0, high));
    }
}
""",
    # Zero and the boundary.
    """\
package demo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class CalcTest {
    @Test
    void zero() {
        assertEquals(0, Calc.sign(0));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2})
    void smallPositives(int value) {
        assertEquals(1, Calc.sign(value));
    }

    @Test
    void clampsInside() {
        assertEquals(5, Calc.clamp(5, 0, 10));
    }
}
""",
]

ANCHOR = "    @Test\n    void positive() {"
CLAMPS = ("    @Test\n    void clamps() {\n"
          "        assertEquals(10, Calc.clamp(50, 0, 10));\n"
          "        assertEquals(0, Calc.clamp(-50, 0, 10));\n"
          "        assertEquals(5, Calc.clamp(5, 0, 10));\n    }\n\n")
ZERO = ("    @Test\n    void zero() {\n"
        "        assertEquals(0, Calc.sign(0));\n    }\n\n")
#: Each agent's turns: read the tests, measure them (the loop's start),
#: one edit, run them, measure, finish.
SCRIPTS = {
    "coverage": [
        ("read_tests", {}),
        ("measure_coverage", {}),
        ("edit_tests", {"old_text": ANCHOR, "new_text": CLAMPS + ANCHOR}),
        ("run_tests", {}),
        ("measure_coverage", {}),
        ("emit_coverage_test_result",
         {"test_file_name": "CalcTest.java",
          "explanation": "Clamp above, below and inside its bounds.",
          "targeted_functions": ["clamp"]}),
    ],
    "mutation": [
        ("read_tests", {}),
        ("measure_mutation", {}),
        ("edit_tests", {"old_text": ANCHOR, "new_text": ZERO + ANCHOR}),
        ("run_tests", {}),
        ("measure_mutation", {}),
        ("emit_mutation_test_result",
         {"mutation_test_file_name": "CalcTest.java",
          "explanation": "sign(0) tells both conditional boundaries apart.",
          "targeted_mutants": ["Calc.java:8:ConditionalsBoundaryMutator",
                               "Calc.java:11:ConditionalsBoundaryMutator"]}),
    ],
}

REQUESTS: List[dict] = []


class ScriptedSource:
    def generate_suite(self, messages, *, temperature, seed, timeout):
        index = len([r for r in REQUESTS if r["kind"] == "seed"])
        REQUESTS.append({"kind": "seed", "seed": seed,
                         "prompt_chars": sum(len(m.get("content") or "") for m in messages)})
        chosen = STARTERS[0] if VARIANT in ("weak", "cancel", "resume") else STARTERS[index % len(STARTERS)]
        text = "```java\n" + chosen + "```\n"
        return text, {"input_tokens": 0, "output_tokens": 0}

    def agent_chat_model(self, phase):
        return ScriptedChat(phase=phase)


class ScriptedChat(BaseChatModel):
    phase: str
    _turn: int = PrivateAttr(default=0)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "java-j8-scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        with self._lock:
            self._turn += 1
            turn = self._turn
        REQUESTS.append({"kind": "agent", "phase": self.phase, "turn": turn,
                         "last": type(messages[-1]).__name__,
                         "last_text": str(getattr(messages[-1], "content", ""))[:300]})
        script = SCRIPTS[self.phase]
        name, args = script[min(turn, len(script)) - 1]
        message = AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"call_{self.phase}_{turn}"}])
        return ChatResult(generations=[ChatGeneration(message=message)])


def _stop_during_the_agents_measurement() -> None:
    """The ``cancel`` variant: a Ctrl+C two seconds into the first mutation
    measurement a tool runs on its own thread, as a terminal's Ctrl+C
    reaches the main thread. The cancels and the stop's time are recorded."""
    import _thread

    from testuary.languages import mutation

    real_run, real_cancel = mutation.MutationSession.run, mutation.MutationSession.cancel

    def run(session, *args, **kwargs):
        if (threading.current_thread() is not threading.main_thread()
                and not any(r["kind"] == "stop" for r in REQUESTS)):
            def stop():
                REQUESTS.append({"kind": "stop", "at": time.monotonic()})
                _thread.interrupt_main()

            threading.Timer(2.0, stop).start()
        return real_run(session, *args, **kwargs)

    def cancel(session):
        REQUESTS.append({"kind": "cancel", "at": time.monotonic()})
        return real_cancel(session)

    mutation.MutationSession.run = run
    mutation.MutationSession.cancel = cancel


def main() -> int:
    # The helpers the test files share are in tests/, as pytest finds them.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from java_fixtures import MAIN, POM

    project = WORK / "projects" / "calc"
    (project / "src/main/java/demo").mkdir(parents=True, exist_ok=True)
    (project / "src/main/java/demo/Calc.java").write_text(MAIN, encoding="utf-8")
    (project / "pom.xml").write_text(POM, encoding="utf-8")
    settings_file = WORK / "preset.ini"
    settings_file.write_text(f"""\
[project]
java_home = {JDK}
maven = {MAVEN}

[starter]
suite_count = 3
concurrent_requests = 1
attempts_per_suite = 1
refill_failed_suites = false

[coverage]
max_requests_per_phase = 8
goal = 80
goal_metric = line

[mutation]
max_requests_per_phase = 8
goal = 100
measurement_timeout_seconds = 900
""", encoding="utf-8")
    (WORK / "models.ini").write_text("""\
[scripted]
provider = openai
model = scripted
api_url = http://127.0.0.1:9
max_output_tokens = 1000
price_input_cache_hit = 0
price_input_cache_miss = 0
price_output = 0
""", encoding="utf-8")
    for name in list(os.environ):
        if name.startswith(("TESTUARY_", "LANGADAPTER_")):
            del os.environ[name]
    os.environ["TESTUARY_CONFIG_FILE"] = str(settings_file)
    # The user directory holds the model catalogue. The run starts in the
    # project's directory, and its results go outside it.
    os.environ["TESTUARY_HOME"] = str(WORK)
    os.chdir(project)
    if VARIANT == "cancel":
        _stop_during_the_agents_measurement()
    from testuary import models as _model_source
    _model_source.install(ScriptedSource())
    from testuary import __main__ as cli
    sys.argv = ["testuary", "--config", str(settings_file), "--model", "scripted",
                "--project-name", "calc", "--source-files",
                "src/main/java/demo/Calc.java",
                "--output-dir", str(WORK / "out")] + (["--resume"] if VARIANT == "resume" else [])
    started = time.perf_counter()
    try:
        status = cli.main()
    finally:
        REQUESTS.append({"kind": "end", "at": time.monotonic()})
        (WORK / f"requests-{VARIANT}.json").write_text(json.dumps(REQUESTS, indent=2),
                                                      encoding="utf-8")
        (WORK / "requests.json").write_text(json.dumps(REQUESTS, indent=2),
                                            encoding="utf-8")
    print(f"STATUS {status} after {time.perf_counter() - started:.0f} s; "
          f"{len(REQUESTS)} scripted model requests")
    return int(status or 0)


if __name__ == "__main__":
    sys.exit(main())
