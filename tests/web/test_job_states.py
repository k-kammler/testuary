"""A job's state has one word: the badge, the label beside it and the
header's running job say the same, and a stopped run's diagram shows no
current step. The page's functions run under Node."""
from __future__ import annotations

import json

from conftest import SOURCE, page_value
import re
import shutil
import subprocess
import pytest

APP = SOURCE / "testuary" / "web" / "static" / "app.js"


def test_each_job_state_has_one_word():
    """Each state's badge says its label's word."""
    statuses = ["running", "stopping", "finished", "cancelled", "stopped", "ended"]
    words = page_value(["JOB_STATES", "incompleteRun", "stateWords"],
                  "%s.map((status) => stateWords({ status }))" % json.dumps(statuses))
    assert [badge for _, badge, _ in words] == [
        "RUNNING", "STOPPING", "FINISHED", "STOPPED", "FORCE-STOPPED", "FAILED"]
    assert all(badge == label.upper() for _, badge, label in words)
    incomplete = page_value(["JOB_STATES", "incompleteRun", "stateWords"],
                       'stateWords({ status: "finished", outcome: { incomplete: true } })')
    assert incomplete == ["incomplete", "MINIMUMS NOT MET", "Finished without a suite that meets the minimums"]
    # The header names the running job's state, Stopping included.
    assert "jobState(state.active).badge" in APP.read_text(encoding="utf-8")


def test_an_ended_jobs_badge_says_testuarys_word():
    """The badge of an ended job says Testuary's word for its status, red
    only for a failure: a refused setting and a problem the check found are
    no failed run."""
    jobs = [{"status": "ended", "word": "refused", "failure": False},
            {"status": "ended", "word": "problem", "failure": False},
            {"status": "ended", "word": "failed", "failure": True},
            {"status": "finished", "word": "ready", "failure": False},
            {"status": "cancelled", "word": "interrupted", "failure": False}]
    assert page_value(["JOB_STATES", "incompleteRun", "stateWords"],
                      "%s.map(stateWords)" % json.dumps(jobs)) == [
        ["incomplete", "REFUSED", "Refused"], ["incomplete", "PROBLEM", "Problem"],
        ["failed", "FAILED", "Failed"], ["finished", "READY", "Ready"],
        ["cancelled", "STOPPED", "Stopped"]]


def test_an_ended_jobs_empty_output_is_no_longer_awaited():
    """An ended job's empty output reads "No output.", not the words for a
    job that still runs."""
    assert page_value(["outputText"], 'outputText([], false)') == "No output yet."
    assert page_value(["outputText"], 'outputText([], true)') == "No output."
    assert page_value(["outputText"], 'outputText(["a", "b"], true)') == "a\nb"


def test_a_phase_whose_steps_stopped_is_shown_stopped():
    """A phase's own event comes after its step's, so a phase with a stopped
    step and none under way shows stopped at once."""
    phases = [
        {"state": "running", "steps": [{"state": "succeeded"}, {"state": "cancelled"}, {"state": "planned"}]},
        {"state": "running", "steps": [{"state": "cancelled"}, {"state": "running"}]},
        {"state": "running", "steps": [{"state": "succeeded"}, {"state": "running"}]},
        {"state": "succeeded", "steps": [{"state": "succeeded"}]},
        # Stopped before any step began: the steps read not run.
        {"state": "running", "steps": [{"state": "skipped"}, {"state": "skipped"}]},
        # Just started, no step yet: still under way.
        {"state": "running", "steps": [{"state": "planned"}, {"state": "planned"}]},
    ]
    assert page_value(["shownState", "phaseState"],
                      "%s.map((phase) => phaseState(phase, false))" % json.dumps(phases)) == [
        "cancelled", "running", "running", "succeeded", "cancelled", "running"]


def test_the_page_has_a_word_and_a_color_for_refused():
    """Testuary's state for a configuration refusal is "refused": the
    diagram has its word, and the badge, the rows, the bars and the diagram
    show it amber, as the console's yellow, not the red of a failure."""
    assert page_value(["STEP_WORDS"], "STEP_WORDS.refused") == "refused"
    style = (SOURCE / "testuary" / "web" / "static" / "style.css").read_text(encoding="utf-8")
    for rule in (".state.refused", ".tone-refused", "progress.tone-refused", ".diagram .state-refused"):
        assert rule in style, rule


def test_a_stopped_runs_diagram_shows_no_current_step():
    """Work still under way when the job ended shows as stopped. Finished
    and upcoming work stay."""
    states = ["running", "waiting", "retrying", "succeeded", "planned", "failed"]
    assert page_value(["shownState"], "%s.map((state) => shownState(state, true))" % json.dumps(states)) == [
        "cancelled", "cancelled", "cancelled", "succeeded", "planned", "failed"]
    assert page_value(["shownState"], "%s.map((state) => shownState(state, false))" % json.dumps(states)) == states
    source = APP.read_text(encoding="utf-8")
    assert "runDiagram(view.diagram, this.ended, view.clock_s, this.kind, this.nested, this.heading)" in source
    assert "progress.ended = true;" in source


def test_a_phase_draws_its_start_its_cycle_and_its_end():
    """A phase's steps in their three groups: the start runs once, the cycle
    repeats and the phase end runs once after it. A plan without an end
    has none, and each step and the phase show their counters."""
    steps = [{"step": name} for name in ("read", "revise", "validate", "consolidation")]
    groups = "(({ start, cycle, end }) => [start, cycle, end].map((group) => group.map((step) => step.step)))"
    assert page_value(["phaseGroups"], f"{groups}(phaseGroups({json.dumps({'steps': steps, 'cycle': 1, 'end': 3})}))") == [
        ["read"], ["revise", "validate"], ["consolidation"]]
    assert page_value(["phaseGroups"], f"{groups}(phaseGroups({json.dumps({'steps': steps, 'cycle': 1})}))") == [
        ["read"], ["revise", "validate", "consolidation"], []]
    source = APP.read_text(encoding="utf-8")
    assert 'class: "step-count", text: item.count' in source
    assert 'class: "small phase-count", text: phase.count' in source


def test_finishing_is_the_fourth_phase_with_its_purpose_while_it_runs():
    """Finishing, the fourth phase, has a box of its own after the plan's
    three, without steps. While it runs, the box says what it does, in
    Testuary's words, beside a spinner. Before and after it, and once the
    job has ended, it does not. The Overall bar's own words say what it
    counts, so no note under the diagram repeats them."""
    finishing = {"label": "Finishing", "purpose": "measuring the delivered suite and writing the results"}
    cases = [({**finishing, "state": state}, ended) for state, ended in (
        ("running", False), ("planned", False), ("succeeded", False), ("running", True), ("reused", False))]
    assert page_value(["finishingText"], "%s.map(([finishing, ended]) => finishingText(finishing, ended))"
                      % json.dumps(cases)) == [
        "measuring the delivered suite and writing the results", None, None, None, None]
    source = APP.read_text(encoding="utf-8")
    assert 'phases.push(el("li", { class: `phase state-${finishing.state}` }' in source
    assert 'class: "finishing small"' in source and "overallNote" not in source


STYLE = SOURCE / "testuary" / "web" / "static" / "style.css"


def _heading(tone):
    """causeHeading(tone) as the page runs it, evaluated by Node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = APP.read_text(encoding="utf-8")
    function = re.search(r"^function causeHeading\(tone, ended\) \{\n.*?^\}\n", source, re.S | re.M)
    assert function, "app.js defines causeHeading"
    script = (function.group(0)
              + f"process.stdout.write(JSON.stringify(causeHeading({json.dumps(tone)})));")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_panel_names_a_succeeded_runs_cause_as_a_note():
    assert _heading("succeeded") == ["Note: ", "panel cause-note"]
    assert _heading("failed") == ["Why the run failed: ", "panel cause"]
    assert _heading("incomplete") == ["What went wrong: ", "panel cause"]
    assert _heading(None) == ["What went wrong: ", "panel cause"]


def test_a_refused_or_problem_job_says_its_word_in_the_body():
    """The body of a refused or problem job says the badge's word: the cause
    heading, the status bar's style and the check report's result."""
    from conftest import page_value

    assert page_value(["causeHeading"], 'causeHeading("failed", { word: "refused", failure: false })') == [
        "Refused: ", "panel cause-warning"]
    assert page_value(["causeHeading"], 'causeHeading("failed", { word: "failed", failure: true })') == [
        "Why the run failed: ", "panel cause"]
    source = APP.read_text(encoding="utf-8")
    assert "panel status-panel ${stateWords(current)[0]}" in source
    assert "job ? jobState(job).badge : stateBadge(report.status)" in source
    assert ".status-panel.failed" in STYLE.read_text(encoding="utf-8")


def test_the_run_view_takes_its_cause_heading_from_the_runs_tone():
    source = APP.read_text(encoding="utf-8")
    assert "causeHeading(view && view.layers[0] ? view.layers[0].tone : null, this.endedWord)" in source
    assert '"What went wrong: "' in source and source.count('"What went wrong: "') == 1
    # The note is neutral: not the cause panel's failure color.
    note = re.search(r"^\.cause-note \{([^}]*)\}", STYLE.read_text(encoding="utf-8"), re.M)
    assert note and "--failed" not in note.group(1)


def test_a_jobs_values_read_as_its_end_of_run_summary_states_them():
    """One file's values stand alone; with several files each line names
    its file. A value the run did not measure says so, a file without
    values says how it ended, and a demonstration's values read as demo
    values."""
    one = {"kind": "run", "values": [{"file": "calc/ops.py", "outcome": "complete", "values": [
        {"name": "Line coverage", "text": "100%"}, {"name": "Mutation score", "text": None}]}]}
    assert page_value(["valueLines"], "valueLines(%s)" % json.dumps(one)) == [
        "Line coverage 100%, Mutation score not measured"]
    two = {"kind": "demo", "values": [
        {"file": "a.py", "outcome": "complete", "values": [{"name": "Line coverage", "text": "80%"}]},
        {"file": "b.py", "outcome": "no_complete_suite", "outcome_text": "no complete suite", "values": []}]}
    assert page_value(["valueLines"], "valueLines(%s)" % json.dumps(two)) == [
        "a.py: Line coverage 80% (demo values)", "b.py: no complete suite"]
    assert page_value(["valueLines"], 'valueLines({ kind: "run", values: null })') == []


def test_a_demonstrations_model_reads_as_the_recorded_one():
    """A demonstration asks no model: the model it names is the recording's."""
    assert page_value(["jobModel"], 'jobModel({ kind: "demo", model: "ollama_deepseek" })') == (
        "ollama_deepseek (recorded)")
    assert page_value(["jobModel"], 'jobModel({ kind: "run", model: "fake" })') == "fake"
    assert page_value(["jobModel"], 'jobModel({ kind: "check", model: null })') == ""


def test_an_open_overview_follows_the_jobs():
    """Each look at the server redraws the recent jobs when they changed or
    one runs, and the projects' latest results once a job has ended."""
    running = {"active": {"id": "a"}, "jobs": [{"id": "a", "status": "running", "ended": None}]}
    ended = {"active": None, "jobs": [{"id": "a", "status": "finished", "ended": 5}]}
    changes = page_value(["overviewChanges"], "[%s].map(([before, state]) => overviewChanges(before, state))"
                         % ", ".join(json.dumps(pair) for pair in (
                             [None, ended], [ended, running], [running, running], [running, ended], [ended, ended])))
    assert changes == [{"jobs": True, "projects": False}, {"jobs": True, "projects": False},
                       {"jobs": True, "projects": False}, {"jobs": True, "projects": True},
                       {"jobs": False, "projects": False}]


def test_a_page_parts_nested_lists_are_its_elements():
    """A part's children may come in lists of lists, such as a job row's
    value lines: each item is drawn, and a part left out (null, undefined
    or false) is not."""
    assert page_value(["childList"], 'childList([1, [null, [2, [3, false]], undefined], "four"])') == [
        1, 2, 3, "four"]
    source = APP.read_text(encoding="utf-8")
    assert "for (const child of childList(children))" in source
    assert "box.replaceChildren(...childList(children))" in source


def test_the_settings_page_shows_the_common_settings_first_and_finds_every_one():
    """A section shows its common settings and every setting a file or the
    environment sets; the search matches every word in a setting's label,
    name, purpose or section; a refusal drops the section and key its row
    names."""
    names = ["COMMON_SETTINGS", "shownFirst"]
    assert page_value(names, 'shownFirst("coverage", { name: "goal", source: "default" })') is True
    assert page_value(names, 'shownFirst("coverage", { name: "restore_count", source: "default" })') is False
    assert page_value(names, 'shownFirst("coverage", { name: "restore_count", source: "C:/p/.testuary/project.ini" })') is True
    setting = json.dumps({"label": "Max requests per phase", "name": "max_requests_per_phase",
                          "purpose": "Maximum model requests in this phase."})
    assert page_value(["settingMatches"], f'settingMatches("Coverage phase", {setting}, "coverage REQUESTS")') is True
    assert page_value(["settingMatches"], f'settingMatches("Coverage phase", {setting}, "mutation requests")') is False
    assert page_value(["refusalText"], 'refusalText("[coverage] goal: expected a percentage from 0 to 100", "coverage", { name: "goal" })') == (
        "Not saved: expected a percentage from 0 to 100.")
    assert page_value(["refusalText"], 'refusalText("The file cannot be written.", "coverage", { name: "goal" })') == (
        "Not saved: The file cannot be written.")
    source = APP.read_text(encoding="utf-8")
    # A Save draws its row anew, not the page: the open sections and the
    # page's place stay.
    row = re.search(r"^function settingRow\(.*?^\}\n", source, re.S | re.M).group(0)
    assert "render()" not in row and "await changed(section, setting.name, words)" in row


def test_the_page_script_parses_as_a_whole():
    """The tests above run single functions of the page's script: the
    browser runs all of it, and one error in it leaves every page empty."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to check the page's script")
    done = subprocess.run([node, "--check", str(APP)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_a_settings_row_says_its_value_and_purpose_as_they_are():
    """An unset choice reads "not set", not as its first option, which a Save
    would write. A short purpose is shown whole, such as the sentence that
    zero disables a limit; a long one shows its first sentence and the rest
    on request."""
    assert page_value(["choiceOptions"], 'choiceOptions({ kind: "choice", choices: ["java", "python"], value: null })') == [
        ["", "not set"], ["java", "java"], ["python", "python"]]
    assert page_value(["choiceOptions"], 'choiceOptions({ kind: "choice", choices: ["java", "python"], value: "python" })') == [
        ["java", "java"], ["python", "python"]]
    assert page_value(["choiceOptions"], 'choiceOptions({ kind: "boolean", choices: [], value: "true" })') == [
        ["true", "true"], ["false", "false"]]
    assert page_value(["purposeParts"], 'purposeParts("Phase wall-clock limit. Zero or unlimited disables this limit.")') == [
        "Phase wall-clock limit. Zero or unlimited disables this limit.", ""]
    long = "First sentence here. " + "More words follow in a second sentence. " * 6
    assert page_value(["purposeParts"], "purposeParts(%s)" % json.dumps(long.strip())) == [
        "First sentence here.", long[len("First sentence here. "):].strip()]


def test_the_setup_offers_each_found_path_in_full():
    """An open question lists what the setup found, each path whole, with
    Testuary's default among them, then another path of the person's own."""
    question = {"choices": ["C:/p/.venv/Scripts/python.exe", "C:/Python312/python.exe"],
                "default": "C:/tools/python.exe"}
    assert page_value(["answerChoices"], "answerChoices(%s)" % json.dumps(question)) == [
        "C:/tools/python.exe", "C:/p/.venv/Scripts/python.exe", "C:/Python312/python.exe"]
    question["default"] = "C:/Python312/python.exe"
    assert page_value(["answerChoices"], "answerChoices(%s)" % json.dumps(question)) == question["choices"]


def test_an_address_in_the_setups_text_is_a_link():
    text = "Settings: https://example.org/settings.md. Nothing else."
    assert page_value(["textParts"], "textParts(%s)" % json.dumps(text)) == [
        ["text", "Settings: "], ["link", "https://example.org/settings.md"], ["text", ". Nothing else."]]
    assert page_value(["textParts"], 'textParts("no address")') == [["text", "no address"]]


def test_another_service_starts_with_its_own_provider_name():
    """Changing to another OpenAI-compatible API leaves no earlier service's
    provider name in the field; a model of that kind keeps its own."""
    custom = {"kind": "openai-compatible", "provider": ""}
    ollama = {"kind": "ollama-local", "provider": "ollama"}
    assert page_value(["providerName"], "providerName(%s, 'ollama', 'ollama-local', null)" % json.dumps(custom)) == ""
    assert page_value(["providerName"], "providerName(%s, 'mine', 'openai-compatible', null)" % json.dumps(custom)) == "mine"
    assert page_value(["providerName"], "providerName(%s, 'mine', 'openai-compatible', null)" % json.dumps(ollama)) == "ollama"
    entry = {"kind": "openai-compatible", "provider": "localfake"}
    assert page_value(["providerName"], "providerName(%s, 'ollama', 'ollama-local', %s)"
                      % (json.dumps(custom), json.dumps(entry))) == "localfake"


def test_a_loop_that_ends_at_its_limit_reads_limit_reached_in_amber():
    """A limit the run's settings set is no failure: the diagram's step says
    "limit reached", and the step, the operation row, its bar and its badge
    are amber, as a refusal is, not the red of a failure."""
    assert page_value(["STEP_WORDS"], "STEP_WORDS.limit_reached") == "limit reached"
    style = (SOURCE / "testuary" / "web" / "static" / "style.css").read_text(encoding="utf-8")
    for rule in (".state.limit_reached", ".tone-limit_reached", "progress.tone-limit_reached",
                 ".diagram .state-limit_reached"):
        assert rule in style, rule


def test_an_error_in_a_group_of_choices_has_its_own_line():
    """In a group of boxes or choices, such as the Run page's files, an
    error starts its own line below the hint, as it does under a field."""
    style = (SOURCE / "testuary" / "web" / "static" / "style.css").read_text(encoding="utf-8")
    assert re.search(r"^fieldset > \.error \{ display: block; \}$", style, re.M)


def test_a_runs_suite_says_how_to_use_its_tests():
    """The guide under a run's suite directory: where the test file goes. A
    demonstration's tests belong to its recorded project."""
    plain = page_value(["suiteGuide"], 'suiteGuide({ tests: ["r0/tests/test_ops.py"], setup: [] }, false)')
    assert plain == ("Use the delivered suite: copy the test file to where the project keeps its tests and run it "
                     "with the project's own test command.")
    assert page_value(["suiteGuide"], 'suiteGuide({ tests: ["x"], setup: [] }, true)') == (
        "These tests belong to the demonstration's recorded project, not to a project of yours.")
    assert page_value(["suiteGuide"], 'suiteGuide({ tests: [], setup: [] }, false)') is None


def test_the_guide_names_the_setup_file_a_test_can_need_and_offers_its_copy():
    """The setup file beside the delivered test file (Python's conftest.py)
    is named with it."""
    suite = '{ tests: ["r0/tests/test_ops.py"], setup: ["r0/tests/conftest.py"] }'
    names = ["suiteGuide"]
    guide = page_value(names, "suiteGuide(%s, false)" % suite)
    assert guide.endswith("Its tests ran with conftest.py beside them and can need it: copy it into the same "
                          "directory too. Where that directory already has a conftest.py of its own, the two have to "
                          "be joined by hand.")
    # Several setup files are named together, and the words follow.
    several = '{ tests: ["r0/T.x"], setup: ["r0/a.cfg", "r0/b.cfg"] }'
    guide = page_value(names, "suiteGuide(%s, false)" % several)
    assert "ran with a.cfg and b.cfg beside them" in guide and "can need them: copy them" in guide
    assert "a file of the same name" in guide


def test_a_models_badge_says_set_up_and_its_test_says_what_the_service_answered():
    """Set up is what the catalogue tells; whether the service answers only
    a connection test tells, in its own words, as a refusal is amber and a
    missing answer red."""
    assert page_value(["modelBadge"], "[modelBadge({ usable: true }), modelBadge({ usable: false })]") == [
        ["succeeded", "SET UP"], ["incomplete", "NO KEY"]]
    assert page_value(["testTone"], "[testTone({ reached: true, listed: true }), testTone({ reached: true, listed: false }),"
                      " testTone({ reached: false, listed: null })]") == ["saved", "warn", "error"]


def test_a_cause_that_the_model_service_gave_no_answer_leads_to_its_connection_test():
    """A cause the server marks as the model service's leads to the Models
    page, where the model's connection can be tested. The page reads the
    mark, never the sentence's words."""
    assert page_value(["serviceCause"], 'serviceCause({ cause: "x", cause_kind: "service" })') is True
    words = "The model service could not be reached after 10 attempts: check it, then continue the run."
    assert page_value(["serviceCause"], "serviceCause(%s)" % json.dumps({"cause": words, "cause_kind": None})) is False
    assert page_value(["serviceCause"], "serviceCause(null)") is False


def test_a_running_rows_time_follows_the_runs_clock_and_the_current_step_counts_its_own():
    """A running row's time starts from the run's clock at its newest
    observation, not the row's own last one, so it moves while other work
    reports; the diagram's current step shows how long it has been under
    way."""
    names = ["CLOCK", "clockBase"]
    assert page_value(names, 'clockBase("00:01:39", true, 300.4)') == 300
    assert page_value(names, 'clockBase("00:01:39", true, null)') == 99
    assert page_value(names, 'clockBase("00:01:39", false, 300)') is None
    assert page_value(names, 'clockBase("playback 00:00:05", true, 7)') == 7
    assert page_value(names, 'clockBase("no clock", true, 7)') is None
    assert page_value(["hms", "stepTime"], 'stepTime({ state: "running", since_s: 99 }, 291.7)') == "for 00:03:12"
    assert page_value(["hms", "stepTime"], 'stepTime({ state: "succeeded", since_s: null }, 291.7)') is None
    assert page_value(["hms", "stepTime"], 'stepTime({ state: "running", since_s: 99 }, null)') is None


def test_a_file_the_settings_name_that_the_project_lacks_is_named():
    """The run and check pages name the files the project's settings name
    that its directory does not hold, instead of only asking for a tick."""
    assert page_value(["missingNamed"], 'missingNamed(["calc/missing.py", "calc/ops.py"], ["calc/ops.py", "calc/more.py"])') == [
        "calc/missing.py"]
    assert page_value(["missingNamed"], 'missingNamed([], ["calc/ops.py"])') == []


#: A document for the page's own drawing code in Node: a part given to
#: append or replaceChildren that is no node becomes a text node of its
#: string, null included, as a browser's DOM makes it.
DOCUMENT = """
class Node {
  constructor() { this.childNodes = []; this.style = { setProperty() {} }; }
  get textContent() { return this.childNodes.map((child) => child.textContent).join(""); }
  set textContent(text) { this.childNodes = [new Text(text)]; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1]; }
  append(...parts) {
    this.childNodes.push(...parts.map((part) => part instanceof Node ? part : new Text(part)));
  }
  replaceChildren(...parts) { this.childNodes = []; this.append(...parts); }
  addEventListener() {}
  setAttribute() {}
  contains() { return false; }
  querySelectorAll() { return []; }
}
class Text extends Node {
  constructor(text) { super(); this.text = String(text); }
  get textContent() { return this.text; }
}
const document = { createElement: () => new Node(), createTextNode: (text) => new Text(text),
                   activeElement: null };
"""


def _definitions(names):
    """app.js's definitions ``names``, as the page has them."""
    source = APP.read_text(encoding="utf-8")
    parts = []
    for name in names:
        found = (re.search(rf"^(const|let) {name} = \{{\n.*?^\}};\n", source, re.S | re.M)
                 or re.search(rf"^(const|let) {name} = .*;\n", source, re.M)
                 or re.search(rf"^(async )?function {name}\(.*?^\}}\n", source, re.S | re.M))
        assert found, f"app.js defines {name}"
        parts.append(found.group(0))
    return "".join(parts)


def _node(script):
    """What ``script`` writes, evaluated by Node, as JSON."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_nested_run_reads_only_while_open():
    """testuaryWeb.nestedRun reads nothing while folded, reads its run when
    opened and stops reading when folded again."""
    script = (DOCUMENT.replace("  addEventListener() {}",
                               "  addEventListener(kind, listener) { this[`on${kind}`] = listener; }")
              + _definitions(["childList", "el", "follow", "nestedRunCount", "nestedRun"]) + """
const asked = [];
globalThis.fetch = (path, options) => { asked.push({ path, signal: options.signal }); return new Promise(() => {}); };
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
const tokenHeaders = () => ({});
const leaving = [];
class ViewPanel { reset() {} }
const part = nestedRun("runs/events.jsonl", "The demonstration");
const before = asked.length;
part.open = true;
part.ontoggle();
const opened = asked.map((item) => item.path);
part.open = false;
part.ontoggle();
process.stdout.write(JSON.stringify([part.className, part.textContent, before, opened,
  asked[0].signal.aborted, leaving.length]));
""")
    assert _node(script) == ["panel nested-run", "Testuary run: The demonstration", 0,
                             ["/api/nested-run?path=runs%2Fevents.jsonl"], True, 1]


def test_a_nested_run_ends_with_its_job_below_the_pages_headings():
    """A nested run of a job that ended ends with it, at once and when it is
    opened again, also when the run wrote no end of its own: its clocks
    stop and its steps show as ended. The job page tells its nested runs
    once the job has ended. The part's headings sit one level below the
    page's."""
    script = (DOCUMENT.replace("  addEventListener() {}",
                               "  addEventListener(kind, listener) { this[`on${kind}`] = listener; }")
              + _definitions(["childList", "el", "nestedRunCount", "nestedRun"]) + """
const made = [];
const cleared = [];
let timers = 0;
globalThis.setInterval = () => { timers += 1; return timers; };
globalThis.clearInterval = (timer) => { cleared.push(timer); };
globalThis.requestAnimationFrame = () => {};
function follow() { return new Promise(() => {}); }
const leaving = [];
class ViewPanel { constructor() { this.ended = false; made.push(this); } reset() {} }
const part = nestedRun("runs/events.jsonl", "The demonstration");
part.open = true;
part.ontoggle();
const opened = [made[0].heading, made[0].ended];
part.onjobended();
const ended = [made[0].ended, cleared.includes(1)];
part.open = false;
part.ontoggle();
part.open = true;
part.ontoggle();
process.stdout.write(JSON.stringify([opened, ended, made[1].ended]));
""")
    assert _node(script) == [["h3", False], [True, True], True]
    source = APP.read_text(encoding="utf-8")
    start = source.index("const nestedRuns = (found) => {")
    told = source[start:source.index("\n  };\n", start)]
    assert 'if (found.ended !== null) for (const run of progress.nested.runs.values()) ' \
           'run.part.dispatchEvent(new Event("jobended"));' in told


def test_a_jobs_nested_runs_are_folded_into_their_phases():
    """The phases and steps fold each nested run into its phase, and a run
    whose phase the plan does not name after the phases, with the error of
    the function that names them."""
    names = ["childList", "el", "STEP_WORDS", "shownState", "phaseState", "phaseGroups", "CLOCK", "hms",
             "clockBase", "clocked", "stepTime", "stepWord", "stepPill", "limitText", "finishingText",
             "nestedParts", "runDiagram"]

    def phase(name, label):
        return {"phase": name, "label": label, "cycle": 0, "end": 0, "state": "running", "steps": [],
                "note": None, "count": None, "values": [], "limits": []}

    diagram = {"target": "notes", "files": ["notes"], "runs": ["r0"],
               "phases": [phase("play", "Play"), phase("note", "Note")],
               "finishing": {"step": "note", "label": "Write a note", "purpose": None, "state": "planned",
                             "steps": [], "values": []}}
    tagged = DOCUMENT.replace("createElement: () => new Node()",
                              "createElement: (tag) => Object.assign(new Node(), { tagName: tag })")
    assert tagged != DOCUMENT
    script = tagged + _definitions(names) + f"""
const nested = {{ runs: new Map([
  ["a", {{ phase: "play", part: el("p", {{}}, "run A") }}],
  ["b", {{ phase: "gone", part: el("p", {{}}, "run B") }}]]), error: "The function failed." }};
const drawn = runDiagram({json.dumps(diagram)}, false, 0, {{ id: "notes/demo" }}, nested);
const list = drawn.childNodes.find((node) => node.className === "phases");
const after = drawn.childNodes.slice(drawn.childNodes.indexOf(list) + 1);
// A nested run's diagram heads one level lower.
const inside = runDiagram({json.dumps(diagram)}, false, 0, null, null, "h3");
process.stdout.write(JSON.stringify([list.childNodes.map((item) => item.textContent.includes("run A")),
  after.map((node) => node.textContent), drawn.childNodes[0].tagName, inside.childNodes[0].tagName]));
"""
    assert _node(script) == [[True, False, False], ["run B", "The function failed."], "h2", "h3"]


def test_a_nested_run_dir_keeps_its_ids_apart_from_the_page():
    """A run directory's parts take their ids from the run's index, which a
    nested run's prefix keeps apart from the Results page's and another
    part's."""
    script = DOCUMENT + _definitions(["childList", "el", "runDirParts"]) + """
const indexes = [];
function runSection(run, summary, open, index) { indexes.push(index); return null; }
function fileView() {}
function fileList() {}
runDirParts({ root: "r", runs: [{}, {}] }, "", null);
runDirParts({ root: "r", runs: [{}, {}] }, "nested-1-", "");
process.stdout.write(JSON.stringify(indexes));
"""
    assert _node(script) == ["0", "1", "nested-1-0", "nested-1-1"]


def _file_box(files, project_files, language="python"):
    """The Run and Check pages' source-file box, as the page draws it for a
    project whose directory holds ``files`` by language and whose settings
    name ``project_files``: its text, evaluated by Node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = APP.read_text(encoding="utf-8")
    parts = []
    for name in ("childList", "el", "missingNamed", "tickAll", "fileChecklist"):
        found = re.search(rf"^function {name}\(.*?^\}}\n", source, re.S | re.M)
        assert found, f"app.js defines {name}"
        parts.append(found.group(0))
    listing = json.dumps({"files": files, "project_files": project_files, "language": language})
    script = (DOCUMENT + "".join(parts) + f"async function call() {{ return {listing}; }}\n"
              + 'const { box } = fileChecklist({ value: "C:/work/calc", addEventListener() {} });\n'
              + "setTimeout(() => process.stdout.write(JSON.stringify(box.textContent)), 0);\n")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_source_file_box_shows_no_null_when_no_named_file_is_missing():
    """The box names a file the settings name that the directory lacks; with
    none missing it shows the files alone, without a stray "null" (seen on
    a Run page on Windows)."""
    shown = _file_box({"python": ["calc/io.py", "calc/ops.py"]}, ["calc/ops.py"])
    assert "null" not in shown
    assert shown.startswith("Source filesSelect allDeselect all") and "tickedcalc/io.py" in shown
    missing = _file_box({"python": ["calc/io.py"]}, ["calc/ops.py"])
    assert "null" not in missing
    assert "The project's settings name calc/ops.py, which its directory does not hold." in missing


def test_a_project_without_a_language_lists_its_files_by_language():
    """Without a language in its settings, a project runs the files of every
    installed language: a group per language, each with its own Select all.
    A set language keeps one list without a heading."""
    shown = _file_box({"java": ["src/main/java/demo/Calc.java"], "python": ["calc.py", "util.py"]},
                      [], language="")
    # The stub keeps no ticks: the counts' totals show each group's own boxes.
    assert shown.startswith("Source filesjavaSelect allDeselect all")
    assert "of 1 tickedsrc/main/java/demo/Calc.javapythonSelect allDeselect all" in shown
    assert "of 2 tickedcalc.pyutil.py" in shown
    one = _file_box({"python": ["calc.py"]}, [])
    assert one.startswith("Source filesSelect allDeselect all") and "of 1 tickedcalc.py" in one


def test_the_jobs_page_follows_each_look_at_the_server():
    """The Jobs page shows a job's status and time as they change, as the
    overview's recent jobs do: a run that ended while the page was open
    shows as finished (on Windows it stayed "RUNNING 0:32")."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    names = ["childList", "el", "hms", "valueLines", "dirName", "jobDir", "jobModel", "JOB_STATES",
             "incompleteRun", "stateWords", "jobState", "leaving", "stateListeners", "onState", "redraw",
             "overviewChanges", "stackedTable"]
    names += [name for name in ("jobsTable",) if re.search(rf"^function {name}\(", APP.read_text(encoding="utf-8"),
                                                            re.M)]
    job = {"id": "a", "kind": "run", "title": "Run of calc", "model": "local", "run_dir": None, "values": [],
           "started": 1, "ended": None, "status": "running", "word": None, "elapsed_s": 32}
    running = {"active": job, "jobs": [job]}
    ended = {**job, "status": "finished", "word": "finished", "ended": 388, "elapsed_s": 387}
    finished = {"active": None, "jobs": [ended]}
    script = (DOCUMENT + _definitions(names) + _definitions(["jobsView"])
              + f"let look = {json.dumps(running)};\nasync function call() {{ return look; }}\n"
              + "(async () => {\n"
              + "  const page = new Node();\n  page.append(...childList(await jobsView()));\n"
              + "  const before = page.textContent;\n"
              + f"  look = {json.dumps(finished)};\n"
              + "  for (const listener of stateListeners) await listener(look);\n"
              + "  process.stdout.write(JSON.stringify([before, page.textContent]));\n})();\n")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    before, after = json.loads(done.stdout)
    assert "RUNNING" in before and "00:00:32" in before
    assert "FINISHED" in after and "00:06:27" in after and "RUNNING" not in after


def test_the_demonstration_says_where_it_writes():
    """The demonstration needs no setup, yet each playback writes a directory:
    the page names where, and the Results page while an extension does not
    hide it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    script = DOCUMENT + _definitions(["childList", "el", "pathText", "shortPath", "extensions",
                                      "addedTab", "tabTitle", "tabShown", "demoWrites"]) + """
const testuaryTitles = new Map([["results", "Results"]]);
const shown = [el("p", {}, demoWrites("C:/work/calc")).textContent,
               el("p", {}, demoWrites("C:/Users/me/work/calc")).textContent];
extensions.hidden_tabs = ["results"];
process.stdout.write(JSON.stringify([...shown, el("p", {}, demoWrites("C:/work/calc")).textContent]));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    near, far, hidden = json.loads(done.stdout)
    written = ("Each playback writes its record into a new directory named after the language and the "
               "time, such as testuary-demo-python-20261031-075100, in C:/work/calc Copy path.")
    assert (near, hidden) == (f"{written} The Results page opens it.", written)
    # A long path is shortened as on every other page, its whole text copied.
    assert ", in .../work/calc Copy path." in far


def test_one_files_project_card_names_its_value_without_statistics_across_files():
    """A median, range and standard deviation describe several files: for one
    file the card names that file's value and that it is one file."""
    one = {"count": 1, "mean": {"text": "87.5%"}, "median": {"text": "87.5%"}, "minimum": {"text": "87.5%"},
           "maximum": {"text": "87.5%"}, "std": None}
    assert page_value(["statRows"], "statRows(%s, null)" % json.dumps(one)) == [["Targets", "1"]]
    two = {**one, "count": 2, "std": {"text": "2.5"}, "minimum": {"text": "85%"}, "maximum": {"text": "90%"}}
    assert page_value(["statRows"], "statRows(%s, null)" % json.dumps(two)) == [
        ["Median", "87.5%"], ["Range", "85% to 90%"], ["Standard deviation", "2.5 points"], ["Targets", "2"]]
    starter = {"count": 3, "median": {"text": "50%"}, "mean": {"text": "48%"}}
    assert page_value(["statRows"], "statRows(%s, %s)" % (json.dumps(one), json.dumps(starter)))[-1] == [
        "Starter suites", "median 50%, mean 48% of the targets' values"]


def test_a_scrolled_text_box_keeps_the_readers_place():
    """The terminal text and the progress log follow new output only while
    the reader is at their end; a reader who scrolled up keeps the place,
    and a new or hidden box starts at its end."""
    assert page_value(["nextScroll"], "[nextScroll(null, 900),"
                      "nextScroll({scrollTop: 480, scrollHeight: 900, clientHeight: 420}, 1200),"
                      "nextScroll({scrollTop: 478, scrollHeight: 900, clientHeight: 420}, 1200),"
                      "nextScroll({scrollTop: 100, scrollHeight: 900, clientHeight: 420}, 1200),"
                      "nextScroll({scrollTop: 0, scrollHeight: 0, clientHeight: 0}, 500)]"
                      ) == [900, 1200, 1200, 100, 500]
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    start = source.index("async function jobView(id) {")
    draw = source[source.index("const draw = () => {", start):source.index("const later = ", start)]
    assert "terminal.scrollTop = nextScroll(" in draw and "redrawKeeping(progressBox, " in draw
    start = source.index("function redrawKeeping(box, drawn) {")
    assert "log.scrollTop = nextScroll(" in source[start:source.index("\n}\n", start)]


TICKING = """
class Node {
  constructor() { this.childNodes = []; this.listeners = {}; this.style = { setProperty() {} }; }
  get textContent() { return this.childNodes.map((child) => child.textContent).join(""); }
  set textContent(text) { this.childNodes = [new Text(text)]; }
  append(...parts) {
    this.childNodes.push(...parts.map((part) => part instanceof Node ? part : new Text(part)));
  }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute() {}
}
class Text extends Node {
  constructor(text) { super(); this.text = String(text); }
  get textContent() { return this.text; }
}
const document = { createElement: () => new Node(), createTextNode: (text) => new Text(text) };
"""


def test_select_all_and_deselect_all_tick_every_file():
    """Wherever files are ticked (the Run and Check pages, the project
    setup's files), Select all and Deselect all tick or clear every one,
    beside how many are ticked; a box ticked by hand updates the count."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    script = TICKING + _definitions(["childList", "el", "tickAll"]) + """
const boxes = [new Node(), new Node(), new Node()];
boxes[0].checked = true;
const bar = tickAll(boxes);
const [select, deselect, count] = bar.childNodes;
const seen = [select.textContent, deselect.textContent, count.textContent];
deselect.listeners.click();
seen.push(boxes.map((box) => box.checked), count.textContent);
select.listeners.click();
seen.push(boxes.map((box) => box.checked), count.textContent);
boxes[1].checked = false;
boxes[1].listeners.change();
seen.push(count.textContent);
process.stdout.write(JSON.stringify(seen));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        "Select all", "Deselect all", "1 of 3 ticked", [False, False, False], "0 of 3 ticked",
        [True, True, True], "3 of 3 ticked", "2 of 3 ticked"]
    source = APP.read_text(encoding="utf-8")
    for owner in ("function fileChecklist(project) {", "function setupInput(view) {"):
        start = source.index(owner)
        assert "tickAll(" in source[start:source.index("\n}\n", start)], owner


DIALOG = """
class Node {
  constructor() { this.childNodes = []; this.listeners = {}; this.style = { setProperty() {} }; }
  get textContent() { return this.childNodes.map((child) => child.textContent).join(""); }
  set textContent(text) { this.childNodes = [new Text(text)]; }
  append(...parts) {
    for (const part of parts.map((item) => item instanceof Node ? item : new Text(item))) {
      part.parent = this;
      this.childNodes.push(part);
    }
  }
  replaceChildren(...parts) { this.childNodes = []; this.append(...parts); }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
  focus() {}
  remove() { this.parent.childNodes = this.parent.childNodes.filter((child) => child !== this); }
}
class Text extends Node {
  constructor(text) { super(); this.text = String(text); }
  get textContent() { return this.text; }
}
const document = { createElement: () => new Node(), createTextNode: (text) => new Text(text),
                   activeElement: null, body: new Node(), addEventListener() {}, removeEventListener() {} };
const window = { listeners: {}, addEventListener(type, listener) { this.listeners[type] = listener; },
                 removeEventListener(type, listener) {
                   if (this.listeners[type] === listener) delete this.listeners[type]; } };
async function call() {
  return { path: "C:/work", crumbs: [], shortcuts: [], parent: null, dirs: [], truncated: false };
}
"""


def test_the_dir_dialog_closes_when_the_page_changes():
    """The directory dialog over a page closes, choosing nothing, when the page
    changes (the browser's Back button), and leaves no listener behind."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    script = DIALOG + _definitions(["childList", "el", "chooseDir"]) + """
const chosen = chooseDir({ title: "Choose the run directory", start: "" });
setTimeout(async () => {
  const open = [document.body.childNodes.length, Object.keys(window.listeners)];
  window.listeners.hashchange();
  const value = await chosen;
  process.stdout.write(JSON.stringify([open, value, document.body.childNodes.length,
                                       Object.keys(window.listeners)]));
}, 0);
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [[1, ["hashchange"]], None, 0, []]


def test_without_a_project_home_offers_to_add_one():
    """The overview's main action is adding a project until one exists:
    starting a run could only say there is none yet."""
    source = APP.read_text(encoding="utf-8")
    start = source.index("async function home() {")
    hero = source[start:source.index("\n}\n", start)]
    assert ('(listed.projects || []).length ? tabShown("run") && el("a", { class: "button primary", '
            'href: "#/run", text: "Start a run" })') in hero
    assert (': tabShown("projects") && el("a", { class: "button primary", href: "#/projects", '
            'text: "Add a project" })') in hero


def test_results_without_a_project_show_no_empty_panel():
    """The Results page lists a project's runs only when a project is
    chosen: without one there is no empty panel."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    script = DOCUMENT + _definitions(["childList", "el", "projectRuns"]) + """
projectRuns({ value: "" }, "").then((box) => process.stdout.write(JSON.stringify(box)));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) is None


def test_the_jobs_page_without_a_job_names_the_way_on():
    """With no job yet, the Jobs page says how to start one: a run or the
    demonstration, the demonstration alone while an extension hides Run."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    script = DOCUMENT + _definitions(["childList", "el", "extensions", "tabShown", "jobsTable"]) + """
const shown = jobsTable([]).textContent;
extensions.hidden_tabs = ["run"];
process.stdout.write(JSON.stringify([shown, jobsTable([]).textContent]));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        "No job started yet in this session. Start a run or play the demonstration.",
        "No job started yet in this session. Play the demonstration."]


def test_links_to_a_hidden_tab_leave_with_their_words():
    """While an extension hides Settings and Results, a run's warning ends
    with its value and a run directory on the Jobs page is plain text."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    script = DOCUMENT + _definitions(["childList", "el", "extensions", "tabShown", "warningSentence",
                                      "warningList", "dirName", "jobDir"]) + """
document.createElement = (tag) => Object.assign(new Node(), { tag });
const warning = { text: "The limit was reached.", setting: "limit", setting_label: "Limit",
                  value: { text: "3", value: "3" } };
const job = { run_dir: "C:/work/out-1", ended: 1, kind: "run" };
const read = () => [warningList([warning]).textContent, jobDir(job).tag];
const shown = read();
extensions.hidden_tabs = ["settings", "results"];
process.stdout.write(JSON.stringify([shown, read()]));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        ["The limit was reached. Limit: 3. Raise it in the settings.", "a"],
        ["The limit was reached. Limit: 3.", "span"]]


def test_a_set_up_project_links_its_settings_page():
    """After the setup, the page links the project's own Settings page for
    every setting the setup left at its default."""
    source = APP.read_text(encoding="utf-8")
    start = source.index('el("h2", { text: "Project set up" })')
    notice = source[start:source.index("\n  }\n", start)]
    assert 'el("a", { href: settingsLink(notice.project_dir), text: "All settings of the project" })' in notice


def test_the_checks_items_are_folded_below_its_summary():
    """The check's page says its findings in the summary; the items table
    repeats them as data and is folded, as the limits are."""
    source = APP.read_text(encoding="utf-8")
    start = source.index("function reportView(")
    view = source[start:source.index("\n}\n", start)]
    assert 'el("details", { class: "panel scroll" }, el("summary", { text: `Items (${items.length})` })' in view


def test_a_narrow_screen_labels_each_cell_of_the_stacked_tables():
    """On a narrow screen the check report's items and limits, the Jobs
    list, how each phase of a target ended and a phase end's candidates
    show each row as a block, each value under its column's name
    (style.css, table.stack). Every cell carries its column's name. A row
    that spans the table, such as a target's details on the Results page,
    carries none."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    report = {"status": "verified", "exit_code": 0,
              "items": [{"key": "source.targets", "status": "verified",
                         "reason": "Each target is in the project directory.", "action": ""}],
              "limits": [{"phase": "setup", "name": "setup_timeout_seconds", "value": 120,
                          "unit": "seconds", "state": "configured"}]}
    job = {"id": "a", "kind": "run", "title": "Run of calc", "model": "local", "run_dir": None, "values": [],
           "started": 1, "ended": None, "status": "running", "word": None, "elapsed_s": 32}
    end = {"label": "Coverage phase end", "summary": "It delivered candidate 1.", "freed": None,
           "candidates": [{"state": "delivered", "label": "c1", "text": "candidate 1", "tests": {"text": "12"},
                           "line": {"text": "90%"}, "branch": None, "mutation": None,
                           "state_text": "chosen", "reason": ""}]}
    item = {"summarized": True, "phases": [{"label": "Starter phase", "status_text": "done", "record": None}],
            "phase_ends": [], "refusals": [], "warnings": [], "details": {}, "row": None}
    names = ["childList", "el", "cell", "stateBadge", "stackedTable", "reportView", "hms", "valueLines",
             "dirName", "jobDir", "jobModel", "JOB_STATES", "incompleteRun", "stateWords", "jobState",
             "jobsTable", "known", "phaseEnd", "candidateScore", "runDetails", "detailGrids"]
    script = (DOCUMENT.replace("  setAttribute() {}", "  setAttribute(name, value) { this[name] = value; }")
              + _definitions(names)
              + f"const parts = reportView({json.dumps(report)}, null, null).childNodes.slice(1);\n"
              + f"const tables = [...parts.map((part) => part.childNodes[1]), jobsTable([{json.dumps(job)}]).childNodes[0],\n"
              + f"  phaseEnd({json.dumps(end)}).childNodes[1].childNodes[0],\n"
              + f"  runDetails({{}}, {json.dumps(item)}, {{ runs: [{{ labels: {{}} }}] }}).childNodes[2].childNodes[0]];\n"
              + 'const spanning = stackedTable(["Target", "Repetition"], [el("tr", {}, el("td", { colspan: 2 }))]);\n'
              + "process.stdout.write(JSON.stringify([...tables.map((table) => [table.className,\n"
              + "  table.childNodes[1].childNodes[0].childNodes.map((item) => item['data-label'])]),\n"
              + "  spanning.childNodes[1].childNodes[0].childNodes[0]['data-label'] || null]));\n")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        ["stack", ["Status", "Item", "Reason", "What to do"]],
        ["stack", ["Phase", "Limit", "Value", "State"]],
        ["stack", ["Started", "Job", "Run directory", "Model", "Values", "Status", "Time"]],
        ["stack", ["Candidate", "Tests", "Line coverage", "Branch coverage", "Mutation score", "Outcome"]],
        ["stack", ["Phase", "How it ended", "Phase record"]],
        None]


def test_a_changed_setting_says_it_is_not_saved_yet():
    """A setting's row marks a value changed but not saved, until it is
    saved (the row is drawn again) or changed back."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    names = ["childList", "el", "settingInput", "choiceOptions", "purposeParts", "canReset", "overrideNote",
             "contextNote", "sourceWords", "sourceBadge", "levelWords", "settingRow"]
    script = TICKING.replace("  setAttribute() {}", "  setAttribute(name, value) { this[name] = value; }") + \
        _definitions(names) + """
const setting = { name: "goal", label: "Coverage goal (%)", kind: "text", value: "90", choices: [],
                  purpose: "Coverage goal in percent.", source: "default", active: true };
const target = { request: {}, file: "global", files: { global: "config.ini", project: null } };
const row = settingRow("coverage", setting, target, undefined, async () => {}, "");
const input = row.childNodes[1].childNodes[0];
const status = row.childNodes[3].childNodes[row.childNodes[3].childNodes.length - 1];
const seen = [];
input.value = "95";
input.listeners.input();
seen.push([status.className, status.textContent]);
input.value = "90";
input.listeners.input();
seen.push([status.className, status.textContent]);
process.stdout.write(JSON.stringify(seen));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [["small unsaved", "Not saved yet."], ["small saved", ""]]


def test_finishings_box_shows_its_step_and_its_values():
    """Finishing's box in the phases and steps shows its observed step and
    the values it measured, as the loops' boxes do."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    names = ["childList", "el", "STEP_WORDS", "shownState", "phaseState", "phaseGroups", "stepTime",
             "stepWord", "stepPill", "limitText", "finishingText", "nestedParts", "runDiagram"]
    diagram = {"target": "calc.py", "files": ["calc.py"], "runs": [], "phases": [], "finishing": {
        "step": "finishing", "label": "Finishing", "purpose": "measuring", "state": "succeeded",
        "steps": [{"step": "step/delivered-measurement", "label": "Checking the delivered suite",
                   "state": "succeeded", "count": None, "since_s": None}],
        "values": ["line coverage: 100% [measured]"]}}
    script = DOCUMENT + _definitions(names) + f"""
const box = runDiagram({json.dumps(diagram)}, true, 10);
process.stdout.write(JSON.stringify(box.textContent));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout).endswith(
        "FinishingdoneChecking the delivered suitedoneline coverage: 100% [measured]")


def test_a_repetition_not_fully_measured_says_so_in_the_summarys_words():
    """A repetition whose summary is not fully measured is marked so, and a
    run directory in the list counts its fully measured repetitions, in
    summary.json's own words."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    script = DOCUMENT + _definitions(["childList", "el", "runName"]) + """
const shown = [false, true, null].map((value) =>
  el("p", {}, runName("calc.py", { name: "r0", fully_measured: value })).textContent);
process.stdout.write(JSON.stringify(shown));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["r0 not fully measured", "r0", "r0"]
    source = APP.read_text(encoding="utf-8")
    assert "${item.fully_measured_runs} fully measured ${item.fully_measured_runs === 1" in source


def test_the_setups_files_question_takes_at_least_one_file():
    """The project setup's "Which files?" refuses no ticked file, with the
    Run page's message, and answers "all" or the ticked files' numbers."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    script = TICKING + _definitions(["childList", "el", "tickAll", "NO_FILE", "setupInput"]) + """
const view = { question: { key: "targets", prompt: "Which files? (numbers)", choices: [], default: "" },
               files: ["calc.py", "util.py"] };
const [group, answer] = setupInput(view);
const [, bar, first] = group.childNodes;
const [select, deselect] = bar.childNodes;
select.listeners.click();
const seen = [answer()];
deselect.listeners.click();
try { seen.push(answer()); } catch (error) { seen.push(error); }
first.childNodes[0].checked = true;
seen.push(answer());
process.stdout.write(JSON.stringify(seen));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        "all", {"field": "answer", "message": "Tick at least one target."}, "1"]


def test_a_step_a_limit_ended_names_the_limit():
    """A step the diagram marks with the limit's words shows them; without
    them, as from an older event file, it says "limit reached"."""
    words = page_value(["STEP_WORDS", "stepWord"], "[" + ", ".join(
        f"stepWord({json.dumps(item)})" for item in (
            {"state": "limit_reached", "reached": "Request limit reached"},
            {"state": "limit_reached", "reached": None},
            {"state": "limit_reached"},
            {"state": "succeeded", "reached": None})) + "]")
    assert words == ["request limit reached", "limit reached", "limit reached", "done"]
