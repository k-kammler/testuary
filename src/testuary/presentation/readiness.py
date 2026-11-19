"""Render supplied readiness facts without inspecting or admitting a run."""
from __future__ import annotations

from dataclasses import dataclass, replace

from testuary import _words
from testuary.readiness import ReadinessItem, ReadinessLimit, RunReadiness
from testuary.settings import PROVIDER_SECTIONS

from .terminal import TerminalPresenter
from .view import phase_label


@dataclass(frozen=True)
class NextSteps:
    """The words a report uses for its reader's next actions: the terminal's
    commands by default, another interface's own controls when it passes
    them. ``verify`` names the tool check, ``check_tools`` asks for one,
    ``verify_named`` follows its mention in the detailed header."""

    verify: str = "testuary check --verify with the same options"
    check_tools: str = "testuary check --verify with the same options"
    verify_named: str = " (testuary check --verify)"
    detailed: str = "the same command with --display detailed"
    read_detailed: str = "with the same command and --display detailed"
    again: str = "run the same command again"
    generate: str = "testuary generate with the same options (without --verify)"


TERMINAL_STEPS = NextSteps()


def _value(value: str | None) -> str:
    return "not reported" if value is None else value or "(empty)"


def _check_name(scope: str) -> str:
    return "quick check" if scope == "inspection" else "tool check"


#: The report's overall status in words, after the check's name.
_OVERALL = {"verified": "no problems found", "unverified": "some required parts not tried yet",
            "missing_setup": "something required is missing",
            "unsupported": "something required is not supported",
            "ambiguous": "something required was found more than once",
            "refused": "refused", "failed": "something required failed"}


def _header(report: RunReadiness, steps: NextSteps = TERMINAL_STEPS, *,
            before_run: bool = False) -> list[str]:
    if before_run:
        # A run's own setup report: no check ran, and the run's exit status
        # is not known yet.
        lines = ["Setup before the run"]
    else:
        overall = _OVERALL.get(report.status, _words.status_word(report.status, report.scope))
        if report.status == "verified" and _left_to_run(report):
            overall = "no problem found in what it tried, some required parts left to a run"
        lines = [f"Testuary {_check_name(report.scope)}: {overall} "
                 f"(exit status {report.exit_code})"]
        disclaimer = "A check does not promise that a run succeeds or what it will measure."
        if report.scope == "inspection":
            disclaimer += (" A quick check does not start the tools, as the tool check"
                           f"{steps.verify_named} does.")
        lines.append(disclaimer)
    where = _EXECUTION.get(report.execution, _value(report.execution))
    lines.append(f"Source directory: {_value(report.source_dir)} | Where it runs: {where}")
    lines.extend(f"Target: {source}" for source in report.source_files)
    if not report.source_files:
        lines.append("Targets: none reported")
    lines.append("Phases: " + (", ".join(phase_label(phase) for phase in report.phases)
                               or "none reported"))
    lines.append(
        f"Model: {_value(report.model_alias)} | provider: {_value(report.provider)} "
        f"| model name at the service: {_value(report.model)}")
    key = {True: "set, not tried", False: "not set", None: "unknown"}[report.credentials_configured]
    lines.append(f"API key: {key}")
    lines.append(f"Run directory: {_value(report.output_dir)} | {_OUTPUT[report.output_mode]}")
    return lines


def _limit_value(limit) -> str:
    return f"{limit.value} {limit.unit}".rstrip()


def _limit_lines(report: RunReadiness) -> list[str]:
    if not report.limits:
        return ["Limits: not reported"]
    return [f"Limit | {phase_label(limit.phase)} | {limit.name}: "
            f"{_limit_value(limit) if limit.state == 'configured' else limit.state}"
            for limit in report.limits]


def _full_items(report: RunReadiness) -> list[str]:
    """Every item by its plain name, its key after it, in the summary's
    status words."""
    if not report.items:
        return ["Items: none reported"]
    lines = []
    for item in report.items:
        required = "required" if item.required else "optional"
        plain = _words.plain_name(item)
        named = f"{plain} ({required})" if plain == item.key else f"{plain} ({item.key}, {required})"
        lines.append(f"{named}: {_words.status_word(item.status, report.scope)}")
        if item.selected is not None or item.version is not None:
            lines.append(f"Selected: {_value(item.selected)} | version: {_value(item.version)}")
        if item.reason:
            lines.append(f"Reason: {item.reason}")
        if item.action:
            lines.append(f"Fix: {item.action}")
        if item.diagnostic_path:
            lines.append(f"Diagnostic: {item.diagnostic_path}")
    return lines


# Plain names for the providers, display text only.
_PROVIDERS = {name: {"openai": "OpenAI", "vllm": "vLLM"}.get(name, name.capitalize())
              for name in PROVIDER_SECTIONS}
# Plain limit wording. Per-request limits and setup timeouts appear only in the
# detailed view (_DETAILED_ONLY).
# Each entry: wording of a ceiling, kind shared by related limits, wording when unlimited.
_LIMITS = {"attempts_per_suite": ("attempts per suite", None, "attempt"),
           "max_requests_per_phase": ("model requests", None, "request"),
           "max_input_tokens_per_phase": ("input tokens", "token", "input token"),
           "max_output_tokens_per_phase": ("output tokens", "token", "output token"),
           "max_total_tokens_per_phase": ("tokens in total", "token", "total token"),
           "max_wall_clock_seconds_per_phase": ("time", "time", "time")}
# The wording of a ceiling of one.
_ONE = {"attempts per suite": "attempt per suite", "model requests": "model request",
        "input tokens": "input token", "output tokens": "output token",
        "tokens in total": "token in total"}
_DETAILED_ONLY = {"request_timeout_seconds", "max_output_tokens",
                  "setup_timeout_seconds", "setup_probe_timeout_seconds"}
# What each check does, after its name (_check_name).
_REACH = {
    "inspection": "runs nothing in your project and contacts no model",
    "verification": ("contacts no model, starts each tool once and, in a copy of your project, "
                     "tries its build where a run builds it and finds its tests without "
                     "running one"),
}
_EXECUTION = {"local": "on this computer", "container": "in a container"}
_OUTPUT = {"fresh": "a new run", "resume": "continues an earlier run in this directory",
           "reuse": "reuses results of an earlier run"}


def _tool(item: ReadinessItem) -> str:
    name = _words.plain_name(item)
    _, kind, raw = _words.key_parts(item.key)
    # Native tool paths are in the detailed view, except where the runtime lives.
    native = item.key.startswith("native.") and kind == "item"
    selected = (item.selected if item.selected not in (None, raw) and
                (not native or raw == "runtime") else None)
    if native and raw == "runtime":
        text = f"{name} {item.version or ''}".rstrip() + (f" ({selected})" if selected else "")
    else:
        text = " ".join(value for value in (name, selected) if value)
        if item.version is not None:
            text = f"{text} ({item.version})" if selected else f"{text} {item.version}"
    return text + ("" if item.required else " (optional)")


def _by_language(items: list[ReadinessItem], *, actions: bool) -> list[str]:
    """One line per language (and supplied action, if shown), then that action.

    Items outside a language get a line each.
    """
    lines: dict[tuple, list[str]] = {}
    for item in items:
        values = lines.setdefault((_words.language_name(item.key), item.action if actions else None), [])
        if _tool(item) not in values:
            values.append(_tool(item))
    result = []
    for (language, action), values in lines.items():
        result += [f"  {language}: {', '.join(values)}"] if language else [
            f"  {value}" for value in values]
        if action:
            result += _indented(action, "    ")
    return result


def _not_tried(items: list[ReadinessItem]) -> list[str]:
    """Plain names per supplied action. Languages with the same names share a line."""
    actions: dict[str | None, dict[str, list[str]]] = {}
    for item in items:
        language = _words.language_name(item.key)
        names = actions.setdefault(item.action, {}).setdefault(language, [])
        # An item on a line of its own says why, when the check gave a reason and a step.
        text = (f"{_tool(item)}: {item.reason}" if not language and item.reason and item.action
                else _tool(item))
        if text not in names:
            names.append(text)
    result = []
    for action, languages in actions.items():
        shared: dict[tuple, list[str]] = {}
        for language, names in languages.items():
            if language:
                shared.setdefault(tuple(names), []).append(language)
        result += [f"  {name}" for name in languages.get("", [])]
        result += [f"  {', '.join(owners)}: {', '.join(names)}" for names, owners in shared.items()]
        if action:
            result += _indented(action, "    ")
    return result


def _indented(text: str, prefix: str, indent: int | None = None) -> list[str]:
    first, *rest = text.splitlines() or [""]
    indent = len(prefix) if indent is None else indent
    return [prefix + first] + [" " * indent + line for line in rest]


def _problem_lines(groups, *, fixes: bool) -> list[str]:
    lines = []
    for members, reason, action in groups:
        # The language is named once before its items.
        names: dict[str, list[str]] = {}
        for item in members:
            language, name = _words.language_name(item.key), _words.plain_name(item)
            # What was selected helps to fix it. A tool's own name adds nothing.
            if item.selected not in (None, _words.key_parts(item.key)[2]) and item.selected not in name:
                name += f" ({item.selected})"
            values = names.setdefault(language if language not in name else "", [])
            if name not in values:
                values.append(name)
        subject = ", ".join(" ".join(filter(None, (language, ", ".join(values))))
                            for language, values in names.items())
        subject = subject[:1].upper() + subject[1:]
        lines.extend(_indented(f"{subject}: {reason}" if reason else subject, "  ", 4))
        if fixes and action:
            lines.extend(_indented(action, "    Fix: "))
        lines.extend(f"    More detail: {path}" for path in
                     dict.fromkeys(item.diagnostic_path for item in members if item.diagnostic_path))
    return lines


def _plain_limits(report: RunReadiness) -> str | None:
    """The limits a user plans a run around. Per-request limits, timeouts and raw
    names are detailed only."""
    phases: dict[str, list[ReadinessLimit]] = {}
    for limit in report.limits:
        if limit.name not in _DETAILED_ONLY:
            phases.setdefault(limit.phase, []).append(limit)
    parts = []
    for phase, limits in phases.items():
        values, open_ended, unclear = [], [], []
        # "No token limit" only when every token limit of the phase is unlimited.
        tokens = [limit for limit in limits if _LIMITS.get(limit.name, (None, None))[1] == "token"]
        shared = all(limit.state == "unlimited" for limit in tokens)
        for limit in limits:
            name = limit.name.replace("_", " ")
            label, kind, unlimited = _LIMITS.get(limit.name, (name, None, name))
            if limit.state == "configured" and limit.name not in _LIMITS:
                # A limit without a plain wording keeps the producer's ``name`` and ``unit``.
                values.append(f"{label}: {_number(limit.value)} {limit.unit}".rstrip())
            elif limit.state == "configured":
                if limit.value == 1:
                    label = _ONE.get(label, label)
                values.append("up to " + (_duration_text(limit.value) if kind == "time" else
                                          f"{_number(limit.value)} {label}"))
            elif limit.state == "unlimited":
                open_ended.append(kind if kind == "time" or kind == "token" and shared
                                  else unlimited)
            else:
                unclear.append(label)
        if "token" in open_ended and "time" in open_ended:
            open_ended = ["token or time"] + [name for name in open_ended
                                              if name not in ("token", "time")]
        values.extend(f"no {name} limit" for name in dict.fromkeys(open_ended))
        values.extend(f"{name} limit unknown" for name in unclear)
        parts.append(f"{_words.phase_words(phase)} " + ", ".join(values))
    return "Limits: " + " | ".join(parts) if parts else None


def _number(value) -> str:
    if isinstance(value, int):
        return str(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _duration_text(seconds) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{_number(seconds / 3600)} h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{_number(seconds / 60)} min"
    return f"{_number(seconds)} s"


def _key_sentence(report: RunReadiness) -> str | None:
    """The reported fact only, for producers without an API key item. No action."""
    return {True: "API key: found, not tried by this check.",
            False: "API key: none found.", None: None}[report.credentials_configured]


def _left_to_run(report: RunReadiness) -> list[ReadinessItem]:
    """The required items that only a run tries: not tried by this check."""
    return [item for item in report.items
            if item.required and item.status == "unverified" and item.tried_by == "run"]


def _unchecked_languages(report: RunReadiness) -> list[str]:
    """The languages whose adapter has no check of its own: the producer leaves
    their assessment to a run."""
    return list(dict.fromkeys(
        _words.language_name(item.key) for item in _left_to_run(report)
        if item.key.startswith("native.") and _words.key_parts(item.key)[1:] == ("", "assessment")))


def _untried_line(check: str, report: RunReadiness, untried: bool) -> str:
    """The first line of a check that found no problem but did not try every
    required part. It never says that no problem was found."""
    if report.exit_code and untried:
        # The first line agrees with the exit status the producer reported.
        return (f"The {check} found no failure, but some required parts were not tried, "
                f"so it ends with exit status {report.exit_code}.")
    unchecked = _unchecked_languages(report)
    has, own = ("has", "its") if len(unchecked) == 1 else ("have", "their")
    languages = (f" {' and '.join(unchecked)} {has} no check of {own} own, so a run tries "
                 f"{own} tools first." if unchecked else "")
    return (f"The {check} found no problem in what it tried.{languages} The required parts "
            "it left to a run are listed under Not tried yet.")


def _summary_header(report: RunReadiness, problems: int, *, refused: bool,
                    before_run: bool = False) -> list[str]:
    found = ("No problems found" if not problems else
             f"{problems} problem{'s' if problems != 1 else ''} found")
    untried = any(item.required and item.status == "unverified" for item in report.items)
    check, reach = _check_name(report.scope), _REACH[report.scope]
    opening = 1
    if before_run:
        # Generate shows this after its own admission: it is not a check of its own.
        lines = [f"Setup before the run: {found[:1].lower() + found[1:]}."]
    elif refused:
        # A settings refusal stops before any tool or project is looked at.
        lines = [f"{found} by a {check}: the settings were refused, so it "
                 "stopped before looking "
                 "at your project."]
    elif not problems and ((report.exit_code and untried) or _left_to_run(report)):
        lines = [_untried_line(check, report, untried), f"A {check} {reach}."]
        opening = 2
    elif problems == 1 and all(item.key.startswith("configuration.model.")
                               for item in _words.listed_problems(list(report.items))):
        # The check went past a missing model to look at the project.
        lines = [f"1 problem found by a {check}: no model is selected.", f"A {check} {reach}."]
        opening = 2
    else:
        lines = [f"{found} by a {check}, which {reach}."]
    if report.source_dir is not None:
        where = _EXECUTION.get(report.execution)
        lines.append(f"Source directory: {report.source_dir}" + (f" ({where})" if where else ""))
    if len(report.source_files) == 1:
        lines.append(f"Target: {report.source_files[0]}")
    elif report.source_files:
        lines += ["Targets:", *(f"  {source}" for source in report.source_files)]
    if report.model is not None:
        provider = _PROVIDERS.get((report.provider or "").lower(), report.provider)
        lines.append(f"Model: {report.model}" + (f" via {provider}" if provider else ""))
    # A producer's API key item states the key itself, with its own action.
    keyed = any(item.key == "configuration.credentials" for item in report.items)
    if not keyed and _key_sentence(report):
        lines.append(_key_sentence(report))
    if report.output_dir is not None:
        lines.append(f"Run directory: {report.output_dir} ({_OUTPUT[report.output_mode]})")
    if report.phases:
        lines.append("Phases: " + ", ".join(phase_label(phase) for phase in report.phases))
    limits = _plain_limits(report)
    if limits:
        lines.append(limits)
    # A blank line separates the facts from the opening lines, if there are any.
    return lines[:opening] + [""] + lines[opening:] if len(lines) > opening else lines


def _summary(report: RunReadiness, *, before_run: bool = False,
             playback: bool = False, steps: NextSteps = TERMINAL_STEPS) -> list[str]:
    items = list(report.items)
    problems = _words.problem_groups(_words.listed_problems(items))
    # Only the settings item is a problem: the producer refused before looking further.
    refused = bool(problems) and all(item.key == "configuration" for item in _words.listed_problems(items))
    lines = _summary_header(report, len(problems), refused=refused, before_run=before_run)
    rest = [item for item in items if not _words.is_problem(item)]
    listed = {id(item) for item in _words.listed_problems(items)}
    # A model's request protocol is a technical detail for the detailed view.
    found = [item for item in rest if item.status == "verified"
             and (item.version is not None or item.selected is not None)
             and not item.key.startswith("configuration.model.")]
    installed = [item for item in rest if item.status == "unverified" and item.version is not None]
    # An optional entry that only names a diagnostic file is listed with the files.
    untried = [item for item in rest if item.status == "unverified" and item.version is None
               and (item.required or item.reason or not item.diagnostic_path)]
    optional = _words.problem_groups([item for item in rest if not item.required
                                      and item.status in _words.PROBLEMS])
    grouped = {id(item) for members, _, _ in problems + optional for item in members}
    notes = [item for item in rest if item.status == "verified" and item not in found
             and not item.required and item.reason]
    diagnostics = list(dict.fromkeys(item.diagnostic_path for item in items
                                     if item.diagnostic_path and id(item) not in grouped))
    if found:
        # A quick check reads metadata. Only a deeper check starts the tools.
        heading = "Found and working:" if report.scope == "verification" else "Found:"
        lines += ["", heading, *_by_language(found, actions=False)]
    if installed:
        lines += ["", "Installed but not tried:", *_by_language(installed, actions=True)]
        if report.scope == "inspection" and not before_run:
            lines.append(f"  The tool check is {steps.verify}. It makes no model requests.")
    if untried:
        lines += ["", "Not tried yet:", *_not_tried(untried)]
    if problems:
        lines += ["", "Missing or blocking:", *_problem_lines(problems, fixes=len(problems) > 1)]
    if optional:
        lines += ["", "Optional:", *_problem_lines(optional, fixes=True)]
    if notes:
        lines += ["", "Notes:", *(line for item in notes for line in _indented(item.reason, "  "))]
    if diagnostics:
        lines += ["", "More detail was saved to:", *(f"  {path}" for path in diagnostics)]
    if lines[-1]:
        lines.append("")
    if len(items) > len(listed):
        lines.append(f"Full setup report: {steps.detailed}")
    lines.append(_next_step(report, problems, before_run=before_run, playback=playback, steps=steps))
    return lines


def _next_step(report: RunReadiness, problems, *, before_run: bool = False,
               playback: bool = False, steps: NextSteps = TERMINAL_STEPS) -> str:
    """One concrete step, from the producer's actions where there are problems.

    Before a run, a report without problems ends by saying that generation
    starts, or in a demonstration's playback that the playback starts.
    """
    if len(problems) > 1:
        return f"Next step: fix the {len(problems)} problems above, then {steps.again}."
    if problems:
        _, _, action = problems[0]
        if action:
            return f"Next step: {action}"
        # No fix was reported: point to everything observed rather than guess one.
        return (f"Next step: read every observation {steps.read_detailed}, "
                "then fix the problem and check again.")
    if before_run and report.exit_code == 0 and playback:
        return "The playback starts now. It sends no model request and costs nothing."
    if before_run and report.exit_code == 0:
        return "Generation starts now. It sends model requests, which may cost money."
    if before_run:
        # A nonzero status without a listed problem: show where it comes from, never a start.
        return (f"Next step: read every observation {steps.read_detailed} "
                f"(the setup report ends with exit status {report.exit_code}).")
    if report.scope == "inspection":
        return (f"Next step: {steps.check_tools} (tries the installed tools, no model "
                "requests).")
    return f"Next step: {steps.generate}. It sends model requests, which may cost money."


def _lines(report: RunReadiness, *, full: bool, before_run: bool = False,
           playback: bool = False, steps: NextSteps = TERMINAL_STEPS) -> list[str]:
    if not full:
        return _summary(report, before_run=before_run, playback=playback, steps=steps)
    problems = _words.problem_groups(_words.listed_problems(list(report.items)))
    return (_header(report, steps, before_run=before_run) + _limit_lines(report) + _full_items(report)
            + ["", _next_step(report, problems, before_run=before_run, playback=playback, steps=steps)])


def present_readiness(report: RunReadiness, *, presenter: TerminalPresenter,
                      before_run: bool = False, playback: bool = False,
                      steps: NextSteps = TERMINAL_STEPS) -> None:
    """Write one report through the presenter's diagnostic boundary.

    Compact and plain output summarize what was found, what is installed but
    not tried, what was not tried yet and each problem with its action,
    ending with one next step. The detailed view lists every item with its
    raw names. Statuses, selections and actions are the producer's: nothing is
    checked, installed or inferred here. ``before_run`` marks the summary
    shown before a run's first model request, and ``playback`` a
    demonstration's playback, which names no API key and ends by saying
    that it costs nothing. ``steps`` words the reader's next actions: the
    terminal's commands unless another interface passes its own.
    """
    if not isinstance(report, RunReadiness):
        raise TypeError("report must be a RunReadiness")
    full = getattr(presenter, "requested_mode", presenter.mode) in ("detailed", "debug")
    if type(before_run) is not bool:
        raise TypeError("before_run must be True or False")
    if type(playback) is not bool:
        raise TypeError("playback must be True or False")
    if playback:
        # A playback uses no API key, so the report names none.
        report = replace(report, credentials_configured=None, items=tuple(
            item for item in report.items if item.key != "configuration.credentials"))
    if not isinstance(steps, NextSteps):
        raise TypeError("steps must be NextSteps")
    lines = _lines(report, full=full, before_run=before_run, playback=playback, steps=steps)

    def write() -> None:
        for line in lines:
            presenter._write_plain(line, preserve_whitespace=True)

    presenter.write_diagnostic(write)
