"""Bundled prompt files for each LLM agent.

One JSON per agent: ``test_initial.json``, ``test_coverage.json``,
``test_mutation.json``.

Each file holds three top-level keys:

  - ``system_prompt``: list of lines, joined with "\\n"
  - ``user_prompt_template``: list of lines, ``string.Template``-style
  - ``tool_messages``: dict of message keys (REJECTED / ERROR /
    BLOCKED hint strings the LLM sees from tool returns), each value
    a string or list of lines

Loaded via ``testuary.prompts.load_agent_prompts(name)``, which resolves the
file through ``importlib.resources.files("testuary.prompts")`` so the
lookup works the same whether the package is installed from a wheel,
sdist, or editable install.
"""
from __future__ import annotations

import json
import importlib.resources
from string import Template
from typing import Any, Dict, Optional, Tuple
from testuary import config
from testuary import hints as hints_module
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from testuary._phase_keys import PHASES, phase_key
from testuary.settings import ConfigurationError, _freeze, phase_section




def load_agent_prompts(agent_name: str) -> Dict[str, Any]:
    """Load an agent's prompts from ``testuary/prompts/<agent_name>.json``.

    Resolved via ``importlib.resources.files("testuary.prompts")`` so the
    lookup works the same from an editable install, a wheel, or an
    sdist. The JSON files live in the ``testuary.prompts`` subpackage and
    are shipped as package data (see pyproject.toml).

    Multi-paragraph prompts are stored as a list of lines (joined here
    with "\n"). That avoids the unreadable single-giant-string form that
    pure JSON forces on multi-paragraph prompts. Editing stays
    line-by-line in any editor.

    Top-level keys:
      - ``system_prompt`` (str | list[str]): joined to a string
      - ``user_prompt_template`` (str | list[str]): joined to a string
      - ``tool_messages`` (dict[str, str | list[str]]): each value
        is joined to a string. Callers look up by key and Template-substitute
        runtime values. The structure lets multi-line LLM-facing messages
        live in JSON without forcing every tool to define its own top-level
        key.

    Either of system_prompt / user_prompt_template may be omitted. Unknown
    top-level keys are passed through unchanged (for future extension).
    """
    ref = importlib.resources.files("testuary.prompts") / f"{agent_name}.json"
    with importlib.resources.as_file(ref) as path:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in ("system_prompt", "user_prompt_template"):
            out[key] = _join_lines(value, where=f"{path.name}:{key}")
        elif key == "tool_messages":
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path.name}: 'tool_messages' must be an object, "
                    f"got {type(value).__name__}"
                )
            out[key] = {
                subkey: _join_lines(subval, where=f"{path.name}:tool_messages.{subkey}")
                for subkey, subval in value.items()
            }
        else:
            out[key] = value
    return out


def _join_lines(value: Any, *, where: str) -> str:
    """Coerce a JSON string or list-of-strings into a single string. Used
    by ``load_agent_prompts`` to keep multi-line prompts editable line-by-
    line in JSON without forcing callers to know whether a particular
    prompt is one line or many."""
    if isinstance(value, list):
        return "\n".join(value)
    if isinstance(value, str):
        return value
    raise ValueError(
        f"{where}: must be a string or list of strings, "
        f"got {type(value).__name__}"
    )


# Per-agent "what to read first" protocol. Single-sources the init-state read
# instruction so the editing agents share one phrasing: the file under test
# enters the message history as a dedup-able read instead of being baked into
# the prompt, where the dedup pass cannot reclaim it. Keyed by agent (prompt)
# name -> (read tool, reason). Only the first read is mandated. The model
# chooses any further reads from the documented tool set. A unit test pins
# each tool to the agent's own read set so the spec cannot drift from the
# tools the agent actually binds.
_READ_PROTOCOL: Dict[str, Tuple[str, str]] = {
    "test_coverage": ("read_tests", "load the suite you are extending"),
    "test_mutation": (
        "read_tests", "load the suite you are extending",
    ),
}


def render_read_protocol(agent_name: str) -> str:
    """The single-sourced first-read instruction for an agent, or ``""`` when
    the agent declares none. The graph starts the agent read-only, so the first
    move must be this read. The file then enters as a dedup-able tool return
    rather than prompt text."""
    spec = _READ_PROTOCOL.get(agent_name)
    if spec is None:
        return ""
    tool, reason = spec
    return (
        f"On the first turn, call only ``{tool}`` to {reason}. It is not in "
        f"this prompt. Do not request another tool in the same response; "
        f"wait for this read to return before continuing. Page large files "
        f"with ``start_line`` / ``end_line``."
    )


#: The names resolve_agent_prompts fills beside a language's SLOTS. With
#: SLOTS they are every name a prompt profile may use.
_DERIVED_FRAGMENTS = ("elided_sentinel", "read_protocol", "validity_rule_item",
                      "validity_rule_lead", "builtin_replacement_rule", "language_level",
                      "coverage_counts_inconsistent_cause_line",
                      "coverage_counts_inconsistent_fix_line", "test_file_form_item")


def _builtin_replacement_guidance(agent_name: str) -> bool:
    """Whether the agent's system prompt carries the guidance on replacing
    built-ins: the ``builtin_replacement_guidance``
    setting of the coverage agent, which the mutation agent inherits. The
    starter phase's prompt has no such rule."""
    if agent_name == "test_coverage":
        cfg = config.get_agent_config("test_coverage")
    elif agent_name == "test_mutation":
        cfg = config.get_agent_config("test_mutation", parent="test_coverage")
    else:
        return False
    return cfg["builtin_replacement_guidance"]


#: The message of the strict-green check on unsound asynchronous tests, which
#: only a language whose adapter reads such warnings can reach.
_UNSOUND_MESSAGE = "tests_strict_green_unsound"


#: The replies to coverage counts the coverage tool could not read, which only
#: a language whose vocabulary names their cause can reach.
_COUNTS_MESSAGES = ("measure_coverage_counts_inconsistent",
                    "measure_mutation_counts_inconsistent")


def _offers_unsound_check(backend: Any, source_dir: Optional[str]) -> bool:
    """Whether ``backend``'s language reports unsound-test warnings in its
    runs (its adapter's facts name ``unsound_test_patterns``), read as
    ``_language_level`` reads the facts. Its prompts carry the strict-green
    message only then."""
    read = getattr(backend, "language_facts", None)
    facts = read(source_dir) if callable(read) else None
    return bool(getattr(facts, "unsound_test_patterns", ()))


def _read_profile_input():
    from testuary import languages

    selected = str(config.load_config("main").get("prompt_profile", "")).strip()
    if not selected:
        return None
    templates = {phase: load_agent_prompts(f"test_{phase}") for phase in PHASES}
    fragments = set(SLOTS) | set(_DERIVED_FRAGMENTS)
    return read_profile(selected, templates, base_dir=config.configuration_base_dir(),
                        languages=languages.available_language_names(), fragments=fragments)


def prompt_inputs():
    """Captured profile and hint bytes for origin records, not fresh file reads."""
    run = config.active_run()
    return (run.profile, run.hints) if run is not None else (None, None)


def _unreachable_messages(backend: Any, source_dir: Optional[str]) -> Tuple[str, ...]:
    """The tool messages ``backend``'s language cannot reach, which its
    prompts leave out: the strict-green message where it reports no
    unsound-test warnings, and the replies to coverage counts the coverage
    tool cannot read where its vocabulary names no cause for them (an empty
    cause names one)."""
    unsound = () if _offers_unsound_check(backend, source_dir) else (_UNSOUND_MESSAGE,)
    counts = (() if "lang_coverage_counts_inconsistent_cause" in _own(backend, source_dir)
              else _COUNTS_MESSAGES)
    return unsound + counts


def _language_level(backend, source_dir: Optional[str]) -> Optional[str]:
    """The language version the tests must be written in, as the adapter's
    facts report it (``LanguageFacts.language_level``: Java's release the
    tests compile for, Python's interpreter version without its patch), or
    None when it reports none (a backend without facts included)."""
    read = getattr(backend, "language_facts", None)
    facts = read(source_dir) if callable(read) else None
    level = getattr(facts, "language_level", None)
    return level.strip() if isinstance(level, str) and level.strip() else None


def resolve_agent_prompts(
    agent_name: str, language: str,
    source_dir: Optional[str] = None,
    hint_profile: Optional[str] = None,
    runtime_version: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
    prompt_profile: Optional["PromptProfile"] = None,
) -> Dict[str, Any]:
    """Load an agent's prompts and resolve the slot surfaces.

    The ``$lang_*`` slots are supplied by the active backend's
    ``prompt_fragments()``, the only language-visible surface.

    The prompt JSON is single-sourced (one file per agent, not one
    per language). ``safe_substitute`` leaves the run-time
    placeholders (``$file_path``, ``$metric_block`` and others) for the
    caller's later ``.substitute``.

    Optional model-facing hints, from the extensions the settings name and
    from ``hints.json``, are appended to the system prompt under
    "## Additional guidance" when at least one of the four cells
    (general / per_profile / per_language / per_profile_per_language) is
    non-empty for the active ``(hint_profile, language)`` pair. No hints
    or all-empty cells -> no append, byte-identical to a run without a
    hints file. See ``resolve_hints``.

    ``prompt_profile`` accepts an already validated text-only composition.
    It is applied to active bundled components before fragment rendering and
    hints. It does not alter tool bindings or descriptions.

    Raises ``ValueError`` for an unknown language."""
    from testuary import languages  # lazy: languages -> agents -> config

    raw = load_agent_prompts(agent_name)
    run = config.active_run()
    if prompt_profile is None and run is not None and run.profile is not None:
        prompt_profile = run.profile.profile
    if prompt_profile is not None:
        if not isinstance(prompt_profile, PromptProfile):
            raise TypeError("prompt_profile must be a validated PromptProfile")
        raw = prompt_profile.compose(phase_key(phase_section(agent_name)), language, raw)
    backend = languages.get_backend(language)
    if backend is None:
        raise ValueError(
            f"resolve_agent_prompts: unknown language {language!r} "
            f"(no registered LanguageBackend)"
        )
    fragments = backend.prompt_fragments(source_dir=source_dir)
    # The history-trim sentinel is single-sourced in _helpers. Surface it
    # to the prompts so no message hardcodes the literal marker.
    from testuary.agent._helpers import ELIDED_SENTINEL  # lazy: agents import config
    fragments = {
        **fragments,
        "elided_sentinel": ELIDED_SENTINEL,
        "read_protocol": render_read_protocol(agent_name),
    }
    # The agents' hard rules gain the backend's guidance
    # on replacing built-ins when the setting is on. Off, the prompts stay
    # unchanged byte for byte.
    # The validity rule as a hard rule of its own, after the line before it.
    # A language that states none gets no line.
    validity_rule = fragments.get("lang_validity_rule", "")
    fragments["validity_rule_item"] = f"\n- {validity_rule}" if validity_rule else ""
    fragments["validity_rule_lead"] = f"{validity_rule} " if validity_rule else ""
    # The form of a test file as a requirement of its own, after the line
    # before it. A language that states none gets no line.
    form = fragments.get("lang_test_file_form", "")
    fragments["test_file_form_item"] = f"\n- {form}" if form else ""
    # The cause of coverage counts the coverage tool cannot read and their
    # fix, each on a line of its own after the line before it. A language
    # without them gets no line.
    cause = fragments.get("lang_coverage_counts_inconsistent_cause", "")
    fragments["coverage_counts_inconsistent_cause_line"] = f"\n{cause}" if cause else ""
    fix = fragments.get("lang_coverage_counts_inconsistent_fix", "")
    fragments["coverage_counts_inconsistent_fix_line"] = f"\n{fix}" if fix else ""
    rule = fragments.get("lang_builtin_replacement_rule", "")
    fragments["builtin_replacement_rule"] = (
        f"\n{rule}" if rule and _builtin_replacement_guidance(agent_name)
        else ""
    )

    # ``lang_runtime_version`` defaults to the execution interpreter's version.
    # A caller that knows the runtime the tests will actually be measured under
    # (a file whose tests are generated on one host but run under the
    # target's pinned interpreter) passes it so the model is told the
    # truth, not the host that happens to assemble the prompt.
    if runtime_version:
        fragments = {**fragments, "lang_runtime_version": runtime_version}
    # ``language_level`` names the language the tests must be written in:
    # the caller's runtime version, else the adapter's language level (for
    # Java the release the tests compile for, not the JDK that runs them),
    # else the runtime line, as for an adapter that reports no level.
    fragments["language_level"] = (
        runtime_version
        or _language_level(backend, source_dir)
        or fragments.get("lang_runtime_version", ""))

    # A text filled again later (the user template at each request, a tool
    # message by tool_msg) keeps $$ doubled here, so it writes one dollar
    # sign at its last fill, from the bundled files and a profile alike.
    def _fill(text: str, *, target_template: bool = False) -> str:
        if target_template:
            text = text.replace("$$", "$$$$")
        return Template(text).safe_substitute(fragments)

    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in ("system_prompt", "user_prompt_template"):
            out[key] = _fill(value, target_template=key == "user_prompt_template")
        elif key == "tool_messages":
            unreachable = _unreachable_messages(backend, source_dir)
            out[key] = {k: _fill(v, target_template=True) for k, v in value.items()
                        if k not in unreachable}
        else:
            out[key] = value

    # Optional hints from hints.json. Walk the 4 cells inside the
    # phase block for this agent, concatenate non-empty entries,
    # append under a single heading. Empty hints -> no append,
    # byte-identical to a run without hints.
    hints_text = hints_module.resolve_hints(
        hint_profile, language, phase=phase_key(phase_section(agent_name)), hints=hints,
    )
    if hints_text and "system_prompt" in out:
        out["system_prompt"] = (
            f"{out['system_prompt']}\n\n"
            f"## Additional guidance\n\n{hints_text}"
        )
    return out


_COMPONENTS = {"system_prompt", "user_prompt_template", "system_prompt_sections", "tool_messages"}
_HEADINGS = re.compile(r"^## .+$", re.MULTILINE)


def _placeholders(text):
    """The ``$name`` and ``${name}`` placeholders of ``text`` in order of first
    use, and whether its placeholder syntax is valid: what
    ``Template.get_identifiers`` and ``Template.is_valid`` report from Python
    3.11 on, read with ``Template.pattern``, which 3.10 has too."""
    names, valid = [], True
    for match in Template.pattern.finditer(text):
        name = match.group("named") or match.group("braced")
        if name is not None:
            if name not in names:
                names.append(name)
        elif match.group("invalid") is not None:
            valid = False
    return names, valid


def _entries(value):
    return {key: item for key, item in value.items() if not isinstance(key, str) or not key.startswith("_")}


def _sections(text):
    matches = list(_HEADINGS.finditer(text))
    return [(match.group(), match.start(), matches[index + 1].start() if index + 1 < len(matches) else len(text))
            for index, match in enumerate(matches)]


def _overlay(general, specific):
    return {**general, **specific}


@dataclass(frozen=True, repr=False)
class PromptProfile:
    """Validated prompt operations per phase, general and per language, held
    as frozen copies. Composition never mutates the supplied templates."""
    name: str
    _general: Mapping = field(repr=False)
    _languages: Mapping = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_general", _freeze(self._general))
        object.__setattr__(self, "_languages", _freeze(self._languages))

    def operations(self, phase: str, language: str) -> Mapping:
        if phase not in PHASES:
            raise ValueError("unknown prompt phase")
        return _freeze(_overlay(self._general.get(phase, {}), self._languages.get(phase, {}).get(language, {})))

    def changes(self, phase: str, language: str) -> tuple[tuple[str, str, str], ...]:
        return tuple((kind, key, operation) for kind, entries in self.operations(phase, language).items()
                     for key, (operation, _) in entries.items())

    def compose(self, phase: str, language: str, templates: Mapping[str, Any]) -> dict[str, Any]:
        operations = self.operations(phase, language)
        output = {key: dict(value) if isinstance(value, Mapping) else value for key, value in templates.items()}
        for kind, entries in operations.items():
            if kind in {"system_prompt_sections", "system_prompt"}:
                continue
            for key, (operation, text) in entries.items():
                if kind not in output or (kind == "tool_messages" and key not in output[kind]):
                    raise ConfigurationError(f"profile component {kind}.{key} is unavailable under the current settings")
                current = output[kind][key] if kind == "tool_messages" else output[kind]
                changed = text if operation == "replace" else f"{current}\n\n{text}"
                if kind == "tool_messages":
                    output[kind][key] = changed
                else:
                    output[kind] = changed
        system_op, system_text = operations.get("system_prompt", {}).get("", (None, ""))
        if system_op == "replace":
            output["system_prompt"] = system_text
        section_ops = operations.get("system_prompt_sections", {})
        if section_ops:
            text = output["system_prompt"]
            available = {heading for heading, _, _ in _sections(text)}
            if section_ops.keys() - available:
                raise ConfigurationError("profile section is unavailable in the current bundled system prompt")
            for heading, start, end in reversed(_sections(text)):
                if heading not in section_ops:
                    continue
                operation, content = section_ops[heading]
                if operation == "remove":
                    changed = ""
                elif operation == "replace":
                    changed = content
                else:
                    changed = f"{text[start:end]}\n\n{content}"
                if changed and end < len(text) and not changed.endswith("\n"):
                    changed += "\n"
                text = text[:start] + changed + text[end:]
            output["system_prompt"] = text
        if system_op == "append":
            output["system_prompt"] = f"{output['system_prompt']}\n\n{system_text}"
        return output


def validate_profile(
    value: Any, templates: Mapping[str, Mapping[str, Any]], *,
    languages: Collection[str], fragments: Collection[str] = (),
) -> PromptProfile:
    """Validate all phases against the bundled, unrendered components.

    The caller supplies the registered languages and the bundled templates.
    An entry for a tool message that a language does not send validates, and
    the prompts composed for that language leave it out. Fragment names are
    the placeholders the prompts fill. An entry may use the placeholders of
    the component it changes and the fragment names.
    Errors name paths and allowed fields, never the supplied text.
    """
    problems = []
    if not isinstance(value, dict):
        raise ConfigurationError("prompt profile must contain a JSON object")
    root = _entries(value)
    for key in root:
        if key not in {"format", "name", "phases"}:
            problems.append(f"profile: unknown key {key!r}")
    if type(root.get("format")) is not int or root.get("format") != 1:
        problems.append("profile.format must be 1")
    if "name" in root and not isinstance(root["name"], str):
        problems.append("profile.name must be text")
    phases = root.get("phases")
    if not isinstance(phases, dict):
        problems.append("profile.phases must be an object")
        phases = {}
    general, localized = {}, {}

    def operation(entry, path, original, *, section=False):
        if not isinstance(entry, dict):
            problems.append(f"{path}: entry must be an object with one operation")
            return None
        entry = _entries(entry)
        if len(entry) != 1:
            problems.append(f"{path}: entry needs exactly one operation")
            return None
        op, raw = next(iter(entry.items()))
        if op not in ({"replace", "append", "remove"} if section else {"replace", "append"}):
            problems.append(f"{path}: unknown or unsupported operation {op!r}")
            return None
        if op == "remove":
            if raw is not True:
                problems.append(f"{path}: remove must be true")
                return None
            return op, ""
        if isinstance(raw, list) and all(isinstance(line, str) for line in raw):
            raw = "\n".join(raw)
        if not isinstance(raw, str):
            problems.append(f"{path}: {op} must be text or a list of text lines")
            return None
        allowed = set(_placeholders(original)[0]).union(fragments)
        names, valid = _placeholders(raw)
        if not valid:
            problems.append(f"{path}: malformed placeholder syntax. Allowed: {', '.join(sorted(allowed))}")
        for name in names:
            if name not in allowed:
                problems.append(f"{path}: unknown placeholder ${name}. Allowed: {', '.join(sorted(allowed))}")
        return op, raw

    def block(raw, path, bundled):
        if not isinstance(raw, dict):
            problems.append(f"{path}: phase/language block must be an object")
            return {}
        out = {}
        for kind, entries in _entries(raw).items():
            if kind not in _COMPONENTS:
                problems.append(f"{path}: unknown component {kind!r}")
                continue
            if kind in {"system_prompt", "user_prompt_template"}:
                original = bundled.get(kind)
                if not isinstance(original, str):
                    problems.append(f"{path}.{kind}: component is unavailable")
                    continue
                result = operation(entries, f"{path}.{kind}", original)
                if result is not None:
                    out[kind] = {"": result}
                continue
            if not isinstance(entries, dict):
                problems.append(f"{path}.{kind}: must be an object of component entries")
                continue
            originals = bundled.get("tool_messages", {}) if kind == "tool_messages" else {
                heading: bundled["system_prompt"][start:end]
                for heading, start, end in _sections(bundled.get("system_prompt", ""))
            }
            for key, entry in _entries(entries).items():
                if key not in originals:
                    problems.append(f"{path}.{kind}: unknown or disabled component {key!r}. Allowed: {', '.join(originals)}")
                    continue
                if kind == "system_prompt_sections" and sum(
                    heading == key for heading, _, _ in _sections(bundled["system_prompt"])
                ) > 1:
                    problems.append(f"{path}.{kind}: ambiguous repeated heading {key!r}")
                    continue
                result = operation(entry, f"{path}.{kind}.{key}", originals[key], section=kind == "system_prompt_sections")
                if result is not None:
                    out.setdefault(kind, {})[key] = result
        return out

    def flatten(operations):
        return {(kind, key): entry for kind, entries in operations.items() for key, entry in entries.items()}

    def expand(operations):
        result = {}
        for (kind, key), entry in operations.items():
            result.setdefault(kind, {})[key] = entry
        return result

    def conflict(operations, path):
        if operations.get("system_prompt", {}).get("", (None,))[0] == "replace" and operations.get("system_prompt_sections"):
            problems.append(f"{path}: system_prompt replace cannot be combined with section operations")

    for phase, raw in _entries(phases).items():
        if phase not in PHASES:
            problems.append(f"profile.phases: unknown phase {phase!r}")
            continue
        if not isinstance(raw, dict):
            problems.append(f"profile.phases.{phase}: must be an object")
            continue
        bundled = templates[phase]
        path = f"profile.phases.{phase}"
        general[phase] = block({key: item for key, item in raw.items() if key != "per_language"}, path, bundled)
        conflict(general[phase], path)
        per_language = raw.get("per_language", {})
        if not isinstance(per_language, dict):
            problems.append(f"{path}.per_language: must be an object")
            continue
        for language, entries in _entries(per_language).items():
            if language not in languages:
                problems.append(f"{path}.per_language: unregistered language {language!r}")
                continue
            local = block(entries, f"{path}.per_language.{language}", bundled)
            merged = expand({**flatten(general[phase]), **flatten(local)})
            conflict(merged, f"{path}.per_language.{language}")
            localized.setdefault(phase, {})[language] = merged
    if problems:
        raise ConfigurationError("invalid prompt profile:\n" + "\n".join(problems))
    return PromptProfile(root.get("name", ""), general, localized)


@dataclass(frozen=True, repr=False)
class ProfileSource:
    """Validated composition and the exact input bytes read for this run."""

    path: Path
    contents: bytes = field(repr=False)
    profile: PromptProfile = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("profile source path must be absolute")
        if not isinstance(self.contents, bytes):
            raise TypeError("profile source contents must be immutable bytes")
        if not isinstance(self.profile, PromptProfile):
            raise TypeError("profile source needs a validated PromptProfile")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.contents).hexdigest()


def read_profile(
    path: str | Path, templates: Mapping[str, Mapping[str, Any]], *,
    base_dir: str | Path, languages: Collection[str], fragments: Collection[str] = (),
) -> ProfileSource:
    """Read once against an explicit base. Refuse malformed or invalid input.

    The templates are the bundled ones, validated as validate_profile says.
    No ambient selection, rendering, result write or reuse decision occurs here.
    """
    selected = Path(path)
    if not selected.is_absolute():
        base = Path(base_dir)
        if not base.is_absolute():
            raise ConfigurationError("prompt profile base_dir must be absolute")
        selected = base / selected
    try:
        contents = selected.read_bytes()
    except FileNotFoundError:
        raise ConfigurationError(f"selected prompt profile is absent: {selected}") from None
    except OSError:
        raise ConfigurationError(f"cannot read prompt profile: {selected}") from None

    def refuse_constant(_value):
        raise ValueError("non-finite JSON number")

    try:
        value = json.loads(contents.decode("utf-8-sig"), parse_constant=refuse_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ConfigurationError(f"{selected}: prompt profile must contain valid UTF-8 JSON") from None
    profile = validate_profile(value, templates, languages=languages, fragments=fragments)
    return ProfileSource(selected, contents, profile)


#: Every slot a language fills. The bundled templates read most of them,
#: and a prompt profile may name each one.
SLOTS = (
    "lang_name", "lang_code_fence", "target_code_fence", "lang_test_framework",
    "lang_valid_source",
    "lang_real_source", "lang_test_decl", "lang_test_decl_full",
    "lang_test_collection_rule", "lang_fixture_term", "lang_parametrize_term",
    "lang_single_file_rule", "lang_coverage_test_file", "lang_mutation_test_file",
    "lang_import_rule", "lang_collection_error", "lang_builtin_replacement_rule",
    "lang_exception_assert_term", "lang_timeout_decorator", "lang_parse_error_hints",
    "lang_runtime_version", "lang_available_test_deps", "lang_validity_rule",
    "lang_stands_alone_advice", "lang_load_error_patterns",
    "lang_coverage_counts_inconsistent_cause", "lang_coverage_counts_inconsistent_fix",
    "lang_test_file_form",
)

#: The slot each test vocabulary entry fills.
ENTRY_SLOTS = {
    "valid_source": ("lang_valid_source",),
    "real_source": ("lang_real_source",),
    "test_declaration": ("lang_test_decl",),
    "test_declaration_full": ("lang_test_decl_full",),
    "collection_rule": ("lang_test_collection_rule",),
    "fixture_term": ("lang_fixture_term",),
    "parametrize_term": ("lang_parametrize_term",),
    "single_file_rule": ("lang_single_file_rule",),
    "test_file": ("lang_coverage_test_file", "lang_mutation_test_file"),
    "import_rule": ("lang_import_rule",),
    "load_error": ("lang_collection_error",),
    "fake_builtin_rule": ("lang_builtin_replacement_rule",),
    "exception_assertion": ("lang_exception_assert_term",),
    "timeout_marker": ("lang_timeout_decorator",),
    "parse_error_hints": ("lang_parse_error_hints",),
    "test_dependencies": ("lang_available_test_deps",),
    "validity_rule": ("lang_validity_rule",),
    "stands_alone_advice": ("lang_stands_alone_advice",),
    "load_error_patterns": ("lang_load_error_patterns",),
    "coverage_counts_inconsistent_cause": ("lang_coverage_counts_inconsistent_cause",),
    "coverage_counts_inconsistent_fix": ("lang_coverage_counts_inconsistent_fix",),
    "test_file_form": ("lang_test_file_form",),
}

#: The per-test timeout in seconds, which the lang_timeout_decorator slot
#: names.
TIMEOUT_SECONDS = "120"


def _own(backend, source_dir) -> Dict[str, str]:
    """The language's test vocabulary for the project, as prompt slots."""
    project = backend.project_backend(source_dir)
    read = getattr(project, "test_vocabulary", None)
    vocabulary = dict(read() if callable(read) else {})
    slots: Dict[str, str] = {}
    for entry, text in vocabulary.items():
        if entry == "timeout_marker":
            text = text.replace("{seconds}", TIMEOUT_SECONDS)
        for slot in ENTRY_SLOTS.get(entry, ()):
            slots[slot] = text
    return slots


def runtime_line(facts, name: str, *, reads_version: bool = True) -> str:
    """The runtime line the prompts carry: ``[project] runtime_version`` when
    it is set, else the runtime the adapter reads, named with the language
    unless its line names it already (Python's ``--version`` line does).
    A version the adapter reads but could not read is ``(unknown)`` after
    the language's name, such as ``Python (unknown)``. A language
    whose adapter reads no runtime version (``reads_version`` false) is
    named alone."""
    from testuary import config

    stated = str(config.get_project_config().get("runtime_version") or "").strip()
    if stated:
        return stated
    runtime = str(getattr(facts, "runtime_version", None) or "").strip()
    if not runtime:
        return f"{name} (unknown)" if reads_version else name
    return runtime if runtime.startswith(name) else f"{name} {runtime}"


def _reads_runtime_version(backend, source_dir) -> bool:
    """Whether the language's adapter reads its runtime's version
    (``runtime_version``, None when it cannot be read)."""
    project = backend.project_backend(source_dir)
    return callable(getattr(project, "runtime_version", None))


def fragments(backend, source_dir: Optional[str] = None) -> Dict[str, str]:
    """The prompt slots for ``backend``'s language: its own vocabulary,
    completed from its facts."""
    facts = backend.language_facts(source_dir)
    # The tests' language names the prompts (Python for Python tests of a C
    # file). The target's source keeps its own fence.
    name = (backend._test_fact("display_name", source_dir=source_dir)
            or backend.name.capitalize())
    fence = (getattr(facts, "code_fence", None)
             or backend._test_fact("code_fence", source_dir=source_dir) or backend.name)
    framework = (getattr(facts, "test_framework", None)
                 or backend._test_fact("test_framework", source_dir=source_dir)
                 or "the test framework")
    declaration = getattr(facts, "test_function_pattern", None) or "a test"
    test_file = backend._test_file_name("<target_stem>", source_dir)
    timeout = (getattr(facts, "timeout_marker", None) or "").replace(
        "{seconds}", TIMEOUT_SECONDS)
    generic = {
        "lang_name": name,
        "lang_code_fence": fence,
        "target_code_fence": getattr(facts, "target_code_fence", None) or fence,
        "lang_test_framework": framework,
        "lang_valid_source": f"valid {name}",
        "lang_real_source": f"real {name}",
        "lang_test_decl": f"``{declaration}``",
        "lang_test_decl_full": f"test written as ``{declaration}``",
        "lang_test_collection_rule": (
            f"{framework} runs the tests written as ``{declaration}`` in the "
            "test file."),
        "lang_fixture_term": "shared setup",
        "lang_parametrize_term": "a test repeated over several inputs",
        "lang_single_file_rule": (
            f"Produce one test file, ``{test_file}``. Do not create helper files."),
        "lang_coverage_test_file": f"``{test_file}``",
        "lang_mutation_test_file": f"``{test_file}``",
        "lang_import_rule": "uses the target the way the language's tests do",
        "lang_collection_error": "load error",
        "lang_builtin_replacement_rule": "",
        "lang_exception_assert_term": "an assertion that the call fails",
        "lang_timeout_decorator": timeout,
        "lang_parse_error_hints": (
            "unbalanced brackets or quotes, a statement outside a test, "
            "accidental code-fence remnants"),
        "lang_runtime_version": runtime_line(
            facts, name, reads_version=_reads_runtime_version(backend, source_dir)),
        "lang_available_test_deps": (
            f"Test toolkit (always available):\n  - {framework}\n"
            "The project's own code is also available; other libraries are "
            "available only if the project declares them."),
        # What a test may and may not read of the target: the language's
        # validity policy in its own terms (``validity_rule``). A language
        # without one states no rule.
        "lang_validity_rule": "",
        # What to do about tests that leave shared state changed, and the
        # common causes of a suite that does not load, in the language's own
        # terms. A language without them gives no list.
        "lang_stands_alone_advice": "",
        "lang_load_error_patterns": "",
        # The usual cause of coverage counts the coverage tool cannot read,
        # and what to change in such a test, in the language's own terms. A
        # language whose coverage reports no such counts gives none.
        "lang_coverage_counts_inconsistent_cause": "",
        "lang_coverage_counts_inconsistent_fix": "",
        # What a test file holds beyond its tests, such as its own entry
        # point, in the language's own terms. A language whose test files
        # need no such form gives none.
        "lang_test_file_form": "",
    }
    own = _own(backend, source_dir)
    return {slot: own.get(slot, generic[slot]) for slot in SLOTS}
