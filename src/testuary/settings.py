"""Validated, immutable configuration and ordered INI resolution."""
from __future__ import annotations

import argparse
import configparser
import contextlib
import json
import math
import re
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import get_close_matches
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Dict, Generic, Iterator, Optional, Callable, List, Tuple, TypeVar
import importlib.resources
import stat

from testuary._words import name_words


Sections = Mapping[str, Mapping[str, Any]]

#: The config.ini section of each phase id ("timing": the suite timing).
#: Code names phases by id or agent name (AGENT_NAMES). Only this mapping
#: names sections.
PHASE_SECTIONS = MappingProxyType({
    "starter": "starter", "coverage": "coverage", "mutation": "mutation",
    "finishing": "finishing", "timing": "timing",
})

#: Each phase's agent name by phase id. Agent names, the prompt files and
#: the hints' phase keys (hints.PHASES) keep their names.
AGENT_NAMES = MappingProxyType({
    "starter": "test_initial", "coverage": "test_coverage", "mutation": "test_mutation",
    "finishing": "test_metrics",
})
_AGENT_PHASES = {agent: phase for phase, agent in AGENT_NAMES.items()}


#: Variables Testuary reads for itself, which set no setting even where
#: their name begins with a section's.
ENVIRONMENT_ONLY = frozenset({
    "TESTUARY_HOME", "TESTUARY_CONFIG_FILE", "TESTUARY_EXECUTION_ID", "TESTUARY_CLOCK_DOMAIN_ID",
    "TESTUARY_MAIN_WORK_DIR_NAMESPACE",
    "TESTUARY_PROJECT_CONTAINER", "TESTUARY_PROJECT_CONTAINER_RUNNER", "TESTUARY_PROJECT_WORK_DIR",
})

#: The configuration's sections of the built-in providers.
PROVIDER_SECTIONS = ("openai", "ollama", "vllm")


def credential_variable(provider: str) -> str:
    """The environment variable that holds ``provider``'s API key when its
    catalogue entry names none (api_key_env)."""
    return f"TESTUARY_{provider.upper()}_API_KEY"


def phase_section(name: str) -> str:
    """The config.ini section of a phase id or agent name. Any other section's
    name unchanged."""
    return PHASE_SECTIONS.get(_AGENT_PHASES.get(name, name), name)


_Installed = TypeVar("_Installed")


class Slot(Generic[_Installed]):
    """A process-wide slot for a stand-in that serves in place of a
    configured service (the demonstration's model and cover sources).
    Installing over another stand-in is refused, so two callers cannot
    silently replace each other."""

    def __init__(self, name: str) -> None:
        self._name, self._value = name, None

    def install(self, value: _Installed) -> None:
        """Install ``value`` for this process."""
        if self._value is not None:
            raise RuntimeError(f"a {self._name} is already installed")
        self._value = value

    def uninstall(self) -> None:
        """Empty the slot."""
        self._value = None

    def current(self) -> Optional[_Installed]:
        """The installed stand-in, or None in a normal run."""
        return self._value

    @contextlib.contextmanager
    def installed(self, value: _Installed) -> Iterator[_Installed]:
        """Install ``value`` for the duration of the block."""
        self.install(value)
        try:
            yield value
        finally:
            self.uninstall()


class ConfigurationError(ValueError):
    """Configuration was refused before a command started."""


class _MissingSetup(ConfigurationError):
    """A run was refused before its work: a selected language lacks a tool
    or an operation the configured workflow requires."""


@dataclass(frozen=True)
class MainSettings:
    """The ``[main]`` section, typed: its values and their defaults are the
    resolved section's (config.ini and the layers over it), its choices
    ``_CHOICES``'."""

    seed: int
    context_rebuild: str
    context_rebuild_trigger_tokens: int
    phase_end_mutation: bool
    reduced_suite_mutation: str
    work_dir_job_namespace: str
    project_exec: str
    pytest_config: str
    strict_canonical_git: bool
    hints_file: str
    prompt_profile: str
    testuary_dir: str


def _freeze(values):
    if isinstance(values, Mapping):
        if any(not isinstance(key, str) for key in values):
            raise TypeError("settings mapping keys must be strings")
        return MappingProxyType({key: _freeze(value) for key, value in values.items()})
    if isinstance(values, (list, tuple)):
        return tuple(_freeze(value) for value in values)
    if isinstance(values, (str, bool, int, float, type(None))):
        return values
    raise TypeError("settings values must be mappings, sequences or scalars")


def _defaults() -> dict[str, dict[str, str]]:

    path = _bundled_data_path("config.ini")
    if path is None:
        raise RuntimeError("the installed package is missing config.ini")
    defaults = _read_ini_file(path)
    defaults.setdefault("project", {}).update(_language_project_settings(defaults))
    return defaults


def _language_project_settings(defaults: Sections) -> dict[str, str]:
    """The ``[project]`` settings and defaults the installed languages
    declare (LangAdapter's ``project_settings``). One Testuary already holds
    keeps its own section and default."""
    from langadapter import names, registration

    held = {key for values in defaults.values() for key in values}
    declared: dict[str, str] = {}
    for name in names():
        for setting in getattr(registration(name).factory, "project_settings", ()) or ():
            if setting.name not in held:
                declared.setdefault(setting.name, setting.default or "")
    return declared


_MODEL_FIELDS = {
    "temperature": "", "top_p": "1.0", "seed": "0",
    "max_output_tokens": "0", "request_timeout_seconds": "3600",
    "reasoning": "", "reasoning_effort": "", "reasoning_budget_tokens": "1",
    "protocol": "", "preserve_reasoning_content": "true",
    "reasoning_field_names": "", "request_options": "",
}

_PROVIDER_FIELDS = {**_MODEL_FIELDS, "reasoning_dialect": "", "reasoning_template": "",
                    "chat_model": "", "deployment": ""}

_CHOICES = {
    "context_rebuild": ("mode_change", "every_turn"),
    "reduced_suite_mutation": ("derive", "measure", "both"),
    "work_dir_job_namespace": ("auto", "always", "never"),
    "project_exec": ("local", "container"),
    "phase_scores": ("full", "coverage", "off"),
    "goal_metric": ("line", "branch", "both", "either"),
    "reduction_solver": ("exact", "greedy"),
    "reduction_tie_break": ("name_order", "solver"),
    "loop_catalogue": ("full", "selective"),
    "cover_attribution": ("first", "full"),
    "frozen_module_strategy": ("unfreeze", "skip"),
    # The Python backend's distributors (langadapter.python.cosmic_ray).
    "cosmic_ray_distributor": ("auto", "local", "http"),
    "suite_mutation": ("measure", "derive", "both"),
    # [web] access: who may use the browser interface.
    "access": ("open", "key", "stable_key"),
}

#: Sections that configure no run: resolved, validated and shown like the
#: others, but left out of a run's records (``RunSettings.export``).
UNRECORDED_SECTIONS = frozenset({"web"})

_OPTIONAL_NUMBERS = {
    "temperature", "top_p", "seed", "reasoning_budget_tokens",
    "reasoning_grace_seconds", "max_output_tokens",
}

_FLOAT_FIELDS = {
    "setup_timeout_seconds", "setup_probe_timeout_seconds",
    "request_timeout_seconds", "max_wall_clock_seconds_per_phase", "temperature", "top_p",
    "reasoning_grace_seconds", "transient_first_wait_seconds",
    "max_transient_wait_seconds", "max_transient_hint_seconds",
    "reduction_solver_time_limit_seconds", "reduction_min_test_time_ms",
    "cosmic_ray_clone_sweep_hours",
    "per_mutant_timeout_factor", "max_per_mutant_timeout_factor",
}

_POSITIVE_FIELDS = {
    "setup_timeout_seconds", "setup_probe_timeout_seconds",
    "concurrent_requests", "concurrent_requests_ollama", "suite_count", "attempts_per_suite",
    "suite_run_timeout_seconds", "coverage_timeout_seconds", "run_count",
    "engine_workers", "request_timeout_seconds", "reasoning_budget_tokens", "max_consecutive_transient_failures",
    "reduction_solver_time_limit_seconds", "cosmic_ray_init_timeout_s",
    "cosmic_ray_worker_ready_timeout_s",
    "measurement_timeout_seconds", "min_per_mutant_timeout_seconds",
    "failing_test_cleanup_rounds",
}

_PERCENT_SHARE = re.compile(r"(\d+(?:\.\d+)?)%")

#: Run k's suite i sends seed_base + 1000*k + i (config.run_seed), so a run
#: requests at most this many suites.
RUN_SEED_STRIDE = 1000


def percent_share(text: str) -> Fraction | None:
    """A max_measurements_without_improvement given as a share of the request limit, such as
    10%, as an exact fraction from 0 to 1, or None for any other text."""
    match = _PERCENT_SHARE.fullmatch(str(text).strip())
    if match is None:
        return None
    share = Fraction(match.group(1)) / 100
    return share if share <= 1 else None


_NONNEGATIVE_FIELDS = {
    "max_output_tokens", "max_input_tokens_per_phase",
    "max_output_tokens_per_phase", "max_total_tokens_per_phase",
    "max_wall_clock_seconds_per_phase", "warmup_count", "max_loop_attempts",
    "max_transient_failures_per_phase", "transient_first_wait_seconds",
    "max_transient_wait_seconds", "max_transient_hint_seconds",
    "temperature", "top_p", "reasoning_grace_seconds",
    "context_rebuild_trigger_tokens", "reduction_min_test_time_ms",
}

#: Scores and thresholds in percent: from 0 to 100.
_PERCENT_FIELDS = {
    "goal", "minimum",
    "selective_switch_threshold",
}

#: Limits that also accept ``unlimited``, for the token and time caps 0.
_LIMIT_FIELDS = {
    "max_requests_per_phase", "max_loop_attempts", "max_input_tokens_per_phase",
    "max_output_tokens_per_phase", "max_total_tokens_per_phase", "max_wall_clock_seconds_per_phase",
}

#: [coverage] keys only the coverage phase reads: [starter] refuses them,
#: [mutation] those it does not hold itself.
_COVERAGE_ONLY_FIELDS = {
    "goal", "goal_metric", "minimum", "restore_count",
    "extra_measurements_after_goal", "complete_tests_below_minimum",
}

#: [coverage] keys of the agents' loop and limits. [starter], with no
#: agent loop, refuses them.
_AGENT_LOOP_FIELDS = {
    "builtin_replacement_guidance", "count_repeated_measurements",
    "allow_early_stop", "enable_trace_log", "count_retries", "report_reverse_order_failures",
    "max_total_tokens_per_phase", "max_failed_test_attempts", "max_input_tokens_per_phase",
    "max_output_tokens_per_phase", "max_loop_attempts", "phase_end_evidence_reuse",
    "progress_heartbeat_requests", "max_requests_per_phase", "min_improvement_points",
    "max_measurements_without_improvement",
    "suite_timeout_stop_after", "max_final_answers_without_measurement",
    "max_wall_clock_seconds_per_phase",
}


def _schema(defaults: Sections) -> dict[str, dict[str, Any]]:
    schema = {name: dict(values) for name, values in defaults.items()}
    for name in ("api", "openai", "ollama", "vllm"):
        schema[name].update(_PROVIDER_FIELDS)
    # A model's service fields are read from its provider's section, the
    # catalogue's provider section or its alias, never from [api].
    for key in ("chat_model", "reasoning_dialect", "reasoning_template"):
        del schema["api"][key]
    schema["api"]["api_url"] = ""
    starter, coverage, mutation = (PHASE_SECTIONS[phase] for phase in ("starter", "coverage", "mutation"))
    for name in (starter, coverage, mutation):
        schema[name].update(schema[coverage])
        schema[name].update(_MODEL_FIELDS)
    # The first number of the starter suites' request seeds.
    schema[starter]["seed_base"] = "0"
    for key in _COVERAGE_ONLY_FIELDS | _AGENT_LOOP_FIELDS:
        schema[starter].pop(key)
    for key in _COVERAGE_ONLY_FIELDS - set(defaults[mutation]):
        schema[mutation].pop(key)
    # The phase sections of the limits take the same keys as [limits].
    for name in ("limits_coverage", "limits_mutation"):
        schema[name] = dict(schema["limits"])
    return schema


def _nearest(label: str, choices: Sequence[str]) -> Optional[str]:
    """The known name nearest ``label``: the one that reads its "folder" as
    "dir", as Testuary's names say it, else the closest by spelling."""
    as_dir = label.replace("folder", "dir").replace("FOLDER", "DIR")
    if as_dir != label and as_dir in choices:
        return as_dir
    close = get_close_matches(label, choices, n=1, cutoff=0.55)
    return close[0] if close else None


def _unknown(source: str, label: str, choices: Sequence[str], kind: str = "setting") -> ValueError:
    near = _nearest(label, choices)
    suggestion = f". The nearest known {kind} is {near!r}" if near else ""
    return ValueError(f"{source}: unknown {kind} {label!r}{suggestion}")


class _OptionParser(argparse.ArgumentParser):
    """An argument parser whose refusal of an unknown option names the
    nearest known one, as an unknown setting's refusal does."""

    def error(self, message: str):
        prefix = "unrecognized arguments: "
        options = tuple(self._option_string_actions)
        for token in message[len(prefix):].split() if message.startswith(prefix) else ():
            near = _nearest(token.split("=", 1)[0], options) if token.startswith("--") else None
            if near:
                message = f"{message}. The nearest known option is {near!r}"
                break
        super().error(message)


def _split_extensions(text: str, label: str = "[main] extensions") -> list[str]:
    """The items of ``[main] extensions``, each trimmed. The check is
    syntactic and loads nothing: an item is not empty and holds no line
    break, which a settings file cannot keep in a value. An empty text names
    no item."""
    items = [item.strip() for item in text.split(",")] if text.strip() else []
    if any(not item or item.splitlines() != [item] for item in items):
        raise ValueError(f"{label}: expected a comma list of files and modules, with no empty "
                         "item and no line break")
    return items


def _numeric(key: str, default: Any) -> bool:
    """Whether ``key`` holds a number: its default is one, or it is an
    optional number, whose blank value leaves it unset."""
    return str(default).replace(".", "", 1).lstrip("-").isdigit() or key in _OPTIONAL_NUMBERS


def _value(section: str, key: str, value: Any, default: Any) -> str:
    label = f"[{section}] {key}"
    if value is None:
        raise ValueError(f"{label}: None is not a setting value. Omit the field instead")
    if not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{label}: expected a scalar setting")
    text = str(value).strip()
    if section in ("limits", "limits_coverage", "limits_mutation"):
        from testuary.agent import _limits

        return _limits.parse(key, value, source=f"[{section}]")
    if key == "reasoning_field_names":
        from testuary.models import field_names
        return ",".join(field_names(text))
    if key == "reasoning":
        if text not in ("", "on", "off"):
            raise ValueError(f"{label}: choose on, off or empty")
        return text
    if key == "reasoning_effort" and (not isinstance(value, str) or text == "0"):
        raise ValueError(f"{label}: expected provider effort text or empty, not zero or a boolean")
    if key == "request_timeout_seconds" and not text:
        raise ValueError(f"{label}: expected a positive value")
    if key in _LIMIT_FIELDS:
        # A request limit of zero allows no request: refused, naming unlimited.
        if text.lower() == "unlimited":
            return "unlimited"
        if key == "max_requests_per_phase":
            try:
                valid = int(text) > 0
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"{label}: expected a positive integer, or unlimited for no limit")
            return text
    if section == "project" and key == "language" and text:
        from langadapter import names

        if text.lower() not in names():
            raise ValueError(f"{label}: {text} is not installed. Choose one of {', '.join(names())}")
        return text.lower()
    if section == "main" and key == "testuary_dir":
        # Strictly below the project directory: not "." and not "~".
        parts = text.replace("\\", "/").split("/")
        if (PurePosixPath(text).is_absolute() or PureWindowsPath(text).anchor
                or text.startswith("~") or any(part in ("", ".", "..") for part in parts)):
            raise ValueError(f"{label}: expected a directory below the project directory, such as .testuary")
        return text
    if section == "main" and key == "extensions":
        return ", ".join(_split_extensions(text, label))
    if str(default).lower() in ("true", "false"):
        states = configparser.ConfigParser.BOOLEAN_STATES
        if text.lower() not in states:
            raise ValueError(f"{label}: expected true or false")
        return "true" if states[text.lower()] else "false"
    if key in _CHOICES:
        if text.lower() not in _CHOICES[key]:
            raise ValueError(f"{label}: choose one of {', '.join(_CHOICES[key])}")
        return text.lower()
    if key == "max_measurements_without_improvement" and text.endswith("%"):
        if percent_share(text) is None:
            raise ValueError(f"{label}: expected a whole number, or a share of max_requests_per_phase "
                             "from 0% to 100%")
        return text
    if key == "per_mutant_timeout_seconds":
        if text.lower() == "auto":
            return "auto"
        try:
            valid = int(text) > 0
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"{label}: expected auto or a positive integer")
        return text
    numeric = _numeric(key, default)
    if key in _OPTIONAL_NUMBERS and not text:
        return ""
    if numeric:
        cast = float if key in _FLOAT_FIELDS or "." in str(default) else int
        expected = f"{label}: expected {'a number' if cast is float else 'a whole number'}"
        try:
            number = cast(text)
        except (ValueError, OverflowError) as exc:
            raise ValueError(expected) from exc
        if isinstance(value, bool) or (isinstance(number, float) and not math.isfinite(number)):
            raise ValueError(expected)
        if key in _POSITIVE_FIELDS and number <= 0:
            if key == "reasoning_budget_tokens":
                raise ValueError(f"{label}: expected a positive value. Use reasoning = off to disable reasoning")
            raise ValueError(f"{label}: expected a positive value")
        if key in _NONNEGATIVE_FIELDS and number < 0:
            raise ValueError(f"{label}: expected a nonnegative value")
        if key in _PERCENT_FIELDS and not 0 <= number <= 100:
            raise ValueError(f"{label}: expected a percentage from 0 to 100")
        if key == "suite_count" and number > RUN_SEED_STRIDE:
            raise ValueError(f"{label}: expected at most {RUN_SEED_STRIDE} suites, "
                             f"the starter suites one run may generate")
        if key == "top_p" and number > 1:
            raise ValueError(f"{label}: expected a value between zero and one")
        if section == "web" and key == "port" and not 0 <= number <= 65535:
            raise ValueError(f"{label}: expected a port from 0 to 65535")
    return text


def _validate(layer: Sections, defaults: Sections, source: str,
              unchecked: list[str] | None = None) -> dict[str, dict[str, str]]:
    """The layer's values as a run takes them. A refusal raises ValueError
    whose ``setting`` is its (section, key), the key None for a section.
    With ``unchecked``, a ``[project]`` key no registered language declares
    and a ``[project] language`` no registered language has are kept as
    written and named there, not refused: for settings that name extensions
    which are not loaded and may add them."""
    schema = _schema(defaults)
    normalized = {}
    for section, values in layer.items():
        if section not in schema:
            raise _naming(_unknown(source, section, tuple(schema), "section"), section, None)
        if not isinstance(values, Mapping):
            raise _naming(ValueError(f"{source}: [{section}] must be a settings mapping"), section, None)
        normalized[section] = {}
        for key, value in values.items():
            if key == "api_key":
                raise _naming(ValueError(
                    f"{source}: [{section}] api_key belongs in the private model catalogue "
                    "or a supported environment variable, not configuration settings"
                ), section, key)
            open_key = unchecked is not None and section == "project" and isinstance(value, str)
            if key not in schema[section]:
                if open_key:
                    normalized[section][key] = value.strip()
                    unchecked.append(f"[project] {key}")
                    continue
                raise _naming(_unknown(f"{source}: [{section}]", key, tuple(schema[section])), section, key)
            try:
                normalized[section][key] = _value(section, key, value, schema[section][key])
            except ValueError as exc:
                if not (open_key and key == "language"):
                    raise _naming(exc, section, key)
                normalized[section][key] = value.strip().lower()
                unchecked.append("[project] language")
    return normalized


def _naming(error: Exception, section: str, key: str | None) -> Exception:
    """``error`` naming the setting it refuses, as ``error.setting``."""
    error.setting = (section, key)
    return error


def _safe(values: Mapping[str, Any]) -> dict[str, Any]:
    from urllib.parse import urlsplit, urlunsplit

    def safe_value(value):
        if isinstance(value, Mapping):
            return _safe(value)
        if isinstance(value, (list, tuple)):
            return [safe_value(item) for item in value]
        return value

    result = {}
    for key, value in values.items():
        if any(word in key.lower() for word in ("key", "secret", "password", "credential", "header", "authorization", "token")) and key not in {
            "max_output_tokens", "reasoning_budget_tokens", "context_rebuild_trigger_tokens",
            "max_input_tokens_per_phase", "max_output_tokens_per_phase", "max_total_tokens_per_phase",
        }:
            continue
        if key == "request_options" and isinstance(value, str) and value.strip():
            try:
                options = json.loads(value)
                result[key] = (json.dumps(_safe(options), sort_keys=True)
                               if isinstance(options, Mapping) else "[invalid request options]")
            except (ValueError, RecursionError):
                result[key] = "[invalid request options]"
        elif key == "api_url" and value:
            try:
                parts = urlsplit(str(value))
                host = parts.hostname or ""
                if ":" in host:
                    host = f"[{host}]"
                port = f":{parts.port}" if parts.port is not None else ""
                result[key] = urlunsplit((parts.scheme, host + port, parts.path, "", ""))
            except ValueError:
                result[key] = "[invalid endpoint]"
        else:
            result[key] = safe_value(value)
    return result


def _model_credentials(settings: RunSettings) -> tuple[frozenset[str], frozenset[str]]:
    """The run's model credentials as (environment variable names, values),
    for redaction and for launch environments: provider keys, variables an
    ``api_key_env`` names, ``api_key`` values and URL passwords or secret
    query values. Names only for credentials the environment supplies. No
    pattern matches ordinary user variables.
    """
    names, secrets = set(), set()
    for name, value in settings.environment.items():
        if name.startswith("TESTUARY_") and name.endswith("_API_KEY") and value:
            names.add(name)
            secrets.add(value)
    for fields in (*settings.models.values(), *settings.sections.values()):
        key = fields.get("api_key")
        if isinstance(key, str) and key:
            secrets.add(key)
            secrets.add(key.strip())
        variable = fields.get("api_key_env")
        if isinstance(variable, str) and variable.strip():
            names.add(variable.strip())
            key = settings.environment.get(variable.strip(), "")
            if key:
                secrets.add(key)
        endpoint = fields.get("api_url")
        if isinstance(endpoint, str) and endpoint:
            secrets.update(_address_credentials(endpoint))
    return frozenset(names), frozenset(secrets)


def _address_credentials(address: str) -> set[str]:
    """The credentials an address holds: its password, also unquoted, and
    each query value whose name holds key, secret, password or token. An
    address that cannot be parsed holds none."""
    from urllib.parse import parse_qsl, unquote, urlsplit

    try:
        parts = urlsplit(address)
        found = {parts.password, unquote(parts.password)} if parts.password else set()
        found.update(value for name, value in parse_qsl(parts.query)
                     if any(word in name.lower() for word in ("key", "secret", "password", "token")))
    except ValueError:
        return set()
    return found


def model_credential_variables(settings: RunSettings) -> frozenset[str]:
    """The variables carrying this run's model credentials, which model
    requests keep and the launch environments of subject code and host
    commands omit."""
    if not isinstance(settings, RunSettings):
        raise ConfigurationError("model credential variables require resolved RunSettings")
    return _model_credentials(settings)[0]


def _without_secrets(text: str, secrets: Iterable[str]) -> str:
    """``text`` with each secret replaced, also as JSON, Python or URL escaping
    would quote it, since exception strings may quote a key so."""
    from urllib.parse import quote

    variants = {variant for secret in secrets if secret for variant in (
        secret, json.dumps(secret)[1:-1], repr(secret)[1:-1], quote(secret, safe=""),
    ) if variant}
    for variant in sorted(variants, key=len, reverse=True):
        text = text.replace(variant, "[redacted]")
    return text


def _redact_diagnostic(text: str, settings: RunSettings | None) -> str:
    """Remove captured credentials and URL user/query data from error text."""
    import re

    text = re.sub(r"https?://[^\s'\"<>]+", lambda match: _safe({"api_url": match.group()})["api_url"],
                  text, flags=re.IGNORECASE)
    return _without_secrets(text, _model_credentials(settings)[1] if settings is not None else ())


#: The environment of settings that captured none (see RunSettings).
_UNCAPTURED: Mapping[str, str] = MappingProxyType({})

#: Variable families that configure Testuary or its language backend.
_INPUT_PREFIXES = ("TESTUARY_", "LANGADAPTER_")


@dataclass(frozen=True, repr=False)
class RunSettings:
    """Resolved settings. Direct construction reads no ambient configuration.

    Sections override packaged defaults. ``models`` is a catalogue, not a
    file name. ``environment`` is the captured launch environment of native
    commands. Omitted, nothing is captured and native execution refuses to
    run: pass ``environment=dict(os.environ)`` or use ``resolve_settings``.
    ``environment_captured`` says whether one was supplied. ``supplied``
    holds the ``(section, key)`` pairs a layer above the packaged defaults
    sets, every key of ``sections`` when omitted. ``configuration_files``
    names the configuration and catalogue files ``resolve_settings`` read,
    which project copies leave out. ``user_dir`` is the user directory whose
    model catalogue and global configuration these settings were resolved
    from, None when constructed directly.
    """

    sections: Sections = field(default_factory=dict)
    models: Sections = field(default_factory=dict)
    environment: Mapping[str, str] | None = None
    base_dir: Path | str = field(default_factory=Path.cwd)
    supplied: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    configuration_files: tuple[str, ...] = ()
    #: The project directory: the one holding the project's Testuary directory
    #: (``[main] testuary_dir``), ``base_dir`` when omitted.
    project_dir: Path | str | None = None
    user_dir: Path | str | None = None
    environment_captured: bool = field(init=False)
    #: The ``[project]`` settings left unchecked, by name, such as
    #: ``[project] language``: those a language an extension adds may hold,
    #: when ``[main] extensions`` names items that were not loaded. A run and
    #: the tool check, which load the extensions, check them.
    unchecked: tuple[str, ...] = field(init=False)
    main: MainSettings = field(init=False)
    _layers: tuple[tuple[str, Sections], ...] = field(init=False, repr=False)
    #: The file each file layer was read from, by the layer's label.
    _layer_files: Mapping[str, str] = field(init=False, repr=False)
    #: The model catalogue files resolve_settings read, or would have read
    #: when given parsed ``models``. None when built directly. Kept through
    #: ``dataclasses.replace``, so a file a catalogue names stays relative
    #: to its directory.
    _catalogue_files: tuple[str, ...] | None = field(default=None, repr=False, compare=False)
    #: The rows of ``extension_items`` as resolve_settings found them, kept
    #: through ``dataclasses.replace`` while ``[main] extensions`` stays.
    _extension_rows: tuple[tuple[str, str, str], ...] | None = field(default=None, repr=False,
                                                                     compare=False)
    #: The last file of ``models_files`` when it does not exist, which reads
    #: as empty and the run's settings record names.
    _absent_models_file: str | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:

        files = self.configuration_files
        if isinstance(files, (str, bytes)) or not all(isinstance(path, str) for path in files):
            raise ValueError("configuration_files must be a sequence of path strings")
        object.__setattr__(self, "configuration_files", tuple(files))

        defaults = _defaults()
        main = self.sections.get("main")
        named = isinstance(main, Mapping) and bool(str(main.get("extensions") or "").strip())
        unchecked: list[str] = []
        values = _validate(self.sections, defaults, "settings", unchecked if named else None)
        object.__setattr__(self, "unchecked", tuple(dict.fromkeys(unchecked)))
        supplied = self.supplied or frozenset(
            (name, key) for name, fields in values.items() for key in fields
        )
        sections = _deep_merge_ini(defaults, values)
        object.__setattr__(self, "sections", _freeze(sections))
        layers = (("defaults", _freeze(defaults)),)
        if values:
            layers += (("overrides", _freeze(values)),)
        object.__setattr__(self, "_layers", layers)
        if not isinstance(self.models, Mapping) or any(
            not isinstance(name, str) or not isinstance(fields, Mapping)
            for name, fields in self.models.items()
        ):
            raise ValueError("models must map names to settings mappings")
        object.__setattr__(self, "models", _freeze(self.models))
        # The shared empty mapping marks "nothing captured" through replace().
        captured = self.environment is not None and self.environment is not _UNCAPTURED
        environment = self.environment if captured else {}
        if not isinstance(environment, Mapping) or any(
                not isinstance(name, str) or not isinstance(value, str)
                for name, value in environment.items()):
            raise ValueError("environment must map variable names to strings")
        object.__setattr__(self, "environment_captured", captured)
        object.__setattr__(self, "environment",
                           MappingProxyType(dict(environment)) if captured else _UNCAPTURED)
        object.__setattr__(self, "base_dir", Path(self.base_dir).resolve())
        object.__setattr__(self, "project_dir", self.base_dir if self.project_dir is None
                           else Path(self.project_dir).resolve())
        if self.user_dir is not None:
            object.__setattr__(self, "user_dir", Path(self.user_dir).resolve())
        object.__setattr__(self, "_layer_files", MappingProxyType({}))
        object.__setattr__(self, "_absent_models_file", None)
        object.__setattr__(self, "supplied", frozenset(supplied))
        main = sections["main"]
        cast = {"bool": lambda text: text == "true", "int": int, "str": str}
        object.__setattr__(self, "main", MainSettings(**{
            key: cast[field.type](main[key]) for key, field in MainSettings.__dataclass_fields__.items()}))

    def configuration_layers(self) -> tuple[tuple[str, Sections], ...]:
        """The captured layers in input order, read-only. Not a safe export."""
        return self._layers

    @property
    def results_dir(self) -> Path:
        """The directory ``results`` in the project's Testuary directory. A run
        writes below it unless ``[project] output_dir`` names another directory."""
        return self.project_dir / self.main.testuary_dir / "results"

    @property
    def consumer_files(self) -> tuple[str, ...]:
        """The files Testuary reads as configuration (settings files,
        catalogues, prompt profile, hints file), never project inputs. Each
        is listed as read, as spelled on disk and resolved, so a copy leaves
        it out however it is spelled or linked."""
        import os

        candidates = [Path(path) for path in self.configuration_files]
        for key in ("prompt_profile", "hints_file"):
            name = str(self.sections["main"].get(key, "")).strip()
            if name:
                candidates.append(Path(name) if Path(name).is_absolute() else self.base_dir / name)
        found = []
        for path in candidates:
            try:
                if path.is_file():
                    found += [os.path.abspath(path), _on_disk_spelling(os.path.abspath(path)),
                              str(path.resolve())]
            except OSError:
                continue
        return tuple(dict.fromkeys(found))

    def export(self) -> dict[str, Any]:
        """Return non-secret effective configuration suitable for result
        records: every section but those that configure no run
        (``UNRECORDED_SECTIONS``)."""
        return _safe({"sections": {name: values for name, values in self.sections.items()
                                   if name not in UNRECORDED_SECTIONS},
                      "models": self.models})

    def __repr__(self) -> str:
        return f"RunSettings({self.export()!r})"


def _on_disk_spelling(path: str) -> str:
    """``path`` spelled as its directories list it, as copies compare names
    (``Models.ini`` may list as ``models.ini``). An unlistable part stays."""
    import os

    anchor, *parts = Path(path).parts
    current = Path(anchor)
    for part in parts:
        candidate = current / part
        try:
            names = os.listdir(current)
        except OSError:
            current = candidate
            continue
        if part not in names:
            for name in names:
                try:
                    if name.lower() == part.lower() and os.path.samefile(current / name, candidate):
                        candidate = current / name
                        break
                except OSError:
                    continue
        current = candidate
    return str(current)


def platform_user_dir() -> Path:
    """The platform's per-user configuration directory for Testuary, as
    platformdirs names it: on Windows ``%APPDATA%\\testuary``, on macOS
    ``~/Library/Application Support/testuary``, elsewhere
    ``$XDG_CONFIG_HOME/testuary`` or ``~/.config/testuary``."""
    import platformdirs

    return Path(platformdirs.user_config_dir("testuary", appauthor=False, roaming=True))


@dataclass(frozen=True)
class FileLocations:
    """Testuary's files for runs started in one directory: the user directory's
    global configuration and catalogue, and the project directory, the nearest
    upward holding a Testuary directory (else the starting directory), whose
    Testuary directory holds the project's settings and results."""

    user: Path
    config: Path
    models: Path
    project: Path
    testuary: Path
    project_settings: Path
    results: Path


def file_locations(base_dir: str | Path | None = None, *, home: str | Path | None = None,
                   environment: Mapping[str, str] | None = None,
                   testuary_dir: str = ".testuary", search: bool = True) -> FileLocations:
    """Where runs started in ``base_dir`` (default: the current directory) find
    their files, and where the setup and changes write them.

    The user directory is ``home``, else ``TESTUARY_HOME`` in ``environment``,
    else ``platform_user_dir()``. The project search goes upward from
    ``base_dir`` and stops below the system user's home directory
    (``Path.home()``), which is never a project. Without ``search`` the
    starting directory is the project. ``--config`` or
    ``TESTUARY_CONFIG_FILE`` replaces the global configuration file for one
    command."""
    root = (Path(base_dir) if base_dir is not None else Path.cwd()).resolve()
    if home is None:
        home = str((os.environ if environment is None else environment).get("TESTUARY_HOME", "")).strip() or None
    user = (Path(home).expanduser() if home is not None else platform_user_dir()).resolve()
    testuary_dir = _value("main", "testuary_dir", testuary_dir, ".testuary")
    project = root
    home_dir = _home_dir()
    for candidate in ((root, *root.parents) if search else ()):
        if candidate == home_dir:
            break
        testuary = candidate / testuary_dir
        # A user directory named like a Testuary directory holds no project.
        if testuary.is_dir() and testuary.resolve() != user:
            project = candidate
            break
    testuary = project / testuary_dir
    return FileLocations(user=user, config=user / "config.ini", models=user / "models.ini",
                         project=project, testuary=testuary,
                         project_settings=testuary / "project.ini", results=testuary / "results")


def _home_dir() -> Path | None:
    """The user's home directory, or None when the system names none."""
    try:
        return Path.home().resolve()
    except (OSError, RuntimeError):
        return None


def project_locations(base_dir: str | Path | None = None, *, home: str | Path | None = None,
                       environment: Mapping[str, str] | None = None) -> FileLocations:
    """``file_locations`` with the Testuary directory's name a run without
    command-line settings uses (its variable, else the global configuration,
    else the default), reading nothing else: for other programs to find a
    project's settings and results."""

    variables = os.environ if environment is None else environment
    testuary_dir = str(variables.get("TESTUARY_MAIN_TESTUARY_DIR", "")).strip()
    if not testuary_dir:
        config = file_locations(base_dir, home=home, environment=variables).config
        testuary_dir = str(_read_ini_file(config).get("main", {}).get("testuary_dir", "")).strip()
    return file_locations(base_dir, home=home, environment=variables,
                          testuary_dir=testuary_dir or ".testuary")


def _read_catalogue(files: Sequence[Path], user_catalogue: Path,
                    project: Path | None) -> Dict[str, Dict[str, Any]]:
    """The catalogue files read in order, a later file's section replacing the
    same section of an earlier file whole, so a redefined alias keeps no
    earlier key. A literal api_key outside the user directory's catalogue is
    refused, as in a configuration file, so that no key lands in a
    project's files. A catalogue file in the ``project`` directory, links
    followed, is checked as a project's (check_project_catalogue_file)."""
    catalogue: Dict[str, Dict[str, Any]] = {}
    for path in files:
        sections = _read_ini_file(path)
        own = bool(sections) and user_catalogue.is_file() and path.samefile(user_catalogue)
        for section, fields in sections.items():
            if "api_key" in fields and not own:
                raise ConfigurationError(
                    f"{path}: [{section}] api_key belongs in the user directory's models.ini, not "
                    "another catalogue file: name the variable that holds it with api_key_env")
        if not own and project is not None and path.resolve().is_relative_to(project):
            check_project_catalogue_file(path)
        catalogue.update(sections)
    return catalogue


def _catalogue_name(settings: RunSettings) -> str:
    """The model catalogue files ``settings`` were resolved from, for the
    refusals that name them: ``models.ini`` for settings built directly."""
    if settings._catalogue_files is not None:
        return ", ".join(settings._catalogue_files) or "an empty models_files"
    return "models.ini" if settings.user_dir is None else str(Path(settings.user_dir) / "models.ini")


def _environment_layer(env: Mapping[str, str], schema: Mapping[str, Any]
                       ) -> tuple[dict[str, dict[str, str]], list[str]]:
    """The ``TESTUARY_<SECTION>_<KEY>`` variables of ``env`` as a layer, and
    the variables of no section, which the caller refuses after the
    catalogue is read, since it may name one as a credential
    (api_key_env)."""
    env_layer: dict[str, dict[str, str]] = {}
    unmatched = []
    for name, value in env.items():
        # A variable ending in _API_KEY is a service's key, a captured
        # execution input, never a setting: no setting ends in api_key, and a
        # service may be named like a section (TESTUARY_MAIN_X_API_KEY).
        if not name.startswith("TESTUARY_") or name in ENVIRONMENT_ONLY or name.endswith("_API_KEY"):
            continue
        if name.replace("FOLDER", "DIR") in ENVIRONMENT_ONLY:
            # It names a variable Testuary reads for itself, with FOLDER for DIR.
            raise _unknown("environment", name, tuple(ENVIRONMENT_ONLY), "variable")
        body = name[len("TESTUARY_"):].lower()
        for section in sorted(schema, key=len, reverse=True):
            if body.startswith(section + "_"):
                env_layer.setdefault(section, {})[body[len(section) + 1:]] = value
                break
        else:
            unmatched.append(name)
    return env_layer, unmatched


def resolve_settings(
    *,
    config_files: Sequence[str | Path] | None = None,
    overrides: Sections | None = None,
    use_environment: bool = True,
    base_dir: str | Path | None = None,
    models: Sections | None = None,
    environment: Mapping[str, str] | None = None,
    home: str | Path | None = None,
    project_settings: bool = True,
    models_files: Sequence[str | Path] | None = None,
    load_extensions: bool = False,
) -> RunSettings:
    """Resolve the settings from these layers, each overriding the one
    before: the packaged defaults, the global or explicit files, the
    project's settings file, the enabled environment and the explicit
    fields. ``[main] extensions`` is the exception: each layer adds its
    items (``extension_items``).

    With ``load_extensions``, for the start of a command that uses them, the
    items are loaded after each layer's ``[main]`` is checked and before
    its other sections are, so ``[project]`` may name a language an
    extension adds and the settings it declares. An item that cannot load
    is refused, naming where it is named. Without it, and always for an
    empty list, nothing is loaded, and settings that name items keep a
    ``[project] language`` or key no registered language has, named in
    ``RunSettings.unchecked``.

    ``file_locations`` names the files: without explicit files the user
    directory's optional ``config.ini``, and the ``project.ini`` found from
    ``base_dir`` upward. With ``project_settings`` false none is read and
    ``base_dir`` is the project. An empty ``models`` mapping skips the
    catalogue. ``models_files`` names the catalogue files read in order in
    place of the user directory's ``models.ini``, relative ones from
    ``base_dir`` as for ``config_files``. A later file's section replaces
    the same section of an earlier one whole, where configuration files
    merge key by key. An empty sequence reads none. An absent last file
    reads as empty, and any other absent file is refused, as are a single
    path and ``models_files`` beside ``models``, and a file in the project
    directory that sets api_url or api_key_env. Relative paths start from
    the project directory.

    The launch environment is always captured (``environment``, else the
    process's). ``use_environment`` decides only whether the ``TESTUARY_``
    and ``LANGADAPTER_`` variables are inputs. Disabled, they are left out
    of it too, and platform variables such as PATH stay.
    """

    root = (Path(base_dir) if base_dir is not None else Path.cwd()).resolve()
    if models is not None and models_files is not None:
        raise TypeError("models and models_files each name the catalogue: pass one")
    if isinstance(models_files, (str, Path)):
        raise TypeError("models_files must be an ordered sequence of paths")
    source = os.environ if environment is None else environment
    launch = dict(source) if use_environment else {
        name: value for name, value in source.items() if not name.startswith(_INPUT_PREFIXES)}
    env = launch if use_environment else {}
    defaults = _defaults()
    schema = _schema(defaults)

    # Settings that name extensions this resolution does not load leave the
    # [project] settings their languages may declare unchecked.
    lenient = False

    def check(layer: Sections, source: str) -> dict[str, dict[str, str]]:
        return _validate(layer, defaults, source, [] if lenient else None)

    def origin(label: str, path: Path | None) -> str:
        """The input a refusal names: a file's path, else the layer."""
        return str(path) if path is not None else {"overrides": "explicit overrides"}.get(label, label)

    def main_of(layer: Sections, label: str, path: Path | None) -> dict[str, str]:
        """A layer's [main], checked before its other sections: it names the
        Testuary directory and the extensions, which add languages and the
        settings they declare."""
        return check({"main": layer["main"]}, origin(label, path))["main"] if "main" in layer else {}

    locations = file_locations(root, home=home, environment=env)
    # Each file layer as read: its sections, its label and its file.
    file_layers: list[tuple[Sections, str, Path]] = []
    files = config_files
    if files is None and env.get("TESTUARY_CONFIG_FILE", "").strip():
        files = [env["TESTUARY_CONFIG_FILE"]]
    if isinstance(files, (str, Path)):
        raise TypeError("config_files must be an ordered sequence of paths")
    if files is None:
        file_layers.append((_read_ini_file(locations.config), "config[0]", locations.config))
    else:
        for index, filename in enumerate(files):
            path = Path(filename)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise ValueError(f"configuration file is absent or not a file: {path}")
            file_layers.append((_read_ini_file(path), f"config[{index}]", path))
    env_layer, unmatched = _environment_layer(env, schema)
    later: list[tuple[Sections, str, None]] = [(env_layer, "environment", None),
                                               (overrides or {}, "overrides", None)]
    # The Testuary directory's name, from every layer but the file it locates.
    testuary_dir = next((main["testuary_dir"] for main in (
        main_of(*layer) for layer in reversed([*file_layers, *later])) if "testuary_dir" in main),
        defaults["main"]["testuary_dir"])
    locations = file_locations(root, home=home, environment=env, testuary_dir=testuary_dir,
                               search=project_settings)
    if project_settings and locations.project != _home_dir() and locations.project_settings.is_file():
        file_layers.append((_read_ini_file(locations.project_settings), "project",
                            locations.project_settings))
    rows = _extension_rows(((main_of(layer, label, path), label, path)
                            for layer, label, path in [*file_layers, *later]), locations.project)
    lenient = bool(rows) and not load_extensions
    if rows and load_extensions:
        from testuary import _extensions

        _extensions.load(rows)
        # The loaded languages declare [project] settings of their own.
        defaults = _defaults()
    inputs = [(check(layer, origin(label, path)), label, path) for layer, label, path in
              [*file_layers, *later]]
    for checked, label, path in inputs:
        if label == "project":
            _check_project_settings(checked, path)

    supplied = set()
    layers: dict[str, dict[str, Any]] = {}
    captured: list[tuple[str, Sections]] = []
    # [main] extensions is the one setting whose layers add their items to
    # those of the earlier layers in place of replacing them.
    combined: list[str] = []
    for checked, label, _ in inputs:
        supplied.update((name, key) for name, values in checked.items() for key in values)
        layers = _deep_merge_ini(layers, checked)
        combined += _split_extensions(checked.get("main", {}).get("extensions", ""))
        if checked:
            captured.append((label, _freeze(checked)))
    if combined:
        layers["main"]["extensions"] = ", ".join(combined)
    read = [path for _, _, path in file_layers]
    catalogue_files = [locations.models]
    absent = None
    if models_files is not None:
        catalogue_files = [Path(name) if Path(name).is_absolute() else root / name
                           for name in models_files]
        # The last file may not be written yet: it reads as empty.
        if catalogue_files and not catalogue_files[-1].exists():
            absent = catalogue_files[-1]
        for path in catalogue_files[:-1] if absent else catalogue_files:
            if not path.is_file():
                raise ConfigurationError(f"model catalogue file is absent or not a file: {path}")
    catalogue = models
    if models is None:
        read.extend(catalogue_files)
        # The home directory is never a project (file_locations).
        project = None if locations.project == _home_dir() else locations.project
        catalogue = _read_catalogue(catalogue_files, locations.models, project)
    named = {str(fields.get("api_key_env", "")) for fields in catalogue.values() if isinstance(fields, Mapping)}
    for name in unmatched:
        if name not in named:
            label = name[len("TESTUARY_"):].lower().split("_")[0]
            raise _unknown(f"environment variable {name}", label, tuple(schema), "section")
    settings = RunSettings(
        sections=layers,
        models=catalogue,
        environment=launch,
        # Relative paths of the settings start from the project directory.
        base_dir=locations.project,
        supplied=frozenset(supplied),
        configuration_files=tuple(str(path) for path in read if path.is_file()),
        project_dir=locations.project,
        user_dir=locations.user,
    )
    # The defaults, then the input layers in place of one merged layer.
    kept = tuple(layer for layer in settings._layers if layer[0] != "overrides")
    object.__setattr__(settings, "_layers", (*kept, *captured))
    object.__setattr__(settings, "_layer_files", MappingProxyType(
        {label: str(path) for _, label, path in file_layers if path.is_file()}))
    object.__setattr__(settings, "_catalogue_files", tuple(map(str, catalogue_files)))
    object.__setattr__(settings, "_extension_rows", extension_items(settings))
    object.__setattr__(settings, "_absent_models_file", None if absent is None else str(absent))
    return settings


@dataclass(frozen=True)
class SettingDescription:
    """One setting: section, name, purpose, kind (``boolean``, ``choice``,
    ``number``, ``text``, and for a language's project setting ``path`` and
    ``executable``), default (empty: unset), a choice's values, its check,
    and every language declaring it (empty: Testuary's own)."""

    section: str
    name: str
    purpose: str
    kind: str
    default: str
    choices: tuple[str, ...] = ()
    check: str = ""
    languages: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """The setting's name in plain words, as a settings view titles it:
        a written label for the settings most runs change, else its name's
        words."""
        return _SETTING_LABELS.get((self.section, self.name)) or _words(self.name)


@dataclass(frozen=True)
class SectionDescription:
    """A section of the configuration: its name, what it is for and its
    settings in order. ``title`` names it in plain words."""

    name: str
    purpose: str
    settings: tuple[SettingDescription, ...]

    @property
    def title(self) -> str:
        """The section's name in plain words, such as "Starter phase"."""
        return _SECTION_TITLES.get(self.name) or _words(self.name)


#: The written labels of the settings most runs change, by section and name,
#: in the words the run plan names its limits with.
_SETTING_LABELS = {
    (PHASE_SECTIONS["starter"], "suite_count"): "Starter suites to generate",
    (PHASE_SECTIONS["coverage"], "goal"): "Coverage goal (%)",
    (PHASE_SECTIONS["coverage"], "minimum"): "Coverage minimum (%)",
    (PHASE_SECTIONS["mutation"], "goal"): "Mutation score goal (%)",
    (PHASE_SECTIONS["mutation"], "minimum"): "Mutation score minimum (%)",
    **{(PHASE_SECTIONS[phase], name): label for phase in ("coverage", "mutation")
       for name, label in (
           ("max_requests_per_phase", "Request limit (model requests per phase)"),
           ("max_wall_clock_seconds_per_phase", "Time limit per phase (seconds, 0 for none)"))},
}

#: Each section's title, as a settings view groups the settings.
_SECTION_TITLES = {
    "api": "Model", "openai": "OpenAI service", "ollama": "Ollama service", "vllm": "vLLM service",
    "main": "General", "project": "Project", PHASE_SECTIONS["starter"]: "Starter phase",
    PHASE_SECTIONS["coverage"]: "Coverage phase", PHASE_SECTIONS["mutation"]: "Mutation phase",
    PHASE_SECTIONS["finishing"]: "Finishing", "limits": "What the model is shown (size limits)",
    "limits_coverage": "What the coverage phase's model is shown",
    "limits_mutation": "What the mutation phase's model is shown",
    PHASE_SECTIONS["timing"]: "Suite timing", "demo": "Demonstration", "web": "Web interface",
}

#: Words a label writes in capitals.
_ACRONYMS = {"api": "API", "url": "URL", "jdk": "JDK", "ini": "INI", "id": "ID", "cpu": "CPU",
             "llm": "LLM", "json": "JSON", "csv": "CSV", "pit": "PIT"}


def _words(name: str) -> str:
    words = [_ACRONYMS.get(word, word) for word in name_words(name).split()]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class SettingValue:
    """A described setting's value and its ``source``: ``default``, the
    path of the file that sets it, ``environment`` or ``overrides``.
    ``sources`` names every layer that sets it, in order and in the same
    terms, the last being ``source``. It holds no value. ``active`` is False
    for a setting the run does not use, ``inactive`` saying why."""

    setting: SettingDescription
    value: str
    source: str
    active: bool = True
    inactive: str = ""
    sources: tuple[str, ...] = ()


def _packaged_entries() -> list[tuple[str, str, list[tuple[str, str, str]]]]:
    """The packaged config.ini's sections in file order, each with the
    comment above it and its keys, each key with its default and the
    comment above it."""

    sections: list[tuple[str, str, list[tuple[str, str, str]]]] = []
    comment: list[str] = []
    for raw in Path(_bundled_data_path("config.ini")).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            comment = []
        elif line.startswith(("#", ";")):
            comment.append(line.lstrip("#;").strip())
        elif line.startswith("[") and line.endswith("]"):
            sections.append((line[1:-1], " ".join(comment), []))
            comment = []
        elif "=" in line and sections:
            key, _, value = line.partition("=")
            sections[-1][2].append((key.strip(), value.strip(), " ".join(comment)))
            comment = []
    return sections


def _kind(section: str, key: str, default: str) -> tuple[str, tuple[str, ...], str]:
    """A packaged setting's kind, choices and check, by the rules the
    resolver validates its values with (``_value``)."""
    if section in ("limits", "limits_coverage", "limits_mutation"):
        return "number", (), "a positive whole number, or unlimited"
    if key == "reasoning":
        return "choice", ("on", "off"), ""
    if key in _LIMIT_FIELDS:
        return "number", (), "a number, or unlimited"
    if section == "project" and key == "language":
        from langadapter import names

        return "choice", tuple(names()), ""
    if section == "main" and key == "testuary_dir":
        return "text", (), "a directory below the project directory"
    if str(default).lower() in ("true", "false"):
        return "boolean", ("true", "false"), ""
    if key in _CHOICES:
        return "choice", tuple(_CHOICES[key]), ""
    if key == "max_measurements_without_improvement":
        return "number", (), "a whole number, or a share of max_requests_per_phase from 0% to 100%"
    if key == "per_mutant_timeout_seconds":
        return "number", (), "auto, or a positive whole number"
    if key in _PERCENT_FIELDS:
        return "number", (), "a percentage from 0 to 100"
    if _numeric(key, default):
        return "number", (), ""
    return "text", (), ""


def reference(*, languages: Sequence[str] | None = None) -> tuple[SectionDescription, ...]:
    """Every setting by section in the packaged ``config.ini``'s order, as
    it and its comments state them, then in ``[project]`` those the installed
    ``languages`` declare. A setting several languages declare names them
    all, and the first declaration gives it its kind, choices and check. The
    reference page and the settings views come from it. The model request
    controls are described with the models."""
    from langadapter import names, registration

    from testuary.languages import declared_settings

    chosen = tuple(names()) if languages is None else tuple(languages)
    declarations: dict[str, tuple[tuple[str, ...], Any]] = {}
    for language in chosen:
        for setting in getattr(registration(language).factory, "project_settings", ()) or ():
            named, first = declarations.get(setting.name, ((), setting))
            declarations[setting.name] = ((*named, language), first)
    sections = []
    for name, purpose, entries in _packaged_entries():
        described = []
        for key, default, comment in entries:
            kind, choices, check = _kind(name, key, default)
            declaring: tuple[str, ...] = ()
            if name == "project" and key in declarations:
                declaring, setting = declarations[key]
                kind, choices, check = setting.kind, tuple(setting.choices), setting.check
            described.append(SettingDescription(name, key, comment, kind, default, choices,
                                                check, declaring))
        if name == "project":
            listed = {item.name for item in described}
            for language in chosen:
                for setting in declared_settings(language):
                    if setting.name not in listed:
                        listed.add(setting.name)
                        described.append(SettingDescription(
                            "project", setting.name, setting.purpose, setting.kind,
                            setting.default or "", tuple(setting.choices), setting.check,
                            declarations[setting.name][0]))
        sections.append(SectionDescription(name, purpose, tuple(described)))
    return tuple(sections)


def setting_values(settings: RunSettings, *,
                   languages: Sequence[str] | None = None) -> tuple[SettingValue, ...]:
    """Each described setting with its value in ``settings`` and the layer
    it comes from, the last that sets it. Values come from the safe view the
    export gives, the sections it leaves out of records included, so no
    credential appears."""
    sections = _safe(settings.sections)

    def sources(section: str, key: str) -> tuple[str, ...]:
        return tuple(settings._layer_files.get(name, name)
                     for name, layer in settings.configuration_layers()
                     if name != "defaults" and key in layer.get(section, {}))

    provider = _provider_in_use(settings, sections)
    engine = str(sections.get(PHASE_SECTIONS["mutation"], {}).get("engine") or "").strip()
    language = str(sections.get("project", {}).get("language") or "").strip()

    def unused(setting: SettingDescription) -> str:
        if setting.section in ("openai", "ollama", "vllm") and provider and setting.section != provider:
            return f"the model's provider is {provider}"
        if setting.section == PHASE_SECTIONS["mutation"] and setting.name.startswith("cosmic_ray_") \
                and engine and engine != "cosmic_ray":
            return f"the mutation engine is {engine}"
        if setting.languages and language and language not in setting.languages:
            return f"the project's language is {language}"
        return ""

    values = []
    for section in reference(languages=languages):
        for setting in section.settings:
            value = sections.get(setting.section, {}).get(setting.name)
            if value is not None:
                reason = unused(setting)
                setting_sources = sources(setting.section, setting.name)
                values.append(SettingValue(setting, str(value),
                                           setting_sources[-1] if setting_sources else "default",
                                           active=not reason, inactive=reason,
                                           sources=setting_sources))
    return tuple(values)


def extension_items(settings: RunSettings) -> tuple[tuple[str, str, str], ...]:
    """Each item of ``[main] extensions`` in the combined list's order, as
    ``(item, source, directory)``: the layer that names it in the terms of
    ``SettingValue.source`` (the path of a file, ``environment`` or
    ``overrides``) and the directory a file item is relative to, that of the
    file that names it, else the project directory. Nothing is loaded, so
    two items that name one file both appear. Settings that
    ``dataclasses.replace`` made keep the rows of their resolution while
    the list stays, and an item a replace adds counts as an override."""
    kept = settings._extension_rows
    if kept is not None and ", ".join(item for item, _, _ in kept) == settings.sections["main"]["extensions"]:
        return kept
    return _extension_rows(((layer.get("main", {}), label, settings._layer_files.get(label))
                            for label, layer in settings.configuration_layers()),
                           settings.project_dir)


def _check_unchecked(settings: RunSettings) -> None:
    """Check the ``[project]`` settings ``settings.unchecked`` names, once
    the extensions are loaded: a language or key that no registered
    language has is refused, as for settings that name no extension."""
    if settings.unchecked:
        try:
            _validate({"project": dict(settings.sections["project"])}, _defaults(), "settings")
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from None


def _naming_dir(settings: RunSettings, section: str, key: str, *, catalogue: bool) -> str:
    """The directory a file named in ``key`` of ``section`` is relative to:
    the directory of the last file that sets it, among the model catalogue's
    files for a ``catalogue`` section and the settings files otherwise, else
    the project directory (the variable, a program's overrides, settings
    built directly)."""
    if catalogue:
        files = [Path(name) for name in settings._catalogue_files or ()]
        holder = next((path for path in reversed(files)
                       if key in _read_ini_file(path).get(section, {})), None)
    else:
        holder = next((settings._layer_files.get(label) for label, layer
                       in reversed(settings.configuration_layers())
                       if key in layer.get(section, {})), None)
    return str(Path(holder).parent if holder else settings.project_dir)


def _extension_rows(layers: Iterable[tuple[Mapping[str, Any], str, str | Path | None]],
                    project_dir: str | Path) -> tuple[tuple[str, str, str], ...]:
    """The rows ``extension_items`` describes, of ``layers`` given as each
    layer's ``[main]``, its label and its file or None."""
    return tuple((item, str(file) if file else label, str(Path(file).parent if file else project_dir))
                 for main, label, file in layers
                 for item in _split_extensions(str(main.get("extensions", ""))))


def _provider_in_use(settings: RunSettings, sections: Mapping[str, Any]) -> str:
    """The provider the run's model goes through: the alias's provider in
    the model catalogue, else ``[api] provider``. Empty when not known."""
    api = sections.get("api", {})
    model = str(api.get("model") or "").strip()
    alias = settings.models.get(model) if model else None
    named = alias.get("provider") if isinstance(alias, Mapping) else None
    return str(named or api.get("provider") or "").strip()


def check_project_settings_file(path: str | Path) -> None:
    """Refuse the settings file ``path`` when any of its sections, such as
    ``[api]`` or a provider's, sets ``api_url``, which decides where the
    model's key goes.

    The caller passes the files that are not the user's own: Testuary
    refuses the addresses of its project's settings file by this rule, and
    a program checks each other settings file it passes to its runs. A
    missing file passes.

    Raises:
        ConfigurationError: naming the file and each address it sets.
        ValueError: when the file is not valid INI.
    """
    _refuse_addresses(_read_ini_file(Path(path)), path)


def check_project_catalogue_file(path: str | Path) -> None:
    """Refuse the model catalogue file ``path`` when any of its sections sets
    ``api_url`` or ``api_key_env``, which decide where the model's key goes.

    The caller passes the files that are not the user's own: settings
    resolution checks each catalogue file in the project directory, and a
    program checks each other catalogue file it passes to its runs. A
    missing file passes.

    Raises:
        ConfigurationError: naming the file and each such value.
        ValueError: when the file is not valid INI.
    """
    named = [f"[{section}] {key}" for section, fields in _read_ini_file(Path(path)).items()
             for key in ("api_url", "api_key_env") if key in fields]
    if named:
        one = len(named) == 1
        raise ConfigurationError(
            f"{path}: {' and '.join(named)} {'decides' if one else 'decide'} where the model's key "
            f"goes. Set {'it' if one else 'them'} in your own models.ini")


def _check_project_settings(values: Mapping[str, Mapping[str, Any]], path: str | Path) -> None:
    """Refuse the values the project's settings file ``path`` may not set:
    [main] testuary_dir, which locates that file, and every api_url
    (_refuse_addresses). The run, the setup and the web interface check the
    project's file through this one rule."""
    if "testuary_dir" in values.get("main", {}):
        raise ConfigurationError(
            f"{path}: [main] testuary_dir locates this file. Set it in the global "
            "configuration, the environment or on the command line")
    _refuse_addresses(values, path)


def _refuse_addresses(values: Mapping[str, Mapping[str, Any]], path: str | Path) -> None:
    """Refuse every api_url of a settings file that is not the user's own,
    which decides where a model's key goes."""
    addresses = [f"[{name}] api_url" for name, keys in values.items() if "api_url" in keys]
    if addresses:
        raise ConfigurationError(
            f"{path}: {' and '.join(addresses)} decide{'s' if len(addresses) == 1 else ''} where "
            "the model's key goes. Set the address in your global configuration, models.ini, "
            "the environment or on the command line")


def _check_configuration_text(text: str, source: str) -> None:
    """Refuse a configuration file's text as the resolver refuses the file."""
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        raise ConfigurationError(f"invalid INI configuration at {source}: "
                                 f"{type(exc).__name__}") from None
    layer = {name: dict(parser[name]) for name in parser.sections()}
    # A file that names extensions may set the [project] settings their
    # languages declare: the save loads nothing.
    named = bool(str(layer.get("main", {}).get("extensions", "")).strip())
    try:
        _validate(layer, _defaults(), source, [] if named else None)
    except ValueError as exc:
        raise _naming(ConfigurationError(str(exc)), *getattr(exc, "setting", (None, None))) from None


def update_file(path: str | Path, section: str, values: Mapping[str, Any] | None = None, *,
                remove: Sequence[str] = ()) -> None:
    """Set ``values`` and remove ``remove`` in ``section`` of the settings
    file ``path``, keeping every other line, and write it only when the
    whole file validates as the resolver's. Else ConfigurationError names
    the problem and the file stays. A missing file is created."""

    target = Path(path)
    text = target.read_bytes().decode("utf-8") if target.is_file() else ""
    items = list((values or {}).items())
    for key, value in items:
        if value is None:
            raise ConfigurationError(f"[{section}] {key}: None is not a setting value. Remove the "
                                     "key to unset it")
    try:
        if remove:
            text = remove_keys(text, section, list(remove))
        if items:
            text = set_values(text, section, [(key, str(value)) for key, value in items])
    except ValueError as exc:
        raise ConfigurationError(f"{target}: {exc}") from None
    _check_configuration_text(text, str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(target, text)


def add_extension(item: str) -> bool:
    """Name ``item``, a file or an installed module, in ``[main]
    extensions`` of the user's config.ini when that file does not name it
    yet, keeping its other items and lines, and tell whether the file
    changed. A run with ``--config`` does not read that file. A file item is
    relative to the user directory."""
    return _change_extensions(item, add=True)


def remove_extension(item: str) -> bool:
    """Take ``item`` out of ``[main] extensions`` of the user's config.ini,
    the key with its last item, and tell whether the file changed."""
    return _change_extensions(item, add=False)


def _change_extensions(item: str, *, add: bool) -> bool:
    """The user's config.ini with ``item`` added or removed through
    ``update_file``. Items compare as written, without outer spaces."""
    try:
        items = _split_extensions(item, f"[main] extensions: {item!r}") if isinstance(item, str) else []
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None
    if len(items) != 1:
        raise ConfigurationError(f"[main] extensions: {item!r} is not one file or module name")
    path = file_locations().config
    try:
        named = _split_extensions(str(_read_ini_file(path).get("main", {}).get("extensions", "")))
    except ValueError as exc:
        raise ConfigurationError(f"{path}: {exc}") from None
    if (items[0] in named) == add:
        return False
    kept = [*named, items[0]] if add else [entry for entry in named if entry != items[0]]
    if kept:
        update_file(path, "main", {"extensions": ", ".join(kept)})
    else:
        update_file(path, "main", remove=["extensions"])
    return True


def _bundled_data_path(name: str) -> Optional[Path]:
    """The path of a bundled file a run reads at the package root, such as
    ``config.ini``, or None."""
    try:
        ref = importlib.resources.files("testuary") / name
        # A file on disk in an unpacked install. A zipped one gives None.
        path = Path(str(ref))
        return path if path.exists() else None
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def _bundled_template_path(name: str) -> Optional[Path]:
    """The path of a scaffold in ``testuary/_templates/`` that ``testuary
    init`` copies out, or None. No run reads them."""
    try:
        ref = importlib.resources.files("testuary._templates") / name
        path = Path(str(ref))
        return path if path.exists() else None
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def _read_ini_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """An INI file as a nested dict, ``{}`` for a missing file."""
    config = configparser.ConfigParser(interpolation=None)
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as stream:
            config.read_file(stream)
    except configparser.Error as exc:
        raise ValueError(f"invalid INI configuration at {path}: {type(exc).__name__}") from None
    return {section: dict(config[section]) for section in config.sections()}


def _deep_merge_ini(
    base: Dict[str, Dict[str, Any]],
    override: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Two INI dicts merged key by key, ``override`` winning, so each layer
    declares only the values it changes."""
    result: Dict[str, Dict[str, Any]] = {
        section: dict(values) for section, values in base.items()
    }
    for section, values in override.items():
        if section not in result:
            result[section] = {}
        for key, value in values.items():
            result[section][key] = value
    return result


#: A section header as configparser reads it: anything after the closing
#: bracket, such as a comment, is not part of the name.
_HEADER = re.compile(r"^\s*\[([^\]]+)\]")
#: A key line: configparser reads key names in any case, and an indented
#: line continues the value before it.
_ANY_KEY = re.compile(r"^[^\s#;\[][^=:]*[=:]")


def _key_pattern(key: str) -> "re.Pattern[str]":
    return re.compile(rf"^{re.escape(key)}\s*[=:]", re.IGNORECASE)


def _span(lines: List[str], section: str) -> Optional[Tuple[int, int]]:
    """The section's header line and the line after its last body line,
    or None. The comments and blank lines that close a section belong to
    the next one, which they usually describe."""
    start = None
    for index, line in enumerate(lines):
        match = _HEADER.match(line)
        if match is None:
            continue
        if start is not None:
            end = index
            while end > start + 1 and (not lines[end - 1].strip()
                                       or lines[end - 1].lstrip().startswith(("#", ";"))):
                end -= 1
            return start, end
        if match.group(1).strip() == section:
            start = index
    if start is None:
        return None
    end = len(lines)
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return start, end


def _one_line(values: Sequence[Tuple[str, str]]) -> None:
    for key, value in values:
        if "\n" in str(value) or "\r" in str(value):
            raise ValueError(f"the value of {key} must be one line")


def set_values(text: str, section: str, values: Sequence[Tuple[str, str]]) -> str:
    """``text`` with each ``key = value`` of ``values`` in ``section``: an
    existing key line of the section replaced, a missing key added after
    the section's last line, and a missing section appended. Every other
    line is unchanged, line endings included."""
    _one_line(values)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    span = _span([line.rstrip("\r\n") for line in lines], section)
    if span is None:
        block = [f"[{section}]"] + [f"{key} = {value}" for key, value in values]
        prefix = text
        if prefix and not prefix.endswith(("\n", "\r")):
            prefix += newline
        if prefix.strip():
            prefix += newline
        return prefix + newline.join(block) + newline
    start, end = span
    for key, value in values:
        pattern = _key_pattern(key)
        line = f"{key} = {value}{newline}"
        found = next((index for index in range(start + 1, end)
                      if pattern.match(lines[index])), None)
        if found is not None:
            ending = newline if lines[found].endswith(("\n", "\r")) else ""
            lines[found] = f"{key} = {value}{ending}"
            continue
        if end > 0 and not lines[end - 1].endswith(("\n", "\r")):
            lines[end - 1] += newline
        lines.insert(end, line)
        end += 1
    return "".join(lines)


def replace_section(text: str, section: str, values: Sequence[Tuple[str, str]]) -> str:
    """``text`` with ``section``'s keys exactly ``values``: every key line of
    the section, with the lines that continue its value, is removed and the
    new key lines follow the header and the comments before the first key.
    Other comments of the section, other sections and line endings stay. A
    missing section is appended."""
    _one_line(values)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    span = _span([line.rstrip("\r\n") for line in lines], section)
    if span is None:
        return set_values(text, section, values)
    start, end = span
    leading: List[str] = []
    trailing: List[str] = []
    seen_key = in_value = False
    for line in lines[start + 1:end]:
        bare = line.rstrip("\r\n")
        if _ANY_KEY.match(bare):
            seen_key = in_value = True
            continue
        if in_value and bare[:1] in (" ", "\t") and bare.strip():
            continue
        in_value = False
        (trailing if seen_key else leading).append(line)
    body = [f"{key} = {value}{newline}" for key, value in values]
    header = lines[start] if lines[start].endswith(("\n", "\r")) else lines[start] + newline
    return "".join(lines[:start] + [header] + leading + body + trailing + lines[end:])


def remove_keys(text: str, section: str, keys: Sequence[str]) -> str:
    """``text`` without the lines of ``keys`` in ``section``, each with the
    lines that continue its value. Every other line stays, line endings
    included. A missing section or key changes nothing."""
    lines = text.splitlines(keepends=True)
    bare = [line.rstrip("\r\n") for line in lines]
    span = _span(bare, section)
    if span is None:
        return text
    start, end = span
    patterns = [_key_pattern(key) for key in keys]
    kept = lines[:start + 1]
    removing = False
    for index in range(start + 1, end):
        if _ANY_KEY.match(bare[index]):
            removing = any(pattern.match(bare[index]) for pattern in patterns)
        elif not (removing and bare[index][:1] in (" ", "\t") and bare[index].strip()):
            removing = False
        if not removing:
            kept.append(lines[index])
    return "".join(kept + lines[end:])


def remove_section(text: str, section: str) -> str:
    """``text`` without ``section``: its header, its body and the comments
    directly above it, but not a comment block starting the file nor the
    comments after the body, which describe the next section. Blank lines
    the removal joins or leaves at the end go."""
    lines = text.splitlines(keepends=True)
    bare = [line.rstrip("\r\n") for line in lines]
    span = _span(bare, section)
    if span is None:
        return text
    start, end = span
    first = start
    while first > 0 and bare[first - 1].lstrip().startswith(("#", ";")):
        first -= 1
    if first > 0:
        start = first
    if end == len(lines):
        while start > 0 and not bare[start - 1].strip():
            start -= 1
    while end < len(lines) and not bare[end].strip() and (start == 0 or not bare[start - 1].strip()):
        end += 1
    return "".join(lines[:start] + lines[end:])


def write_atomic(path: Path, text: str, *, restrict: Optional[Callable[[Path], bool]] = None) -> None:
    """Write ``text`` to ``path`` through a temporary file and a rename, so
    a reader sees the old or the new file. ``restrict`` limits the temporary
    file to its owner before any content goes in, else nothing is written.
    Without it an existing file keeps its POSIX mode and a new file gets
    the mode the umask leaves of 0o666. The system applies the umask as it
    creates the temporary file, since reading the umask would change it for
    every thread of the process."""
    path = Path(path)
    existing = path.exists()
    temporary = path.with_name(f".{path.name}.{os.urandom(4).hex()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o600 if restrict is not None or existing else 0o666)
    try:
        if restrict is not None:
            os.close(descriptor)
            descriptor = None
            if not restrict(temporary):
                raise OSError(f"{path.name} could not be restricted to its owner, so it was not written")
            stream = open(temporary, "w", encoding="utf-8", newline="")
        else:
            if existing and os.name == "posix":
                os.chmod(temporary, stat.S_IMODE(os.stat(path).st_mode))
            stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
            descriptor = None
        with stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
