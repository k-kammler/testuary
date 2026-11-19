"""The records that tie a phase's output to how it was made: the validity
records of generated tests, bound to their source, and each phase attempt's
origin with its generation condition (the settings sent, the composed
prompts, the limits, the measurement settings and the language package),
with the checks that refuse a resume or reuse under other inputs."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from testuary.languages import SuiteAuditFlag, SuiteValidityFinding, TargetIdentity
from testuary.phase_state import timing_fields
from testuary.generation_evidence import source_sha256
from collections.abc import Mapping
import math
from urllib.parse import urlsplit
from testuary.models import PhaseModelSettings, resolve_model_identity
from testuary.settings import (
    PHASE_SECTIONS, ConfigurationError, _freeze, _model_credentials, _safe, _without_secrets)


RECORD_SCHEMA_VERSION = 1


def direct_record(
    *,
    identity: TargetIdentity,
    analyzed_source: str,
    findings: Sequence[SuiteValidityFinding] = (),
    removed_tests: Iterable[str] = (),
    final_source: Optional[str] = None,
    audit_flags: Sequence[SuiteAuditFlag] = (),
    stage: str,
) -> dict[str, Any]:
    """A validity record of a source checked directly: the findings in
    ``analyzed_source``, the tests removed and the clean source that
    remains (``final_source``, the analyzed source when None), bound to the
    target ``identity``."""
    final_source = analyzed_source if final_source is None else final_source
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_type": "direct",
        "stage": stage,
        "policy_version": identity.policy_version,
        "target_identity": identity.record(),
        "analyzed_source_sha256": source_sha256(analyzed_source),
        "findings": [asdict(item) for item in findings],
        "removed_tests": sorted(set(removed_tests)),
        "final_clean_source_sha256": source_sha256(final_source),
        "audit_flags": [asdict(item) for item in audit_flags],
        **timing_fields(),
    }


def derivation_record(
    *,
    identity: TargetIdentity,
    parent_source: str,
    child_source: str,
    operation: str,
    removed_tests: Iterable[str] = (),
    restored_tests: Iterable[str] = (),
    stage: str,
) -> dict[str, Any]:
    """A validity record of a source derived from another by ``operation``:
    the parent's and the child's hashes and the tests removed and restored,
    bound to the target ``identity``."""
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_type": "derivation",
        "stage": stage,
        "policy_version": identity.policy_version,
        "target_identity": identity.record(),
        "parent_source_sha256": source_sha256(parent_source),
        "child_source_sha256": source_sha256(child_source),
        "operation": operation,
        "removed_tests": sorted(set(removed_tests)),
        "restored_tests": sorted(set(restored_tests)),
        **timing_fields(),
    }


# The version of the generated-test prompts that each prompt transport
# record names.
PROMPT_TEMPLATE_VERSION = "generated-test-prompts-v1"


def prompt_transport_record(
    *,
    phase: str,
    request_idx: int,
    messages: Sequence[Mapping[str, str]],
    prompt_template_version: str,
    provider_transport: str,
) -> dict[str, Any]:
    """The record of one model request's prompt: its messages' roles, the hash
    of each message as canonical JSON, the prompt template version and the
    provider's transport."""
    canonical_messages = [
        json.dumps(
            dict(message), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        )
        for message in messages
    ]
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_type": "prompt_transport",
        "phase": phase,
        "request_idx": int(request_idx),
        "roles": [message.get("role", "") for message in messages],
        "message_sha256": [
            source_sha256(message) for message in canonical_messages
        ],
        "prompt_template_version": prompt_template_version,
        "provider_transport": provider_transport,
        **timing_fields(),
    }


def record_binds_source(
    record: Mapping[str, Any], source: str, identity: TargetIdentity,
) -> bool:
    """Whether the validity record ``record`` describes ``source`` for the
    target ``identity``: the same policy version and target, and the
    source's hash as its final source (a direct record) or its child
    source (a derivation record)."""
    if record.get("policy_version") != identity.policy_version:
        return False
    if record.get("target_identity") != identity.record():
        return False
    digest = source_sha256(source)
    if record.get("record_type") == "direct":
        return record.get("final_clean_source_sha256") == digest
    if record.get("record_type") == "derivation":
        return record.get("child_source_sha256") == digest
    return False


def write_json_atomic(path: Path, record: Mapping[str, Any]) -> None:
    """Write ``record`` to ``path`` as indented JSON with sorted keys, through
    a temporary file that replaces it. A missing directory is created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append ``record`` to ``path`` as one JSON line with sorted keys and
    flush it to the disk. A missing directory is created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_record(path: Path) -> dict[str, Any]:
    """The JSON object in ``path``. A file that holds another JSON value
    raises ValueError."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"validity record is not an object: {path}")
    return value


def verify_delivered_archive(
    *,
    archive_dir: Path,
    source: str,
    identity: TargetIdentity,
) -> tuple[bool, str]:
    """Verify a restored delivered suite against its validity chain."""
    try:
        consolidation = load_record(archive_dir / "consolidation_record.json")
        delivered = load_record(archive_dir / "validity_delivered.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"validity record missing or unreadable: {exc}"
    if not record_binds_source(delivered, source, identity):
        return False, "delivered validity record does not bind the restored source"

    if consolidation.get("delivered_from") != "reduced":
        return True, ""

    try:
        merged = load_record(archive_dir / "validity_merged.json")
        derivation_container = load_record(
            archive_dir / "validity_derivations.json"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"reduced validity chain missing or unreadable: {exc}"
    if merged.get("record_type") != "direct":
        return False, "merged validity record is not direct"
    if merged.get("policy_version") != identity.policy_version:
        return False, "merged validity policy version is stale"
    if merged.get("target_identity") != identity.record():
        return False, "merged validity target identity is stale"
    if merged.get("findings"):
        return False, "merged validity record contains rejected findings"

    records = derivation_container.get("records")
    if not isinstance(records, list) or not records:
        return False, "reduced output has no validity derivation records"
    current_hash = merged.get("final_clean_source_sha256")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return False, f"validity derivation {index} is not an object"
        if record.get("record_type") != "derivation":
            return False, f"validity derivation {index} has the wrong type"
        if record.get("policy_version") != identity.policy_version:
            return False, f"validity derivation {index} has a stale policy"
        if record.get("target_identity") != identity.record():
            return False, f"validity derivation {index} has a stale target"
        if record.get("parent_source_sha256") != current_hash:
            return False, f"validity derivation {index} breaks the hash chain"
        current_hash = record.get("child_source_sha256")
    if current_hash != source_sha256(source):
        return False, "validity derivation chain does not reach restored source"
    return True, ""


def language_package(language: str) -> Mapping[str, Any] | None:
    """The distribution and version that provide ``language``'s adapter, or
    for a language in one extension file its SHA-256 and item, or None for a
    language LangAdapter itself provides. A resume on another package,
    another version of it or an edited file is refused."""
    from importlib import metadata

    from langadapter import registration

    found = registration(language)
    extension = found.extension
    if extension is not None and extension.kind == "file":
        # A language in one file: its SHA-256 in place of a version and the
        # item as named, no resolved path, so a moved project still resumes.
        return {"distribution": None, "module": None, "version": None,
                "sha256": extension.sha256, "item": extension.item}
    module = getattr(found.factory, "__module__", "") or ""
    top = module.split(".")[0]
    if top == "langadapter":
        return None
    distributions = metadata.packages_distributions().get(top) or []
    distribution = distributions[0] if distributions else None
    try:
        version = metadata.version(distribution) if distribution else None
    except metadata.PackageNotFoundError:
        version = None
    return {"distribution": distribution, "module": module, "version": version}


def _detached(value):
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_detached(item) for item in value]
    return value


def _encoded(value) -> bytes:
    return json.dumps(_detached(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def composition_digest(templates: Mapping[str, Any]) -> str:
    """The digest of the composed prompts before their target slots are
    filled: the system prompt, the user prompt template and the optional tool
    messages, after the language fragments and hints. Tool bindings, the
    execution identity and the target checks are recorded separately.
    """
    required = {"system_prompt", "user_prompt_template"}
    allowed = required | {"tool_messages"}
    if (not isinstance(templates, Mapping) or required - templates.keys()
            or templates.keys() - allowed):
        raise ConfigurationError("prompt composition needs its system/user templates and optional tool_messages")
    if any(not isinstance(templates[key], str) for key in required):
        raise ConfigurationError("prompt composition templates must be text")
    messages = templates.get("tool_messages", {})
    if not isinstance(messages, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in messages.items()):
        raise ConfigurationError("prompt composition tool_messages must map names to text")
    try:
        encoded = _encoded({**templates, "tool_messages": messages})
    except UnicodeError:
        raise ConfigurationError("prompt composition must contain valid UTF-8 text") from None
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, repr=False)
class GenerationCondition:
    phase: str
    language: str
    model: Mapping[str, Any] = field(repr=False)
    request: Mapping[str, Any] = field(repr=False)
    prompt_composition: str
    service_identity_known: bool
    request_options_known: bool
    language_package: Mapping[str, Any] | None = field(default=None, repr=False)
    #: Every limit on model-facing text, for the editing loops. A resume on
    #: other limits, other defaults included, is refused.
    limits: Mapping[str, str] | None = field(default=None, repr=False)
    #: The settings that decide how the phase's suites are measured
    #: (:func:`measurement_settings`). A resume on other values is refused.
    measurement: Mapping[str, str] | None = field(default=None, repr=False)
    #: The registered language the tests are written in, where it is not
    #: ``language`` (``python`` for Python tests of a C file), else None.
    test_language: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "model", _freeze(self.model))
        object.__setattr__(self, "request", _freeze(self.request))
        if self.language_package is not None:
            object.__setattr__(self, "language_package", _freeze(self.language_package))

    def export(self) -> dict[str, Any]:
        exported = {
            "format": 1, "phase": self.phase, "language": self.language,
            "model": _detached(self.model), "request": _detached(self.request),
            "prompt_composition": self.prompt_composition,
            "service_identity_known": self.service_identity_known,
            "request_options_known": self.request_options_known,
        }
        if self.test_language is not None:
            exported["test_language"] = self.test_language
        if self.language_package is not None:
            exported["language_package"] = _detached(self.language_package)
        if self.limits:
            exported["limits"] = dict(self.limits)
        if self.measurement:
            exported["measurement"] = dict(self.measurement)
        return exported


#: The mutation settings that decide a measurement's verdicts: the engine,
#: the catalogue and the per-mutant timeout rule. The worker count and the
#: distributor change waiting time, not verdicts, and are left out.
_MUTATION_MEASUREMENT = (
    "engine", "loop_catalogue", "cosmic_ray_operators",
    "pit_mutators", "cover_attribution", "full_after_selective_goal",
    "selective_switch_threshold", "per_mutant_timeout_seconds",
    "per_mutant_timeout_factor", "min_per_mutant_timeout_seconds",
    "mutation_max_timeout", "measurement_timeout_seconds",
)


#: A phase's settings that decide how its suites are run, cleaned and
#: reduced: the suite-run timeout, the loops' reverse-order note and the
#: reduction solver.
_PHASE_MEASUREMENT = (
    "suite_run_timeout_seconds", "report_reverse_order_failures",
    "reduction_solver", "reduction_solver_time_limit_seconds", "reduction_min_test_time_ms",
    "reduction_solver_random_seed", "reduction_tie_break",
)
#: The starter phase's settings that decide how a generated starter is
#: cleaned before it is accepted. A saved starter pool keeps them, so its
#: reuse under other values is refused (``generation_only``).
_STARTER_CLEANING = (
    "suite_run_timeout_seconds", "failing_test_cleanup_rounds",
)
#: Finishing's settings that decide its numbers, its own and the suite
#: timing's, by the field name its completion records them under
#: (phase_originals is [finishing] phase_scores). They are no part of a
#: generation condition: a resume measures the delivered suites again when
#: one changed (``finishing_settings``).
_FINISHING_MEASUREMENT = {
    "measure_mutation": "measure_mutation", "phase_originals": "phase_scores",
    "measure_lint": "measure_lint", "coverage_timeout_seconds": "coverage_timeout_seconds",
    "suite_run_timeout_seconds": "suite_run_timeout_seconds",
}
_FINISHING_TIMING = ("measure", "run_count", "warmup_count")


def _engine_that_ran(language: str | None, configured: Any) -> Any:
    """The mutation engine a run of ``language`` uses for the configured
    ``engine`` value, which its records name (Cosmic Ray for Python, PIT for
    Java and Mull for C under the shipped default). ``configured`` as it is
    for no language or for an engine the language does not have."""
    from testuary import languages
    from testuary.languages.generic import _mutation_engine

    backend = languages.get_backend(language) if language else None
    if backend is None:
        return configured
    return _mutation_engine(backend._adapter_class(), configured) or configured


def _settings_naming_engine(echo: Any, language: str | None) -> None:
    """Name in the settings echo ``echo`` (:func:`recorded_settings`) the
    mutation engine that runs for ``language`` (:func:`_engine_that_ran`),
    in place."""
    sections = echo.get("sections") if isinstance(echo, Mapping) else None
    mutation = sections.get(PHASE_SECTIONS["mutation"]) if isinstance(sections, Mapping) else None
    if isinstance(mutation, dict) and "engine" in mutation:
        mutation["engine"] = _engine_that_ran(language, mutation["engine"])


def _measurement_naming_engine(measurement: Mapping[str, Any], language: str) -> dict[str, Any]:
    """Recorded measurement settings with their engine read as the engine
    that ran for ``language``, so that a record of the shipped default
    compares equal to the language's own engine."""
    values = dict(measurement)
    try:
        engine = json.loads(values["engine"])
    except (KeyError, TypeError, ValueError):
        return values
    values["engine"] = json.dumps(_engine_that_ran(language, engine))
    return values


def measurement_settings(phase: str, language: str | None) -> dict[str, str]:
    """The resolved settings that decide how ``phase``'s suites are run,
    cleaned, reduced and measured, as text: whether and how phase ends
    measure mutation, the mutation settings above, the phase's suite-run,
    delivered-suite and reduction settings its section holds, and for the
    starter suites the starter-pool mode and the starter cleaning. The engine
    is the one that runs for ``language``. Results measured under other
    values would mix with these on a resume, so a resume under other values
    is refused."""
    from testuary import config
    from testuary.settings import AGENT_NAMES

    phase_config = config.get_agent_config(
        AGENT_NAMES[phase], parent="test_coverage" if phase == "mutation" else None)
    mutation = config.get_agent_config("test_mutation", parent="test_coverage")
    values: dict[str, Any] = {
        "phase_end_mutation": config.consolidation_measure_mutation(),
        "reduced_suite_mutation": config.consolidation_mutation_method(),
        **{key: phase_config[key] for key in _PHASE_MEASUREMENT if key in phase_config},
        **{key: mutation.get(key) for key in _MUTATION_MEASUREMENT},
    }
    values["engine"] = _engine_that_ran(language, values["engine"])
    if phase == "starter":
        values["suite_mutation"] = phase_config.get("suite_mutation")
        values["failing_test_cleanup_rounds"] = phase_config.get("failing_test_cleanup_rounds")
    return {key: json.dumps(value) for key, value in values.items()}


def finishing_settings() -> dict[str, str]:
    """Finishing's resolved settings, as text, as its completion records
    them."""
    from testuary import config

    finishing, timing = config.load_config("finishing"), config.load_config("timing")
    values = {field: finishing.get(key) for field, key in _FINISHING_MEASUREMENT.items()}
    values.update({f"[{config.phase_section('timing')}] {key}": timing.get(key)
                   for key in _FINISHING_TIMING})
    return {key: json.dumps(value) for key, value in values.items()}


def _check_test_language(test_language: str | None, language: str) -> None:
    """Refuse a tests' language that is no text, empty or the target's own
    ``language`` (GenerationCondition.test_language)."""
    if test_language is not None and (
            not isinstance(test_language, str) or not test_language.strip()
            or test_language == language):
        raise ConfigurationError("a tests' language must be another language than the target's")


def capture_condition(
    settings: PhaseModelSettings, *, language: str,
    templates: Mapping[str, Any], extension_version: str | None = None,
    extension_sha256: str | None = None,
    initial_generation: Mapping[str, Any] | None = None,
    language_package: Mapping[str, Any] | None = None,
    limits: Mapping[str, str] | None = None,
    measurement: Mapping[str, str] | None = None,
    test_language: str | None = None,
) -> GenerationCondition:
    """The generation condition of one phase from its resolved settings:
    what is sent to the model, the composed prompts, the limits and the
    measurement settings. Alias labels, layer provenance, prices, the model
    and phase caps, request timeouts and endpoint queries are left out, and
    redacted request options cannot be compared. An added provider's
    distribution version, or the SHA-256 of its file, comes from the caller
    (``extension_version``, ``extension_sha256``). The
    starter suites pass their sampling seeds' base and their reasoning grace (the cutoff for a
    streamed answer that does not start) as ``initial_generation``. Without
    it they are recorded as unknown, not as defaults. A language from another
    package carries that package's distribution and version
    (:func:`language_package`), and tests in another language than the
    target's carry theirs (``test_language``).
    """
    if not isinstance(settings, PhaseModelSettings):
        raise TypeError("condition capture requires resolved phase model settings")
    if not isinstance(language, str) or not language.strip():
        raise ConfigurationError("generation condition needs a language")
    _check_test_language(test_language, language)
    package_record = dict(language_package) if language_package is not None else None
    if extension_version is not None and (
            not isinstance(extension_version, str) or not extension_version.strip()):
        raise ConfigurationError("the client factory's version must be text or unavailable")
    if settings.phase != "starter" and initial_generation is not None:
        raise ConfigurationError("starter generation controls are only for the starter phase")
    initial_controls = None
    if initial_generation is not None:
        if not isinstance(initial_generation, Mapping) or initial_generation.keys() != {
                "seed_base", "reasoning_grace_seconds"}:
            raise ConfigurationError("starter generation controls need exactly seed_base and reasoning_grace_seconds")
        seed_base = initial_generation["seed_base"]
        grace = initial_generation["reasoning_grace_seconds"]
        if type(seed_base) is not int:
            raise ConfigurationError("starter generation seed_base must be an integer")
        if grace is not None:
            if type(grace) not in {int, float}:
                raise ConfigurationError("starter generation reasoning_grace_seconds must be a nonnegative finite number or unset")
            try:
                grace = float(grace)
            except OverflowError:
                raise ConfigurationError("starter generation reasoning_grace_seconds must be a nonnegative finite number or unset") from None
            if not math.isfinite(grace) or grace < 0:
                raise ConfigurationError("starter generation reasoning_grace_seconds must be a nonnegative finite number or unset")
            grace = grace or None
        initial_controls = {"seed_base": seed_base, "reasoning_grace_seconds": grace}
    identity = settings.identity.export()
    model = {key: identity[key] for key in ("provider", "model", "api_url", "deployment")}
    model.update({
        "protocol": settings.protocol, "reasoning_dialect": settings.reasoning_dialect,
        "chat_model": settings.chat_model,
        "extension_version": extension_version if settings.protocol == "added" else None,
    })
    if settings.protocol == "added" and extension_sha256:
        # Only a provider in one file records it, so older records compare as before.
        model["extension_sha256"] = extension_sha256
    values = {key: value for key, value in settings.values.items() if key not in {
        "request_timeout_seconds", "model_max_output_tokens", "phase_max_output_tokens",
    }}
    options = _safe(settings.request_options)
    request = {"values": values, "request_options": options,
               "reasoning_body": settings.reasoning_body}
    if settings.phase == "starter":
        request["initial_generation"] = initial_controls
    return GenerationCondition(
        settings.phase, language, model, request,
        composition_digest(templates), not bool(urlsplit(settings.identity.api_url).query),
        _encoded(options) == _encoded(settings.request_options),
        package_record, dict(limits) if limits else None,
        dict(measurement) if measurement else None, test_language,
    )


#: Where the service is reached, not which service answers: a reuse does
#: not compare it, it reports a change (:func:`reuse_warnings`).
_ADDRESS = "model.api_url"


def reuse_problems(recorded: Any, current: GenerationCondition) -> tuple[str, ...]:
    """The unknown or changed inputs that refuse a reuse, by name, without
    settings values or prompt contents. A result without a condition, or
    with one in an unknown format, is refused. The model address is not
    one of them: the provider and the model name decide which service
    answers."""
    if not isinstance(recorded, Mapping):
        return ("recorded generation condition is unknown: these results have no generation "
                "records, so they cannot be resumed or reused. Start a fresh run in an empty "
                "output directory",)
    if type(recorded.get("format")) is not int or recorded.get("format") != 1:
        return ("recorded generation condition format is unknown, so these results cannot be "
                "resumed or reused. Start a fresh run in an empty output directory",)
    problems = []
    if recorded.get("service_identity_known") is not True or not current.service_identity_known:
        problems.append("service identity is unknown (unrecorded endpoint routing)")
    if recorded.get("request_options_known") is not True or not current.request_options_known:
        problems.append("request options are unknown after record redaction")
    previous_package = recorded.get("language_package")
    current_package = (dict(current.language_package)
                       if current.language_package is not None else None)
    if current_package is not None and not (current_package.get("version")
                                            or current_package.get("sha256")):
        problems.append("the language package's version is unknown")
    elif previous_package != current_package:
        problems.append(f"the language file {current_package['item']} changed"
                        if current_package and current_package.get("sha256")
                        else "the language package changed")
    if current.phase == "starter" and current.request.get("initial_generation") is None:
        problems.append("starter generation controls are unknown")
    if current.model["protocol"] == "added" and not (current.model["extension_version"]
                                                     or current.model.get("extension_sha256")):
        problems.append("the client factory's current distribution version is unknown")
    previous_model = recorded.get("model")
    if isinstance(previous_model, Mapping) and previous_model.get("protocol") == "added" and not (
            previous_model.get("extension_version") or previous_model.get("extension_sha256")):
        problems.append("the client factory's recorded distribution version is unknown")
    elif (isinstance(previous_model, Mapping) and current.model.get("extension_sha256")
            and previous_model.get("extension_sha256") != current.model["extension_sha256"]):
        problems.append(f"the client factory's file {current.model['chat_model'].rpartition(':')[0]} "
                        "changed")
    previous_limits = recorded.get("limits") or {}
    current_limits = dict(current.limits or {})
    if not isinstance(previous_limits, Mapping):
        problems.append("recorded limits are invalid")
    elif current_limits and not previous_limits:
        problems.append("recorded limits are unknown")
    elif dict(previous_limits) != current_limits:
        changed = sorted(set(previous_limits) | set(current_limits),
                         key=lambda name: name)
        changed = [name for name in changed
                   if previous_limits.get(name) != current_limits.get(name)]
        problems.append(f"limits changed: {', '.join(changed)}")
    previous_measurement = recorded.get("measurement") or {}
    current_measurement = dict(current.measurement or {})
    if current.measurement is None:
        # A generation-only condition (a reused starter pool) does not judge
        # how the suites are measured.
        pass
    elif not isinstance(previous_measurement, Mapping):
        problems.append("recorded measurement settings are invalid")
    elif current_measurement and not previous_measurement:
        problems.append("recorded measurement settings are unknown")
    else:
        # The settings the current condition names: all of them for a phase,
        # the cleaning settings for a reused starter pool (generation_only). A
        # setting the record lacks is reported. A recorded engine reads as the
        # engine that ran, so a record naming the shipped default matches.
        previous_measurement = _measurement_naming_engine(previous_measurement, current.language)
        missing = sorted(name for name in current_measurement if name not in previous_measurement)
        changed = sorted(name for name in current_measurement if name in previous_measurement
                         and previous_measurement[name] != current_measurement[name])
        if missing:
            problems.append(f"recorded measurement settings lack: {', '.join(missing)}")
        if changed:
            problems.append(f"measurement settings changed: {', '.join(changed)}")
    # The limits and the measurement settings are compared above.
    recorded = {key: value for key, value in recorded.items()
                if key not in ("limits", "measurement")}

    def compare(previous, expected, path):
        if isinstance(expected, Mapping):
            if not isinstance(previous, Mapping):
                problems.append(f"{path or 'generation condition'} is unknown or invalid")
                return
            if previous.keys() - expected.keys():
                problems.append(f"{path or 'generation condition'} has unsupported fields")
            for key, value in expected.items():
                nested = f"{path}.{key}" if path else key
                if nested == _ADDRESS:
                    continue
                if key not in previous:
                    problems.append(f"{nested} is unknown")
                else:
                    compare(previous[key], value, nested)
            return
        try:
            equal = _encoded(previous) == _encoded(expected)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            equal = False
        if not equal:
            problems.append(f"{path} changed")

    expected = current.export()
    expected.pop("limits", None)
    expected.pop("measurement", None)
    compare(recorded, expected, "")
    return tuple(problems)


def reuse_warnings(recorded: Any, current: GenerationCondition) -> tuple[str, ...]:
    """What a reuse accepts but reports, without settings values: a model
    address (``api_url``) other than the recorded one, such as a local proxy
    on another port."""
    previous = recorded.get("model") if isinstance(recorded, Mapping) else None
    if isinstance(previous, Mapping) and previous.get("api_url") != current.model["api_url"]:
        return ("the model address (api_url) differs from the one they were recorded with",)
    return ()


def require_matching_condition(recorded: Any, current: GenerationCondition) -> None:
    """Refuse, with ConfigurationError, a reuse whose recorded generation
    condition ``recorded`` differs from ``current``, naming each
    difference."""
    problems = reuse_problems(recorded, current)
    if problems:
        raise ConfigurationError(
            f"requested {current.phase} reuse refused: " + ". ".join(problems)
            + ". Use a new output/run or select an explicitly compatible stage."
        )


def provider_extension(model: PhaseModelSettings, secrets: Sequence[str] = ()):
    """The ``Extension`` of an added provider's ``chat_model`` item, the text
    before its last colon: a Python file relative to ``model.chat_model_dir``
    or a module name, loaded through LangAdapter's ``load_extension``, which
    runs its top-level code once per process. It gives the resolved file,
    that file's SHA-256 and an installed module's distribution and version.
    None for another protocol. An item that cannot load raises
    ConfigurationError naming it, with the error's type and message and
    ``secrets`` (the request's credential) removed."""
    if model.protocol != "added":
        return None
    from langadapter import InvalidRequest, RegistrationConflict, load_extension

    item = model.chat_model.rpartition(":")[0]
    try:
        return load_extension(item, relative_to=model.chat_model_dir)
    except (InvalidRequest, RegistrationConflict) as exc:
        detail = exc.message if isinstance(exc, InvalidRequest) else f"{item}: {exc}"
        raise ConfigurationError(f"chat_model {model.chat_model}: "
                                 f"{_without_secrets(detail, secrets)}") from None


def current_condition(phase: str, language: str, source_dir: str, *, templates=None):
    """The condition of ``phase`` from the templates the phase itself
    composes. Explicit model sources and direct calls carry no configured
    provider's identity, also inside an invocation.
    """
    from testuary.agent import _limits
    from testuary import models as _model_source
    from testuary import config
    from testuary import hints as hints_module
    from testuary import prompts
    from testuary.models import resolve_phase_model

    settings = config.active_settings()
    if settings is None or _model_source.current() is not None:
        return None
    model = resolve_phase_model(settings, phase)
    controls = None
    if phase == "starter":
        cfg = config.get_agent_config("test_initial")
        grace = cfg.get("reasoning_grace_seconds") or None
        # The record keeps its field names when the settings are renamed.
        controls = {"seed_base": cfg["seed_base"],
                    "reasoning_grace_seconds": grace}
    if templates is None:
        from testuary.settings import AGENT_NAMES

        templates = prompts.resolve_agent_prompts(
            AGENT_NAMES[phase], language, source_dir=source_dir,
            hint_profile=hints_module.hint_profile(),
        )
    extension = provider_extension(model, _model_credentials(settings)[1])
    return capture_condition(model, language=language,
                             templates={key: templates[key] for key in
                                        ("system_prompt", "user_prompt_template", "tool_messages") if key in templates},
                             extension_version=extension.version if extension else None,
                             extension_sha256=(extension.sha256 if extension and extension.kind == "file"
                                               else None),
                             initial_generation=controls,
                             language_package=language_package(language),
                             limits=_limits.recorded(settings, phase) if phase != "starter" else None,
                             measurement=measurement_settings(phase, language),
                             test_language=_test_language(language, source_dir))


def _test_language(language: str, source_dir) -> str | None:
    """The language the tests of ``language``'s project are written in,
    where it is not ``language`` itself, else None."""
    from testuary import languages

    tests = languages.get_backend(language).test_language(source_dir)
    return tests if tests != language else None


def require_origin(path: Path, condition: GenerationCondition | None) -> tuple[str, ...]:
    """Refuse a reuse whose origin at ``path`` records a condition that does
    not match ``condition``, before any origin or artifact is written, and
    return what the reuse reports (:func:`reuse_warnings`). Nothing is
    checked without ``condition``."""
    if condition is None:
        return ()
    try:
        origin = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        origin = None
    except (OSError, ValueError):
        raise ConfigurationError(f"requested {condition.phase} reuse refused: generation origin cannot be read") from None
    recorded = origin.get("condition") if isinstance(origin, dict) else None
    require_matching_condition(recorded, condition)
    return reuse_warnings(recorded, condition)


def recorded_condition(path: Path):
    """The condition an origin file records, None without the file. It
    describes the generation, not the execution."""
    try:
        origin = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise ConfigurationError("generation origin cannot be read") from None
    return origin.get("condition") if isinstance(origin, dict) else None


def generation_only(condition: GenerationCondition | None) -> GenerationCondition | None:
    """``condition`` with only the measurement settings that decided how
    its starters were cleaned: a starter pool records how its suites were
    generated and accepted, not how a run measures them, so reusing it under
    other measurement settings is a new measurement, not a mix, while other
    cleaning settings would leave starters cleaned another way."""
    if condition is None:
        return None
    cleaning = {key: value for key, value in (condition.measurement or {}).items()
                if key in _STARTER_CLEANING}
    return replace(condition, measurement=cleaning or None)


def copy_origin(source: Path, destination: Path, condition: GenerationCondition | None) -> None:
    """Carry a reused starter's original settings and captured inputs forward,
    with the measurement settings of the run that reuses it."""
    if condition is None:
        return

    require_origin(source, generation_only(condition))
    try:
        origin = json.loads(source.read_bytes())
        inputs = {}
        captured = [(key, origin.get(key)) for key in ("prompt_profile", "hints")]
        # An origin recorded before extensions keeps its package_hints.
        captured += [(f"{key}_{index}", item) for key in ("extension_hints", "package_hints")
                     for index, item in enumerate(origin.get(key) or [], start=1)]
        for key, item in captured:
            if item is None:
                continue
            name = item.get("copy") if isinstance(item, dict) else None
            if (not isinstance(name, str) or not name or name in {".", ".."}
                    or "/" in name or "\\" in name):
                raise ValueError("invalid captured-input name")
            data = (source.parent / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != item.get("sha256"):
                raise ValueError("captured input does not match origin")
            retained_name = destination.stem + f".{key}.json"
            inputs[retained_name] = data
            item["copy"] = retained_name
    except (OSError, ValueError, AttributeError):
        raise ConfigurationError("reused generation origin has unavailable captured inputs") from None
    recorded = origin.get("condition")
    if isinstance(recorded, dict):
        if condition.measurement:
            recorded["measurement"] = dict(condition.measurement)
        else:
            recorded.pop("measurement", None)
    # The carried settings name the engine that runs, also where the reused
    # origin names the shipped default.
    _settings_naming_engine(origin.get("settings"), condition.language)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name, data in inputs.items():
        (destination.parent / name).write_bytes(data)
    write_json_atomic(destination, origin)


def recorded_settings(settings, language: str | None) -> dict[str, Any]:
    """The run's settings as its records hold them (generation_origin.json
    and resolved_settings.json): the export without secrets, with only what
    the run's model goes through, and the mutation engine that runs for
    ``language``. Of the model catalogue that is the alias
    and its provider's section, and of the provider sections only that
    provider's, since another one's address can name a private host. A
    model that a model source supplies, or that does not resolve, goes
    through none of them."""
    from testuary import models as _model_source
    from testuary.settings import PROVIDER_SECTIONS

    used: set[str] = set()
    if _model_source.current() is None:
        try:
            identity = resolve_model_identity(settings)
        except ConfigurationError:
            pass
        else:
            used = {identity.model_alias, identity.provider}
    recorded = settings.export()
    recorded["models"] = {name: fields for name, fields in recorded["models"].items() if name in used}
    recorded["sections"] = {name: fields for name, fields in recorded["sections"].items()
                            if name not in PROVIDER_SECTIONS or name in used}
    _settings_naming_engine(recorded, language)
    return recorded


def write_origin(path: Path, condition: GenerationCondition | None) -> None:
    """Record one phase attempt and its captured inputs, never on a reuse path."""
    if condition is None:
        return
    from testuary import _extensions, config
    from testuary import hints as hints_module
    from testuary import prompts
    from testuary._phase_keys import phase_key
    from testuary.settings import extension_items

    profile, hints = prompts.prompt_inputs()
    origin = {"format": 1, "condition": condition.export(),
              "settings": recorded_settings(config.active_settings(), condition.language),
              "prompt_profile": None,
              "hints": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    if profile is not None:
        filename = path.stem + ".profile.json"
        (path.parent / filename).write_bytes(profile.contents)
        origin["prompt_profile"] = {
            "path": str(profile.path), "name": profile.profile.name, "sha256": profile.sha256,
            "copy": filename, "changes": profile.profile.changes(
                phase_key(condition.phase), condition.language),
        }
    if hints is not None and hints_module.captured_hints_path() is not None:
        filename = path.stem + ".hints.json"
        (path.parent / filename).write_bytes(hints)
        origin["hints"] = {"path": str(hints_module.captured_hints_path()),
                           "sha256": hashlib.sha256(hints).hexdigest(), "copy": filename}
    # The extensions the run loaded and their hints, which stand before the
    # user's in the prompts. Recorded only when the run loaded some.
    loaded = _extensions.loaded(extension_items(config.active_settings()))
    if loaded:
        origin["extensions"] = [{"item": item, "file": extension.file, "sha256": extension.sha256,
                                 "distribution": extension.distribution,
                                 "version": extension.version}
                                for item, _, extension, _ in loaded]
    captured = []
    for index, item in enumerate(hints_module._captured_extension_hints(), start=1):
        filename = path.stem + f".extension_hints_{index}.json"
        (path.parent / filename).write_bytes(item.contents)
        captured.append({"item": item.item, "sha256": hashlib.sha256(item.contents).hexdigest(),
                         "copy": filename})
    if captured:
        origin["extension_hints"] = captured
    write_json_atomic(path, origin)


def phase_origin_path(target_dir: Path, phase: str, run_idx: int = 0) -> Path:
    """The generation record of ``phase`` in the run directory of repetition
    ``run_idx`` of ``target_dir``."""
    from testuary import project

    return project.run_dir(target_dir, run_idx) / "archive" / phase / "generation" / "record.json"
