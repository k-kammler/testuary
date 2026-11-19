"""The setup report of ``testuary check`` and of a run before its first model
request, as immutable values. Building one inspects nothing: its producers
supply the facts."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
import time
from typing import Literal


ReadinessStatus = Literal["verified", "missing_setup", "unsupported", "ambiguous", "unverified", "failed",
                          "refused"]
_STATUSES = {"verified", "missing_setup", "unsupported", "ambiguous", "unverified", "failed", "refused"}


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"invalid {name}: expected {', '.join(sorted(choices))}")


def _text(value, name, *, optional=False):
    if not isinstance(value, str) and not (optional and value is None):
        raise ValueError(f"{name} must be text" + (" or None" if optional else ""))


def _tuple(values, item_type, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        copied = tuple(values)
    except TypeError:
        raise ValueError(f"{name} must be a sequence") from None
    if any(type(value) is not item_type for value in copied):
        raise ValueError(f"invalid {name} item")
    return copied


def _settings_scope(settings):
    from testuary import config as config_module
    from testuary.settings import ConfigurationError

    active = config_module.active_settings()
    if active is not None and active is not settings:
        raise ConfigurationError("readiness conflicts with the active run settings")
    return nullcontext() if active is settings else config_module.use_run_settings(settings)


@dataclass(frozen=True)
class ReadinessItem:
    """One item of the report: an operation, tool or setting by ``key``,
    whether the run needs it, its status, what was selected and its version,
    the reason, the next action and a diagnostic file (``diagnostic_path``).
    It speaks for that operation only, not for the language as a whole. The
    status is one of "verified", "missing_setup", "unsupported", "ambiguous",
    "unverified", "failed" and "refused" (``ReadinessStatus``). ``tried_by``
    names what first tries an item not tried yet: "verification"
    (``testuary check --verify``) or "run" (only a real run), otherwise None.
    It matters only while the item is unverified."""

    key: str
    required: bool
    status: ReadinessStatus
    selected: str | None = None
    version: str | None = None
    reason: str = ""
    action: str | None = None
    diagnostic_path: str | None = None
    tried_by: str | None = None

    def __post_init__(self):
        _text(self.key, "key")
        if not self.key or type(self.required) is not bool:
            raise ValueError("readiness items require a key and boolean required flag")
        _choice(self.status, _STATUSES, "readiness status")
        _text(self.reason, "reason")
        for name in ("selected", "version", "action", "diagnostic_path"):
            _text(getattr(self, name), name, optional=True)
        if self.tried_by not in (None, "verification", "run"):
            raise ValueError("tried_by names verification or run")


@dataclass(frozen=True)
class ReadinessLimit:
    """A limit as configured for ``phase``: a value in ``unit``
    (``configured``), ``unlimited`` or ``unknown``. Only a configured limit
    has a value, so None is never read as zero."""

    phase: str
    name: str
    value: int | float | None
    unit: str
    state: Literal["configured", "unlimited", "unknown"]

    def __post_init__(self):
        for name in ("phase", "name", "unit"):
            _text(getattr(self, name), name)
        _choice(self.state, {"configured", "unlimited", "unknown"}, "limit state")
        if self.state == "configured":
            if (type(self.value) not in (int, float) or self.value < 0
                    or type(self.value) is float and not math.isfinite(self.value)):
                raise ValueError("configured limits require a finite nonnegative value")
        elif self.value is not None:
            raise ValueError("unknown/unlimited limits must not invent a numeric value")


@dataclass(frozen=True)
class RunReadiness:
    """The setup report of ``testuary check``, or of a run before its first
    model request: its scope, status and exit code, the selections (project,
    execution, source files, phases, model, run directory and whether it starts
    fresh, resumes or reuses), the
    limits and the items. A configured key is reported as configured, not as
    accepted by the provider. The report holds the redacted facts its
    producer observed and promises nothing about later requests, cost or
    provider capacity.
    """

    scope: Literal["inspection", "verification"]
    status: ReadinessStatus
    exit_code: int
    source_dir: str | None
    execution: Literal["local", "container"] | None
    source_files: tuple[str, ...]
    phases: tuple[str, ...]
    model_alias: str | None
    provider: str | None
    model: str | None
    credentials_configured: bool | None
    output_dir: str | None
    output_mode: Literal["fresh", "resume", "reuse"]
    limits: tuple[ReadinessLimit, ...]
    items: tuple[ReadinessItem, ...]

    def __post_init__(self):
        _choice(self.scope, {"inspection", "verification"}, "readiness scope")
        _choice(self.status, _STATUSES, "readiness status")
        if type(self.exit_code) is not int or self.exit_code not in (0, 1, 2):
            raise ValueError("exit_code must be 0, 1 or 2")
        if self.execution is not None:
            _choice(self.execution, {"local", "container"}, "execution")
        _choice(self.output_mode, {"fresh", "resume", "reuse"}, "output mode")
        for name in ("source_dir", "model_alias", "provider", "model", "output_dir"):
            _text(getattr(self, name), name, optional=True)
        if self.credentials_configured is not None and type(self.credentials_configured) is not bool:
            raise ValueError("credentials_configured must be boolean or None")
        for name, item_type in (("source_files", str), ("phases", str),
                                ("limits", ReadinessLimit), ("items", ReadinessItem)):
            object.__setattr__(self, name, _tuple(getattr(self, name), item_type, name))

    def as_dict(self) -> dict:
        """The report as detached JSON data, without settings values,
        environment or keys."""
        import json

        return json.loads(json.dumps(asdict(self), allow_nan=False))


def _no_model_selected(settings) -> bool:
    """Whether ``[api] model`` names no model, and models.ini has no problem
    of its own: then a missing model is the only model problem."""
    from testuary.models import catalogue_problems

    value = settings.sections.get("api", {}).get("model", "")
    return isinstance(value, str) and not value.strip() and not catalogue_problems(settings.models)


def _configuration_report(request, settings) -> RunReadiness:
    """The report of the configuration alone: the model each phase resolves
    (an added provider's factory is imported, no client is built and no key
    is checked), the limits, the key's presence, the targets and the run
    directory. Files, the run directory and the language's operations are left
    to the adapter's inspection.
    """
    from testuary import orchestrator
    from testuary import config as config_module
    from testuary import project as project_module
    from testuary.orchestrator import PHASES, RunRequest, _invocation
    from testuary.settings import AGENT_NAMES, ConfigurationError, RunSettings, _redact_diagnostic

    if not isinstance(request, RunRequest) or not isinstance(settings, RunSettings):
        raise ConfigurationError("readiness requires RunRequest and explicitly resolved RunSettings")

    def safe(value):
        return _redact_diagnostic(str(value), settings)

    args = _invocation(request)
    with _settings_scope(settings):
        models, problems, credential = orchestrator._resolve_generation_models(args, settings)
        # No model selected is the one model problem the check reports and goes
        # past, to look at the project. Every other one refuses the check.
        unselected = bool(problems) and _no_model_selected(settings)
        if not problems:
            problems = _added_factory_problems(models, credential)
        if problems and not unselected:
            raise ConfigurationError(safe("model configuration refused:\n" + "\n".join(problems)))
        project = config_module.get_project_config()
        output = project_module.output_dir()
        phases = PHASES if request.stop_after_phase is None else PHASES[:PHASES.index(request.stop_after_phase) + 1]
        identity = next(iter(models.values())).identity if models else None
        limits = [ReadinessLimit("setup", name, float(project[name]), "seconds", "configured")
                  for name in ("setup_timeout_seconds", "setup_probe_timeout_seconds")]
        items = [ReadinessItem(
            f"configuration.model.{phase}", True, "missing_setup",
            reason="No model is selected: [api] model names no alias of models.ini and no model.",
            action=("Run testuary setup model, or run testuary init and add an alias to models.ini, "
                    "then select the alias with --model or [api] model."),
        ) for phase in (orchestrator._generation_model_phases(args) if unselected else ())]
        for phase, model in models.items():
            items.append(ReadinessItem(
                f"configuration.model.{phase}", True, "verified", selected=safe(model.protocol),
                reason=("The request settings resolve and the client factory imports. Nothing "
                        "was sent." if model.protocol == "added" else
                        "The request settings resolve. Nothing was sent."),
            ))
            for name, unit in (("request_timeout_seconds", "seconds"), ("max_output_tokens", "tokens")):
                value = model.values[name]
                limits.append(ReadinessLimit(phase, name, value, unit,
                                             "configured" if value is not None else "unlimited"))
            config = config_module.get_agent_config(
                AGENT_NAMES[phase], parent="test_coverage" if phase == "mutation" else None)
            if phase == "starter":
                limits.append(ReadinessLimit(phase, "attempts_per_suite", config["attempts_per_suite"],
                                             "attempts per suite", "configured"))
            else:
                for name, unit in (("max_requests_per_phase", "model requests"),
                                   ("max_input_tokens_per_phase", "tokens"),
                                   ("max_output_tokens_per_phase", "tokens"),
                                   ("max_total_tokens_per_phase", "tokens"),
                                   ("max_wall_clock_seconds_per_phase", "seconds")):
                    value = config[name]
                    if name == "max_requests_per_phase":
                        # A positive ceiling or unlimited (None). Settings
                        # refuse zero and negative values before this report.
                        limits.append(ReadinessLimit(phase, name, value, unit,
                                                     "configured" if value is not None else "unlimited"))
                    else:
                        limits.append(ReadinessLimit(phase, name, value if value > 0 else None, unit,
                                                     "configured" if value > 0 else "unlimited"))
        if identity is not None:
            # The key itself is private request input. The report says only
            # whether one is configured.
            if credential and credential.strip():
                items.append(ReadinessItem(
                    "configuration.credentials", True, "unverified",
                    reason="An API key is configured. This check does not try it.",
                    action="Nothing to do now. Generation uses the API key for its first model request.",
                    tried_by="run",
                ))
            else:
                # An empty key reaches the client unchanged, and a service
                # without keys (a local Ollama) accepts it. Only the service
                # can tell, so the first request is the check.
                from testuary.settings import credential_variable

                provider = safe(identity.provider)
                items.append(ReadinessItem(
                    "configuration.credentials", True, "unverified",
                    reason=("No API key is configured. A service without API keys works, and a service "
                            "that needs one refuses the first model request."),
                    action=(f"If the service needs an API key, set {credential_variable(provider)}, or name "
                            f"the key's variable with api_key_env in the [{safe(identity.model_alias)}] "
                            "section of models.ini."),
                    tried_by="run",
                ))
        items.extend((
            # Each language's check looks up its targets (_targets_item), and
            # the run checks them again before any work.
            ReadinessItem("source.targets", True, "unverified",
                          reason="The check stopped before it confirmed the targets.",
                          action=("Nothing to do now. The run checks that each target is in the "
                                  "project before any work."),
                          tried_by="run"),
            ReadinessItem("output.admission", True, "unverified",
                          reason="This check has not looked at the run directory yet. The run checks it before any work."),
            ReadinessItem("native.execution", True, "unverified",
                          reason="The checks of each language's tools have not run yet."),
        ))
        mode = "resume" if request.resume else "reuse" if orchestrator._cleanup_preserve_flags(args) else "fresh"
        target_numbers = config_module.parse_targets_spec(request.targets, len(request.source_files)) if request.targets else range(1, len(request.source_files) + 1)
        return RunReadiness(
            scope="inspection", status="unverified", exit_code=0,
            source_dir=safe(project["source_dir"]), execution=settings.main.project_exec,
            source_files=tuple(safe(request.source_files[target_number - 1]) for target_number in target_numbers),
            phases=phases, model_alias=safe(identity.model_alias) if identity else None,
            provider=safe(identity.provider) if identity else None, model=safe(identity.model) if identity else None,
            credentials_configured=bool(credential and credential.strip()) if models else None,
            output_dir=safe(output), output_mode=mode, limits=tuple(limits), items=tuple(items),
        )


_ADMITTED = {
    "source.targets": "The run found each target in the project.",
    "output.admission": "The run checked the run directory and kept what it reuses.",
    "native.execution": "The run's checks before it started found the tools each language needs.",
}


def pre_run_report(request, settings) -> RunReadiness:
    """The summary generate shows before its first model request.

    Called only after the run's own admission and preflight passed: the
    configuration facts of ``check`` with the items that admission
    established marked verified. Items only a run tries stay as they are.
    """
    from dataclasses import replace

    report = _configuration_report(request, settings)
    items = tuple(replace(item, status="verified", reason=_ADMITTED[item.key], action=None, tried_by=None)
                  if item.key in _ADMITTED else item for item in report.items)
    # No check ran: say what generation does with a configured key.
    items = tuple(replace(item, reason="An API key is configured. Generation uses it for its first "
                                       "model request.", action=None)
                  if item.key == "configuration.credentials" and report.credentials_configured else item
                  for item in items)
    status, exit_code = _readiness_summary(items, scope=report.scope)
    return replace(report, status=status, exit_code=exit_code, items=items)


def _added_factory_problems(models, credential=None) -> list:
    """Load each added provider's client factory once, without calling
    it, and list the failures with the phases they affect. The request's
    credential never appears in them."""
    from testuary.models import resolve_chat_model_factory
    from testuary.settings import ConfigurationError

    failures: dict = {}
    for phase, model in models.items():
        if model.protocol != "added":
            continue
        try:
            resolve_chat_model_factory(model, (credential,) if credential else ())
        except ConfigurationError as exc:
            failures.setdefault(str(exc), []).append(phase)
    return [f"{', '.join(phases)}: {message}" for message, phases in failures.items()]


def _configuration_error_report(error, *, settings=None, scope="inspection",
                                before_run=False) -> RunReadiness:
    """The report of a refused configuration: no selections and no
    diagnostic file. ``before_run``: a run was refused, not a check. A run
    refused for missing setup names it on the tools each language needs."""
    from testuary.settings import _MissingSetup, _redact_diagnostic

    reason = _redact_diagnostic(str(error), settings)
    item = (ReadinessItem("native.execution", True, "missing_setup", reason=reason,
                          action="Install what is missing or change the settings, then run the command again.")
            if isinstance(error, _MissingSetup) else
            ReadinessItem("configuration", True, "refused", reason=reason,
                          action=("Correct it, then run the command again." if before_run else
                                  "Correct it, then run the check again.")))
    # A refused configuration is no failure: the word of its exit status.
    return RunReadiness(
        scope=scope, status="refused", exit_code=2, source_dir=None, execution=None,
        source_files=(), phases=(), model_alias=None, provider=None, model=None,
        credentials_configured=None, output_dir=None, output_mode="fresh", limits=(),
        items=(item,),
    )


def _configuration_inputs(context, settings):
    """Leave Testuary's configuration and catalogue files out of the copy a
    verification prepares. A guest project's copy is the container's."""
    from langadapter import ContainerExecution, ProjectInputSelection
    from testuary import config as config_module

    if isinstance(context.execution, ContainerExecution):
        return None
    excluded = config_module.configuration_exclusions(context.root, settings)
    return ProjectInputSelection(exclude=excluded) if excluded else None


def _setup_request(invocation, settings, *, context, required_operations, inputs=None,
                   required_facets=(), allow_acquisition=False):
    """The setup request for one captured context and its selections. The
    language's adapter selects contexts and tools. The execution resolvers are
    not called here, since selecting a configured Python version can
    provision an environment. ``allow_acquisition`` lets the adapter
    download what a run would, which the tool check asks for and inspection
    never does.
    """
    from langadapter import ContainerExecution, Environment, ProjectContext, ProjectSetupRequest
    from pathlib import Path, PurePosixPath
    from testuary import config as config_module
    from testuary.orchestrator import RunRequest
    from testuary.settings import ConfigurationError, RunSettings

    if (not isinstance(invocation, RunRequest) or not isinstance(settings, RunSettings)
            or not isinstance(context, ProjectContext)):
        raise ConfigurationError("native setup requires RunRequest, RunSettings and a selected ProjectContext")
    if not isinstance(context.environment, Environment) or context.environment.inherit:
        raise ConfigurationError("native setup requires a captured target environment without ambient inheritance")
    if isinstance(context.execution, ContainerExecution):
        control = context.execution.control_environment
        if control is None or control.inherit:
            raise ConfigurationError("container native setup requires a separate captured host control environment")
    with _settings_scope(settings):
        project = config_module.get_project_config()
        target_numbers = (config_module.parse_targets_spec(invocation.targets, len(invocation.source_files))
                          if invocation.targets else range(1, len(invocation.source_files) + 1))
        path_type = PurePosixPath if isinstance(context.execution, ContainerExecution) else Path
        sources = []
        for target_number in target_numbers:
            path = path_type(invocation.source_files[target_number - 1])
            if path.is_absolute():
                try:
                    path = path.relative_to(path_type(context.root))
                except ValueError:
                    raise ConfigurationError("a target must lie inside the source directory") from None
            # Serialization is lexical only. Common native admission checks
            # root and target routes before any canonicalization or execution.
            sources.append(path.as_posix())
        return ProjectSetupRequest(
            context=context, source_files=tuple(sources),
            required_operations=required_operations, required_facets=required_facets,
            inputs=inputs,
            probe_timeout_s=float(project["setup_probe_timeout_seconds"]),
            total_timeout_s=float(project["setup_timeout_seconds"]),
            allow_acquisition=allow_acquisition,
        )


def _setup_item_status(item) -> ReadinessStatus:
    from langadapter import Availability, Outcome

    if item.ambiguous:
        return "ambiguous"
    if item.outcome is Outcome.NO_TESTS and not item.required:
        # A project without tests is no failure of an optional step.
        return "verified"
    if item.outcome not in (None, Outcome.COMPLETED, Outcome.MISSING_SETUP, Outcome.UNSUPPORTED):
        return "failed"
    if item.outcome is Outcome.MISSING_SETUP or item.readiness is Availability.MISSING_SETUP:
        return "missing_setup"
    if item.outcome is Outcome.UNSUPPORTED or item.readiness is Availability.UNSUPPORTED:
        return "unsupported"
    if item.outcome is Outcome.COMPLETED and item.readiness is Availability.AVAILABLE:
        return "verified"
    return "unverified"


def _tried_by(values, status):
    """LangAdapter's marker of what first tries an item not tried yet."""
    return values["tried_by"] if status == "unverified" else None


def _readiness_summary(items, *, scope):
    # Verification passes when everything in its own scope succeeded: an item
    # that only a real run tries is listed, not failed.
    required = {item.status for item in items if item.required and not (
        scope == "verification" and item.status == "unverified" and item.tried_by == "run")}
    for status in ("refused", "ambiguous", "failed", "missing_setup", "unsupported", "unverified"):
        if status in required:
            return status, (2 if status in ("refused", "ambiguous")
                            else 0 if status == "unverified" and scope == "inspection" else 1)
    return "verified", 0


def _named_after_missing_model(project) -> bool:
    """Whether the run directory is named after the model alias while no
    model is selected: the default, results/<model alias>, or an output_dir
    with the ${model} placeholder (project.output_dir)."""
    import re

    from testuary import config as config_module

    if str(config_module.load_config("api").get("model") or "").strip():
        return False
    raw = str(project.get("output_dir") or "")
    return not raw or re.search(r"\$(\{model\}|model\b)", raw) is not None


def _output_report(configuration: RunReadiness, invocation, settings) -> RunReadiness:
    """The run directory's item: whether the run may write there, checked
    without cleanup, allocation or a resume. A kept directory does not show
    that its saved artifacts fit the run. In a container the separation of
    results and project stays unverified: the run directory is a host path
    and the project a container path.
    """
    from dataclasses import replace
    from testuary import orchestrator
    from testuary import config as config_module
    from testuary import project as project_module
    from testuary.orchestrator import RunRequest, _invocation
    from testuary.settings import ConfigurationError, RunSettings, _redact_diagnostic

    if (not isinstance(configuration, RunReadiness) or not isinstance(invocation, RunRequest)
            or not isinstance(settings, RunSettings)):
        raise ConfigurationError("output inspection requires RunReadiness, RunRequest and RunSettings")
    with _settings_scope(settings):
        if _named_after_missing_model(config_module.get_project_config()):
            # Its name is not known yet, so the check looks at no directory.
            items = [item for item in configuration.items if item.key not in {
                "output.admission", "output.original_separation", "output.saved_artifacts"}]
            items.append(ReadinessItem(
                "output.admission", True, "unverified",
                reason="It is named after the model alias, and no model is selected yet.",
                action="Nothing to do now. The check looks at the run directory once a model is selected.",
            ))
            status, exit_code = _readiness_summary(items, scope=configuration.scope)
            return replace(configuration, status=status, exit_code=exit_code, output_dir=None,
                           items=tuple(items))
        preserve = orchestrator._cleanup_preserve_flags(_invocation(invocation))
        output = orchestrator._admit_output(project_module.output_dir(), preserve=preserve)
        project = config_module.get_project_config()
        items = [item for item in configuration.items if item.key not in {
            "output.admission", "output.original_separation", "output.saved_artifacts"}]
        items.append(ReadinessItem(
            "output.admission", True, "verified",
            reason="The run directory's path and what the run does with a directory in use were checked. No files were changed.",
        ))
        if settings.main.project_exec == "container":
            # The run directory is a host path and the project a path inside
            # the container, so the run has no separation requirement to meet:
            # it writes only to the host run directory (no host work copy).
            separation = ReadinessItem(
                "output.original_separation", False, "unverified",
                reason=("The run directory is on this computer and the project is inside the "
                        "container, so this check does not compare their locations."),
                action="Nothing to do now. The run writes only to the run directory.",
            )
        else:
            root = project_module.resolve_project_source_path(project["source_dir"])
            if root is None:
                separation = ReadinessItem(
                    "output.original_separation", True, "unverified",
                    reason=("The selected source directory was not found, so this check cannot confirm "
                            "that the run directory is kept apart from it."),
                    action="Select an existing --source-dir before the tool check.",
                )
            else:
                orchestrator._check_output_outside_source(output, root)
                separation = ReadinessItem(
                    "output.original_separation", True, "verified",
                    reason=("The run directory is in the project's Testuary directory, which "
                            "the run keeps apart from the project's files."
                            if output.is_relative_to(root) else
                            "The run directory is outside the source directory."),
                )
        items.append(separation)
        if preserve:
            items.append(ReadinessItem(
                "output.saved_artifacts", True, "unverified",
                reason=("Saved results are kept for this run. This check does not open them to "
                        "confirm that they fit it."),
                action=("Nothing to do now. The run validates the saved results before it reuses "
                        "them and stops if they do not fit."),
                tried_by="run",
            ))
        status, exit_code = _readiness_summary(items, scope=configuration.scope)
        return replace(configuration, status=status, exit_code=exit_code,
                       output_dir=_redact_diagnostic(str(output), settings), items=tuple(items))


_ASSESSMENT_ACTIONS = {
    "missing_setup": "Follow the actions of this language's items below, then run the check again.",
    "unsupported": ("This project setup is not supported for the language. The items below name "
                    "what is unsupported. Select a supported setup or leave the language out."),
    "failed": ("Read the reason and any diagnostic files listed below, correct the cause, then "
               "run the check again."),
}


def _fallback_items(operations, safe, *, mutation: bool) -> list:
    """An optional item for each optional family of Testuary's profile whose
    operations the language does not offer and whose feature the configured
    run uses (``profile.in_use``, ``mutation`` when the workflow measures
    mutation): what a run does without it, as the run announces it. A
    fallback that changes the method says so."""
    from testuary import languages

    items = []
    for family in languages.families():
        if family["tier"] != "optional":
            continue
        if not languages.in_use(family["family"], mutation=mutation):
            continue

        def issue_of(name):
            found = operations.get(name)
            if found is None or found[0].supported:
                return None
            return safe(found[1]["reason"]) or f"{name} is missing"

        issue = languages.hooks_issue(family["family"], issue_of)
        if issue is None:
            continue
        reason = (f"The language adapter lacks {family['family']}: {issue}. "
                  f"Without it, {family['fallback']}.")
        if family["family"] in languages.METHOD_CHANGING:
            reason += " This changes Testuary's method."
        items.append(ReadinessItem("native.fallback." + family["family"].replace(" ", "_"),
                                   False, "unsupported", reason=reason))
    return items


def _compose_setup_report(configuration: RunReadiness, report, *, request, settings) -> RunReadiness:
    """``_composed_setup_report`` read in ``settings``: the active run's, or
    entered for the call. Untyped settings reach the report's own refusal."""
    from testuary.settings import RunSettings

    with _settings_scope(settings) if isinstance(settings, RunSettings) else nullcontext():
        return _composed_setup_report(configuration, report, request=request, settings=settings)


def _composed_setup_report(configuration: RunReadiness, report, *, request, settings) -> RunReadiness:
    """The report from the adapter's typed setup observations, without
    admitting or starting a run. The adapters keep selection, inspection and
    offline verification. Nothing here builds a context, probes a tool or
    provisions one. Configuration, source and output items pass through
    unchanged.
    """
    from dataclasses import replace
    from langadapter import Availability, Outcome, ProjectSetupReport, ProjectSetupRequest
    from testuary.settings import ConfigurationError, RunSettings, _redact_diagnostic

    if (not isinstance(configuration, RunReadiness) or not isinstance(report, ProjectSetupReport)
            or not isinstance(request, ProjectSetupRequest) or not isinstance(settings, RunSettings)):
        raise ConfigurationError("native readiness requires typed configuration, setup request/report and RunSettings")

    def safe(value):
        return None if value is None else _redact_diagnostic(str(value), settings)

    exported = report.safe_export()
    items = [item for item in configuration.items if item.key != "native.execution"]
    # A language whose adapter has no check of its own is not refused
    # for it: its operations are tried by the run.
    no_own_check = report.outcome is Outcome.UNSUPPORTED and any(
        item.key == "provider" and item.readiness is Availability.UNSUPPORTED
        for item in report.items)
    assessment = ("verified" if report.outcome is Outcome.COMPLETED else
                  "missing_setup" if report.outcome is Outcome.MISSING_SETUP else
                  "unverified" if no_own_check else
                  "unsupported" if report.outcome is Outcome.UNSUPPORTED else "failed")
    items.append(ReadinessItem("native.assessment", True, assessment, reason=safe(exported["reason"]),
                               action=_ASSESSMENT_ACTIONS.get(assessment),
                               tried_by="run" if no_own_check else None))
    for item, values in zip(report.items, exported["items"]):
        if no_own_check and values["key"] == "provider":
            continue
        reason = safe(values["reason"])
        if values["candidates"]:
            # The choices found for an ambiguous item, else the places the check looked in.
            label = "Candidates: " if item.ambiguous else "Looked in: "
            reason = (reason + "\n" if reason else "") + label + ", ".join(safe(value) for value in values["candidates"])
        status = _setup_item_status(item)
        items.append(ReadinessItem(
            "native.item." + safe(values["key"]), item.required, status,
            selected=safe(values["selected"]), version=safe(values["version"]),
            reason=reason, action=safe(values["action"]),
            tried_by=_tried_by(values, status),
        ))
        if values["key"] == "runtime" and _testuary_environment_selected(
                values["selected"], request.context.root, settings):
            items.append(ReadinessItem(
                "native.runtime_selection", True, "missing_setup", selected=safe(values["selected"]),
                reason=("The only environment found is the one Testuary is installed in. A run uses "
                        "it for the project only when you name it."),
                action=("Set [project] python_executable (or TESTUARY_PROJECT_PYTHON_EXECUTABLE) to the "
                        "project's interpreter, or create the project's .venv. To use this "
                        "environment on purpose, select its interpreter there."),
            ))
    operations = {item.name: (item, values) for item, values in zip(report.operations, exported["operations"])}
    for name in request.required_operations:
        found = operations.get(name)
        if found is None and report.outcome is not Outcome.COMPLETED:
            # The assessment stopped at an earlier item, whose action is the
            # one to take. Updating LangAdapter would not help.
            status, action = "unverified", None
            reason = "Not tried: this language's check stopped at an earlier problem."
        elif found is None:
            status, reason = "unverified", "This language's check did not report this required operation."
            action = "Update the language adapter. This version does not report the operation."
        else:
            operation, values = found
            status = ("unsupported" if not operation.supported or operation.readiness is Availability.UNSUPPORTED else
                      "missing_setup" if operation.readiness is Availability.MISSING_SETUP else
                      "verified" if operation.readiness is Availability.AVAILABLE and report.outcome is Outcome.COMPLETED else
                      "unverified")
            reason, action = safe(values["reason"]), safe(values["action"])
        tried_by = _tried_by(values, status) if found is not None else None
        items.append(ReadinessItem("native.operation." + name, True, status, reason=reason, action=action,
                                   tried_by=tried_by))
    items.extend(_fallback_items(
        operations, safe, mutation="mutations.open_session" in request.required_operations))
    for index, path in enumerate(exported["diagnostic_paths"], 1):
        items.append(ReadinessItem(f"native.diagnostic.{index}", False, "unverified", diagnostic_path=safe(path)))
    status, exit_code = _readiness_summary(items, scope=report.scope)
    # Keep original configuration selections, never expose the executable native
    # context or substitute a disposable execution copy for the source directory.
    return replace(configuration, scope=report.scope, status=status, exit_code=exit_code, items=tuple(items))


def _testuary_environment_selected(selected, root, settings) -> bool:
    """Whether setup discovered Testuary's own environment for the project
    without an explicit selection in ``[project]``. Execution refuses it."""
    from testuary import project as project_module

    project = settings.sections.get("project", {})
    if not selected or str(project.get("python_executable") or "").strip():
        return False
    try:
        return project_module._is_testuary_environment(str(selected)) and not project_module._is_project_venv(
            str(selected), str(root or ""))
    except (OSError, ValueError):
        return False


def workflow_requirements(args):
    """What the configured workflow measures, as requirements on the adapter,
    for check, verify and a run alike. ``args`` is the parsed invocation
    (reuse options and stop phase). It covers the configured work, not a
    guess at how far a saved run got. Coverage is measured whenever the
    workflow generates suites or runs finishing.
    """
    from testuary import orchestrator
    from testuary import config as config_module
    from testuary.languages import setup_requirements

    phases = orchestrator._generation_model_phases(args)
    metrics = ("finishing" not in orchestrator._reused_phases(args)
               and getattr(args, "stop_after_phase", None) in (None, "finishing"))
    final_mutation = metrics and config_module.finishing_measures_mutation()
    mutation = ("mutation" in phases or final_mutation
                or bool(phases) and config_module.consolidation_measure_mutation())
    # Delivered runs and suite timing are optional families: a language
    # without them takes their fallback rather than being refused.
    return setup_requirements(coverage=bool(phases) or bool(metrics), mutation=bool(mutation))


def refuse_unselected_languages(groups, selected, settings) -> None:
    """Refuse selected targets outside an explicit language filter.

    A filter that excludes a named target is a conflict to resolve, not a
    reason to skip the target or its checks.
    """
    from testuary.settings import ConfigurationError, _redact_diagnostic

    if selected is None:
        return
    allowed = {str(name).strip().lower() for name in selected}
    outside = [source for group in groups if group.language not in allowed for source in group.sources]
    if outside:
        raise ConfigurationError(_redact_diagnostic(
            f"the language selection ({', '.join(sorted(allowed))}) excludes "
            f"targets: {', '.join(outside)}", settings))


def _mutation_hook(language: str, hook: str):
    """``language``'s adapter's mutation-settings check ``hook``, bound to an
    adapter without a runtime (the checks read settings only), or None when
    the language keeps LangAdapter's default, which refuses nothing."""
    from langadapter.adapter import LangAdapter

    from testuary import languages

    factory = registration_of(language)
    if getattr(factory, hook, None) is getattr(LangAdapter, hook):
        return None
    return getattr(languages.get_backend(language).source_backend(), hook)


def registration_of(language: str):
    """The adapter factory LangAdapter registered for ``language``."""
    from langadapter import registration
    return registration(language).factory


def session_dir_refusal(languages, *, create: bool) -> str | None:
    """Why a selected language's mutation tool cannot hold its local session
    in the configured directory (its adapter's ``mutation_dir_refusal``, in
    the tool's own words), or None. ``create`` makes a missing directory, as a
    run does. Without it, a missing directory that a run could create reads
    ``"missing"``."""
    from testuary import config as config_module

    config = config_module.get_agent_config("test_mutation", parent="test_coverage")
    for language in sorted(languages):
        check = _mutation_hook(language, "mutation_dir_refusal")
        refusal = check(config, create=create) if check is not None else None
        if refusal is not None:
            return refusal
    return None


def distributor_refusal(languages) -> str | None:
    """Why a selected language's mutation tool cannot run the configured
    workers (its adapter's ``mutation_workers_refusal``: Cosmic Ray's local
    distributor with more than one), or None. A run is refused with it before
    any request."""
    from testuary import config as config_module

    config = config_module.get_agent_config("test_mutation", parent="test_coverage")
    for language in sorted(languages):
        check = _mutation_hook(language, "mutation_workers_refusal")
        refusal = check(config) if check is not None else None
        if refusal is not None:
            return refusal
    return None


def _session_dir_item(languages, requirements, settings, *, verify: bool) -> ReadinessItem | None:
    """The configured session directory of a selected language's mutation tool
    (one whose adapter checks it, Python's Cosmic Ray), checked before any
    model request: a run would otherwise refuse it at its first mutation
    measurement. Inside a container the directory is the container's, which
    this check cannot see."""
    from testuary import config as config_module

    checked = {language for language in languages
               if _mutation_hook(language, "mutation_dir_refusal") is not None}
    if (not checked or "mutations.open_session" not in requirements.operations
            or not str(config_module.get_agent_config("test_mutation", parent="test_coverage").get(
                "cosmic_ray_local_session_dir") or "").strip()):
        return None
    key = "mutation.session_folder"
    if settings.main.project_exec == "container":
        return ReadinessItem(
            key, True, "unverified",
            reason=("The mutation session directory is inside the container, so this check cannot "
                    "test it. The run refuses an unusable directory at its first mutation measurement."),
            action="Nothing to do now if the directory exists and is writable in the container.",
            tried_by="run")
    refusal = session_dir_refusal(checked, create=verify)
    if refusal == "missing":
        return ReadinessItem(
            key, True, "unverified",
            reason="The mutation session directory does not exist yet. A run creates it.",
            action="Run testuary check --verify (the tool check) to create and test it.",
            tried_by="verification")
    if refusal is not None:
        return ReadinessItem(
            key, True, "failed", reason=refusal,
            action=("Select a writable directory on this computer's own disk (not a network drive) "
                    "for [mutation] cosmic_ray_local_session_dir, or leave it empty for the "
                    "system's temporary directory."))
    return ReadinessItem(key, True, "verified",
                         reason="The mutation session directory exists and is writable.")


def _extension_admission(sources, settings, *, verify: bool) -> tuple:
    """The targets the language checks take, and an item for each thing
    only the extensions decide. The tool check loads the extensions the
    settings name and checks the ``[project]`` settings their languages
    declare, as a run does. The quick check loads none: while the settings
    name extensions, it leaves a target no registered language handles and
    each setting of ``RunSettings.unchecked`` to the tool check."""
    from langadapter import language_for
    from langadapter.errors import UnsupportedLanguage

    from testuary import _extensions
    from testuary.settings import _check_unchecked, _redact_diagnostic, extension_items

    rows = extension_items(settings)
    if verify:
        _extensions.load(rows)
        _check_unchecked(settings)
    if verify or not rows:
        return list(sources), []
    later = "Nothing to do now. The tool check (testuary check --verify) loads the extensions."
    because = "An extension the settings name may add it, and this check loads no extension."
    selected, items = [], []
    for source in sources:
        try:
            language_for(source) if isinstance(source, str) else None
            selected.append(source)
        except UnsupportedLanguage:
            items.append(ReadinessItem(f"configuration.target.{len(items) + 1}", True, "unverified",
                                       selected=_redact_diagnostic(source, settings), tried_by="verification",
                                       reason=f"No registered language handles it. {because}", action=later))
    for index, text in enumerate(_unchecked_values(settings), start=1):
        items.append(ReadinessItem(f"configuration.unchecked.{index}", True, "unverified",
                                   selected=_redact_diagnostic(text, settings),
                                   tried_by="verification", action=later,
                                   reason=f"No registered language has it. {because}"))
    return selected, items


def _unchecked_values(settings) -> list:
    """Each setting of ``RunSettings.unchecked`` with its value, such as
    ``[project] language = toy``."""
    return [f"{name} = {settings.sections['project'].get(name.removeprefix('[project] '), '')}"
            for name in settings.unchecked]


def _extension_check_items(settings, *, verify: bool) -> list:
    """An item for each item of ``[main] extensions``, in the order the
    settings combine them. The quick check resolves an item through
    LangAdapter's ``resolve_extension``, which runs no extension code, and
    leaves one found only by importing its package to the tool check. The
    tool check validates the hints of each item _extension_admission loaded,
    each resolved file once."""
    from langadapter import ExtensionFoundOnlyByImport, InvalidRequest, resolve_extension

    from testuary import _extensions, hints
    from testuary.settings import ConfigurationError, extension_items

    later = "Nothing to do now. The tool check (testuary check --verify) loads it."
    items = []
    rows = _extensions.loaded(extension_items(settings)) if verify else extension_items(settings)
    for index, row in enumerate(rows, start=1):
        item, source = row[0], row[1]
        key, named = f"configuration.extensions.{index}", f"It is named in {_extensions.place(source)}."
        if verify:
            try:
                hints._captured((row,))
            except ConfigurationError as exc:
                items.append(ReadinessItem(key, True, "failed", selected=item, reason=str(exc),
                                           action="Correct the extension or its item, then run the check again."))
            else:
                items.append(ReadinessItem(key, True, "verified", selected=item,
                                           reason=f"It loads, and its hints are valid. {named}"))
            continue
        try:
            found = resolve_extension(item, relative_to=row[2])
        except ExtensionFoundOnlyByImport:
            items.append(ReadinessItem(key, True, "unverified", selected=item, tried_by="verification", action=later,
                                       reason=f"Only an import of its package finds it, which this check does not do. {named}"))
        except InvalidRequest as exc:
            items.append(ReadinessItem(key, True, "missing_setup", selected=item, reason=f"{exc.message}. {named}",
                                       action="Correct the item where it is named, or remove it."))
        else:
            items.append(ReadinessItem(key, True, "unverified", selected=item, tried_by="verification", action=later,
                                       reason=f"It is {found.file}. This check runs no extension code. {named}"))
    return items


def _declared_setting_items(languages, settings) -> list:
    """An item for each project setting a language's adapter declares
    (``project_settings``, Python's and Java's being Testuary's own): a set
    value checked by its kind, an unset one left to the language's
    discovery, which the run tries. A relative path is read from the
    project directory, where the adapter reads it and the setup asks it, not
    from the configuration's directory. Inside a container a path is the
    container's, which this check cannot look up: the run tries it."""
    from langadapter import registration

    from testuary import config as config_module
    from testuary import project as project_module
    from testuary.languages import language_setting_values
    from testuary.languages import declared_settings
    from testuary.setup import Probe
    from testuary.settings import _redact_diagnostic

    container = settings.main.project_exec == "container"
    root = (None if container else project_module.resolve_project_source_path(
        config_module.get_project_config()["source_dir"]))
    items = []
    for language in sorted(languages):
        values = language_setting_values(language)
        for setting in declared_settings(language):
            key = f"native.{_redact_diagnostic(language, settings)}.setting.{setting.name}"
            value = values.get(setting.name, "")
            if not value:
                items.append(ReadinessItem(
                    key, False, "unverified", reason=f"{setting.purpose} Not set.",
                    tried_by="run"))
                continue
            if setting.kind in ("path", "executable") and root is None:
                where = ("inside the container" if container else
                         "in a project directory that was not found")
                items.append(ReadinessItem(
                    key, True, "unverified", selected=_redact_diagnostic(value, settings),
                    reason=f"{setting.purpose} The path lies {where}, so this check does "
                           "not look it up.", tried_by="run"))
                continue
            # A kind without a path reads no base.
            problem = registration(language).factory.setup_problem(setting.name, value, (
                root if root is not None else config_module.configuration_base_dir()), Probe())
            items.append(ReadinessItem(
                key, True, "missing_setup" if problem else "verified",
                selected=_redact_diagnostic(value, settings),
                reason=setting.purpose + (f" {problem}" if problem else ""),
                action=(f"Set [project] {setting.name} to {setting.check or 'a fitting value'}."
                        if problem else None)))
    return items


def _targets_item(item: ReadinessItem, reports, languages: int, settings) -> ReadinessItem:
    """``item`` (``source.targets``) from the languages' checks, which look
    up every selected input before a language's own part runs: missing
    setup with the check's reason and action when an input is missing,
    verified when the own part of each of the ``languages`` checks ran,
    else ``item``. The report of an own part names the requested
    operations, and a check that stopped earlier names none."""
    from dataclasses import replace
    from langadapter import Availability
    from testuary.settings import _redact_diagnostic

    def safe(value):
        return None if value is None else _redact_diagnostic(str(value), settings)

    for report in reports:
        missing = next((values for found, values in zip(report.items, report.safe_export()["items"])
                        if found.key == "inputs" and found.readiness is Availability.MISSING_SETUP), None)
        if missing is not None:
            return replace(item, status="missing_setup", reason=safe(missing["reason"]),
                           action=safe(missing["action"]), tried_by=None)
    if len(reports) == languages and all(report.operations for report in reports):
        return replace(item, status="verified", reason="Each target is in the project directory.",
                       action=None, tried_by=None)
    return item


def _native_report(invocation, settings, *, verify=False) -> RunReadiness:
    """The adapter's setup report for one captured invocation, through
    LangAdapter's public setup functions, with the targets as the
    languages' checks found them. It does not admit a run: the run checks
    saved-work compatibility, source membership and the separation of guest
    and output again. Verification runs only on request, and it downloads
    what a run would, as a run does.
    """
    from dataclasses import replace
    from langadapter import InvalidRequest, inspect_project, verify_project
    from testuary import config as config_module
    from testuary.orchestrator import RunRequest, _invocation
    from testuary.languages import selected_groups
    from testuary.settings import ConfigurationError, RunSettings, _redact_diagnostic

    if (not isinstance(invocation, RunRequest) or not isinstance(settings, RunSettings)
            or type(verify) is not bool):
        raise ConfigurationError("native inspection requires RunRequest, RunSettings and a boolean verify flag")
    scope = "verification" if verify else "inspection"
    started = time.monotonic()
    # The public invocation guard owns the entire observation, including all
    # language callbacks. Private helpers reuse only this same settings object.
    with config_module.use_run_settings(settings):
        configuration = _configuration_report(invocation, settings)
        configuration = _output_report(configuration, invocation, settings)
        project = config_module.get_project_config()
        deadline = started + float(project["setup_timeout_seconds"])
        requirements = workflow_requirements(_invocation(invocation))
        target_numbers = (config_module.parse_targets_spec(invocation.targets, len(invocation.source_files))
                          if invocation.targets else range(1, len(invocation.source_files) + 1))
        # Construct every selection before the first native assessment. A later
        # invalid context or target must not cause an earlier native probe.
        admitted, left = _extension_admission(
            [invocation.source_files[target_number - 1] for target_number in target_numbers], settings,
            verify=verify)
        try:
            groups = selected_groups(admitted, settings)
        except ConfigurationError as exc:
            raise ConfigurationError(_redact_diagnostic(str(exc), settings)) from exc
        refuse_unselected_languages(groups, invocation.languages, settings)
        requests = []
        for group in groups:
            selected = replace(invocation, source_files=group.sources, targets=None)
            try:
                request = _setup_request(selected, settings, context=group.context,
                                         required_operations=requirements.operations,
                                         required_facets=requirements.facets,
                                         inputs=_configuration_inputs(group.context, settings),
                                         allow_acquisition=verify)
            except InvalidRequest as exc:
                raise ConfigurationError(_redact_diagnostic(str(exc), settings)) from exc
            requests.append((group.language, request))
        items = [item for item in configuration.items if item.key != "native.execution"]
        native_facts = replace(configuration, items=())
        assess = verify_project if verify else inspect_project
        reports = []
        for language, request in requests:
            namespace = "native." + _redact_diagnostic(language, settings) + "."
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                items.append(ReadinessItem(
                    namespace + "assessment", True, "failed",
                    reason=("The checks took longer than [project] setup_timeout_seconds "
                            "before this language's check could start."),
                    action=("Raise [project] setup_timeout_seconds, or select fewer languages or "
                            "targets, then run the check again."),
                ))
                continue
            request = replace(request, total_timeout_s=remaining,
                              probe_timeout_s=min(request.probe_timeout_s, remaining))
            try:
                report = assess(language, request)
                reports.append(report)
            except InvalidRequest as exc:
                # For example a host LangAdapter runs no project command on.
                raise ConfigurationError(_redact_diagnostic(str(exc), settings)) from exc
            converted = _compose_setup_report(native_facts, report, request=request, settings=settings)
            items.extend(replace(item, key=namespace + item.key.removeprefix("native."))
                         for item in converted.items)
        # A target left to the tool check keeps the targets unchecked.
        items = [_targets_item(item, reports, len(requests), settings)
                 if item.key == "source.targets" and len(admitted) == len(target_numbers)
                 else item for item in items] + left
        items.extend(_declared_setting_items({group.language for group in groups}, settings))
        items.extend(_extension_check_items(settings, verify=verify))
        session_dir_item = _session_dir_item({group.language for group in groups}, requirements,
                                      settings, verify=verify)
        if session_dir_item is not None:
            items.append(session_dir_item)
        status, exit_code = _readiness_summary(items, scope=scope)
        return replace(configuration, scope=scope, status=status, exit_code=exit_code, items=tuple(items))
