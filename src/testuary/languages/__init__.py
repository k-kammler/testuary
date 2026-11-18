"""Testuary's languages: their backends, the project context each runs in,
Testuary's requirement profile and the project settings a language declares.

A language is found through LangAdapter's registration (``get_backend``,
``detect_language``), which alone does not establish that a run can use it.
The contexts translate Testuary's settings into an explicit
``ProjectContext``: the captured route, work directory and environment, with the
runtime Testuary resolves for Python and Java, or the one another
language's adapter discovers from its declared settings.

The profile (``testuary/langadapter_profile.json``, which LangAdapter finds
in the package's directory) names capability families in
Testuary's levels: required (a run cannot start without them) or optional
(what each improves, and the fallback a run takes without it). ``assess``
decides from a language's capability report which required families are
missing and which fallbacks apply, and the current run's applied fallbacks
are kept for its records. Which operations make up a family, and whether it
can be assessed at all, is LangAdapter's (``langadapter.hooks``,
``langadapter.check``). An optional family is assessed only when the
configured run uses its feature (``in_use``).

A language's adapter declares its project settings (LangAdapter's
``ProjectSetting``). Python's and Java's are part of Testuary's own
configuration. Those another language declares join ``[project]``:
``testuary setup`` asks them and ``testuary check`` reports each as the
adapter's ``setup_problem`` checks it (``declared_settings``).
"""
from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING, Mapping, Sequence, Any, Callable, Tuple
from langadapter import language_for, names, registration
from langadapter.errors import UnsupportedLanguage
# Testuary's own modules import these classes from here.
from testuary.languages.generic import (
    DeliveredRun, LanguageBackend, SuiteAuditFlag, SuiteValidityFinding, SourceValidation,  # noqa: F401
    TargetIdentity, TestFailure, TestRunResult, GenericBackend,  # noqa: F401
)
from contextlib import nullcontext
from dataclasses import dataclass, replace, field
import os
from pathlib import PurePosixPath
from langadapter.context import (
    ContainerExecution,
    Environment,
    LocalExecution,
    ProjectContext,
)
from testuary import config as config_module, project as project_module
from langadapter.execution import container_work_dir_for
import json
from functools import lru_cache
from importlib import resources
from langadapter.capabilities import CapabilityReport
from langadapter.check import assessed, family_hooks, family_issue, hook_issue  # re-exported
from langadapter.outcomes import Outcome


# One backend per registered language, created when it is first asked for. A
# test or the demo may override one (override_backend).
_ADAPTERS: Dict[str, LanguageBackend] = {}


def supported_file_extensions() -> tuple[str, ...]:
    """The registered languages' file extensions, read at each call."""
    return tuple(sorted({extension for name in names()
                         for extension in registration(name).file_extensions}))


def get_backend(name: str) -> Optional[LanguageBackend]:
    """The backend of the registered language ``name``, with its current
    file extensions, or None, so that a caller can refuse an unsupported
    language without try/except.
    """
    try:
        selected = registration(name)
    except UnsupportedLanguage:
        return None
    adapter = _ADAPTERS.get(selected.name)
    if adapter is None:
        adapter = GenericBackend(selected.name, selected.file_extensions)
        _ADAPTERS[selected.name] = adapter
    adapter.file_extensions = selected.file_extensions
    return adapter


def override_backend(
    name: str, backend: LanguageBackend,
) -> LanguageBackend:
    """Replace the registered backend for ``name`` for this process and
    return the one it replaced, which the caller passes back to restore it.
    Only an already registered language can be overridden. Registering a new
    language is a separate operation."""
    previous = get_backend(name)
    if previous is None:
        raise KeyError(f"no backend registered for {name!r}")
    _ADAPTERS[registration(name).name] = backend
    return previous


def detect_language(file_path: str) -> str:
    """The registered language name for ``file_path``'s file extension, or
    ``"unknown"``, so that no caller carries its own file extension map.
    """
    try:
        return language_for(file_path)
    except UnsupportedLanguage:
        return "unknown"


def available_language_names() -> list[str]:
    """Every registered language name, sorted: what ``--languages``
    accepts, and its default."""
    return list(names())


if TYPE_CHECKING:
    from testuary.settings import RunSettings


def setup_context(language: str, settings: RunSettings) -> ProjectContext:
    """The setup context of the selected original project, without resolving
    or acquiring tools: the runtime the language's adapter describes from its
    declared settings (``describe_runtime``), so an unselected runtime stays
    absent and a requested Python version is a provisioning description,
    never an install. The
    execution contexts below share its captured route, work directory, framework
    and environment, and differ in root and in the runtime they resolve.
    """
    from testuary.settings import ConfigurationError, RunSettings

    if not isinstance(settings, RunSettings):
        raise ConfigurationError("setup context requires explicitly resolved RunSettings")
    language = registration(language).name
    active = config_module.active_settings()
    if active is not None and active is not settings:
        raise ConfigurationError("setup context conflicts with the active run settings")
    scope = nullcontext() if active is settings else config_module.use_run_settings(settings)
    with scope:
        project = config_module.get_project_config()
        root = project["source_dir"]
        guest = settings.main.project_exec == "container"
        execution = _execution(root, guest, settings)
        environment = _environment(settings, guest, setup=True)
        return _language_context(language, root, guest, settings, execution, environment,
                                 project=project, describe=True)


@dataclass(frozen=True)
class SelectedGroup:
    """Ranked targets of one language, with their ``context``: the setup
    context of the selected original project (``setup_context``).
    """

    language: str
    context: ProjectContext
    sources: tuple[str, ...]


def selected_groups(sources: Sequence[str], settings: RunSettings) -> tuple[SelectedGroup, ...]:
    """The ranked ``sources`` grouped by registered language, in first-seen
    order, each group with its setup context over the selected source directory.
    The contexts are built before any native assessment, so an invalid
    selection is refused (ConfigurationError) before any work.
    """
    from testuary.settings import ConfigurationError

    if isinstance(sources, (str, bytes)):
        raise TypeError("sources must be a sequence of target paths")
    grouped: dict[str, list[str]] = {}
    for source in sources:
        if not isinstance(source, str) or not source.strip():
            raise ConfigurationError("source targets must be nonempty paths")
        try:
            language = language_for(source)
        except UnsupportedLanguage:
            raise ConfigurationError(f"no installed language handles the file {source}") from None
        grouped.setdefault(language, []).append(source)
    return tuple(SelectedGroup(language, setup_context(language, settings), tuple(selected))
                 for language, selected in grouped.items())


@dataclass(frozen=True)
class SetupRequirements:
    """What a selected workflow requires of a language: ``operations`` and
    ``facets`` for the native setup request, and ``capabilities``, the same
    requirements as the capability report names them for generation's
    preflight. One projection serves check, verify and generate.
    """

    operations: tuple[str, ...]
    facets: tuple[tuple[str, tuple[str, ...]], ...]
    capabilities: tuple[str, ...]


def setup_requirements(*, coverage: bool, mutation: bool) -> SetupRequirements:
    """The requirements of a workflow that measures ``coverage`` and
    ``mutation`` as the caller chose: the families the profile requires,
    running the tests with their test units always, coverage and mutation
    when measured. Every other family is optional, with its fallback.
    """

    if any(type(value) is not bool for value in (coverage, mutation)):
        raise TypeError("setup requirement selections must be booleans")
    operations = required_operations("run tests")
    capabilities = ("suite_runs",)
    if mutation:
        operations += ("mutations.operators", "mutations.open_session")
        capabilities += ("mutation",)
    facets = ()
    if coverage:
        facets = (("tests.run_suite", ("coverage",)),)
        capabilities += ("coverage",)
    return SetupRequirements(operations, facets, capabilities)


def _without_credentials(values: Mapping[str, str], settings: RunSettings) -> dict[str, str]:
    """``values`` without the run's model credentials, which only model
    requests keep. Names compare case-insensitively on Windows.
    """
    from testuary.settings import model_credential_variables

    def key(name: str) -> str:
        return name.upper() if os.name == "nt" else name

    excluded = {key(name) for name in model_credential_variables(settings)}
    return {name: value for name, value in values.items() if key(name) not in excluded}


def _launch_values(settings: RunSettings) -> dict[str, str]:
    """The captured launch environment for subject and host control commands."""
    from testuary.settings import ConfigurationError

    if not settings.environment_captured:
        raise ConfigurationError(
            "native execution requires a captured launch environment: pass "
            "environment=dict(os.environ) to RunSettings or use resolve_settings()")
    return _without_credentials(settings.environment, settings)


def _work_dir(settings: RunSettings) -> Optional[str]:
    """The host work directory, resolved once from the captured selection."""
    value = settings.environment.get("TESTUARY_PROJECT_WORK_DIR", "").strip()
    return str(project_module.resolve_host_path(value).resolve()) if value else None


def _execution(root, guest: bool, settings: RunSettings):
    control = _launch_values(settings)
    work_dir = _work_dir(settings)
    if not guest:
        return LocalExecution(work_dir=work_dir)
    execution = _container_execution(root, work_dir, settings.environment)
    # The host runner receives only the captured values, never later ambient ones.
    return replace(execution, control_environment=Environment(control, inherit=False))


def _environment(settings: RunSettings, guest: bool, *, setup: bool = False) -> Environment:
    """The target's environment: the captured values. A guest target gets
    only the selected tool values, never host values. For execution the
    guest environment stays an overlay, since a captured one would admit
    only the supervised route. Setup describes it as captured, as its guest
    admission already requires the selected supervisor. The language's
    tool variables are added by ``_language_context``.
    """
    # Host search/import values are not forwarded to guest commands.
    environment = {} if guest else _launch_values(settings)
    return Environment(environment, inherit=False if setup or not guest else None)


def _container_execution(root, work_dir, values: Mapping[str, str]):
    container = values.get("TESTUARY_PROJECT_CONTAINER", "").strip()
    if not container or not work_dir:
        raise ValueError("container execution requires a container and a work directory")
    if not root:
        raise ValueError("container execution requires a source directory")
    return ContainerExecution(
        container=container, work_dir=work_dir,
        runner=values.get("TESTUARY_PROJECT_CONTAINER_RUNNER", "").strip() or "docker",
        repository_dir=root,
        container_work_dir=container_work_dir_for(work_dir),
        helper_python=values.get("LANGADAPTER_HELPER_PYTHON", "").strip() or None,
    )


def _with_guest_helper(execution, python: Optional[str]):
    """A container's own project interpreter supervises its commands unless
    LANGADAPTER_HELPER_PYTHON selects another (an interpreter without
    process descriptors supervises by process group)."""
    if (not isinstance(execution, ContainerExecution) or execution.helper_python is not None
            or not python or "\\" in python or not PurePosixPath(python).is_absolute()
            or ".." in PurePosixPath(python).parts):
        return execution
    return replace(execution, helper_python=python)


def _execution_root(source_dir: str | None, guest: bool) -> str:
    if guest:
        return source_dir or project_module.project_repo_dir()
    return str(project_module.resolve_host_path(source_dir or ".").resolve())


def language_project_context(language: str, source_dir: str | None = None) -> ProjectContext:
    """The execution context of ``language``'s project at ``source_dir``:
    the captured route, work directory and environment, with the runtime the
    language's adapter discovers from its project settings
    (``project.project_runtime``). Every language takes this path."""
    settings = config_module.run_settings()
    guest = config_module.project_exec_mode() == "container"
    root = _execution_root(source_dir, guest)
    return _language_context(language, root, guest, settings, _execution(root, guest, settings),
                             _environment(settings, guest))


def language_setting_values(language: str, project: Optional[Mapping] = None,
                            settings: Optional[RunSettings] = None) -> dict[str, str]:
    """The values of ``language``'s declared project settings
    (``project_settings``), each setting's default where none is set: from
    the resolved project configuration (``project``, else read), or from the
    section Testuary's own configuration keeps a setting in (``[main]
    pytest_config``)."""
    project = config_module.get_project_config() if project is None else project
    own = _own_sections()
    values: dict[str, str] = {}
    for setting in getattr(registration(language).factory, "project_settings", ()) or ():
        section = own.get(setting.name)
        source = (project if section in (None, "project") else
                  config_module.load_config(section) if settings is None
                  else settings.sections[section])
        values[setting.name] = str(source.get(setting.name) or setting.default or "")
    return values


def _language_context(language: str, root: str, guest: bool, settings: RunSettings,
                      execution, environment: Environment, *,
                      project: Optional[Mapping] = None, describe: bool = False) -> ProjectContext:
    """A language's context: the runtime its adapter chooses from its
    project settings, for a run discovered at the selected project
    (``project.project_runtime``) or, for a setup, described
    (``describe_runtime``, a class method, which refuses with a
    ConfigurationError), the interpreter that can supervise a container's
    commands when the route has none, and the variables its tools read
    (``context_environment``) over the captured environment. A language
    registered with a factory function has none of them."""
    from langadapter.adapter import RuntimeChoice
    from langadapter.errors import InvalidRequest

    from testuary.settings import ConfigurationError

    factory = registration(language).factory
    project = config_module.get_project_config() if project is None else project
    values = language_setting_values(language, project, settings)
    if describe:
        choose = getattr(factory, "describe_runtime", None)
        try:
            choice = (choose(root, values, container=guest, environment=settings.environment,
                             resolve_path=project_module.resolve_host_path)
                      if choose is not None else RuntimeChoice())
        except InvalidRequest as refusal:
            raise ConfigurationError(f"[project] {refusal.field}: {refusal.message}") from refusal
    else:
        choice = project_module.project_runtime(language, values,
                                                root=str(project.get("source_dir") or ""))
    execution = _with_guest_helper(execution, choice.helper_python)
    environment_of = getattr(factory, "context_environment", None)
    tools = environment_of(values, container=guest) if environment_of is not None else {}
    if tools:
        environment = Environment({**environment, **tools}, inherit=environment.inherit)
    framework = {"test_framework": choice.test_framework} if choice.test_framework is not None else {}
    return ProjectContext(root=root, interpreter=choice.interpreter, execution=execution,
                          environment=environment,
                          interpreter_flags=tuple(choice.interpreter_flags),
                          runtime_version=_stated_runtime(project), **framework)


def _stated_runtime(project: Optional[Mapping] = None) -> Optional[str]:
    """``[project] runtime_version``, which states the runtime's version line
    instead of asking the runtime (for a runtime that cannot start when a
    prompt is written), or None: the adapter reads it as its own."""
    project = config_module.get_project_config() if project is None else project
    return str(project.get("runtime_version") or "").strip() or None


def required_operations(family: str) -> Tuple[str, ...]:
    """The operations a setup request requires for ``family``: each hook's
    first alternative."""
    return tuple(dict.fromkeys(operation for alternatives in family_hooks(family)
                               for operation in alternatives[0]))


def hooks_issue(family: str, issue_of: Callable[[str], Optional[str]]) -> Optional[str]:
    """Why a hook of ``family`` is missing, by ``issue_of`` (an operation's
    problem or None), or None when every hook has a supported alternative:
    LangAdapter's ``hook_issue`` over ``family_hooks``, as ``family_issue``
    reads a capability report. The native readiness check judges its own
    operation records with it."""
    return next((issue for issue in (hook_issue(alternatives, issue_of)
                                     for alternatives in family_hooks(family)) if issue), None)


#: Fallbacks that change Testuary's method, shown in the run's summary and
#: the readiness report as well as recorded.
METHOD_CHANGING = frozenset({"merging", "validity screening", "kill attribution"})

#: A workflow requirement (``SetupRequirements.capabilities``) and the
#: required family it selects.
_WORKFLOW_FAMILIES = {"suite_runs": "run tests", "coverage": "coverage",
                      "mutation": "mutation"}


@lru_cache(maxsize=1)
def profile() -> Dict[str, Any]:
    """Testuary's profile as published."""
    text = resources.files("testuary").joinpath("langadapter_profile.json").read_text(encoding="utf-8")
    return json.loads(text)


def families() -> Tuple[Dict[str, Any], ...]:
    """Every family of the profile, with its level."""
    return tuple({**item, "level": level["level"]}
                 for level in profile()["levels"] for item in level["families"])


@dataclass(frozen=True)
class Fallback:
    """An optional family a language lacks, and what the run does without it."""

    family: str
    level: int
    fallback: str
    reason: str

    @property
    def changes_method(self) -> bool:
        return self.family in METHOD_CHANGING

    def record(self) -> Dict[str, Any]:
        return {"family": self.family, "level": self.level,
                "fallback": self.fallback, "reason": self.reason}


@dataclass(frozen=True)
class Assessment:
    """What a capability report means for a run: why it cannot start, if it
    cannot, and the fallbacks it takes (as lines, and as records)."""

    refusals: Tuple[str, ...]
    fallbacks: Tuple[str, ...]
    applied: Tuple[Fallback, ...] = field(default=())


def _setting_on(section: str, key: str) -> bool:
    from testuary import config
    value = config.load_config(section).get(key, "")
    return str(value).strip().lower() in ("true", "1", "yes", "on")


#: The loop sections whose reverse-order note
#: (``report_reverse_order_failures``) runs suites as delivered.
_DELIVERED_RUN_SECTIONS = ("coverage", "mutation")


def in_use(family: str, *, mutation: Optional[bool] = None) -> bool:
    """Whether the configured run uses ``family``'s feature, so that its
    fallback applies. Code smells fill the lint columns only with
    ``[finishing] measure_lint``. Delivered runs are the normal run of a
    language whose runs as delivered take a session seed
    (``seeds_delivered_runs``), and any other language's
    normal run is ``run_test_files`` (``normal_run``), so a language
    without them loses only the loops' reverse-order note, on with
    ``report_reverse_order_failures``. Kill attribution
    serves the reductions that keep every kill, so only a workflow that
    measures mutation uses it (``mutation``, assumed when None). Every
    other optional family serves every run."""
    if family == "code smells":
        return _setting_on("finishing", "measure_lint")
    if family == "delivered runs":
        return any(_setting_on(section, "report_reverse_order_failures")
                   for section in _DELIVERED_RUN_SECTIONS)
    if family == "kill attribution":
        return mutation is not False
    return True


def assess(report: CapabilityReport, requirements=None) -> Assessment:
    """The profile against ``report``. The capabilities of
    ``requirements`` (``setup_requirements``) select the required families
    the workflow needs. Without it every required family is needed."""
    if report.outcome is not Outcome.COMPLETED:
        return Assessment(
            (f"the project environment could not be probed: {report.reason}",), ())
    needed = (None if requirements is None else
              {_WORKFLOW_FAMILIES[name] for name in requirements.capabilities
               if name in _WORKFLOW_FAMILIES})
    mutation = None if requirements is None else "mutation" in requirements.capabilities
    refusals, lines, applied = [], [], []
    for item in families():
        family = item["family"]
        if not assessed(family):
            continue
        if item["tier"] != "required" and not in_use(family, mutation=mutation):
            continue
        issue = family_issue(report, family)
        if issue is None:
            continue
        if item["tier"] == "required":
            if needed is None or family in needed:
                refusals.append(f"The language adapter lacks {family}, which Testuary "
                                f"needs: {issue}")
            continue
        fallback = Fallback(family, item["level"], item["fallback"], issue)
        applied.append(fallback)
        lines.append(f"The language adapter lacks {family}: {issue}. "
                     f"Without it, {item['fallback']}.")
    return Assessment(tuple(refusals), tuple(lines), tuple(applied))


# ---- the applied fallbacks of the current run --------------------------------

# One run is active per process at a time. The store holds the run's scope
# object itself, not its id, which a later run's scope can reuse.
_APPLIED: Dict[str, Tuple[Fallback, ...]] = {}
_APPLIED_SCOPE: list = [None]


def _current() -> Dict[str, Tuple[Fallback, ...]]:
    """The applied fallbacks of the current run, emptied when a new run's
    scope begins."""
    from testuary import config
    scope = config.active_run_scope()
    if _APPLIED_SCOPE[0] is not scope:
        _APPLIED.clear()
        _APPLIED_SCOPE[0] = scope
    return _APPLIED


def remember(language: str, applied: Tuple[Fallback, ...]) -> None:
    """Keep ``language``'s applied fallbacks for the current run's records."""
    _current()[language] = tuple(applied)


def applied(language: str) -> Tuple[Fallback, ...]:
    """The fallbacks the current run applies for ``language``."""
    return _current().get(language, ())


def records(language: str) -> Optional[list]:
    """The ``fallbacks`` record for ``language``: None when no fallback
    applies, so a run of a language with every family records nothing new."""
    taken = applied(language)
    return [fallback.record() for fallback in taken] or None


@lru_cache(maxsize=None)
def _own_keys() -> frozenset:
    from testuary import settings

    bundled = settings._read_ini_file(settings._bundled_data_path("config.ini"))
    return frozenset(key for values in bundled.values() for key in values)


@lru_cache(maxsize=None)
def _own_sections() -> Mapping[str, str]:
    """The section of Testuary's own configuration that holds each of its
    keys, read once from the packaged configuration."""
    from types import MappingProxyType

    from testuary import settings

    bundled = settings._read_ini_file(settings._bundled_data_path("config.ini"))
    return MappingProxyType({key: section for section, values in bundled.items() for key in values})


def declared_settings(language: str) -> Tuple:
    """The settings ``language``'s adapter declares that Testuary's own
    configuration does not hold."""
    from langadapter import registration

    return tuple(setting for setting in
                 getattr(registration(language).factory, "project_settings", ()) or ()
                 if setting.name not in _own_keys())

