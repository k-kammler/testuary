"""The resolved configuration a phase reads: the run's settings and state,
the phase and agent configuration and the main settings. Every accessor
reads the settings of the active run (``use_run_settings``)."""

import contextlib
import dataclasses
import math
import os
import re
import sys
import threading
from collections.abc import Mapping, Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from testuary.settings import phase_section
from testuary import hints
from testuary import models
from testuary import project as project_module
from testuary import prompts
from testuary import settings as settings_module
if TYPE_CHECKING:
    from testuary.consolidation import ReductionSettings
    from testuary.settings import RunSettings
    from testuary.hints import ExtensionHints

@dataclasses.dataclass(eq=False)
class Run:
    """One invocation while ``use_run_settings`` holds it: its settings, an
    identity its worker threads share, the hint and prompt-profile inputs
    captured at its start and the extensions' hints a run captured at its
    start (``run_pipeline``), the source directory of a target's work copy
    (``set_invocation_source_dir``), and what holds for the whole run: the
    runtime each language's adapter chose for a selection
    (``project.project_runtime``)."""
    settings: "RunSettings"
    scope: object = dataclasses.field(default_factory=object)
    hints: Optional[bytes] = None
    hints_path: Optional[Path] = None
    extension_hints: Tuple["ExtensionHints", ...] = ()
    profile: Any = None
    source_dir_override: str = ""
    project_runtimes: Dict[tuple, Any] = dataclasses.field(default_factory=dict)


#: The active run, process-wide: the starter phase's worker threads read it.
_RUN: Optional[Run] = None

_RUN_GUARD = threading.Lock()


def active_run() -> Optional[Run]:
    """The active run, or None outside a run."""
    return _RUN


def current_run() -> Run:
    """The active run. Testuary's configuration is read inside a run only."""
    if _RUN is None:
        raise RuntimeError("no run settings are active: resolve them with "
                           "testuary.settings.resolve_settings and enter config.use_run_settings")
    return _RUN


def active_settings() -> Optional["RunSettings"]:
    """The active run's settings, or None outside a run."""
    return _RUN.settings if _RUN is not None else None


def run_settings() -> "RunSettings":
    """The active run's settings."""
    return current_run().settings


def active_run_scope() -> Optional[object]:
    """Identity of the active invocation, shared with its worker threads."""
    return _RUN.scope if _RUN is not None else None


def configured_environment() -> Mapping[str, str]:
    """The launch environment the active run's settings captured."""
    return run_settings().environment


def environment_value(name: str, default: Any = None) -> Any:
    """The variable ``name`` of the active run's captured environment, or
    ``default`` when it holds none."""
    return configured_environment().get(name, default)


def configuration_base_dir() -> Path:
    """The captured host base for paths, independent of later chdir calls."""
    return run_settings().base_dir


def configuration_input_files(settings=None) -> tuple[Path, ...]:
    """``RunSettings.consumer_files`` of ``settings`` or the active run."""
    settings = settings if settings is not None else active_settings()
    return () if settings is None else tuple(Path(path) for path in settings.consumer_files)


def configuration_exclusions(root, settings=None) -> tuple[str, ...]:
    """Root-relative POSIX paths of the configuration input files inside
    ``root``, for project copies to leave out, and of the project's Testuary
    directory with its settings and results. A file read through a link
    inside the project is excluded at that in-project path, not only where
    its target lies."""
    from testuary.settings import _on_disk_spelling

    bases = {Path(os.path.abspath(root)), Path(_on_disk_spelling(os.path.abspath(root)))}
    try:
        bases.add(Path(root).resolve())
    except OSError:
        pass
    files = configuration_input_files(settings)
    excluded = set(configuration_aliases(root, files))
    settings = settings if settings is not None else active_settings()
    if settings is not None and (settings.project_dir / settings.main.testuary_dir).is_dir():
        files = (*files, settings.project_dir / settings.main.testuary_dir)
    for path in files:
        for base in bases:
            try:
                excluded.add(path.relative_to(base).as_posix())
            except ValueError:
                continue
    return tuple(sorted(excluded))


def configuration_aliases(root, files) -> tuple[str, ...]:
    """Root-relative POSIX paths of the files inside ``root`` that are the
    same file as one of ``files`` under another name, a hard link. A copy
    matched by name alone would copy such a file with its contents, keys
    included. Only a file with more than one link can have such a name, so
    the tree is searched only when one of ``files`` has one."""
    identities = set()
    for path in files:
        try:
            info = os.stat(path)
        except OSError:
            continue
        if info.st_nlink > 1 and info.st_ino:
            identities.add((info.st_dev, info.st_ino))
    if not identities:
        return ()
    found = []
    for top, _dirs, names in os.walk(root):
        for name in names:
            entry = os.path.join(top, name)
            try:
                info = os.lstat(entry)
            except OSError:
                continue
            if (info.st_dev, info.st_ino) in identities:
                found.append(Path(os.path.relpath(entry, root)).as_posix())
    return tuple(sorted(found))


@contextlib.contextmanager
def use_run_settings(settings: "RunSettings"):
    """Isolate one invocation, including worker threads, and restore its state.

    This binding is process-wide because the starter phase uses
    worker threads. Until its remaining shared consumers are instance-owned,
    overlapping same-process invocations are explicitly unsupported.
    """
    from testuary.settings import RunSettings

    if not isinstance(settings, RunSettings):
        raise TypeError("settings must be a RunSettings object")
    if not _RUN_GUARD.acquire(blocking=False):
        raise RuntimeError("a Testuary run is already active in this process. Use a separate process")
    global _RUN
    previous = _RUN
    log_states = []
    module_names = ("testuary.orchestrator", "testuary.agent.coverage_agent",
                    "testuary.agent.mutation_agent", "testuary.measurement")
    existing_modules = {name for name in module_names if name in sys.modules}
    try:
        for name in module_names:
            module = sys.modules.get(name)
            if module is None:
                continue
            for attr in ("_mutation_session",):
                if hasattr(module, attr):
                    log_states.append((module, attr, getattr(module, attr)))
                    setattr(module, attr, None)
            phase_logger = getattr(module, "_log", None)
            if phase_logger is not None:
                for attr in ("logger", "log_dir_override", "log_filename_override", "_config"):
                    log_states.append((phase_logger, attr, getattr(phase_logger, attr)))
                    if not attr.startswith("_"):
                        setattr(phase_logger, attr, None)
        run = _RUN = Run(settings)
        run.hints = hints._read_hint_input()
        run.profile = prompts._read_profile_input()
        yield settings
    finally:
        try:
            _RUN = previous
            for target, attr, value in reversed(log_states):
                setattr(target, attr, value)
            # Modules imported during this invocation must not keep its loggers.
            for name in set(module_names) - existing_modules:
                module = sys.modules.get(name)
                if module is None:
                    continue
                for target, attrs in ((module, ("_mutation_session",)),
                                      (getattr(module, "_log", None),
                                       ("logger", "log_dir_override", "log_filename_override"))):
                    for attr in attrs:
                        if target is not None and hasattr(target, attr):
                            setattr(target, attr, None)
        finally:
            _RUN_GUARD.release()


class PhaseConfiguration(Mapping[str, Any]):
    """A phase's settings as a read-only mapping of the run's settings, read
    at each access."""

    def __init__(self, section: str, *, parent: Optional[str] = None, api: bool = False,
                 raw: bool = False):
        self.section, self.parent, self.api, self.raw = section, parent, api, raw

    def _values(self) -> Dict[str, Any]:
        return dict(models.get_api_config() if self.api else load_config(self.section) if self.raw
                    else get_agent_config(self.section, parent=self.parent))

    def __getitem__(self, key: str) -> Any:
        return self._values()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values())

    def __len__(self) -> int:
        return len(self._values())

    def copy(self) -> Dict[str, Any]:
        return dict(self._values())


def load_config(section: str = None) -> Dict[str, Any]:
    """The run's resolved settings as copies: one section's (an empty dict
    for a section no layer has. A phase id or agent name names its phase's
    section, ``phase_section``), or all of them."""
    values = run_settings().sections
    if section:
        return dict(values.get(phase_section(section), {}))
    return {name: dict(fields) for name, fields in values.items()}



_TARGETS_RANGE_RE = re.compile(r"^\s*(\d*)\s*-\s*(\d*)\s*$")

_TARGETS_SINGLE_RE = re.compile(r"^\s*(\d+)\s*$")


def parse_targets_spec(spec: str, total: int) -> List[int]:
    """Parse a ``--targets`` argument into a sorted list of 1-based
    positions in the target list.

    Grammar:

      * ``N``: target N.
      * ``A-B``: inclusive range from target A to B.
      * ``A-``: open range, target A to ``total``.
      * ``-B``: open range, target 1 to B.

    Args:
        spec: the raw CLI / config value (e.g. ``"6-10"``).
        total: how many targets are available. Used to resolve open
            ranges. Values above ``total`` are
            silently clamped because the user might know the spec is
            intentionally loose (e.g. ``5-20`` on a list of 12).

    Returns: a sorted, deduplicated list of 1-based target numbers. May
    be empty when the requested range falls entirely outside
    ``[1, total]``.

    Raises ``ValueError`` for any input that does not match a grammar
    rule, contains a non-positive integer, or has ``A > B`` in a
    closed range.
    """
    if spec is None or not str(spec).strip():
        raise ValueError("--targets: no position or range given. Use N, A-B, A- or -B")
    if total < 0:
        raise ValueError(f"--targets: total must be >= 0, got {total}")

    s = str(spec).strip()

    m_single = _TARGETS_SINGLE_RE.match(s)
    if m_single is not None:
        v = int(m_single.group(1))
        if v < 1:
            raise ValueError(
                f"--targets: a position is 1 or more, not {v}"
            )
        return [v] if v <= total else []

    m_range = _TARGETS_RANGE_RE.match(s)
    if m_range is not None:
        a_raw, b_raw = m_range.group(1), m_range.group(2)
        if a_raw == "" and b_raw == "":
            raise ValueError(
                "--targets: '-' alone is not a range. "
                "Use A-B, A- (A to the end) or -B (the start to B)"
            )
        a = int(a_raw) if a_raw else 1
        b_explicit = int(b_raw) if b_raw else None
        if a < 1:
            raise ValueError(
                f"--targets: a range starts at 1 or more, not {a}"
            )
        if b_explicit is not None and b_explicit < 1:
            raise ValueError(
                f"--targets: a range ends at 1 or more, not {b_explicit}"
            )
        # Only fail on inverted ranges when both ends were specified.
        # An open-ended ``A-`` on a smaller-than-A total resolves to an
        # empty list rather than an error (the operator's intent was
        # "from A onward, however many there are").
        if b_explicit is not None and a > b_explicit:
            raise ValueError(
                f"--targets: a range's start {a} is after its end {b_explicit}"
            )
        if total == 0:
            return []
        b = b_explicit if b_explicit is not None else total
        b = min(b, total)
        if a > b:
            return []
        return list(range(a, b + 1))

    raise ValueError(
        f"--targets: {spec!r} is not a position or range. Use N, A-B, A- or -B"
    )


def set_invocation_source_dir(path: str) -> None:
    """Redirect one target's work tree without modifying resolved configuration."""
    current_run().source_dir_override = str(path)


def global_seed() -> int:
    """The single deterministic seed for the whole project (``[main] seed``,
    default 0). It is the default ``seed_base`` of the starter phase's
    generation requests, so one value keeps a run reproducible. A consumer
    reads it as its default and may still be overridden on its own. The
    settings hold it as a whole number."""
    return int(load_config("main").get("seed", 0))


def run_seed(seed_base: int, run_idx: int, slot: int) -> int:
    """The seed of starter suite ``slot`` (from 1) in run ``run_idx`` (0 for
    r0): ``seed_base + 1000 * k + slot`` for run rk. Run r0 sends the seeds of
    a single run, every run draws its own, and a run draws the same seeds
    whether it runs alone (``--run K``) or among others (``--runs N``)."""
    from testuary.settings import RUN_SEED_STRIDE

    return int(seed_base) + RUN_SEED_STRIDE * int(run_idx) + int(slot)


def get_project_config() -> Dict[str, str]:
    """Project configuration: the resolved ``[project]`` settings (the
    project's settings file .testuary/project.ini, the configuration files,
    the environment and the command line), then the invocation's project
    root override (a target's work copy, ``set_invocation_source_dir``).

    Which of the ``--source-files`` targets run is not returned by this
    function. It is a per-run operator decision set via the ``--targets``
    option and read directly from ``args.targets``.

    Returned fields:
      - ``project_name``: the value of ``--project-name`` / ``[project].name``,
        a label in results and records, else the source directory's name.
      - ``source_dir`` / ``target_repo``: the source directory selection,
        resolved: an explicit [project] source_dir (an absolute path, or
        the container's POSIX path) wins, otherwise the project directory (the
        captured repo selector in container mode). The setting's own text
        is not kept here: the run settings' [project] section holds it.
      - ``source_dir_override``: private execution root. Explicit source or
        label selection supplies its initial value. Without either, direct
        agent callers retain target-based discovery until the pipeline sets it.
      - ``hint_profile``: the hints file's ``per_profile`` cells to use.
        Empty uses none.
      - ``python_import_root``: the directory below the source directory a
        Python package is imported from (e.g. ``src``, or ``Lib`` for
        CPython): first on the Python runs' PYTHONPATH, before the source
        directory, in which the tests run.
      - ``frozen_module_strategy``: ``unfreeze`` (default) or ``skip``.
      - ``output_dir``: where the run writes its output (``--output-dir``
        or ``[project] output_dir``). Blank uses ``results/<model>`` in the
        project's Testuary directory (see ``output_dir()``).
      - each installed language's declared settings (``project_settings``)
        without surrounding blanks, a relative host path resolved by its
        declared rule (``ProjectSetting.host_path``).
    """
    config_section = load_config("project")
    override = current_run().source_dir_override
    runtime_overrides = {"source_dir_override": override} if override else {}

    # The project's label, used in results and records.
    project_name = str(config_section.get("name") or "").strip()
    if project_name in {".", ".."} or any(char in project_name for char in "/\\:\0"):
        from testuary.settings import ConfigurationError

        raise ConfigurationError("[project] name must be a single directory name. Select paths with --source-dir")

    # Layer: resolved settings <- the invocation's source directory override.
    # An empty string counts as unset, so it does not replace a value from
    # an earlier layer.
    settings = run_settings()

    def _layer(*sources: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for source in sources:
            for k, v in source.items():
                if v is None:
                    continue
                explicit = ("project", k) in settings.supplied and source is config_section
                if isinstance(v, str) and v.strip() == "" and k != "name" and not explicit:
                    continue
                merged[k] = v
        return merged

    merged = _layer(config_section, runtime_overrides)

    container = project_exec_mode() == "container"
    captured_repo = project_module._captured_source_dir()
    source_dir = str(merged.get("source_dir", "")).strip()
    selected_root = bool(source_dir or project_name or (container and captured_repo))
    if not source_dir:
        source_dir = captured_repo if container else str(configuration_base_dir())
    if container:
        if source_dir and ("\\" in source_dir or not PurePosixPath(source_dir).is_absolute()):
            from testuary.settings import ConfigurationError

            raise ConfigurationError("container source_dir must be an absolute POSIX directory")
        source_dir = str(PurePosixPath(source_dir)) if source_dir else ""
    elif source_dir:
        # Preserve aliases until nonfollowing admission, rather than adopting
        # their targets while resolving settings.
        source_dir = str(project_module.resolve_host_path(source_dir).absolute())
    if not project_name:
        root_name = (PurePosixPath(source_dir).name if container else Path(source_dir).name)
        project_name = project_module.sanitize_path_component(root_name) if root_name else "project"
    target_repo = source_dir
    source_dir_override = (
        str(merged.get("source_dir_override", "")).strip()
        or (target_repo if selected_root else "")
    )

    resolved = {
        **merged,
        "project_name": project_name,
        "source_dir": source_dir,
        "target_repo": target_repo,
        "hint_profile": str(merged.get("hint_profile", "")).strip(),
        "source_dir_override": source_dir_override,
        "python_import_root": str(merged.get("python_import_root", "")).strip(),
        # ``unfreeze`` (default) emits ``-X frozen_modules=off`` to every
        # subprocess so source edits to CPython stdlib modules like
        # ``importlib._bootstrap`` actually take runtime effect, so the
        # interpreter runs the edited source instead of the bytecode
        # embedded in the binary. ``skip`` instead drops frozen-module
        # candidates and emits no flag, the right setting for
        # third-party projects where no module is frozen
        # so the workaround is a noisy no-op.
        "frozen_module_strategy": str(
            merged.get("frozen_module_strategy", "unfreeze")
        ).strip() or "unfreeze",
        "output_dir": str(merged.get("output_dir", "")).strip(),
    }
    # The installed languages' settings, as given without surrounding blanks.
    for key in _declared_project_settings():
        if isinstance(resolved.get(key), str):
            resolved[key] = resolved[key].strip()
    # A relative host path lies under the captured base, as each setting
    # declares (``host_path``). A container's paths are POSIX addresses,
    # except a file of this machine such as requirements.
    rules = {"source_dir_override": "path", **_declared_host_paths()}
    for key, rule in rules.items():
        value = str(resolved.get(key) or "").strip()
        if not value or (container and rule != "host file"):
            continue
        if rule == "command" and "/" not in value and "\\" not in value:
            continue  # a bare name is a command on PATH
        path = project_module.resolve_host_path(value)
        # Following an environment's interpreter link would select its
        # base interpreter: only a host file's links are followed.
        resolved[key] = str(path.resolve() if rule == "host file" else path.absolute())
    return resolved


def _declared_host_paths() -> Dict[str, str]:
    """How a relative host path in each installed language's project
    settings resolves (``ProjectSetting.host_path``), by setting name. A name
    several languages declare is one setting, and the first language by name
    that declares it gives its rule, as it gives its kind and default."""
    return {name: rule for name, rule in _declared_project_settings().items() if rule}


def _declared_project_settings() -> Dict[str, str]:
    """Each installed language's declared project setting with its host-path
    rule ("" for none), the first language by name giving a shared name's."""
    from langadapter import names, registration

    found: Dict[str, str] = {}
    for language in names():
        for setting in getattr(registration(language).factory, "project_settings", ()) or ():
            found.setdefault(setting.name, getattr(setting, "host_path", "") or "")
    return found


def project_exec_mode() -> str:
    """Where project commands run: ``local`` (default) or ``container``
    (``[main] project_exec``)."""
    mode = str(load_config("main").get("project_exec", "local")).strip().lower()
    return mode or "local"


def device_name() -> str:
    """The name this computer has in the run's records (``[main]
    device_name``): empty by default and outside a run, so that the
    computer's own name reaches no record unless the user names it."""
    if active_settings() is None:
        return ""
    return str(load_config("main").get("device_name", "")).strip()


def consolidation_measure_mutation() -> bool:
    """Whether every phase-end consolidation measures the full mutation
    catalogue on the union, so the set-cover keeps a killer for every killed
    mutant (kill-lossless) rather than reducing on coverage alone
    (``[main] phase_end_mutation``, on by default)."""
    raw = str(load_config("main").get("phase_end_mutation", "true"))
    return raw.strip().lower() in ("true", "1", "yes", "on")


def finishing_measures_mutation() -> bool:
    """Whether Finishing measures the delivered suite's mutation score
    (``measure_mutation``, on by default). Off measures coverage only."""
    raw = str(load_config("finishing").get("measure_mutation", "true")).strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def solver_settings(cfg: Mapping[str, Any]) -> "ReductionSettings":
    """The five suite reduction settings in a phase's configuration ``cfg``,
    as the phase ends pass them."""
    from testuary.consolidation import ReductionSettings
    return ReductionSettings(
        solver=str(cfg["reduction_solver"]),
        solver_time_limit_seconds=float(cfg["reduction_solver_time_limit_seconds"]),
        minimum_test_runtime_ms=float(cfg["reduction_min_test_time_ms"]),
        solver_random_seed=int(cfg["reduction_solver_random_seed"]),
        tie_break=str(cfg["reduction_tie_break"]),
    )


#: The ways a phase end obtains the reduced suite's mutation score.
REDUCED_SUITE_MUTATION_METHODS = settings_module._CHOICES["reduced_suite_mutation"]


def consolidation_mutation_method() -> str:
    """How a consolidation produces the reduced suite's mutation score: ``derive``
    (restrict the merged matrix to the kept tests), ``measure`` (re-measure the
    reduced suite with the engine), or ``both`` (re-measure and deliver that, keeping
    the derivation as a logged cross-check that flags any derive-vs-measure gap),
    from ``[main] reduced_suite_mutation``, ``both`` when it is empty. Any
    other value raises ``ValueError`` naming it, its source and the choices: a
    mistyped method would otherwise select another measurement without a
    word."""
    raw = str(load_config("main").get("reduced_suite_mutation", "")).strip()
    source = "[main] reduced_suite_mutation"
    method = raw.lower() or "both"
    if method not in REDUCED_SUITE_MUTATION_METHODS:
        raise ValueError(
            f"{source} = {raw!r} is not a reduced-suite mutation method. "
            f"Choose one of {', '.join(REDUCED_SUITE_MUTATION_METHODS)}"
        )
    return method


def context_rebuild() -> str:
    """When the agents rebuild their conversation context: ``mode_change``
    (on the turn the bound tool mode changes, holding the result between)
    or ``every_turn``, from ``[main] context_rebuild``, ``mode_change`` when it
    is empty. Any other value raises ``ValueError`` naming it, its source and
    the choices."""
    from testuary.agent._context import CONTEXT_REBUILDS

    raw = str(load_config("main").get("context_rebuild", "")).strip()
    source = "[main] context_rebuild"
    value = raw.lower() or "mode_change"
    if value not in CONTEXT_REBUILDS:
        raise ValueError(
            f"{source} = {raw!r} is not a context rebuild policy. "
            f"Choose one of {', '.join(CONTEXT_REBUILDS)}"
        )
    return value


def _reduction_tie_break_setting(raw: object) -> str:
    """``reduction_tie_break``: how the exact reduction chooses among
    selections equal in runtime and test count."""
    from testuary.reduction import TIE_BREAKS
    value = str(raw).strip().lower()
    if value not in TIE_BREAKS:
        raise ValueError(
            f"reduction_tie_break must be one of {', '.join(TIE_BREAKS)}, "
            f"not {raw!r}")
    return value


def _per_mutant_timeout_setting(raw: object) -> object:
    """Normalize the ``per_mutant_timeout_seconds`` config value to its
    semantic type: a positive int when the value is numeric, else the
    lowercased keyword (``auto``). ``per_mutant_timeout`` accepts either, so the
    resolved value carries the same type the operator meant rather than a
    stringified int next to the other integer knobs."""
    s = str(raw).strip()
    if s.lstrip("+").isdigit() and int(s) > 0:
        return int(s)
    return s.lower()


def fixed_per_mutant_timeout(configured: object) -> int:
    """The pinned per-mutant timeout from the ``per_mutant_timeout_seconds``
    config value, or 0 when it selects the dynamic rule. ``auto`` (the default)
    and any non-positive or unparseable value return 0. A positive integer
    returns itself."""
    try:
        v = int(str(configured).strip())
    except (TypeError, ValueError):
        return 0
    return max(0, v)


def per_mutant_timeout(
    configured: object,
    suite_exec_s: Optional[float],
    *,
    cap_s: Optional[int] = None,
    floor_s: int,
    multiplier: float,
) -> int:
    """The cosmic-ray per-mutant timeout in seconds.

    A pinned ``configured`` (a positive integer) sets a fixed timeout. ``auto``
    scales the timeout with the suite's own green execution time: ``multiplier``
    times the green run time, rounded up, with a ``floor_s`` minimum for a
    near-instant suite (headroom for worker startup and shared-filesystem
    latency). A larger multiplier (``per_mutant_timeout_factor``) absorbs the per-mutant
    slowdown when many workers run the suite concurrently, so a slow-but-green
    mutant is judged by its assertions rather than the clock. ``cap_s`` (the
    ceiling resolve_mutation_max_timeout computes) bounds the auto value: a
    single mutant's test run cannot exceed this ceiling, so one that does is
    genuinely stuck.
    The floor is applied before the cap, so a cap below the floor still wins. A
    missing suite time falls back to the floor. Used by the in-loop mutation
    agent and Finishing so every suite is scored
    under the same per-mutant rule, in every environment (venv, Docker,
    Apptainer)."""
    fixed = fixed_per_mutant_timeout(configured)
    if fixed > 0:
        return fixed
    t = suite_exec_s if suite_exec_s is not None else 0.0
    val = max(int(floor_s), math.ceil(multiplier * t))
    if cap_s is not None and cap_s > 0:
        val = min(val, int(cap_s))
    return val


def require_config(config: Dict[str, Any], key: str) -> Any:
    """A measurement-policy config value. The bundled config.ini defines
    every such key, so a miss means the code and the config drifted (a rename,
    a typo, or a hand-built partial dict): hard fail naming the key. A silent
    code-side default would change the measurement policy without any
    signal."""
    try:
        return config[key]
    except KeyError:
        raise KeyError(
            f"config key {key!r} is missing. Methodology-relevant keys have "
            f"no code-side default. The bundled config.ini (or the dict "
            f"passed in) must define it.") from None


def resolve_mutation_max_timeout(config: Dict[str, Any]) -> int:
    """The per-mutant timeout ceiling: ``max_per_mutant_timeout_factor`` times the
    suite run timeout. It is computed from those two knobs, not stored as its
    own key, so the agent config and Finishing's config resolve the same
    ceiling from the same source."""
    return int(
        float(require_config(config, "max_per_mutant_timeout_factor"))
        * int(require_config(config, "suite_run_timeout_seconds")))


def _limit(cast: Callable[[Any], Any], unlimited: Any) -> Callable[[Any], Any]:
    """A limit's cast that reads ``unlimited`` as ``unlimited``."""
    return lambda value: unlimited if str(value).strip().lower() == "unlimited" else cast(value)


#: The measurement-policy keys and their casts.
#: ``get_agent_config`` resolves each strictly from the section that owns
#: it. ``tests/unit/test_config_strictness.py`` pins that none is ever read
#: through a code-side default. The per-mutant timeout floor is
#: ``min_per_mutant_timeout_seconds`` (30 s in the bundled config, for worker startup
#: and shared-FS latency). The auto timeout is
#: ``per_mutant_timeout_factor`` x the suite's own green time,
#: capped at ``max_per_mutant_timeout_factor`` x ``suite_run_timeout_seconds``.
_METHODOLOGY_CASTS: Dict[str, Callable[[Any], Any]] = {
    # None is no request limit.
    "max_requests_per_phase": _limit(int, None),
    "suite_count": int,
    "suite_run_timeout_seconds": int,
    "coverage_timeout_seconds": int,
    "goal": float,
    "minimum": float,
    "min_improvement_points": float,
    "goal_metric": str,
    "reduction_solver": lambda v: str(v).strip().lower(),
    "reduction_solver_time_limit_seconds": float,
    "reduction_min_test_time_ms": float,
    "reduction_solver_random_seed": int,
    "reduction_tie_break": _reduction_tie_break_setting,
    "cosmic_ray_operators": lambda v: str(v).strip(),
    "pit_mutators": lambda v: str(v).strip(),
    "per_mutant_timeout_seconds": lambda v: _per_mutant_timeout_setting(v),
    "min_per_mutant_timeout_seconds": int,
    "per_mutant_timeout_factor": float,
    "max_per_mutant_timeout_factor": float,
    "measurement_timeout_seconds": int,
}


def _max_measurements_without_improvement(value: Any, request_limit: Optional[int], section: str) -> int:
    """max_measurements_without_improvement as a number of measurements. A share such as 10%
    is that share of the phase's request limit, rounded up, and needs a
    request limit."""
    text = str(value).strip()
    if not text.endswith("%"):
        return int(text)
    from testuary.settings import ConfigurationError, percent_share

    share = percent_share(text)
    if share is None:
        raise ConfigurationError(f"[{section}] max_measurements_without_improvement: expected a whole number, "
                                 "or a share of max_requests_per_phase from 0% to 100%")
    if request_limit is None:
        raise ConfigurationError(
            f"[{section}] max_measurements_without_improvement = {text} is a share of max_requests_per_phase, but this "
            "phase has no request limit: give a number of measurements instead")
    return math.ceil(share * request_limit)


def _flag(value: Any) -> bool:
    """A switch as the run settings normalize it (true or false)."""
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _text(value: Any) -> str:
    return str(value).strip()


def _lower(value: Any) -> str:
    return str(value).strip().lower()


#: get_agent_config's keys in the order it returns them: (key, cast), the
#: cast taking the section's value where the sections hold the key. A row
#: whose cast is None is computed in get_agent_config from other values.
_AGENT_KEYS: Tuple[Tuple[str, Optional[Callable[[Any], Any]]], ...] = (
    # Unlimited graph attempts end only at a non-retryable failure or a limit.
    ("max_loop_attempts", _limit(int, sys.maxsize)),
    # The phase's token and wall-clock limits, 0 for none, checked before
    # every model request and retry wait.
    ("max_input_tokens_per_phase", _limit(int, 0)),
    ("max_output_tokens_per_phase", _limit(int, 0)),
    ("max_total_tokens_per_phase", _limit(int, 0)),
    ("max_wall_clock_seconds_per_phase", _limit(float, 0.0)),
    ("concurrent_requests", int),
    ("concurrent_requests_ollama", int),
    # The starter suites: the refill, each slot's own attempts, the cleanup
    # rounds, reuse and the cache warmup's first request alone.
    ("refill_failed_suites", _flag),
    ("attempts_per_suite", int),
    ("failing_test_cleanup_rounds", int),
    ("reuse_suites", _flag),
    ("cache_warmup", _flag),
    ("temperature", None),
    ("seed_base", None),
    # Cut a stream that gave no answer content for this many seconds.
    ("reasoning_grace_seconds", lambda value: float(value) if value else None),
    # How each starter suite's mutation score is obtained.
    ("suite_mutation", _lower),
    # Whether the remaining phases still run after a missed floor.
    ("complete_tests_below_minimum", _flag),
    ("max_final_answers_without_measurement", int),
    ("cosmic_ray_progress_interval_s", int),
    # Revert to the best green suite after this many worse measurements.
    ("restore_count", int),
    # Measurements of polish at or above the goal before the handoff.
    ("extra_measurements_after_goal", int),
    ("engine", _lower),
    ("engine_workers", int),
    ("cosmic_ray_distributor", _lower),
    ("cosmic_ray_clone_dir", _text),
    ("cosmic_ray_local_session_dir", _text),
    ("allow_sequential_fallback", _flag),
    ("cosmic_ray_keep_output", _flag),
    ("full_after_selective_goal", None),
    ("selective_switch_threshold", float),
    # Per-test kill attribution of the cover: all killers, or the first.
    ("cover_attribution", lambda value: "first" if _lower(value) == "first" else "full"),
    # The loop's catalogue: full, or else the selective proxy.
    ("loop_catalogue", lambda value: "full" if _lower(value) == "full" else "selective"),
    ("cosmic_ray_session_dir", _text),
    ("cosmic_ray_init_timeout_s", int),
    ("cosmic_ray_worker_ready_timeout_s", int),
    ("cosmic_ray_clone_sweep_hours", float),
    ("cosmic_ray_verbose", _flag),
    ("cosmic_ray_per_file_adjudication", _flag),
    # auto: <output_dir>/cache/mutation/cosmic_ray. Empty: no cache.
    ("cosmic_ray_catalogue_cache_dir", _text),
    ("cosmic_ray_cache_max_entries", int),
    ("cosmic_ray_cache_max_age_days", int),
    ("count_repeated_measurements", _flag),
    ("allow_early_stop", _flag),
    # The policy for sending the phase's model requests again
    # (model_requests.RetrySettings).
    ("max_consecutive_transient_failures", int),
    ("max_transient_failures_per_phase", int),
    ("transient_first_wait_seconds", float),
    ("max_transient_wait_seconds", float),
    ("max_transient_hint_seconds", float),
    ("count_retries", _flag),
    ("phase_end_evidence_reuse", _flag),
    ("max_failed_test_attempts", int),
    ("suite_timeout_stop_after", int),
    ("reduction_solver_time_limit_seconds", float),
    ("reduction_min_test_time_ms", float),
    ("reduction_solver_random_seed", int),
    ("progress_heartbeat_requests", int),
    ("reject_unsound_async_green", _flag),
    ("report_reverse_order_failures", _flag),
    ("builtin_replacement_guidance", _flag),
    ("max_output_tokens", None),
    ("reasoning_field_names", lambda value: models.field_names(value)),
    ("enable_trace_log", _flag),
)


def get_agent_config(
    agent_name: str,
    parent: Optional[str] = None,
) -> Dict[str, Any]:
    """The run's settings of an agent's section, typed (``_AGENT_KEYS``).
    With ``parent`` the agent's section lies over the parent's, so a key the
    agent's section omits comes from the parent's (test_mutation over
    test_coverage). A typed key comes only from the sections that hold it,
    and one they lack is absent. The computed keys are always present."""
    config = load_config(agent_name)
    if parent:
        config = {**load_config(parent), **config}
    full_loop = _lower(config.get("loop_catalogue", "full")) == "full"
    computed = {
        "temperature": lambda: _temperature(config.get("temperature", "")),
        "seed_base": lambda: int(config.get("seed_base", global_seed())),
        # A full-catalogue loop has no selective phase to switch out of.
        "full_after_selective_goal": lambda: (
            False if full_loop else _flag(config.get("full_after_selective_goal", "false"))),
        # The phase's own cap. Where the requests are built it combines with
        # the model alias's to the smaller of the two (models.resolve_output_cap).
        "max_output_tokens": lambda: models.resolve_output_cap(config.get("max_output_tokens")),
    }
    resolved = {key: computed[key]() if cast is None else cast(config[key])
                for key, cast in _AGENT_KEYS if cast is None or key in config}

    # Measurement-policy keys resolve strictly and only from the section that owns
    # them: present in the merged section -> cast and included. Absent ->
    # absent from the dict, so a consumer reading a key its section does not
    # own fails naming the key instead of receiving an invented value.
    for _key, _cast in _METHODOLOGY_CASTS.items():
        if _key in config:
            resolved[_key] = _cast(require_config(config, _key))
    if "max_measurements_without_improvement" in config:
        resolved["max_measurements_without_improvement"] = _max_measurements_without_improvement(
            require_config(config, "max_measurements_without_improvement"), resolved.get("max_requests_per_phase"),
            phase_section(agent_name))
    if ("max_per_mutant_timeout_factor" in config
            and "suite_run_timeout_seconds" in config):
        resolved["mutation_max_timeout"] = resolve_mutation_max_timeout(config)
    return resolved


def _temperature(value: Any) -> Optional[float]:
    """The sampling temperature, None for an empty setting."""
    return float(value) if str(value).strip() else None


def _validity_policy_version(language: Optional[str]) -> Optional[str]:
    """The validity policy version ``language``'s registered adapter
    declares, None without one."""
    if not language:
        return None
    from langadapter.errors import UnsupportedLanguage
    from langadapter.registry import registration

    try:
        factory = registration(language).factory
    except UnsupportedLanguage:
        return None
    return getattr(factory, "validity_policy_version", None) or None


#: The key of resolved_settings.json that names the last file of
#: models_files when it does not exist.
_ABSENT_MODELS_FILE_KEY = "absent_models_file"


def resolved_test_preset(*, model_phases: tuple[str, ...] = (),
                         language: Optional[str] = None) -> Dict[str, Any]:
    """The run's settings as configured, for its resolved-settings record:
    every section and key as the run resolved them (the settings echo, its
    secrets left out, of the provider sections only the one the run's
    model goes through, and the mutation engine that runs for
    ``language``, ``evidence.recorded_settings``), beside what the
    settings alone do not show: the generated-test validity policy, the
    mutation phase's effective catalogue and Cosmic Ray distributor, the
    model limits per phase and, for the configured generation phases
    ``model_phases``, each phase's model, and the last catalogue file of
    ``models_files`` when it does not exist. The validity policy is
    ``language``'s adapter's (``validity_policy_version``), ``none`` for an
    adapter without one or no language. It does not describe a reused
    suite's ancestry."""
    from testuary.agent import _limits
    from testuary.evidence import recorded_settings
    from testuary.languages.generic import NO_VALIDITY_POLICY
    from testuary.languages.mutation import cosmic_ray_distributor

    settings = run_settings()
    sections = recorded_settings(settings, language)["sections"]
    mut = get_agent_config("test_mutation", parent="test_coverage")
    record: Dict[str, Any] = {
        "sections": sections,
        "generated_test_validity": {
            "policy_version": _validity_policy_version(language) or NO_VALIDITY_POLICY},
        "effective": {
            "mutation_catalogue": "full" if mut["loop_catalogue"] == "full"
            and not str(mut["cosmic_ray_operators"]).strip() else "filtered",
            # The distributor a Python run uses for the configured workers,
            # auto resolved, or refused for a local one with several workers.
            "cosmic_ray_distributor": cosmic_ray_distributor(
                mut["cosmic_ray_distributor"], int(mut["engine_workers"])) or "refused",
        },
        "limits": {phase: _limits.export(phase) for phase in ("coverage", "mutation")},
    }
    if settings._absent_models_file is not None:
        # Named so that a mistyped path in models_files stays visible.
        record[_ABSENT_MODELS_FILE_KEY] = settings._absent_models_file
    if model_phases:
        from testuary import models as _model_source
        from testuary.evidence import provider_extension
        from testuary.models import resolve_phase_model

        model_records = {}
        for phase in model_phases:
            if _model_source.current() is not None:
                model_records[phase] = {"source": "model_source"}
            else:
                model = resolve_phase_model(settings, phase)
                extension = provider_extension(model, settings_module._model_credentials(settings)[1])
                model_records[phase] = {**model.export(), "source": "configuration",
                                        "extension_version": extension.version if extension else None}
                if extension is not None:
                    # Only an added provider records its file, so other runs record as before.
                    model_records[phase].update(extension_file=extension.file,
                                                extension_sha256=extension.sha256)
        record["model"] = model_records
    return record
