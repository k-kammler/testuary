"""Where and with what a project's commands run: its directories, interpreter,
executors and files."""
from __future__ import annotations

from importlib import metadata
import subprocess
from typing import Final, Any, Dict, Optional, Sequence, Iterator
import os
import re
import shutil
import sys
import stat
import contextlib
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from string import Template
from testuary import config
from langadapter.context import Environment
from langadapter.execution import ENV_PREFIX, Executor as ExecutionBackend, LocalExecutor
from langadapter.execution import ContainerExecutor as _ContainerExecutor
from contextlib import contextmanager
from langadapter import placement as _placement


HOST_SOLVER_VERSIONS: Final[dict[str, str]] = {
    "PuLP": "3.3.2",
    "CBC": "2.10.3",
}


def host_solver_versions() -> dict[str, str]:
    """Return the host-side exact-reduction tool versions. A CBC probe that
    fails keeps its reason: the error that stopped it, or the exit code and
    the first output line when the solver printed no version."""
    try:
        import pulp
    except ImportError:
        return {"PuLP": "not installed", "CBC": "not installed"}
    from testuary.reduction import cbc_version

    pulp_version = metadata.version("PuLP")
    try:
        # The constructor's flag keeps the bundled solver without changing
        # the process's warning filters. A PuLP other than the pinned one
        # may lack it, and the report names it as a mismatch.
        solver = pulp.PULP_CBC_CMD(msg=False, _skip_v4_deprecation=True)
    except TypeError as exc:
        return {"PuLP": pulp_version, "CBC": f"unavailable (TypeError: {exc})"}
    version, reason = cbc_version(str(solver.path))
    return {"PuLP": pulp_version, "CBC": version or reason}


def host_solver_mismatches() -> tuple[str, ...]:
    """Each solver part (PuLP, CBC) whose version on this host differs from
    ``HOST_SOLVER_VERSIONS``, as "name: expected version, observed
    version"."""
    observed = host_solver_versions()
    return tuple(
        f"{name}: expected {expected}, observed {observed.get(name)}"
        for name, expected in HOST_SOLVER_VERSIONS.items()
        if observed.get(name) != expected
    )




def resolve_host_path(value: str) -> Path:
    """Expand a host path against the invocation's captured inputs."""
    env = config.configured_environment()
    expanded = Template(value).safe_substitute(env)
    expanded = re.sub(r"%([^%]+)%", lambda match: env.get(match[1], match[0]), expanded)
    if expanded == "~" or expanded.startswith(("~/", "~\\")):
        home = env.get("USERPROFILE") if os.name == "nt" else env.get("HOME")
        if not home:
            raise ValueError("a home-relative path needs HOME or USERPROFILE in the captured environment")
        expanded = home + expanded[1:]
    path = Path(expanded)
    return path if path.is_absolute() else config.configuration_base_dir() / path


def sanitize_path_component(s: str) -> str:
    """Map a string to a safe single path component: letters and digits
    (``str.isalnum``, non-ASCII included), ``-`` and ``_``
    survive, everything else collapses to ``_``. An empty input
    returns the literal ``_`` so the caller never gets an empty path
    segment (which would silently disappear in ``Path("a") / ""``).

    Used to derive path-safe pieces from the model alias and the
    project name in a templated ``output_dir`` (``${model}`` and
    ``${project}``) and the model's directory of the default. Shares the exact rule with
    ``resolve_work_dir_namespace`` so the work-dir namespace + the
    output root stay consistent."""
    if not s:
        return "_"
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in s
    ) or "_"


def output_dir() -> Path:
    """Return the configured output directory as an absolute Path.

    Reads ``[project].output_dir`` (``--output-dir`` sets it). The
    value may contain ``${model}`` and
    ``${project}`` placeholders, filled with the active
    model alias (``--model`` / ``[api].model``) and project name
    (``--project-name`` / ``[project].name``), both passed through
    ``sanitize_path_component``. Empty, the bundled default, is
    ``results/<model>`` in the project's Testuary directory
    (``RunSettings.results_dir``), one directory per model, ready for
    cross-model evaluation.

    Relative paths are anchored to the captured configuration base directory.
    Lexical parent traversal remains visible until output admission. This
    function neither inspects nor creates the directory.
    """
    proj_cfg = config.get_project_config()
    raw = proj_cfg.get("output_dir") or ""
    # The model piece is the alias the user typed (``--model``
    # ``ollama_qwen3_coder_next`` / ``[api].model``), not the resolved
    # underlying API model string (which can contain spaces or
    # slashes from a vendor's display name). Aliases are
    # sanitized before substitution into the output path.
    model_alias = config.load_config("api").get("model") or ""
    if not raw:
        results = config.run_settings().results_dir
        return (results / sanitize_path_component(str(model_alias))).absolute()

    if "$" in raw:  # templated value: fill model and project
        from string import Template
        project_name = proj_cfg["project_name"]
        raw = Template(raw).safe_substitute(
            model=sanitize_path_component(str(model_alias)),
            project=sanitize_path_component(str(project_name)),
        )

    p = Path(raw)
    if not p.is_absolute():
        p = config.configuration_base_dir() / p
    # Admission must see every selected ancestor before collapsing '..'.
    return p.absolute()


#: This process's part of an own work directory's name, which keeps runs on
#: several computers that share one directory apart.
_PROCESS_PART = f"{os.getpid()}_{os.urandom(4).hex()}"


def resolve_work_dir_namespace() -> str:
    """Return the namespace used for transient work and its cleanup.

    ``work_dir_job_namespace=never`` disables namespacing. Otherwise an
    explicit ``TESTUARY_MAIN_WORK_DIR_NAMESPACE`` takes precedence over the caller's
    ``TESTUARY_EXECUTION_ID``. Execution IDs are sanitized path components
    under the ``job_`` prefix. An explicit workspace label uses
    ``job_explicit_`` and must satisfy its stricter length/character limit.

    Without either input, ``always`` names this process (its ID and a random
    part), after ``[main] device_name`` when set, never the computer's own
    name, and ``auto`` returns an empty string. Callers provide a stable execution ID
    when repeated invocations must address the same transient work tree.
    """
    mode = str(
        config.load_config("main").get("work_dir_job_namespace", "auto")
    ).strip().lower()
    if mode == "never":
        return ""

    env = config.configured_environment()
    explicit = env.get("TESTUARY_MAIN_WORK_DIR_NAMESPACE", "").strip()
    if explicit:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", explicit):
            raise ValueError("invalid TESTUARY_MAIN_WORK_DIR_NAMESPACE")
        return f"job_explicit_{explicit}"
    execution_id = env.get("TESTUARY_EXECUTION_ID", "").strip()
    if execution_id:
        return f"job_{sanitize_path_component(execution_id)}"

    if mode == "always":
        # This process, when no stable label was supplied.
        device = config.device_name()
        label = f"{sanitize_path_component(device)}_" if device else ""
        return f"job_local_{label}{_PROCESS_PART}"
    return ""


def resolve_source_dir(file_path: str) -> str:
    """The directory ``file_path``'s tests run in, with the project's own test
    configuration and build files: in a run, the run's own copy of the
    source directory, and in container mode the source directory inside the
    container.

    When ``[project].source_dir_override`` is set, it is used directly
    (resolved relative to the configuration base directory if not
    absolute). Otherwise
    walks up from ``file_path`` looking for common project markers.
    ``[project].python_import_root`` may place the directory a Python
    package is imported from below the source directory.
    """
    proj_cfg = config.get_project_config()
    override = proj_cfg.get("source_dir_override")

    # Container mode: file_path is an in-container POSIX path. On a Windows host
    # pathlib would rewrite "/" to "\" and prepend a drive (and the marker walk
    # cannot see the container's filesystem anyway). The caller sets
    # TESTUARY_PROJECT_SOURCE_DIR to the in-container repo root. Use it verbatim and
    # join POSIX-style, which matches what the marker walk finds on a POSIX host.
    if config.project_exec_mode() == "container":
        repo_dir = proj_cfg.get("source_dir") or _captured_source_dir()
        return repo_dir or str(PurePosixPath(file_path).parent)

    if override:
        base = Path(override)
        if not base.is_absolute():
            base = (config.configuration_base_dir() / base).resolve()
        else:
            base = base.resolve()
    else:
        # Walk up from the file looking for markers.
        markers = {
            "setup.py", "setup.cfg", "pyproject.toml", "pom.xml",
            "build.gradle", "build.gradle.kts", "Makefile", "CMakeLists.txt",
            ".git", "Cargo.toml", "go.mod", "package.json",
        }
        cur = Path(file_path).resolve().parent
        base = None
        for _ in range(15):
            if any((cur / m).exists() for m in markers):
                base = cur
                break
            if cur.parent == cur:
                break
            cur = cur.parent
        if base is None:
            base = Path(file_path).resolve().parent.parent

    return str(base)


def _python_import_dir(file_path: str) -> str:
    """``[project] python_import_root`` for a Python target, else empty."""
    from testuary.languages import detect_language

    if detect_language(file_path) != "python":
        return ""
    return str(config.get_project_config().get("python_import_root") or "").strip()


def _below(root: str, sub_dir: str) -> str:
    """``sub_dir`` below ``root``. In container mode ``root`` is an
    in-container POSIX path, joined POSIX-style. A local directory is honored
    even where it does not exist, as for a results file read on another
    machine than the one that produced it."""
    if not sub_dir:
        return root
    if config.project_exec_mode() == "container":
        return root.rstrip("/") + "/" + sub_dir.strip("/")
    candidate = Path(root) / sub_dir
    return str(candidate.resolve() if candidate.exists() else candidate)


def display_path(target_file_path: str, source_dir: str) -> str:
    """The path shown to the model as the file under test: relative to the
    directory its package is imported from (``source_dir``, with
    ``[project] python_import_root`` below it for a Python target), with
    forward slashes, so every phase names the file by its place in the
    project rather than by an absolute filesystem location. In container mode
    the absolute path is the in-container path, which is meaningless to a reader
    and differs from the host. The relative form is stable across host and
    container. Falls back to the basename
    when the target is not under that directory."""
    root = _below(source_dir, _python_import_dir(target_file_path))
    try:
        return Path(target_file_path).resolve().relative_to(
            Path(root).resolve()).as_posix()
    except ValueError:
        return Path(target_file_path).name


def project_runtime(language: str, values: Mapping[str, Any], *, root: str = ""):
    """The runtime ``language``'s adapter chooses for a run
    (``discover_runtime``) from its setting ``values`` for the project at
    ``root``, on the run's route with its captured environment, a relative
    host path lying under the configuration base
    (``_configured_runtime_path``), and a provisioning request fulfilled
    (``provision_runtime``). It is chosen once per selection and
    environment while the run settings stay active. A language registered
    with a factory function chooses none. A refusal is a RuntimeError with
    the adapter's reason and the setting to change."""
    from dataclasses import replace

    from langadapter import registration
    from langadapter.adapter import RuntimeChoice
    from langadapter.context import Interpreter
    from langadapter.errors import InvalidRequest

    guest = config.project_exec_mode() == "container"
    environment = dict(config.configured_environment())
    key = (language, root, guest, tuple(sorted((name, str(value)) for name, value in values.items())),
           tuple(sorted(environment.items())))
    runtimes = config.current_run().project_runtimes
    cached = runtimes.get(key)
    if cached is not None:
        return cached
    discover = getattr(registration(language).factory, "discover_runtime", None)
    try:
        choice = (discover(root, values, container=guest, environment=environment,
                           resolve_path=_configured_runtime_path)
                  if discover is not None else RuntimeChoice())
    except InvalidRequest as refusal:
        raise RuntimeError(f"[project] {refusal.field}: {refusal.message} (set in the project's "
                           f"settings, or with {_setting_variable(refusal.field)})") from refusal
    request = getattr(choice.interpreter, "provisioning", None)
    if request is not None:
        choice = replace(choice, interpreter=Interpreter(path=provision_runtime(language, request)))
    runtimes[key] = choice
    return choice


def _setting_variable(name: str) -> str:
    """The variable that sets the setting ``name`` at launch, in the section
    Testuary's configuration keeps it in (``[project]`` for a language's
    own)."""
    from testuary.languages import _own_sections

    return f"TESTUARY_{_own_sections().get(name, 'project').upper()}_{name.upper()}"


def _configured_runtime_path(value: str) -> Path:
    """A selected host runtime path: an absolute one as given, a relative
    one under the configuration base. Nothing is expanded: the project
    configuration has expanded the [project] values it resolves, and
    JAVA_HOME is taken as set."""
    path = Path(value)
    return path if path.is_absolute() else config.configuration_base_dir() / path


def _environment_dirs(interpreter: str) -> set:
    # A virtual environment keeps its interpreter in Scripts or bin, a Windows
    # Conda environment keeps it at its root.
    path = Path(interpreter)
    return {os.path.normcase(os.path.realpath(env_dir)) for env_dir in (path.parent, path.parent.parent)}


def _is_testuary_environment(interpreter: str) -> bool:
    """Whether ``interpreter`` belongs to the environment Testuary runs in."""
    return os.path.normcase(os.path.realpath(sys.prefix)) in _environment_dirs(interpreter)


def _is_project_venv(interpreter: str, source: str) -> bool:
    """Whether ``interpreter``'s environment lies inside the source directory,
    whatever its name (.venv, venv, env or a Conda prefix there)."""
    if not source:
        return False
    root = os.path.normcase(os.path.realpath(source))
    for env_dir in _environment_dirs(interpreter):
        try:
            if env_dir != root and os.path.commonpath([root, env_dir]) == root:
                return True
        except ValueError:
            continue  # another drive is never inside the project
    return False


def launch_environment() -> Dict[str, str]:
    """The environment Testuary's own project commands are launched with:
    the run's captured launch environment without its model-credential
    variables. Settings constructed without an environment refuse, since
    tools cannot start in an empty one."""
    settings = config.run_settings()
    if not settings.environment_captured:
        raise RuntimeError(
            "these run settings captured no launch environment, so native commands "
            "cannot start; construct RunSettings with environment=dict(os.environ) "
            "or use resolve_settings()")
    from testuary.settings import model_credential_variables

    hidden = model_credential_variables(settings)
    return {name: value for name, value in settings.environment.items()
            if name not in hidden}


def get_executor() -> "ExecutionBackend":
    """The execution backend project subprocesses run through, per
    ``project_exec_mode()``. Container mode requires ``TESTUARY_PROJECT_CONTAINER``
    (the running container) and runs the in-container interpreter that
    ``TESTUARY_PROJECT_PYTHON_EXECUTABLE`` names. A local command that names
    no environment, and the container runner itself, use
    :func:`launch_environment`."""

    environment = launch_environment()
    if config.project_exec_mode() == "container":
        container = config.environment_value("TESTUARY_PROJECT_CONTAINER", "").strip()
        if not container:
            raise RuntimeError(
                "project_exec=container requires TESTUARY_PROJECT_CONTAINER "
                "(the running container name or id)."
            )
        runner = (config.environment_value("TESTUARY_PROJECT_CONTAINER_RUNNER", "").strip()
                  or "docker")
        return ContainerExecutor(container, runner=runner,
                                 work_dir=project_work_dir(),
                                 control_environment=environment)
    return CapturedExecutor(LocalExecutor(), environment)


def project_work_dir() -> Optional[str]:
    """The host<->container shared work directory for container mode, bind-
    mounted at an identical path so test files and coverage data are visible on
    both sides. Temp dirs nest under it there. ``TESTUARY_PROJECT_WORK_DIR`` env.
    None in local mode, where temp dirs use the system default location."""
    value = config.environment_value("TESTUARY_PROJECT_WORK_DIR", "").strip()
    return str(resolve_host_path(value).resolve()) if value else None


def _captured_source_dir() -> str:
    """The source directory TESTUARY_PROJECT_SOURCE_DIR names, as given."""
    return config.environment_value("TESTUARY_PROJECT_SOURCE_DIR", "").strip()


def project_repo_dir() -> Optional[str]:
    """The selected guest source directory, falling back to the captured repo input.

    The selected source directory wins over its variable in container mode.
    Private execution overrides never relocate this source directory selection.
    """
    if config.project_exec_mode() == "container":
        selected = config.get_project_config().get("source_dir")
        if selected:
            return selected
    return _captured_source_dir() or None


@contextlib.contextmanager
def scratch_dir(prefix: str = "testuary_"):
    """A temporary directory under the project work directory, torn down by whoever
    wrote into it.

    A command run through the executor writes its outputs (pytest caches,
    coverage data) into this directory. In container mode that process may run
    as a different uid than the host (some testbed images run as root), so the
    teardown removes the tree through the executor: the writer's own user can
    delete what it created, where a host-side delete across the uid boundary
    cannot. Local mode is a plain tempdir under the work directory (None falls back
    to the system default location). Its removal is attempted on every
    exit, also after an exception, and a failed removal is ignored."""
    d = tempfile.mkdtemp(prefix=prefix, dir=project_work_dir())
    try:
        yield d
    finally:
        if config.project_exec_mode() == "container":
            try:
                get_executor().run(["rm", "-rf", d], timeout=120)
            except Exception:
                pass
        shutil.rmtree(d, ignore_errors=True)


def read_project_text(path: str) -> str:
    """Read a project file's text. One rule: everything testuary does to the
    project goes through the executor. In container mode the file lives in the
    container, so the read runs there (``_container_file_bytes``). Locally it
    is a direct read. Both read UTF-8 with undecodable bytes replaced and
    CRLF or CR as LF, so both modes give the same text."""
    if config.project_exec_mode() != "container":
        return Path(path).read_text(encoding="utf-8", errors="replace")
    text = _container_file_bytes(path).decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_project_file(path: str) -> Optional[str]:
    """Hex SHA-256 of a project file's bytes, through the executor in
    container mode (the file lives in the container, so a direct host
    read cannot see it). Locally a direct read. Returns None when the
    file cannot be read in the active mode, so a caller comparing
    hashes fails closed on an unreadable source."""
    import hashlib
    if config.project_exec_mode() != "container":
        p = Path(path)
        if not p.is_file():
            return None
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            return None
    try:
        return hashlib.sha256(_container_file_bytes(path)).hexdigest()
    except Exception:
        return None


def _container_file_bytes(path: str) -> bytes:
    """The bytes of a file in the container, read through the executor with
    coreutils' base64, which keeps them exact over the text channel. The
    container's other commands use coreutils too (cat, cp, rm), so no
    language's runtime is needed. OSError when the read fails."""
    import base64
    r = get_executor().run(["base64", "--", str(path)], timeout=60)
    if r.returncode != 0:
        raise OSError(
            f"reading {path} in the container failed (rc={r.returncode}): "
            f"{(r.stderr or '')[-300:]}")
    return base64.b64decode(r.stdout or "")


def _venv_home() -> Path:
    """Where provisioned runtimes live: the user's cache directory for Testuary,
    as platformdirs names it, outside every project, so no copy of a project
    holds one and no runtime directory is taken for a project's Testuary directory."""
    import platformdirs

    return Path(platformdirs.user_cache_dir("testuary", appauthor=False)) / "venvs"


def provision_runtime(language: str, request) -> str:
    """The runtime that fulfils ``language``'s provisioning ``request`` (a
    version and a requirements file), made once by its adapter
    (``provision_runtime``, for Python uv, the requirements and the tools
    LangAdapter pins for that Python) in the language's directory of Testuary's
    cache, named after the configured project and reused while its inputs
    stay the same. A relative requirements path lies under the configuration
    directory. A refusal, a language that provisions nothing or a failed
    provisioning is a RuntimeError naming the setting or saying why."""
    from langadapter import registration
    from langadapter.context import ProvisioningRequest
    from langadapter.errors import InvalidRequest

    proj_cfg = config.get_project_config()
    key = str(proj_cfg.get("project_name") or "").strip() or f"{language}{request.version}"
    requirements = request.requirements
    if requirements:
        path = Path(requirements)
        requirements = str(path if path.is_absolute() else
                           (config.configuration_base_dir() / path).resolve())
    provision = getattr(registration(language).factory, "provision_runtime", None)
    if provision is None:
        raise RuntimeError(f"[project] runtime: the {language} adapter provisions no runtime: "
                           "select one")
    try:
        return provision(ProvisioningRequest(request.version, requirements),
                         store=_venv_home() / language, name=key,
                         environment=launch_environment(), log=lambda line: print(f"[env] {line}"))
    except InvalidRequest as refusal:
        raise RuntimeError(f"[project] {refusal.field}: {refusal.message}") from refusal
    except RuntimeError as failure:
        raise RuntimeError(f"[env] {failure}") from failure


def _source_dir_base() -> Optional[Path]:
    """Resolve the configured ``source_dir_override`` to an absolute Path.

    Returns ``None`` when the project configures no source directory override
    (paths then cannot be made relative safely and stay absolute).
    Differs from ``resolve_source_dir`` in returning the project
    directory (e.g. ``.../Python-3.12.13``), not the import-root subdir
    (``.../Python-3.12.13/Lib``). Relativizing against the import root
    would produce paths like ``_bootstrap.py`` which lose the ``Lib/``
    prefix and break ``file_to_module_name``.

    Per-target context: ``set_invocation_source_dir`` points
    ``source_dir_override`` at the target's copy of the project for
    the duration of a target file's phases, so callers of this function get
    the copy's root, not the source directory.
    """
    proj_cfg = config.get_project_config()
    override = (proj_cfg.get("source_dir_override") or "").strip()
    if not override:
        return None
    base = Path(override)
    if not base.is_absolute():
        base = (config.configuration_base_dir() / base).resolve()
    else:
        base = base.resolve()
    return base


def relativize_path(file_path: str) -> str:
    """Convert an absolute file path to a path relative to the source directory.

    Used when persisting paths to CSVs / logs / backup manifests so that
    artifacts stay valid after the project namespace directory is renamed
    (e.g. ``minimax/Python-3.12.13/Lib/x.py`` -> ``Lib/x.py``). Falls
    back to the original string when:

      - ``file_path`` is empty / already relative
      - no ``source_dir_override`` is configured
      - the file is not under the source directory (cross-project reference)

    Forward slashes are emitted unconditionally so the same CSV is
    portable between Windows and POSIX runners.
    """
    if not file_path:
        return file_path
    p = Path(file_path)
    if not p.is_absolute():
        return file_path
    base = _source_dir_base()
    if base is None:
        return file_path
    try:
        rel = p.resolve().relative_to(base)
    except ValueError:
        return file_path
    return str(rel).replace("\\", "/")


def workspace_name(file_path: str) -> str:
    """Stable per-target directory name: the target's path relative to
    the source directory, slugified. ``pkg/core/module.py`` becomes
    ``pkg_core_module``.

    Path-derived so the directory is stable across reruns: a target's number
    (the target_number column) is its position in the run's target list, which can differ between
    runs, so it is recorded as a column in the CSVs rather than baked into
    the directory name. Accepts an absolute
    path or an already-relative one (an absolute path is first reduced
    via :func:`relativize_path`). Each path segment's leading underscores
    are stripped so directory listings sort sensibly.
    """
    rel = Path(relativize_path(file_path))
    if rel.suffix:
        rel = rel.with_suffix("")
    segments = []
    for seg in rel.parts:
        if seg == rel.anchor or seg in ("/", "\\"):
            continue
        seg = seg.lstrip("_")
        if seg and seg not in (".", ".."):
            segments.append(seg)
    return "_".join(segments) or Path(file_path).stem.lstrip("_") or "target"


def target_workspace_name(target, projects_root=None) -> str:
    """The per-target workspace directory name for ``target``.

    With ``projects_root`` given and the target under it, the name slugifies the
    target's path relative to that root: a container-absolute target such as
    ``/home/user/project/pkg/core/groupby.py`` under the root
    ``/home/user/project`` yields ``pkg_core_groupby``, not the whole mount
    path (``home_user_project_pkg_core_groupby``). ``target`` and
    ``projects_root`` are passed as the caller's own path objects so the relative
    step stays in one path flavor (a POSIX target in the copy relativizes against
    a POSIX root, a host path against a host root). Falls back to
    :func:`workspace_name` of the path as given when no root applies (none given,
    the target sits elsewhere, or the two are of incompatible flavors)."""
    if projects_root is not None:
        try:
            return workspace_name(str(target.relative_to(projects_root)))
        except (ValueError, TypeError):
            pass
    return workspace_name(str(target))


def run_name(run_idx: int) -> str:
    """The directory name of a run: ``r0`` for the first run, ``r1`` for the
    second. ``run_idx`` is the run's number, from 0."""
    run_idx = int(run_idx)
    if run_idx < 0:
        raise ValueError(f"a run index counts from 0, not {run_idx}")
    return f"r{run_idx}"


def run_dir(target_dir: Path, run_idx: int) -> Path:
    """The directory of one run of a file below the file's results or log
    directory, such as ``results/<workspace>/r0``."""
    return Path(target_dir) / run_name(run_idx)


def phase_artifact_path(
    base_dir: Path,
    artifact_name: str,
    run_idx: int = 0,
) -> Path:
    """The path of the artifact ``artifact_name`` of run ``run_idx`` below
    the file's directory ``base_dir``: inside the run's own directory, such as
    ``tests -> r0/tests`` for the first run and ``r1/tests`` for the
    second. A file and a directory are named alike."""
    return run_dir(base_dir, run_idx) / artifact_name


def run_idx_from_artifact_path(path: Path) -> int:
    """Inverse of ``run_dir``: the run index of a run directory (``.../r2`` ->
    2), read from its own name only. Any other directory gives 0, the first
    run."""
    name = Path(path).name
    return int(name[1:]) if len(name) > 1 and name[0] == "r" and name[1:].isdigit() else 0


def pick_phase_artifact(
    base_dir: Path,
    artifact_name: str,
    current_run_idx: int,
) -> Path:
    """Resolve which run's version of an upstream-phase artifact the
    current run reads: its own run's, else the first run's (r0), else
    ``FileNotFoundError`` naming both. It never reads another later run's
    artifact."""
    same = phase_artifact_path(
        base_dir, artifact_name, current_run_idx,
    )
    if same.exists():
        return same
    canonical = phase_artifact_path(
        base_dir, artifact_name, 0,
    )
    if canonical.exists():
        return canonical
    raise FileNotFoundError(
        f"pick_phase_artifact: needed {same} or the first run's "
        f"{canonical}, and neither exists. Run the upstream phase first."
    )


def _check_directory_route(path: Path, *, purpose: str, action: str) -> None:
    """Inspect lexical ancestors before following a selected filesystem path."""
    from testuary.settings import ConfigurationError

    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            raise ConfigurationError(f"{purpose} route is not an ordinary directory: {component}. {action}") from None
        except OSError as exc:
            # A name Windows cannot represent (a URL's "https:") names no file
            # and so no link. Path.exists treats it as absent the same way.
            if getattr(exc, "winerror", None) != 123:
                raise
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ConfigurationError(
                f"{purpose} route is not an ordinary directory: {component}. "
                f"Links and reparse points cannot select an execution or cleanup root. {action}"
            )


def resolve_project_source_path(target_repo: str, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve target_repo to an existing local directory if one is configured."""
    target_repo = (target_repo or "").strip()
    if not target_repo:
        return None

    expanded = resolve_host_path(target_repo)
    anchor = base_dir or config.configuration_base_dir()
    candidates = [expanded]
    if not expanded.is_absolute():
        candidates.insert(0, anchor / expanded)

    for candidate in candidates:
        _check_directory_route(candidate, purpose="source", action="Select an ordinary --source-dir")
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def read_file_content(file_path: str) -> str:
    """Read a project file's content.

    Every phase loads the file under test through here, so it follows the one
    rule: in container mode the file lives in the container and the read runs
    there. Locally it is a direct read.

    Raises:
        FileNotFoundError / OSError: when the file cannot be read.
    """
    if config.project_exec_mode() != "container" and not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return read_project_text(file_path)


#: The variables Testuary's own child processes read inside a container.
TESTUARY_ENV_PREFIX = "TESTUARY_"


class ContainerExecutor(_ContainerExecutor):
    """The backend's container executor, forwarding Testuary's variables
    too. ``control_environment`` is the captured host environment the
    container runner itself is launched with. Omitted, it inherits."""

    def __init__(self, container: str, *, runner: str = "docker",
                 work_dir: Optional[str] = None,
                 container_work_dir: Optional[str] = None,
                 control_environment: Optional[Mapping[str, str]] = None) -> None:
        super().__init__(
            container, runner=runner, work_dir=work_dir,
            container_work_dir=container_work_dir,
            forward_prefixes=(TESTUARY_ENV_PREFIX, ENV_PREFIX),
            control_environment=(None if control_environment is None
                                 else Environment(dict(control_environment), inherit=False)))


class CapturedExecutor(ExecutionBackend):
    """Runs through another executor. A command that names no environment is
    launched with the run's captured launch environment, never the live
    process environment."""

    def __init__(self, executor: ExecutionBackend, environment: Mapping[str, str]) -> None:
        self._executor = executor
        self._environment = dict(environment)

    def run(self, argv: Sequence[str], *, cwd: Optional[str] = None,
            env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None,
            input: Optional[str] = None) -> subprocess.CompletedProcess:
        return self._executor.run(
            argv, cwd=cwd, env=dict(self._environment) if env is None else env,
            timeout=timeout, input=input)


def _serialize_enabled() -> bool:
    """The run's ``[timing] serialize`` as the settings check resolved and
    the run records it, on by default, so parallel runs are safe by
    construction."""
    from testuary import config
    return config.load_config("timing")["serialize"] == "true"


@contextmanager
def acquire() -> Iterator[None]:
    """Hold the process-wide benchmark lock for the ``with`` block when
    serialization is on. Otherwise it does nothing."""
    with _placement.serialized(_serialize_enabled()):
        yield
