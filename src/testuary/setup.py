"""Testuary's setup for other programs.

The walk ``testuary setup`` runs is ``run_setup``: every question is a
``Question`` answered through one ``ask`` function, every look at the
machine goes through a ``Probe``, and the result names what was written and
what a run still lacks. A program shows the questions its own way and gets
the same checks and writes as the terminal.

The model catalogue is listed, and its aliases created, edited and removed
one at a time as the setup changes them: ``models``, ``usable_aliases``,
``set_model`` and ``remove_model``. A project's settings are read and
changed with ``project`` and ``set_project``, with ``languages`` naming the
languages a project can take. Their files are the ones
``testuary.settings.file_locations`` names: the model catalogue in the user
directory (``home`` when given), a project's settings file in its Testuary
directory. Every change keeps the file's comments, line endings and other
entries, and a model catalogue that holds a key is readable by its owner
only. A refused change raises ``SetupError``, whose ``problems`` name each
refused field. No function here returns a stored key.
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from langadapter.resources import allowed_cpus
from testuary import __version__
from testuary import settings as _ini_text


PARTS = ("model", "machine", "project")
"""The setup's parts, as ``run_setup``'s ``part`` names them."""
#: How many invalid answers a question accepts before the wizard stops.
_ATTEMPTS = 5
#: This version's documentation, online: an installed package has no docs directory.
_DOCS = f"https://github.com/k-kammler/testuary/blob/v{__version__}/docs/"
#: The settings reference of this version.
SETTINGS_REFERENCE = _DOCS + "reference/settings.md"


class SetupError(RuntimeError):
    """The wizard stopped, or a catalogue change was refused. The message
    says why and what to do. For a refused change, ``problems`` names each
    refused value as (field, problem), the field being the argument of
    ``set_model`` or ``set_project`` it concerns, or the project key."""

    def __init__(self, message: str, problems: Sequence[Tuple[str, str]] = ()) -> None:
        super().__init__(message)
        self.problems = tuple(problems)


def _refused(problems: Sequence[Tuple[str, str]]) -> SetupError:
    return SetupError(". ".join(f"{field}: {problem}" for field, problem in problems), problems)


def _refused_setting(error: Exception, suffix: str = "") -> SetupError:
    """A run's refusal of a value as the setup's: the message, and the
    setting it names (``error.setting``, set by the settings' check) as the
    problem's field, the key, or the section for a refused section."""
    section, key = getattr(error, "setting", None) or (None, None)
    field = key or section
    message = f"{error}{suffix}"
    return SetupError(message, ((field, str(error)),) if field else ())


@dataclass(frozen=True)
class Question:
    """One question: its stable key, the text shown, the default Enter
    accepts, its choices, and the reason shown with it, if any. A closed
    question takes one of its choices. An ``open`` one also takes another
    answer, such as a path, that its check accepts: its choices are what
    the setup found. A secret answer is read without echo and never shown."""

    key: str
    prompt: str
    default: str = ""
    choices: Tuple[str, ...] = ()
    secret: bool = False
    reason: str = ""
    open: bool = False


Ask = Callable[[Question], str]
"""The question function ``run_setup`` takes: it returns the answer to a
``Question`` as text, an empty text for the question's default."""
Say = Callable[[str], None]
"""The function ``run_setup`` shows one line to the person with."""


@dataclass(frozen=True)
class Provider:
    """A kind of model service the wizard offers."""

    kind: str
    label: str
    provider: str
    address: str
    key: str  # "required", "optional" or "none"
    custom_address: bool = False
    #: The concurrent starter requests to propose, with the reason shown.
    starters: int = 4
    starters_reason: str = ""


PROVIDERS: Tuple[Provider, ...] = (
    Provider("ollama-local", "Ollama on this machine", "ollama", "http://localhost:11434/v1", "none",
             starters=1, starters_reason="a local Ollama answers one request at a time"),
    Provider("ollama-cloud", "Ollama Cloud", "ollama", "https://ollama.com/v1", "required",
             starters=4, starters_reason="Ollama Cloud refuses more than a few concurrent requests"),
    Provider("openai", "OpenAI", "openai", "https://api.openai.com/v1", "required",
             starters=4, starters_reason="a cautious start, raise it within your account's rate limit"),
    Provider("deepseek", "DeepSeek API", "deepseek", "https://api.deepseek.com/v1", "required",
             starters=4, starters_reason="DeepSeek states a limit of 25 parallel requests"),
    Provider("vllm", "vLLM server", "vllm", "http://localhost:8000/v1", "optional", custom_address=True,
             starters=10, starters_reason="a vLLM server sustains parallel requests"),
    Provider("openai-compatible", "another OpenAI-compatible API", "", "http://localhost:8000/v1", "optional",
             custom_address=True,
             starters=4, starters_reason="a cautious start, raise it within the service's limit"),
)
"""The kinds of model service the setup offers, in the order it offers them."""


def provider_kind(kind: str) -> Provider:
    """The provider of ``PROVIDERS`` whose kind is ``kind``."""
    return next(provider for provider in PROVIDERS if provider.kind == kind)


#: The root of /proc and of the cgroup file systems the CPU quota is read from.
_SYSTEM_ROOT = Path("/")


def _cpu_limits(version: int, directory: Path) -> Optional[float]:
    """The CPUs the quota of the cgroup ``directory`` allows, None for no
    quota: cpu.max under cgroups v2, cpu.cfs_quota_us over cpu.cfs_period_us
    under v1. A file that is absent sets no quota."""
    if version == 2:
        if not (directory / "cpu.max").is_file():
            return None
        quota, period = (directory / "cpu.max").read_text(encoding="utf-8").split()
        return None if quota == "max" else int(quota) / int(period)
    if not (directory / "cpu.cfs_quota_us").is_file():
        return None
    quota = int((directory / "cpu.cfs_quota_us").read_text(encoding="utf-8"))
    period = int((directory / "cpu.cfs_period_us").read_text(encoding="utf-8"))
    return None if quota < 0 else quota / period


def _cgroup_members(proc: Path) -> Dict[str, PurePosixPath]:
    """This process's cgroup for each controller, ``cgroup2`` for the v2
    one, from /proc/self/cgroup."""
    members = {}
    for line in (proc / "cgroup").read_text(encoding="utf-8").splitlines():
        _number, controllers, member = line.split(":", 2)
        for name in controllers.split(",") if controllers else ("cgroup2",):
            members[name] = PurePosixPath(member)
    return members


def _cpu_cgroups(proc: Path):
    """Each version and directory of this process's CPU cgroup and its
    ancestors up to the mount, from /proc/self/mountinfo. A container's
    cgroup namespace mounts its own cgroup as the mount's root."""
    members = _cgroup_members(proc)
    for line in (proc / "mountinfo").read_text(encoding="utf-8").splitlines():
        fields, _, source = line.partition(" - ")
        fields, source = fields.split(), source.split()
        version = {"cgroup2": 2, "cgroup": 1}.get(source[0])
        member = members.get("cgroup2" if version == 2 else "cpu")
        if version is None or member is None or (version == 1 and "cpu" not in source[2].split(",")):
            continue
        root = PurePosixPath(fields[3])
        inside = member.relative_to(root) if member.is_relative_to(root) else PurePosixPath()
        point = _SYSTEM_ROOT / fields[4].lstrip("/")
        for depth in range(len(inside.parts), -1, -1):
            yield version, point.joinpath(*inside.parts[:depth])


def _quota_cpus() -> Optional[int]:
    """The CPUs the tightest CPU quota of this process's cgroup and its
    ancestors leaves, rounded up, under cgroups v1 or v2, else None: no
    cgroups, no quota, or files that cannot be read."""
    try:
        limits = [_cpu_limits(version, directory)
                  for version, directory in _cpu_cgroups(_SYSTEM_ROOT / "proc" / "self")]
    except (OSError, ValueError, IndexError):
        return None
    limits = [limit for limit in limits if limit is not None]
    return max(1, math.ceil(min(limits))) if limits else None


class Probe:
    """Looks at the machine for defaults and checks. The demonstration
    replaces it with answers from fixtures. ``cpu_count`` returns the number
    of CPUs this process may use, and ``local_ollama`` whether an Ollama on
    this computer lists its models at ``http://localhost:11434`` within two
    seconds."""

    def cpu_count(self) -> int:
        """The number of CPUs this process may use: those its affinity
        allows (``allowed_cpus``), or the machine's where the platform has
        no affinity, at most what a CPU quota leaves (``_quota_cpus``), 1
        when none can be read."""
        allowed = allowed_cpus()
        count = len(allowed) if allowed else (os.cpu_count() or 1)
        quota = _quota_cpus()
        return min(count, quota) if quota else count

    def environment(self, name: str) -> str:
        """The environment variable ``name``, empty when it is not set."""
        return os.environ.get(name, "")

    def is_dir(self, path: Path) -> bool:
        """Whether ``path`` is a directory."""
        return Path(path).is_dir()

    def is_file(self, path: Path) -> bool:
        """Whether ``path`` is a file."""
        return Path(path).is_file()

    def source_files(self, root: Path, language: str) -> List[str]:
        """The language's source files below ``root``, relative to it with
        forward slashes, as the language's adapter tells them: files with its
        file extensions outside the directories it ignores (``ignored_dirs``, and
        dot directories), except the names it declares as no sources
        (``non_source_files``), its test files (``is_test_file``) and empty
        files, which hold nothing to test."""
        from langadapter import registration
        from langadapter.adapter import LangAdapter

        from testuary import languages

        registered = registration(language)
        factory = registered.factory
        backend = languages.get_backend(language)
        skipped = set(getattr(factory, "ignored_dirs", LangAdapter.ignored_dirs))
        excluded = set(getattr(factory, "non_source_files", ()))
        suffixes = tuple(registered.file_extensions)
        found = []
        # Ignored and dot directories are never entered, and a link that leads
        # nowhere is no source file.
        for dirpath, dirnames, names in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in skipped and not name.startswith(".")]
            for name in names:
                path = Path(dirpath) / name
                if not name.endswith(suffixes) or name in excluded:
                    continue
                try:
                    if not path.is_file() or path.stat().st_size == 0:
                        continue
                except OSError:
                    continue
                if not backend.is_test_file(path):
                    found.append(path)
        return [path.relative_to(root).as_posix() for path in sorted(found)]

    def read_text(self, path: Path) -> Optional[str]:
        """The text of the file ``path``, or None when it cannot be read."""
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def run(self, argv: Sequence[str], timeout: float) -> Optional[Tuple[int, str]]:
        """The exit code and output of ``argv``, or None when it does not
        start or does not end within ``timeout`` seconds. It gets no input,
        and output the locale cannot decode is kept with replacements."""
        import subprocess

        try:
            done = subprocess.run(list(argv), capture_output=True, text=True, errors="replace",
                                  stdin=subprocess.DEVNULL, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.returncode, done.stdout

    def which(self, command: str) -> str:
        """The path of the program ``command`` on the search path, empty when it
        is not found."""
        import shutil

        return shutil.which(command) or ""

    def local_ollama(self) -> bool:
        """Whether an Ollama on this computer lists its models at
        ``http://localhost:11434`` within two seconds."""
        return self.model_list(provider_kind("ollama-local"), "http://localhost:11434/v1", "",
                               timeout=2) is not None

    def model_list(self, provider: Provider, address: str, key: str, *,
                   timeout: float = 15) -> Optional[List[str]]:
        """The model names the service lists, a request that generates
        nothing, or None when it cannot be read. The key goes into the
        request header only."""
        import httpx

        headers = {"Authorization": f"Bearer {key}"} if key else {}
        base = address.rstrip("/")
        try:
            if provider.kind == "ollama-local":
                response = httpx.get(base[:-3] + "/api/tags" if base.endswith("/v1") else base + "/api/tags",
                                     timeout=timeout)
                response.raise_for_status()
                return [str(item.get("name")) for item in response.json().get("models", [])]
            response = httpx.get(base + "/models", headers=headers, timeout=timeout)
            response.raise_for_status()
            return [str(item.get("id")) for item in response.json().get("data", [])]
        except Exception:
            return None

    def restrict_to_user(self, path: Path) -> bool:
        """Leave ``path`` readable by its owner only, where the system
        allows it. POSIX files are written with permissions 0600. On Windows the
        file's inherited access is replaced by its owner's."""
        if os.name == "posix":
            os.chmod(path, 0o600)
            return True
        if os.name == "nt":
            import getpass
            import subprocess

            try:
                done = subprocess.run(
                    ["icacls", str(path), "/inheritance:r", "/grant:r", f"{getpass.getuser()}:F"],
                    capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                return False
            return done.returncode == 0
        return False


@dataclass
class SetupResult:
    """What the wizard wrote and what a run still lacks. ``project`` is the
    directory of the project it wrote, where ``run`` runs it."""

    written: List[str] = field(default_factory=list)
    project: str = ""
    alias: str = ""
    run: Tuple[str, ...] = ()
    missing: List[str] = field(default_factory=list)


class _Files:
    """The setup's files for runs started in ``base_dir``, where the settings
    resolver names them (``settings.file_locations``): models.ini and
    config.ini in the user directory (``home``, else TESTUARY_HOME, else the
    platform's). Each read sees the file as it is now, and each part writes
    its file back after the user confirms."""

    def __init__(self, base_dir: Path, home: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir)
        self.home = home

    def path(self, name: str) -> Path:
        from testuary.settings import file_locations

        locations = file_locations(self.base_dir, home=self.home)
        return {"config.ini": locations.config, "models.ini": locations.models}[name]

    def text(self, name: str) -> str:
        path = self.path(name)
        if path.is_file():
            # Bytes, so the file's own line endings are kept.
            return path.read_bytes().decode("utf-8")
        from testuary.settings import _bundled_template_path

        template = _bundled_template_path(name) if name != "config.ini" else None
        if not template:
            return ""
        # Only the template's header comments, up to its first example
        # section, which is commented out.
        header = []
        for line in Path(template).read_text(encoding="utf-8").splitlines():
            if re.match(r"\s*(#\s*)?\[[^\]]+\]\s*$", line):
                break
            header.append(line)
        text = "\n".join(_without_example_sentences(header)).rstrip()
        return text + "\n" if text else ""

    def values(self, name: str) -> Dict[str, Dict[str, str]]:
        import configparser

        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(self.text(name) if self.path(name).is_file() else "")
        except configparser.Error as exc:
            raise SetupError(f"{name} cannot be read ({type(exc).__name__}): fix it, "
                             "then run the setup again") from None
        return {section: dict(parser[section]) for section in parser.sections()}


def _without_example_sentences(lines: Sequence[str]) -> List[str]:
    """A template's header comments without what they say about the
    template's examples, which a new file does not have. A paragraph of
    prose keeps its other complete sentences, rewrapped. A paragraph laid
    out as a table loses only its lines about examples. Separator lines go."""
    import textwrap

    paragraphs: List[List[str]] = [[]]
    for line in lines:
        if line.strip() in ("", "#"):
            paragraphs.append([])
        elif not re.match(r"^#\s*-{3,}", line):
            paragraphs[-1].append(line)
    kept: List[List[str]] = []
    for paragraph in paragraphs:
        if not any(re.search(r"(?i)example", line) for line in paragraph):
            kept.append(paragraph)
        elif any(re.match(r"^#\s{2,}\S", line) for line in paragraph):
            kept.append([line for line in paragraph if not re.search(r"(?i)example", line)])
        else:
            prose = " ".join(line.lstrip("#").strip() for line in paragraph)
            sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", prose)
                         if sentence and not re.search(r"(?i)example", sentence)]
            if sentences:
                kept.append(textwrap.wrap(" ".join(sentences), width=78,
                                          initial_indent="# ", subsequent_indent="# "))
    kept = [paragraph for paragraph in kept if paragraph]
    result: List[str] = []
    for paragraph in kept:
        if result:
            result.append("#")
        result.extend(paragraph)
    return result


def _asked(ask: Ask, question: Question, check: Callable[[str], Optional[str]], say: Say) -> str:
    """Ask until the answer passes ``check``, Enter taking the default."""
    for _ in range(_ATTEMPTS):
        answer = ask(question)
        answer = question.default if answer.strip() == "" else answer.strip()
        problem = check(answer)
        if problem is None:
            return answer
        say(f"  {problem}")
    raise SetupError(f"no valid answer to '{question.prompt}' after {_ATTEMPTS} tries")


def _yes(answer: str) -> bool:
    return answer.strip().lower() in ("y", "yes")


def _yes_no(answer: str) -> Optional[str]:
    return None if answer.strip().lower() in ("y", "yes", "n", "no") else "answer yes or no"


def _choice(choices: Sequence[str]) -> Callable[[str], Optional[str]]:
    def check(answer: str) -> Optional[str]:
        return None if answer in choices else f"choose one of {', '.join(choices)}"
    return check


def _label(value: str) -> str:
    """A name a run can use everywhere it appears (a section, a directory, a
    TESTUARY_<NAME>_API_KEY variable): lowercase letters, digits and _."""
    text = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return text or "project"


def _label_check(refused: Optional[Callable[[str], Optional[str]]] = None) -> Callable[[str], Optional[str]]:
    """The check of a name: its form and ``refused``. The name it suggests
    for a malformed answer passes the whole check."""

    def problem(answer: str) -> Optional[str]:
        return refused(answer) if refused is not None else None

    def suggestion(answer: str) -> str:
        base = _label(answer)
        candidate, number = base, 1
        while problem(candidate) is not None:
            number += 1
            candidate = f"{base}{number}"
        return candidate

    def check(answer: str) -> Optional[str]:
        if answer != _label(answer):
            return f"use lowercase letters, digits and _ only, for example {suggestion(answer)}"
        return problem(answer)
    return check


def _section_text(section: str, values: Sequence[Tuple[str, str]], hidden: Sequence[str] = ()) -> str:
    lines = [f"[{section}]"] + [f"{key} = {'(hidden)' if key in hidden else value}"
                                for key, value in values]
    return "\n".join(lines)


# The catalogues

#: The providers whose endpoint the packaged configuration holds.
_BUILT_IN_PROVIDERS = ("ollama", "openai", "vllm")


@dataclass(frozen=True)
class ModelEntry:
    """One alias of the model catalogue as the setup sees it. ``address`` is
    the address the catalogue names for the alias or its provider, never
    with credentials, and empty for the provider's configured endpoint.
    ``key_source`` says where its key comes from: ``models.ini``,
    ``environment``, ``not needed`` for a local address, or ``missing``.
    ``key_variable`` names the variable the key is read from, or that a run
    would read it from when it is missing. ``usable`` says whether a run can
    use the alias. ``file`` is the catalogue file the alias's section comes
    from. It holds no key."""

    alias: str
    provider: str
    model: str
    address: str
    key_source: str
    key_variable: str
    usable: bool
    file: str


@dataclass(frozen=True)
class ProjectEntry:
    """A project's settings as the setup writes them: its directory
    (``project_dir``), its settings file (``settings_file``), and the file's
    ``[project]`` values with the model alias of its ``[api]`` section as
    ``model`` (``fields``)."""

    project_dir: Path
    settings_file: Path
    fields: Dict[str, str]


def _dir(base_dir: Optional[Path]) -> Path:
    return Path(base_dir) if base_dir is not None else Path.cwd()


def _key_source(provider: str, fields: Dict[str, str], provider_fields: Dict[str, str],
                endpoint: str, probe: Probe) -> Tuple[str, str]:
    """Where a run takes an alias's key from and the variable it reads, by
    the run's own rule (``models.resolve_provider_credential``) over the
    probe's environment. A variable the alias or its provider names refuses
    every run while it is unset, so the key is missing. Without a key, an
    alias that names no variable needs none at a local address."""
    from testuary.models import resolve_provider_credential

    variable = str(fields.get("api_key_env") or provider_fields.get("api_key_env") or "")
    default_variable = _ini_text.credential_variable(provider)
    environment = {name: probe.environment(name) for name in (variable, default_variable) if name}
    try:
        key = resolve_provider_credential(provider, provider_fields, fields, environment)
    except _ini_text.ConfigurationError:
        return "missing", variable
    if fields.get("api_key") or provider_fields.get("api_key"):
        return "models.ini", ""
    if key:
        return "environment", variable or default_variable
    if not variable and _is_local(endpoint):
        return "not needed", ""
    return "missing", variable or default_variable


def _model_entries(files: _Files, probe: Probe, listed: Optional[_Listed] = None) -> List[ModelEntry]:
    """The aliases of the model catalogue, or of the ``listed`` catalogue
    files joined, each naming a provider other than itself, with where
    their keys come from (``_key_source``) and the file its section comes
    from. Without a key, the local address that lets an alias need none is
    its own, else its provider's configured endpoint (config.ini's, else
    the packaged one), which a vLLM server on this machine has by
    default."""
    from testuary.settings import _safe

    if listed is None:
        catalogue = files.values("models.ini")
        origin = {name: str(files.path("models.ini")) for name in catalogue}
    else:
        catalogue = listed[2]
        origin = {name: str(listed[0][_origin(listed[1], name)]) for name in catalogue}
    configured = files.values("config.ini")
    entries = []
    for alias, fields in catalogue.items():
        provider = str(fields.get("provider") or "").strip()
        if not provider or alias == provider:
            continue
        provider_fields = catalogue.get(provider, {})
        address = str(fields.get("api_url") or provider_fields.get("api_url") or "")
        endpoint = (address or str(configured.get(provider, {}).get("api_url") or "")
                    or _config_address(provider))
        source, read_from = _key_source(provider, fields, provider_fields, endpoint, probe)
        entries.append(ModelEntry(alias, provider, str(fields.get("model") or ""),
                                  _safe({"api_url": address})["api_url"] if address else "",
                                  source, read_from, source != "missing", origin[alias]))
    return entries


def _usable_aliases(files: _Files, probe: Probe) -> List[str]:
    """Aliases of models.ini that name a provider and can reach their
    credentials, or need none (a local address)."""
    return [entry.alias for entry in _model_entries(files, probe) if entry.usable]


def _write_catalogue(files: _Files, probe: Probe, name: str, text: str) -> Path:
    """Write a catalogue, a model catalogue holding a key readable by its
    owner only. A user directory a fresh machine does not have yet is made."""
    path = files.path(name)
    restrict = probe.restrict_to_user if name == "models.ini" and _holds_key(text) else None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ini_text.write_atomic(path, text, restrict=restrict)
    except OSError as exc:
        raise SetupError(str(exc)) from None
    return path


def _model_values(provider_name: str, address: str, name: str, key: str,
                  key_variable: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """The provider section's and the alias section's values for a model:
    a provider other than the built-in ones keeps its address, and a stored
    key goes to the provider. A variable other than the provider's default
    belongs to the alias, and so does a built-in provider's address the
    setup was given or confirmed, so that the alias does not follow another
    address a settings file gives its provider: in the provider's section
    they would change every alias of the service."""
    provider_values: List[Tuple[str, str]] = []
    if provider_name not in _BUILT_IN_PROVIDERS:
        provider_values.append(("api_url", address))
    if key:
        provider_values.append(("api_key", key))
    alias_values = [("provider", provider_name), ("model", name)]
    if key_variable and key_variable != _ini_text.credential_variable(provider_name):
        alias_values.append(("api_key_env", key_variable))
    if provider_name in _BUILT_IN_PROVIDERS and address:
        alias_values.append(("api_url", address))
    return provider_values, alias_values


def _model_text(files: _Files, provider_name: str, provider_values: Sequence[Tuple[str, str]],
                alias: str, alias_values: Sequence[Tuple[str, str]]) -> str:
    """The model catalogue's text with a model's provider and alias sections.
    An existing alias's section is replaced as a whole, and the provider's
    keeps its other keys."""
    text = files.text("models.ini")
    if provider_values:
        text = _ini_text.set_values(text, provider_name, provider_values)
    edit = _ini_text.replace_section if alias in files.values("models.ini") else _ini_text.set_values
    return edit(text, alias, alias_values)


def _write_model(files: _Files, probe: Probe, provider_name: str,
                 provider_values: Sequence[Tuple[str, str]], alias: str,
                 alias_values: Sequence[Tuple[str, str]]) -> Path:
    """Write a model's provider and alias sections (``_model_text``)."""
    return _write_catalogue(files, probe, "models.ini",
                            _model_text(files, provider_name, provider_values, alias, alias_values))


# A list of catalogue files (models_files): runs read them in order, and a
# change goes to the last one.

#: The catalogue files of a list, each one's own sections, and the catalogue
#: its runs read (``_listed``).
_Listed = Tuple[List[Path], List[Dict[str, Dict[str, str]]], Dict[str, Dict[str, str]]]

#: The keys of an alias's section the setup's model fields set.
_MODEL_KEYS = ("provider", "model", "api_url", "api_key", "api_key_env")


def _listed(base_dir: Path, home: Optional[Path], models_files: Sequence[Union[str, Path]],
            **settings_files) -> _Listed:
    """The catalogue files ``models_files`` names, relative ones from
    ``base_dir``, each with its own sections, and the catalogue the run's
    own resolution of the list (``resolve_settings(models_files=...)``,
    with the settings files ``settings_files`` names) reads: an absent last
    file has none. SetupError gives the run's refusal."""
    from testuary.settings import _read_ini_file, resolve_settings

    if isinstance(models_files, (str, Path)) or not models_files:
        raise SetupError("models_files names the catalogue files in order, the one a change goes to last")
    paths = [Path(name) if Path(name).is_absolute() else base_dir.resolve() / name for name in models_files]
    try:
        settings = resolve_settings(base_dir=base_dir, home=home, models_files=paths, **settings_files)
        return paths, [_read_ini_file(path) for path in paths], _catalogue(settings)
    except (ValueError, OSError) as exc:
        raise SetupError(str(exc)) from None


def _catalogue(settings) -> Dict[str, Dict[str, str]]:
    """The model catalogue resolved settings hold, as plain sections."""
    return {name: dict(fields) for name, fields in settings.models.items()}


def _origin(layers: Sequence[Dict[str, Dict[str, str]]], name: str, before: Optional[int] = None) -> Optional[int]:
    """The position of the last of ``layers``, up to ``before``, that holds
    the section ``name``, the file its runs read it from, or None."""
    found = [position for position, sections in enumerate(layers[:before]) if name in sections]
    return found[-1] if found else None


def _resolved_with(listed: _Listed, text: str, base_dir: Path, home: Optional[Path],
                   alone: bool = False) -> Dict[str, Dict[str, str]]:
    """The catalogue the list's runs would read with ``text`` as its last
    file, or with that file ``alone``: the run's own resolution, ``text`` in
    a temporary file in the last file's nearest existing directory, which
    lies in the project exactly when the last file does, made new and
    readable by its owner only. A refusal names the last file."""
    import tempfile

    from testuary.settings import resolve_settings

    last = listed[0][-1]
    place = next(directory for directory in (last.parent, *last.parent.parents) if directory.is_dir())
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{last.name}.", dir=place)
    except OSError as exc:
        raise SetupError(str(exc)) from None
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(text.encode("utf-8"))
        files = [temporary] if alone else [*listed[0][:-1], temporary]
        return _catalogue(resolve_settings(base_dir=base_dir, home=home, models_files=files))
    except (ValueError, OSError) as exc:
        raise SetupError(str(exc).replace(str(temporary), str(last))) from None
    finally:
        temporary.unlink(missing_ok=True)


def _resolving(catalogue: Dict[str, Dict[str, str]], options: Mapping[str, object]) -> set:
    """The aliases of ``catalogue`` that resolve their provider, model,
    address and protocol as a run whose settings ``resolve_settings`` reads
    with ``options`` resolves them (``models.resolve_model_identity``). No
    credential is looked up."""
    import dataclasses

    from testuary.models import resolve_model_identity
    from testuary.settings import resolve_settings

    settings = resolve_settings(models=catalogue, **options)
    found = set()
    for alias, fields in catalogue.items():
        if not str(fields.get("provider") or "").strip():
            continue
        api = {**settings.sections["api"], "model": alias}
        try:
            resolve_model_identity(dataclasses.replace(settings, sections={**settings.sections, "api": api}))
        except ValueError:
            continue
        found.add(alias)
    return found


def _keep_resolving(before: Dict[str, Dict[str, str]], after: Dict[str, Dict[str, str]],
                    changed: Sequence[str], base_dir: Path, home: Optional[Path],
                    removed: Sequence[str] = ()) -> None:
    """Refuse a catalogue change that leaves an alias of ``changed``, or an
    alias that resolved before it, unresolved (``_resolving``), but an alias
    ``removed`` that ``after`` no longer holds. An alias that did not
    resolve before blocks no change to another."""
    try:
        earlier = _resolving(before, {"base_dir": base_dir, "home": home}) - (set(removed) - set(after))
    except (ValueError, OSError):
        earlier = set()
    try:
        broken = sorted((earlier | set(changed)) - _resolving(after, {"base_dir": base_dir, "home": home}))
    except (ValueError, OSError) as exc:
        raise SetupError(str(exc)) from None
    if broken:
        raise SetupError(f"This change leaves {', '.join(broken)} without a provider, model, address or "
                         "protocol a run can use, so nothing was written",
                         [("alias", f"{alias} would no longer resolve") for alias in broken])


def _held_keys(path: Path, sections: Dict[str, Dict[str, str]]) -> None:
    """Refuse a catalogue file changed through a list while it holds a key."""
    held = [name for name, fields in sections.items() if "api_key" in fields]
    if held:
        raise SetupError(f"{path}: [{held[0]}] holds an api_key, which a file changed through a list of "
                         "catalogue files does not keep: name the variable that holds it with api_key_env "
                         "in its place")


def _override(listed: _Listed, name: str, values: Dict[str, str],
              key_variable: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """The section ``name`` with ``values`` as the list's last file holds it,
    and its problems: an override of an earlier file's section starts from
    its keys, a key held there is not copied and needs ``key_variable`` as
    its api_key_env, and no address that holds a credential is written."""
    from testuary.settings import _address_credentials

    paths, layers, _ = listed
    position = _origin(layers, name, -1)
    last = paths[-1]
    problems: List[Tuple[str, str]] = []
    fields = dict(values)
    source = layers[position][name] if position is not None and name not in layers[-1] else None
    if source is not None:
        fields.update({key: value for key, value in source.items() if key not in values and key != "api_key"})
        if "api_key" in source and key_variable:
            fields["api_key_env"] = key_variable
        elif "api_key" in source:
            problems.append(("key_variable", f"{paths[position]} holds the key of [{name}], which is not "
                                             f"copied into {last}: name the variable that holds it"))
    address = fields.get("api_url", "")
    if address and _address_credentials(address):
        copied = source is not None and source.get("api_url") == address
        problems.append(("address", (
            f"[{name}] in {paths[position]} has an address that holds a credential, which is not copied "
            f"into {last}. Change its address in {paths[position]} to one without the credential, and "
            "give the key through a key variable") if copied else (
            f"The address holds a credential, which {last} does not keep. Name an address without it, "
            "and give the key through a key variable")))
    return fields, problems


def _listed_model_text(listed: _Listed, provider: str, provider_values: Sequence[Tuple[str, str]],
                       alias: str, alias_values: Sequence[Tuple[str, str]], key_variable: str) -> str:
    """The list's last file's text with a model's provider and alias
    sections (``_override``). The alias's section is written whole, its
    other keys kept from an earlier file's section it overrides. The
    provider's section changes only when its values do."""
    paths, layers, joined = listed
    own = layers[-1]
    _held_keys(paths[-1], own)
    text = paths[-1].read_bytes().decode("utf-8") if paths[-1].is_file() else ""
    problems: List[Tuple[str, str]] = []
    values = dict(provider_values)
    if any(joined.get(provider, {}).get(key) != value for key, value in values.items()):
        fields, found = _override(listed, provider, {**own.get(provider, {}), **values}, key_variable)
        problems += found
        text = _ini_text.replace_section(text, provider, list(fields.items()))
    kept = {} if alias in own else {key: value for key, value in joined.get(alias, {}).items()
                                    if key not in _MODEL_KEYS}
    fields, found = _override(listed, alias, {**dict(alias_values), **kept}, key_variable)
    problems += found
    if problems:
        raise _refused(problems)
    return _ini_text.replace_section(text, alias, list(fields.items()))


def _ended_import_text(listed: _Listed, alias: str, key_variables: Mapping[str, str],
                       base_dir: Path, home: Optional[Path]) -> str:
    """The list's last file's text once the import of the earlier files ends
    without ``alias``: every other section of the joined list, an earlier
    one written whole as an override with the variable ``key_variables``
    names for its section in place of a key it holds. Refused, naming every
    section, when a variable is missing or an address holds a credential,
    and when an alias that resolved before would not resolve from the last
    file read alone."""
    paths, layers, joined = listed
    _held_keys(paths[-1], layers[-1])
    _alias_fields(joined, alias)
    text = paths[-1].read_bytes().decode("utf-8") if paths[-1].is_file() else ""
    problems: List[Tuple[str, str]] = []
    for name in joined:
        if name != alias and name not in layers[-1]:
            fields, found = _override(listed, name, {}, str(key_variables.get(name) or "").strip())
            problems += [(name, problem) for _, problem in found]
            text = _ini_text.replace_section(text, name, list(fields.items()))
    if problems:
        raise _refused(problems)
    _keep_resolving(joined, _resolved_with(listed, text, base_dir, home, alone=True), (), base_dir, home, [alias])
    return text


def _project_settings_file(base_dir: Path, *, own: bool,
                           home: Optional[Path] = None) -> Tuple[Path, Path]:
    """The project's directory and its settings file: ``base_dir`` itself when
    ``own``, else the project found from ``base_dir`` upward as a run finds
    it, with the Testuary directory's name from the user directory ``home``'s
    configuration when given."""
    from testuary.settings import project_locations

    try:
        locations = project_locations(base_dir, home=home)
    except (OSError, ValueError) as exc:
        raise SetupError(f"{exc}: fix it, then run the setup again") from None
    project = Path(base_dir).resolve() if own else locations.project
    return project, project / locations.testuary.relative_to(locations.project) / "project.ini"


def _write_project(path: Path, values: Sequence[Tuple[str, str]],
                   remove: Sequence[str] = ()) -> Path:
    """Write a project's values into its settings file ``path``: the model
    alias as ``[api] model``, every other value in ``[project]``, keeping
    the file's other lines. The whole file must then validate as a run
    validates it. The results directory beside it gets a ``.gitignore`` that
    keeps the results out of version control."""
    from testuary.settings import ConfigurationError, _check_configuration_text, _check_project_settings

    text = path.read_bytes().decode("utf-8") if path.is_file() else ""
    project = [(key, value) for key, value in values if key != "model"]
    model = [("model", value) for key, value in values if key == "model"]
    removed = [key for key in remove if key != "model"]
    if removed:
        text = _ini_text.remove_keys(text, "project", removed)
    if "model" in remove:
        text = _ini_text.remove_keys(text, "api", ["model"])
    if project:
        text = _ini_text.set_values(text, "project", project)
    if model:
        text = _ini_text.set_values(text, "api", model)
    try:
        _check_configuration_text(text, str(path))
        _check_project_settings(_ini_values(text), path)
    except ConfigurationError as exc:
        raise _refused_setting(exc) from None
    path.parent.mkdir(parents=True, exist_ok=True)
    _ini_text.write_atomic(path, text)
    ignore = path.parent / "results" / ".gitignore"
    if not ignore.exists():
        ignore.parent.mkdir(exist_ok=True)
        ignore.write_bytes(b"*\n")
    return path


def _ini_values(text: str) -> Dict[str, Dict[str, str]]:
    import configparser

    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text)
    return {name: dict(parser[name]) for name in parser.sections()}


def _alias_fields(catalogue: Dict[str, Dict[str, str]], alias: str) -> Dict[str, str]:
    fields = catalogue.get(alias) or {}
    provider = str(fields.get("provider") or "").strip()
    if not provider or provider == alias:
        raise SetupError(f"models.ini holds no model alias {alias}")
    return fields


def models(base_dir: Optional[Path] = None, *, home: Optional[Path] = None,
           probe: Optional[Probe] = None,
           models_files: Optional[Sequence[Union[str, Path]]] = None) -> Tuple[ModelEntry, ...]:
    """The aliases of the model catalogue of runs in ``base_dir`` (the
    current directory when omitted), the user directory's (``home`` when given),
    with where each key comes from and whether a run can use the alias, by
    the rule the setup offers aliases by. ``probe`` reads the environment.
    ``models_files`` names the catalogue files read in order in place of
    the user directory's, joined as ``resolve_settings(models_files=...)``
    joins them."""
    files = _Files(_dir(base_dir), home)
    listed = None if models_files is None else _listed(files.base_dir, home, models_files)
    return tuple(_model_entries(files, probe or Probe(), listed))


def usable_aliases(base_dir: Optional[Path] = None, *, home: Optional[Path] = None,
                   probe: Optional[Probe] = None) -> Tuple[str, ...]:
    """The aliases a run can use: the setup offers these."""
    return tuple(entry.alias for entry in models(base_dir, home=home, probe=probe) if entry.usable)


def _alias_name_check(catalogue: Dict[str, Dict[str, str]], provider: str) -> Callable[[str], Optional[str]]:
    """The check of an alias's name, the walk's and set_model's: its form,
    and none of a built-in provider, its own provider or a provider section
    of models.ini, which writing the alias would replace with its key."""
    from testuary.models import is_provider_section

    def refused(answer: str) -> Optional[str]:
        if answer in _BUILT_IN_PROVIDERS:
            return f"{answer} is a built-in provider: name the alias otherwise"
        if answer == provider:
            return "an alias and its provider need different names"
        if is_provider_section(catalogue, answer):
            return f"{answer} is a provider section in models.ini: name the alias otherwise"
        return None
    return _label_check(refused)


def set_model(alias: str, *, provider: str, model: str, address: str = "", key: str = "",
              key_variable: str = "", base_dir: Optional[Path] = None,
              home: Optional[Path] = None, probe: Optional[Probe] = None,
              models_files: Optional[Sequence[Union[str, Path]]] = None) -> Path:
    """Create or replace ``alias`` in the model catalogue, as the setup's
    model part writes it: the alias's provider and model, an address (empty:
    the provider's configured one, or the address its section already
    holds), a key stored in the provider's section, or the variable that
    holds the key. The file keeps its comments and other sections and is
    restricted to its owner while it holds a key. SetupError names a value
    the setup would refuse, and an alias that resolved before and would no
    longer resolve as a run resolves it.

    With ``models_files``, the catalogue files read in order (``models``),
    the change goes to the last file, which it creates when absent. A
    section only an earlier file defines is written into it whole from the
    earlier section's keys. No key is written: a key is refused, and so is
    a last file that holds one. A section whose key an earlier file holds
    needs ``key_variable``, and no address that holds a credential is
    written."""
    files = _Files(_dir(base_dir), home)
    listed = None if models_files is None else _listed(files.base_dir, home, models_files)
    catalogue = files.values("models.ini") if listed is None else listed[2]
    given = {"alias": alias, "provider": provider, "model": model, "address": address,
             "key": key, "key_variable": key_variable}
    problems = [(field, "must be one line") for field, value in given.items()
                if "\n" in value or "\r" in value]
    problem = _alias_name_check(catalogue, provider)(alias)
    if problem:
        problems.append(("alias", problem))
    if provider not in _BUILT_IN_PROVIDERS:
        problem = _service_name_check(catalogue)(provider)
        if problem:
            problems.append(("provider", problem))
    if not model.strip():
        problems.append(("model", "a model name is needed"))
    if address and _address_check(address):
        problems.append(("address", _address_check(address)))
    if key_variable and _variable_check(key_variable):
        problems.append(("key_variable", _variable_check(key_variable)))
    if listed is not None and key:
        problems.append(("key", f"{listed[0][-1]} keeps no key: name the variable that holds it"))
    if listed is not None:
        address = _unchanged_address(catalogue, provider, alias, address)
    # Without an address a built-in provider's alias follows its provider's
    # configured endpoint: no address line for the alias.
    if not address and provider not in _BUILT_IN_PROVIDERS:
        address = str(catalogue.get(provider, {}).get("api_url") or "")
        if not address:
            problems.append(("address", f"models.ini holds no address for {provider}, so give one"))
    if problems:
        raise _refused(problems)
    provider_values, alias_values = _model_values(provider, address, model.strip(), key, key_variable)
    try:
        if listed is None:
            text = _model_text(files, provider, provider_values, alias, alias_values)
            _keep_resolving(catalogue, _ini_values(text), [alias], files.base_dir, home)
            return _write_catalogue(files, probe or Probe(), "models.ini", text)
        text = _listed_model_text(listed, provider, provider_values, alias, alias_values, key_variable)
        return _write_listed(listed, text, [alias], files.base_dir, home)
    except ValueError as exc:
        raise _refused_setting(exc) from None


def remove_model(alias: str, *, base_dir: Optional[Path] = None, home: Optional[Path] = None,
                 probe: Optional[Probe] = None,
                 models_files: Optional[Sequence[Union[str, Path]]] = None) -> Path:
    """Remove ``alias``'s section from the model catalogue, keeping every
    other line. Its provider's section stays: other aliases can use it.
    SetupError refuses a removal that leaves an alias that resolved before
    unresolved. With ``models_files`` (``set_model``), only a section of the
    last file is removed, and an earlier file's section of the alias
    applies again."""
    files = _Files(_dir(base_dir), home)
    if models_files is None:
        catalogue = files.values("models.ini")
        _alias_fields(catalogue, alias)
        text = _ini_text.remove_section(files.text("models.ini"), alias)
        _keep_resolving(catalogue, _ini_values(text), (), files.base_dir, home, [alias])
        return _write_catalogue(files, probe or Probe(), "models.ini", text)
    listed = _listed(files.base_dir, home, models_files)
    paths, layers, joined = listed
    _held_keys(paths[-1], layers[-1])
    if alias not in layers[-1]:
        _alias_fields(joined, alias)
        raise SetupError(f"{alias} comes from an earlier file than {paths[-1]}, which alone changes "
                         "through this list, so it cannot be removed here")
    text = _ini_text.remove_section(paths[-1].read_bytes().decode("utf-8"), alias)
    return _write_listed(listed, text, (), files.base_dir, home, [alias])


def _unchanged_address(catalogue: Dict[str, Dict[str, str]], provider: str, alias: str, address: str) -> str:
    """The address a list's change keeps: the address the alias's or its
    provider's section holds when ``address`` is that address as the model
    list shows it, without its credentials, so that an unchanged field
    writes no override."""
    from testuary.settings import _safe

    for fields in (catalogue.get(alias, {}), catalogue.get(provider, {})):
        held = str(fields.get("api_url") or "")
        if held and address and _safe({"api_url": held})["api_url"] == address:
            return held
    return address


def _write_listed(listed: _Listed, text: str, changed: Sequence[str], base_dir: Path,
                  home: Optional[Path], removed: Sequence[str] = ()) -> Path:
    """Write ``text`` as the list's last file once the run's resolution of
    the list with it (``_resolved_with``) accepts it and still resolves
    every alias that resolved before and those ``changed``
    (``_keep_resolving``). No key is ever written to it."""
    paths = listed[0]
    _keep_resolving(listed[2], _resolved_with(listed, text, base_dir, home), changed, base_dir, home, removed)
    try:
        paths[-1].parent.mkdir(parents=True, exist_ok=True)
        _ini_text.write_atomic(paths[-1], text)
    except OSError as exc:
        raise SetupError(str(exc)) from None
    return paths[-1]


def _project_fields() -> frozenset:
    """The values a project's settings take through the setup: the
    [project] settings and the model alias, ``model``."""
    from testuary.settings import _defaults, _schema

    return frozenset(_schema(_defaults())["project"]) | {"model"}


def project(base_dir: Optional[Path] = None, *, home: Optional[Path] = None) -> ProjectEntry:
    """The settings of the project of runs in ``base_dir`` (the current
    directory when omitted), found from there upward as a run finds it, with
    the user directory ``home`` when given."""
    project_dir, path = _project_settings_file(_dir(base_dir), own=False, home=home)
    sections = _ini_values(path.read_bytes().decode("utf-8")) if path.is_file() else {}
    fields = dict(sections.get("project", {}))
    if sections.get("api", {}).get("model"):
        fields["model"] = sections["api"]["model"]
    return ProjectEntry(project_dir, path, fields)


def set_project(values: Dict[str, str], *, remove: Sequence[str] = (),
                base_dir: Optional[Path] = None, home: Optional[Path] = None,
                probe: Optional[Probe] = None) -> Path:
    """Set ``values`` and remove the keys ``remove`` in the settings file of
    the project of runs in ``base_dir``, as the setup's project part writes
    them: ``model`` as ``[api] model``, every other key in ``[project]``.
    Without a project there, ``base_dir`` becomes one. A key a run would
    refuse, a value of more than one line, a language that is not
    installed and a value the setup's project part refuses (an import
    directory, interpreter, JDK or build tool that does not exist, checked
    with ``probe``) are refused, each named, and nothing is written. The
    file keeps its comments and other lines. ``home`` is the user directory,
    whose configuration names the Testuary directory."""
    from testuary.settings import _unknown

    problems = []
    allowed = _project_fields()
    project_dir, path = _project_settings_file(_dir(base_dir), own=False, home=home)
    probe = probe or Probe()
    for key, value in [*values.items(), *((key, "") for key in remove)]:
        if key not in allowed:
            problems.append((key, str(_unknown("the project's settings", key, tuple(sorted(allowed))))))
        elif "\n" in str(value) or "\r" in str(value):
            problems.append((key, "must be one line"))
        elif str(value):
            problem = _project_value_problem(key, str(value), project_dir, probe)
            if problem:
                problems.append((key, problem))
    language = str(values.get("language") or "")
    if language and language not in languages():
        problems.append(("language", f"{language} is not installed: choose one of "
                                     f"{', '.join(languages())}"))
    if problems:
        raise _refused(problems)
    return _write_project(path, [(key, str(value)) for key, value in values.items()], remove)


def _project_value_problem(key: str, value: str, project_dir: Path, probe: Probe) -> Optional[str]:
    """Why the setup's project part refuses ``value`` for ``key`` in the
    project ``project_dir``, or None: the installed language that declares the
    setting checks it (``setup_problem``, by the setting's kind unless the
    language checks it itself). Every other value is the run's to check."""
    from langadapter import registration

    for language in languages():
        factory = registration(language).factory
        if any(setting.name == key for setting in getattr(factory, "project_settings", ()) or ()):
            return factory.setup_problem(key, value, project_dir, probe)
    return None


# Part 1: model access


def _model_part(ask: Ask, probe: Probe, say: Say, files: _Files, result: SetupResult) -> bool:
    say("Model access (models.ini)")
    detected = "ollama-local" if probe.local_ollama() else ""
    kinds = tuple(provider.kind for provider in PROVIDERS)
    say("\n".join(f"  {index}. {provider.label} ({provider.kind})"
                   for index, provider in enumerate(PROVIDERS, 1)))
    # The list above is the menu, so the question lists no choices of its
    # own, and its check takes a number or a name.
    numbers = {str(index): kind for index, kind in enumerate(kinds, 1)}
    kind = _asked(ask, Question(
        "model.provider", f"Which model service? A number from 1 to {len(kinds)}, or its name",
        default=detected,
        reason="Ollama answers on this machine" if detected else ""),
        lambda answer: None if numbers.get(answer, answer) in kinds else
        f"a number from 1 to {len(kinds)} or one of {', '.join(kinds)}", say)
    kind = numbers.get(kind, kind)
    provider = provider_kind(kind)
    provider_name = provider.provider
    address = provider.address
    if provider.custom_address:
        address = _asked(ask, Question("model.address", "The service's API address",
                                       default=provider.address), _address_check, say)
    catalogue = files.values("models.ini")
    if kind == "openai-compatible":
        provider_name = _asked(ask, Question(
            "model.provider_name",
            "A short name for this service. It names its section in models.ini and its key "
            "variable TESTUARY_<NAME>_API_KEY",
            default=_free_service_name(catalogue, _service_name(address), address)),
            _service_name_check(catalogue), say)
    key, stored_key, key_variable = "", "", ""
    pending: List[str] = []
    if provider.key != "none":
        default_variable = _ini_text.credential_variable(provider_name)
        storage = _asked(ask, Question(
            "model.key_storage",
            f"Keep the API key in models.ini, or in an environment variable ({default_variable})?",
            default="environment", choices=("file", "environment")),
            _choice(("file", "environment")), say)
        if storage == "file":
            key = _asked(ask, Question("model.key", "The API key (not shown while typed)",
                                       secret=True),
                         (lambda answer: None) if provider.key == "optional"
                         else (lambda answer: None if answer else "the service needs an API key"), say)
            stored_key = key
        else:
            key_variable = _asked(ask, Question(
                "model.key_variable", "The environment variable that holds the API key",
                default=default_variable), _variable_check, say)
            key = probe.environment(key_variable)
            if catalogue.get(provider_name, {}).get("api_key"):
                say(f"  models.ini already holds an api_key in [{provider_name}]. It takes "
                    f"precedence over {key_variable} for every alias of {provider_name}: "
                    "remove it there to use the variable.")
                pending.append(f"remove api_key from [{provider_name}] in models.ini "
                               f"to use {key_variable}")
            if not key and provider.key == "required":
                say(f"  {key_variable} is not set here: set it before a run. "
                    "The model-list check is skipped.")
                pending.append(f"set the environment variable {key_variable}")
    listed = probe.model_list(provider, address, key) if (key or provider.key != "required") else None
    if listed is None and (key or provider.key != "required"):
        say("  The service's model list could not be read (no request was charged).")
        if key_variable and not key:
            say(f"  {key_variable} is not set here: set it before a run if the service needs an API key.")
            pending.append(f"set the environment variable {key_variable}")
    elif listed is not None and key_variable and not key:
        # The service answered without a key: a variable named for it would
        # refuse every run while it is unset.
        say(f"  The service lists {len(listed)} model(s) without an API key, so none is needed.")
        key_variable = ""
    elif listed is not None:
        say(f"  Access confirmed: the service lists {len(listed)} model(s).")
    name = _asked(ask, Question(
        "model.name", "The model's name" + (f" ({', '.join(listed[:8])})" if listed else ""),
        default=listed[0] if listed and len(listed) == 1 else ""),
        lambda answer: (None if answer and (not listed or answer in listed)
                        else "name a model the service lists" if answer else "a model name is needed"),
        say)
    existing = list(catalogue)
    alias = _asked(ask, Question("model.alias", "A short name for this model in models.ini, "
                                 "which projects use (its alias)",
                                 default=_label(f"{provider_name if kind == 'openai-compatible' else kind.split('-')[0]}_{name}"))
                   , _alias_name_check(catalogue, provider_name), say)
    provider_values, alias_values = _model_values(provider_name, address, name, stored_key,
                                                  key_variable)
    shown = []
    if provider_values:
        shown.append(_section_text(provider_name, provider_values, hidden=("api_key",)))
    shown.append(_section_text(alias, alias_values))
    say("\n".join(shown))
    current = catalogue.get(provider_name)
    changed = [key for key, value in provider_values
               if current is not None and key in current and current[key] != value]
    if changed:
        users = sorted(name for name, fields in catalogue.items()
                       if name != provider_name and fields.get("provider") == provider_name)
        if not _yes(_asked(ask, Question(
                "model.provider_confirm",
                f"Replace {' and '.join(changed)} of [{provider_name}] in models.ini? "
                f"{'The aliases ' + ', '.join(users) + ' use it' if users else 'No alias uses it'}",
                default="no"), _yes_no, say)):
            say("  Nothing written to models.ini.")
            return False
    replacing = alias in existing
    if not _yes(_asked(ask, Question(
            "model.confirm", ("Replace" if replacing else "Write") + " this in models.ini?",
            default="yes" if not replacing else "no"), _yes_no, say)):
        say("  Nothing written to models.ini.")
        return False
    path = _write_model(files, probe, provider_name, provider_values, alias, alias_values)
    if _holds_key(files.text("models.ini")):
        say("  models.ini holds a key and is readable by you only.")
    result.missing.extend(pending)
    result.written.append(str(path))
    result.alias = alias
    return True


def _config_address(provider: str) -> str:
    """A built-in provider's packaged endpoint, from the bundled config.ini."""
    from testuary.settings import _defaults

    return str(_defaults().get(provider, {}).get("api_url") or "")


#: The names of this machine as an address's host.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def _host(address: str) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(address).hostname or "").lower()


def _is_local(address: str) -> bool:
    """Whether ``address`` names a service on this machine, by its host."""
    return _host(address) in _LOCAL_HOSTS


def _service_name(address: str) -> str:
    """A name for a service from its address: the host's first meaningful
    label, or local for this machine."""
    host = _host(address)
    if host in ("", *_LOCAL_HOSTS):
        return "local"
    labels = [label for label in host.split(".") if label not in ("api", "www")]
    return _label(labels[0]) if labels else "service"


def _holds_key(text: str) -> bool:
    """Whether any section of the models.ini text holds an api_key."""
    import configparser

    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error:
        return True
    return any(parser.has_option(section, "api_key") for section in parser.sections())


def _free_service_name(catalogue: Dict[str, Dict[str, str]], base: str, address: str) -> str:
    """``base``, or the first of base2, base3 and on that no other service's
    section holds and that names no built-in provider or alias: a second
    service on this machine gets its own section, and the default passes
    the name's check."""
    candidate, number = base, 1
    while (candidate in ("ollama", "openai", "vllm") or catalogue.get(candidate, {}).get("provider")
           or (candidate in catalogue and catalogue[candidate].get("api_url") != address)):
        number += 1
        candidate = f"{base}{number}"
    return candidate


def _service_name_check(catalogue: Dict[str, Dict[str, str]]) -> Callable[[str], Optional[str]]:
    def refused(answer: str) -> Optional[str]:
        if answer in ("ollama", "openai", "vllm"):
            return f"{answer} is a built-in provider: name this service otherwise"
        if catalogue.get(answer, {}).get("provider"):
            return f"{answer} is a model alias in models.ini: name this service otherwise"
        return None
    return _label_check(refused)


def _address_check(answer: str) -> Optional[str]:
    return (None if re.match(r"^https?://[^\s/]+", answer)
            else "an address such as http://localhost:8000/v1")


def _variable_check(answer: str) -> Optional[str]:
    return None if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", answer) else "a variable name"


# Part 2: this machine

def _machine_unset(files: _Files, probe: Probe) -> List[str]:
    config = files.values("config.ini")
    unset = []
    if not (config.get(_ini_text.phase_section("mutation"), {}).get("engine_workers")
            or probe.environment("TESTUARY_MUTATION_ENGINE_WORKERS")):
        unset.append("engine_workers")
    starter = config.get(_ini_text.phase_section("starter"), {})
    if not (starter.get("concurrent_requests") or starter.get("concurrent_requests_ollama")
            or probe.environment("TESTUARY_STARTER_CONCURRENT_REQUESTS")
            or probe.environment("TESTUARY_STARTER_CONCURRENT_REQUESTS_OLLAMA")):
        unset.append("starter_requests")
    return unset


def _alias_provider(files: _Files, alias: str) -> Optional[Provider]:
    """The kind of service an alias reaches. An Ollama alias is local or in
    the cloud by the address its run sends to: its own, else its
    provider's in the catalogue or the configuration, else the packaged
    one."""
    catalogue = files.values("models.ini")
    fields = catalogue.get(alias, {})
    provider = str(fields.get("provider") or "")
    address = str(fields.get("api_url") or catalogue.get(provider, {}).get("api_url")
                  or files.values("config.ini").get(provider, {}).get("api_url")
                  or _config_address(provider))
    if provider == "ollama":
        return provider_kind("ollama-local" if _is_local(address) else "ollama-cloud")
    return next((item for item in PROVIDERS if item.provider == provider), None)


def _machine_part(ask: Ask, probe: Probe, say: Say, files: _Files, result: SetupResult,
                  only_unset: bool) -> bool:
    say("This machine (config.ini)")
    unset = _machine_unset(files, probe) if only_unset else ["engine_workers", "starter_requests"]
    values: List[Tuple[str, str, str]] = []
    if "engine_workers" in unset:
        cpus = probe.cpu_count()
        default = max(1, min(15, cpus // 2))
        workers = _asked(ask, Question(
            "machine.mutation_workers", "How many mutation-testing processes may run at once?",
            default=str(default),
            reason=f"Testuary may use {cpus} of this machine's CPUs, the default is about half"),
            _positive, say)
        values.append((_ini_text.phase_section("mutation"), "engine_workers", workers))
    if "starter_requests" in unset:
        alias = result.alias or str(files.values("config.ini").get("api", {}).get("model") or "")
        if not alias:
            usable = _usable_aliases(files, probe)
            alias = usable[0] if len(usable) == 1 else ""
        provider = _alias_provider(files, alias) if alias else None
        default = provider.starters if provider else 4
        reason = provider.starters_reason if provider else "a cautious start"
        starters = _asked(ask, Question(
            "machine.starter_requests",
            "How many model requests may run at once while the starter suites are written?",
            default=str(default), reason=reason), _positive, say)
        # Ollama runs read concurrent_requests_ollama, every other service
        # concurrent_requests.
        keys = (("concurrent_requests_ollama",) if provider is not None and provider.provider == "ollama"
                else ("concurrent_requests",) if provider is not None
                else ("concurrent_requests", "concurrent_requests_ollama"))
        values.extend((_ini_text.phase_section("starter"), key, starters) for key in keys)
    if not values:
        return False
    say("\n".join(f"[{section}] {key} = {value}" for section, key, value in values))
    if not _yes(_asked(ask, Question("machine.confirm", "Write this in config.ini?", default="yes"),
                       _yes_no, say)):
        say("  Nothing written to config.ini.")
        return False
    from testuary.settings import ConfigurationError, update_file

    path = files.path("config.ini")
    try:
        for section in dict.fromkeys(section for section, _, _ in values):
            update_file(path, section, {key: value for name, key, value in values if name == section})
    except ConfigurationError as exc:
        raise _refused_setting(exc, ": fix config.ini, then run the setup again") from None
    result.written.append(str(path))
    return True


def _positive(answer: str) -> Optional[str]:
    return None if answer.isdigit() and int(answer) > 0 else "a whole number above zero"


# Part 3: the project

def languages() -> Tuple[str, ...]:
    """The languages the setup offers: the installed ones, as LangAdapter
    lists them (in name order)."""
    from langadapter import names

    return tuple(names())


def _detect_language(probe: Probe, project_dir: Path) -> str:
    """The language with the most source files in ``project_dir``, the earlier
    in ``languages()`` on a tie, or "" without any."""
    ordered = _languages_by_files(probe, project_dir)
    return ordered[0][0] if ordered and ordered[0][1] else ""


def _languages_by_files(probe: Probe, project_dir: Path) -> List[Tuple[str, int]]:
    """The installed languages with how many source files of each
    ``project_dir`` holds, the most first, in ``languages()``'s order on a tie."""
    counts = [(language, len(probe.source_files(project_dir, language))) for language in languages()]
    return sorted(counts, key=lambda item: -item[1])


def _selection(answer: str, count: int) -> Optional[List[int]]:
    """The 1-based positions ``answer`` names ("all", "3", "1-4,7"), or None."""
    if answer.strip().lower() == "all":
        return list(range(1, count + 1))
    chosen: List[int] = []
    for part in answer.replace(" ", "").split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            return None
        low, high = int(match.group(1)), int(match.group(2) or match.group(1))
        if not 1 <= low <= high <= count:
            return None
        chosen.extend(index for index in range(low, high + 1) if index not in chosen)
    return chosen or None


def _project_part(ask: Ask, probe: Probe, say: Say, files: _Files, result: SetupResult,
                  name: str = "") -> bool:
    say("The project (its settings file, .testuary/project.ini in its directory)")
    dir_text = _asked(ask, Question("project.folder", "The project directory", default="."),
                         lambda answer: (None if probe.is_dir(files.base_dir / answer)
                                         else "no such directory"), say)
    project_dir = (files.base_dir / dir_text).resolve()
    # The languages the project's files are written in come first.
    ordered = _languages_by_files(probe, project_dir)
    choices = tuple(language for language, _ in ordered)
    found = ", ".join(f"{language} ({count} {'file' if count == 1 else 'files'})"
                      for language, count in ordered if count)
    language = _asked(ask, Question("project.language", "Its language",
                                    default=_detect_language(probe, project_dir), choices=choices,
                                    reason=f"found: {found}" if found else "no source files found"),
                      _choice(choices), say)
    values: List[Tuple[str, str]] = [("name", name)] if name else []
    values.append(("language", language))
    root = project_dir
    values.extend(_language_part(ask, probe, say, language, project_dir, result))
    sources = probe.source_files(root, language)
    if not sources:
        raise SetupError(f"no {language} source files below {project_dir}")
    say("\n".join(f"  {index}. {path}" for index, path in enumerate(sources, 1)))
    picked = _asked(ask, Question(
        "targets", f"Which targets? Enter takes all {len(sources)}, or name numbers and ranges such as 1-3,5",
        default="all"),
        lambda answer: None if _selection(answer, len(sources)) else "numbers and ranges from the list", say)
    chosen = [sources[index - 1] for index in _selection(picked, len(sources))]
    values.append(("source_files", ", ".join(chosen)))
    aliases = _usable_aliases(files, probe)
    if result.alias and result.alias not in aliases:
        aliases.append(result.alias)
    known = {name for name, fields in files.values("models.ini").items() if fields.get("provider")}
    known.update(aliases)
    # The usable aliases first, then the catalogue's others.
    offered = tuple(dict.fromkeys([*aliases, *sorted(known)]))
    model = _asked(ask, Question(
        "model", "The model, an alias of models.ini" + ("" if offered else " (none set up yet)"),
        default=result.alias or (aliases[0] if len(aliases) == 1 else ""), choices=offered),
        lambda answer: None if not answer or answer in known else "name an alias of models.ini", say)
    _, path = _project_settings_file(project_dir, own=True, home=files.home)
    shown = _section_text("project", values)
    if model:
        values.append(("model", model))
        shown += "\n\n" + _section_text("api", [("model", model)])
    say(shown)
    replacing = path.is_file()
    if not _yes(_asked(ask, Question(
            "project.confirm", ("Update" if replacing else "Write") + f" this in {path}?",
            default="no" if replacing else "yes"), _yes_no, say)):
        say(f"  Nothing written to {path}.")
        return False
    # The keys this part asks about hold exactly the answers: one left out
    # now goes. Every other setting of the file stays.
    _write_project(path, values,
                   sorted(_project_part_keys(project_dir, probe) - {key for key, _ in values}))
    result.written.append(str(path))
    result.project = str(project_dir)
    result.run = ("testuary",)
    if not model:
        result.missing.append("a model alias: run testuary setup model")
    return True


def _project_part_keys(project_dir: Path, probe: Probe) -> set:
    """The keys the project part asks about: the files, the language, the
    model alias and the settings the installed languages' setups ask."""
    from langadapter import registration
    from langadapter.adapter import SetupQuestion

    keys = {"source_files", "language", "model"}
    for language in languages():
        questions = getattr(registration(language).factory, "setup_questions", None)
        if questions is not None:
            keys.update(item.setting for item in questions(project_dir, probe, consumer="Testuary")
                        if isinstance(item, SetupQuestion) and item.setting)
    return keys


def _language_part(ask: Ask, probe: Probe, say: Say, language: str, project_dir: Path,
                   result: SetupResult) -> List[Tuple[str, str]]:
    """The questions ``language``'s adapter asks for the project in
    ``project_dir`` (``setup_questions``), each answer checked as the question
    says or by its setting's kind, and its notes said, with what they name
    as missing. Returns the settings to write. An empty value writes
    nothing."""
    from langadapter import registration
    from langadapter.adapter import SetupNote

    factory = registration(language).factory
    questions = getattr(factory, "setup_questions", None)
    values: List[Tuple[str, str]] = []

    def noted(note: SetupNote) -> None:
        for line in note.lines:
            say(line)
        result.missing.extend(note.missing)

    for item in (questions(project_dir, probe, consumer="Testuary") if questions is not None else ()):
        if isinstance(item, SetupNote):
            noted(item)
            continue
        check = item.check
        if check is None:
            def check(answer: str, setting=item.setting) -> Optional[str]:
                return factory.setup_problem(setting, answer, project_dir, probe)
        choices = item.choices() if callable(item.choices) else item.choices
        answer = _asked(ask, Question(item.key, item.prompt, default=item.default,
                                      choices=tuple(choices), reason=item.reason,
                                      open=item.open), check, say)
        if item.after is not None:
            noted(item.after(answer))
        value = item.written(answer) if item.written is not None else answer
        if item.setting and value:
            values.append((item.setting, value))
    return values


# The walk

def run_setup(ask: Ask, probe: Probe, say: Say, base_dir: Path, *,
              part: Optional[str] = None, name: str = "",
              announce_run: bool = True, home: Optional[Path] = None) -> SetupResult:
    """Run the named ``part``, or all three, for runs started in ``base_dir``:
    models.ini and config.ini in the user directory (``home`` when given),
    the project's settings file in the project's directory. ``part`` is
    "model", "machine" or "project" (``PARTS``).

    ``ask(question)`` returns the answer to a ``Question`` as text, an
    empty text for its default. ``probe`` tries the model and the tools the
    answers name. ``say(text)`` shows one line to the person.

    Run alone, a part always asks. In the walk of all three, a part already
    set is passed with one line, and every other part offers a skip. The
    result lists what was written and what a run still lacks.
    ``announce_run`` names the run command at the end. ``testuary setup``
    switches it off and names the command after its check of the project."""
    if part is not None and part not in PARTS:
        raise SetupError(f"choose one of {', '.join(PARTS)}")
    files = _Files(base_dir, home)
    result = SetupResult()
    walk = part is None
    for current in (PARTS if walk else (part,)):
        if current == "model":
            usable = _usable_aliases(files, probe)
            if walk and usable:
                say(f"Model access: {', '.join(usable)}, set.")
                result.alias = usable[0] if len(usable) == 1 else ""
                continue
            if walk and not _offer(ask, say, "model", "Set up model access now?"):
                result.missing.append("model access: run testuary setup model")
                continue
            if not _model_part(ask, probe, say, files, result) and not usable:
                result.missing.append("model access: run testuary setup model")
        elif current == "machine":
            unset = _machine_unset(files, probe)
            if walk and not unset:
                say("This machine: parallel work set.")
                continue
            if walk and not _offer(ask, say, "machine", "Set what runs in parallel on this machine now?"):
                result.missing.append("parallel work on this machine (defaults apply): run testuary "
                                      "setup machine")
                continue
            if not _machine_part(ask, probe, say, files, result, only_unset=walk) and _machine_unset(files, probe):
                result.missing.append("parallel work on this machine (defaults apply): run testuary "
                                      "setup machine")
        else:
            if walk and not _offer(ask, say, "project", "Set up a project now?"):
                result.missing.append("a project: run testuary setup project")
                continue
            if not _project_part(ask, probe, say, files, result, name=name):
                result.missing.append("a project: run testuary setup project")
    if result.missing:
        say("Still missing:\n" + "\n".join(f"  {item}" for item in result.missing))
    if result.run and announce_run:
        say(f"Run it{_in_dir(result.project, base_dir)} with: " + " ".join(result.run))
    say("Request limits, presets and measurement settings keep their defaults: "
        f"see the settings reference ({SETTINGS_REFERENCE}).")
    return result


def _in_dir(project: str, base_dir: Path) -> str:
    """" in DIR" when the project is not the directory the setup ran in."""
    return "" if Path(project) == Path(base_dir).resolve() else f" in {project}"


def _offer(ask: Ask, say: Say, part: str, prompt: str) -> bool:
    return _yes(_asked(ask, Question(f"{part}.run", prompt, default="yes"), _yes_no, say))


# The terminal

def _question_lines(question: Question) -> Tuple[List[str], str]:
    """The lines shown before a question, its reason and its numbered
    choices, and its prompt, with the default in brackets unless the
    answer is secret."""
    lines = [f"  ({question.reason})"] if question.reason else []
    # The choices, numbered: a number answers with its choice.
    lines += [f"  {index}. {choice}" for index, choice in enumerate(question.choices, 1)]
    hint = f" [{question.default}]" if question.default and not question.secret else ""
    return lines, f"{question.prompt}{hint}: "


def terminal_ask(question: Question) -> str:
    """Ask on the terminal: the default in brackets, a secret without
    echo. Without terminal input the wizard stops instead of writing."""
    import getpass

    lines, prompt = _question_lines(question)
    for line in lines:
        print(line)
    try:
        if question.secret:
            return getpass.getpass(prompt)
        answer = input(prompt)
    except EOFError:
        raise SetupError("the setup needs answers from a terminal") from None
    picked = answer.strip()
    if picked.isdigit() and 1 <= int(picked) <= len(question.choices):
        return question.choices[int(picked) - 1]
    return answer


def _parser():
    import argparse

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        prog="testuary setup",
        description=("Ask for and write what a run needs: model access in models.ini and what "
                     "runs in parallel on this machine in config.ini, both in your user directory, "
                     "and a project's settings in .testuary/project.ini in its directory. Without a "
                     "part, walk all three, skipping the parts already set."),
        epilog=("testuary setup --demo [--demo-language java] [--output-dir DIR] plays the setup "
                "of the demonstration project with its answers typed visibly, then its run. "
                "--demo-speed, --demo-playback-seconds and --demo-step pace the playback as for "
                "testuary --demo."))
    parser.add_argument("part", nargs="?", choices=PARTS, help="Run only this part.")
    parser.add_argument("name", nargs="?", help="With the project part, the project's name in results "
                                                "and records (default: its directory's name).")
    return parser


def _check(project_dir: str) -> int:
    """The read-only check of the project just written, in its own process
    and its directory so it reads the files as a run there will."""
    import subprocess

    return subprocess.run([sys.executable, "-m", "testuary", "check"], cwd=project_dir).returncode


def main(argv: Sequence[str], *, ask: Ask = terminal_ask, probe: Optional[Probe] = None,
         check: Callable[[str], int] = _check) -> int:
    """``testuary setup [model | machine | project [NAME]]``."""
    if "--demo" in argv:
        from testuary.demo import setup_demo

        return setup_demo.main(argv)
    parser = _parser()
    args = parser.parse_args(list(argv))
    if args.name and args.part != "project":
        parser.error("a name goes with the project part: testuary setup project [NAME]")
    try:
        result = run_setup(ask, probe or Probe(), print, Path.cwd(), part=args.part,
                           name=args.name or "", announce_run=False)
    except SetupError as exc:
        print(f"testuary setup: {exc}", file=sys.stderr)
        return 1
    if not result.project:
        return 0
    where = _in_dir(result.project, Path.cwd())
    print(f"Checking the project{where} (quick check):", flush=True)
    status = check(result.project)
    if status == 0:
        print(f"Run it{where} with: " + " ".join(result.run) + ". To try the project's tools "
              "first: testuary check --verify")
        return 0
    print(f"Fix the problems above, then check again{where} with: testuary check")
    return 1


__all__ = [
    "PARTS", "PROVIDERS", "Ask", "ModelEntry", "Probe", "ProjectEntry", "Provider",
    "Question", "Say", "SetupError", "SetupResult", "languages", "models",
    "project", "remove_model", "run_setup", "set_model", "set_project",
    "terminal_ask", "usable_aliases",
]
