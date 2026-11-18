"""The model catalogue, the projects and their settings, through Testuary's
own ``testuary.setup`` and ``testuary.settings``: every check, message and
write is Testuary's, in the files its resolver names.

- Models: the catalogue without keys. No answer holds a stored key, which
  goes only to its model's service in a Test, and nothing here logs one.
- Projects: project directories, found from a directory upward as Testuary finds
  a project. The recent ones are kept in Testuary's user directory
  (``web/projects.json``, directories only), and forgetting one only takes it
  off the list. Runs and checks start in the project's directory, as in a
  terminal there.
- Setup: Testuary's project part (``run_setup`` with ``part="project"``),
  one question at a time with the page's answers, up to the first
  unanswered question or one Testuary refused. With every answer confirmed,
  Testuary writes the settings file and the directory joins the recent
  projects. It asks for no key.
"""
from __future__ import annotations

import contextlib
import dataclasses
import re
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Dict
import json
import os
import secrets
import shutil
import string
import subprocess
import threading
from testuary.web import _results


#: The text fields of a model change, by the argument of ``set_model``.
MODEL_FIELDS = ("alias", "provider", "model", "address", "key", "key_variable")


class Refusal(Exception):
    """A change Testuary refused: its message and each refused value's field
    and problem."""

    def __init__(self, message: str, problems: Sequence[Tuple[str, str]] = ()) -> None:
        super().__init__(message)
        self.problems = tuple(problems)


def _refusal(error) -> Refusal:
    return Refusal(str(error), getattr(error, "problems", ()))


def _alias(text: object) -> str:
    """A model alias as a catalogue names it: at most 128 characters on one
    line. Whether the catalogue holds it is the setup's and the run's
    question."""
    if not isinstance(text, str) or not text.strip() or len(text) > 128 or "\n" in text or "\r" in text:
        raise Refusal("Name a model alias of at most 128 characters on one line.")
    return text


def _text(body: Mapping[str, object], field: str) -> str:
    value = body.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > 4096:
        raise Refusal("Enter text of at most 4096 characters.", ((field, "too long or not text"),))
    return value.strip()


def _kind(entry, kinds) -> str:
    """The setup's kind of service of a catalogue entry: by provider and
    address, else by provider, else another OpenAI-compatible API."""
    same = [kind for kind in kinds if kind.provider and kind.provider == entry.provider]
    by_address = [kind for kind in same if not entry.address or kind.address == entry.address]
    chosen = by_address or same
    return chosen[0].kind if chosen else "openai-compatible"


def models(base_dir: Path, models_files: Optional[Sequence[str]] = None, **settings_files) -> dict:
    """The catalogue as the setup sees it, without keys: each alias with its
    provider, model, address, key source, usability, kind of service, its
    file and whether its provider has a section of its own
    (``provider_section``, which stays when the model goes), the kinds the
    setup offers, the provider sections no model names (``unused``) and the
    file a change goes to (``file``): models.ini, or the last of the
    ``models_files`` read in its place (``_listed_facts``), with the
    settings files ``settings_files`` names as ``resolve_settings`` takes
    them (``config_files``, ``project_settings``)."""
    from testuary import setup
    from testuary.settings import ConfigurationError, file_locations, resolve_settings

    where = str(file_locations(base_dir).models) if models_files is None else models_files[-1]
    providers = [{"kind": item.kind, "label": item.label, "provider": item.provider,
                  "address": item.address, "key": item.key} for item in setup.PROVIDERS]
    try:
        listed = None if models_files is None else setup._listed(base_dir, None, models_files, **settings_files)
        entries = (setup.models(base_dir) if listed is None
                   else setup._model_entries(setup._Files(base_dir), setup.Probe(), listed))
    except setup.SetupError as error:
        return {"error": str(error), "models": [], "providers": providers, "unused": [],
                "file": where}
    named = {entry.alias for entry in entries} | {entry.provider for entry in entries}
    try:
        sections = resolve_settings(base_dir=base_dir).export().get("models", {}) if listed is None else listed[2]
    except (ConfigurationError, ValueError, OSError):
        sections = {}
    view = {"error": None, "providers": providers,
            "models": [{**dataclasses.asdict(entry), "kind": _kind(entry, setup.PROVIDERS),
                        "provider_section": entry.provider in sections}
                       for entry in entries],
            "unused": sorted(name for name in sections if name not in named), "file": where}
    return view if listed is None else _listed_facts(view, listed)


def _listed_facts(view: dict, listed) -> dict:
    """The model list ``view`` of a list of catalogue files with what its
    page asks. For each alias: the earlier file whose section of the alias
    its section in the last file replaces (``overrides``), the file its
    provider's section comes from (``provider_from``), whether its section
    in an earlier file sets an address, a key or a key variable
    (``key_places``), and the earlier file holding a key that a copy of the
    alias's section (``key_file``) or of its provider's section
    (``provider_key_file``) into the last file leaves out. For the list: the
    last file's provider sections that take the place of an earlier file's
    (``provider_overrides``), and what an end of the import cannot copy
    (``_import_facts``)."""
    from testuary import setup

    paths, layers, catalogue = listed
    own = layers[-1]
    places = ("api_url", "api_key", "api_key_env")

    def earlier(name: str) -> Tuple[Optional[str], dict]:
        position = setup._origin(layers, name, -1)
        return (None, {}) if position is None else (str(paths[position]), layers[position][name])

    inherited = {name: earlier(name) for name in catalogue if name not in own}

    def held(name: str) -> Optional[str]:
        file, fields = inherited.get(name, (None, {}))
        return file if "api_key" in fields else None

    for entry in view["models"]:
        alias, provider = entry["alias"], entry["provider"]
        entry.update(overrides=earlier(alias)[0] if alias in own else None,
                     provider_from=str(paths[setup._origin(layers, provider)]) if provider in catalogue else None,
                     key_places=any(key in inherited.get(alias, (None, {}))[1] for key in places),
                     key_file=held(alias), provider_key_file=held(provider))
    view["provider_overrides"] = [{"section": name, "file": earlier(name)[0]} for name in own
                                  if not own[name].get("provider") and earlier(name)[0]]
    return {**view, **_import_facts(inherited)}


def _import_facts(inherited: Mapping[str, Tuple[str, dict]]) -> dict:
    """What an end of an import cannot copy of the ``inherited`` sections,
    each with its earlier file and keys: a key (``import_keys``), an
    address with a credential (``import_blocked``), and an address, a key or
    a key variable, which a last file in the project directory may not hold
    (``import_places``)."""
    from testuary.settings import _address_credentials

    def where(test) -> List[dict]:
        return [{"section": name, "file": file} for name, (file, fields) in inherited.items() if test(fields)]

    return {"import_keys": where(lambda fields: "api_key" in fields),
            "import_blocked": where(lambda fields: _address_credentials(str(fields.get("api_url") or ""))),
            "import_places": where(lambda fields: any(key in fields for key in ("api_url", "api_key", "api_key_env")))}


def set_model(base_dir: Path, body: Mapping[str, object], models_files: Optional[Sequence[str]] = None) -> None:
    """Create or replace an alias as the setup writes it, in models.ini or
    in the last of the ``models_files`` read in its place. An empty key
    keeps the stored one. A new model (``new``) refuses an alias that
    exists."""
    from testuary import setup

    values = {field: _text(body, field) for field in MODEL_FIELDS}
    new = body.get("new", False)
    if not isinstance(new, bool):
        raise Refusal("Say whether the model is new.")
    try:
        if new and values["alias"] in {entry.alias for entry in setup.models(base_dir, models_files=models_files)}:
            raise Refusal(f"A model named {values['alias']} exists already. Edit changes it.",
                          (("alias", "exists already: Edit changes it"),))
        setup.set_model(values.pop("alias"), base_dir=base_dir, models_files=models_files, **values)
    except setup.SetupError as error:
        raise _refusal(error) from None


def remove_model(base_dir: Path, alias: object, models_files: Optional[Sequence[str]] = None) -> None:
    from testuary import setup

    try:
        setup.remove_model(_alias(alias), base_dir=base_dir, models_files=models_files)
    except setup.SetupError as error:
        raise _refusal(error) from None


def test_model(base_dir: Path, alias: object, models_files: Optional[Sequence[str]] = None,
               config_files: Optional[Sequence[str]] = None) -> dict:
    """Whether the service a run with ``alias`` asks answers: the run's own
    resolution of its address and key (``models.resolve_model_identity``,
    ``resolve_provider_credential``), then the setup's request for the
    models the service lists, which generates nothing (``Probe.model_list``).
    ``models_files`` and ``config_files`` name the catalogue and settings
    files an extension's runs read in place of Testuary's own, as
    ``resolve_settings`` takes them. The answer names the address without
    credentials and holds no key."""
    from testuary import setup
    from testuary.models import resolve_model_identity, resolve_provider_credential
    from testuary.settings import ConfigurationError, resolve_settings

    name = _alias(alias)
    try:
        settings = resolve_settings(base_dir=base_dir, overrides={"api": {"model": name}},
                                    models_files=models_files, config_files=config_files,
                                    project_settings=config_files is None)
        identity = resolve_model_identity(settings)
        key = resolve_provider_credential(identity.provider, settings.models.get(identity.provider, {}),
                                          settings.models.get(identity.model_alias, {}),
                                          settings.environment)
    except (ConfigurationError, ValueError) as error:
        raise Refusal(f"Testuary cannot resolve {name} as a run would: {error}") from None
    address = identity.export()["api_url"]
    # The kind of service at that address, for its way of listing models.
    kind = next((item for item in setup.PROVIDERS if item.provider == identity.provider
                 and item.address.rstrip("/") == identity.api_url.rstrip("/")),
                setup.provider_kind("openai-compatible"))
    listed = setup.Probe().model_list(kind, identity.api_url, key, timeout=10)
    if listed is None:
        text = (f"No answer from {address}, or the service refused the key: check that it runs "
                "and that its address and key are right.")
    elif identity.model in listed:
        text = f"The service at {address} answered and lists {identity.model}."
    else:
        text = (f"The service at {address} answered, but does not list {identity.model} among "
                f"the {len(listed)} models it serves: check the model's name.")
    return {"reached": listed is not None,
            "listed": None if listed is None else identity.model in listed,
            "address": address, "model": identity.model, "text": text}


def field_labels() -> dict:
    """Testuary's plain name of each setting a project's settings file holds."""
    from testuary import settings

    labels: dict = {}
    for section in settings.reference():
        for setting in section.settings:
            if section.name in ("project", "api") or (section.name, setting.name) == ("main", "extensions"):
                labels.setdefault(setting.name, setting.label)
    return labels


def project_settings() -> List[dict]:
    """The [project] settings a project's form offers, each language's own
    included, with Testuary's plain names."""
    from testuary import settings

    project = next(section for section in settings.reference() if section.name == "project")
    return [{**dataclasses.asdict(setting), "label": setting.label} for setting in project.settings]


def language_settings(languages: Sequence[str]) -> dict:
    """Each language's own settings for a project's form, once each, in the
    order of Testuary's description. A setting several languages declare
    is offered under each of them."""
    described = project_settings()
    return {language: [setting for setting in described if language in setting["languages"]]
            for language in languages}


def set_project(base: Path, body: Mapping[str, object]) -> None:
    """Set a listed project's values through the setup (``model`` is the
    alias). An empty value and the keys in ``remove`` leave the file."""
    from testuary import setup

    project_dir = known(base, body.get("project_dir"))
    values = body.get("values") or {}
    remove = body.get("remove") or []
    if (not isinstance(values, dict) or not isinstance(remove, list)
            or len(values) + len(remove) > 200 or not all(isinstance(key, str) for key in remove)):
        raise Refusal("Send the project's values as a mapping and the keys to remove as a list.")
    cleaned, removed = {}, list(remove)
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str) or len(value) > 4096:
            raise Refusal("Every value is text of at most 4096 characters.", ((str(key), "not text"),))
        if value.strip():
            cleaned[key] = value.strip()
        else:
            removed.append(key)
    try:
        setup.set_project(cleaned, remove=removed, base_dir=project_dir)
    except setup.SetupError as error:
        raise _refusal(error) from None


def _settings_files(base: Path, project_dir: Optional[Path], named: Optional[Sequence[Path]] = None) -> dict:
    """The global configuration file and the project's settings file when a
    project is named, or the ``named`` files read in place of both, a change
    going to the last of them."""
    from testuary.settings import file_locations, project_locations

    if named is not None:
        return {"global": None, "project": None, "named": str(named[-1]),
                "list": [str(path) for path in named]}
    files = {"global": str(file_locations(base).config), "project": None}
    if project_dir is not None:
        files["project"] = str(project_locations(project_dir).project_settings)
    return files


def settings_view(base: Path, dir_text: object = None, named: Optional[Sequence[Path]] = None) -> dict:
    """Every setting by section with its value and source, for runs in the
    named project, else without any project's settings file, and the files
    a change can go to. A ``named`` list of settings files is read in place
    of both the global configuration and the project's settings file, its
    last file once it exists."""
    from testuary import settings

    project_dir = known(base, dir_text) if dir_text else None
    config = None if named is None else list(named) if named[-1].is_file() else list(named[:-1])
    try:
        resolved = settings.resolve_settings(base_dir=project_dir or base, config_files=config,
                                             project_settings=project_dir is not None and named is None)
        values = settings.setting_values(resolved)
    except Exception:
        # No details: the check names the problem, and an error text
        # can quote a configuration value.
        return {"error": "Testuary cannot read these settings. The check names the "
                         "problem.", "sections": [], "files": _settings_files(base, project_dir, named)}
    by_setting = {(item.setting.section, item.setting.name): item for item in values}
    files = _settings_files(base, project_dir, named)
    sections = []
    for section in settings.reference():
        rows = [_setting_row(setting, by_setting.get((setting.section, setting.name)), resolved, files)
                for setting in section.settings]
        sections.append({"name": section.name, "title": section.title, "purpose": section.purpose,
                         "settings": rows})
    from testuary.readiness import _unchecked_values

    # Settings that an extension's language may declare, which the tool
    # check checks once it has loaded the extensions.
    return {"error": None, "sections": sections, "files": files, "unchecked": _unchecked_values(resolved)}


def _setting_row(setting, item, resolved, files: dict) -> dict:
    """A setting's row: its description, its value and where the value comes
    from. ``[main] extensions`` adds each level's items, so its row edits
    the file's own items, and the combined list is the value in force."""
    row = {**dataclasses.asdict(setting), "label": setting.label,
           "value": item.value if item else None,
           "source": item.source if item else None,
           "sources": list(item.sources) if item else [],
           "active": item.active if item else True,
           "inactive": item.inactive if item else ""}
    if (setting.section, setting.name) == ("main", "extensions"):
        row.update(combined=row["value"], value=_own_extensions(resolved, files))
    return row


def _own_extensions(resolved, files: dict) -> str:
    """The items of ``[main] extensions`` that the file a change goes to
    names itself: a named list's last file, else the project's settings
    file, else the global configuration."""
    from testuary.settings import extension_items

    target = files.get("named") or files.get("project") or files.get("global")
    return ", ".join(item for item, source, _ in extension_items(resolved)
                     if target and os.path.normcase(str(source)) == os.path.normcase(target))


def update_settings(base: Path, body: Mapping[str, object], named: Optional[Path] = None) -> None:
    """Set and reset one section's settings in the global configuration, a
    listed project's settings file or the ``named`` settings file of
    another scope, through Testuary's validated update. A named file, which
    lies in the project's directory (named_file), takes no api_url, as
    the runs that read it refuse one there (``settings.check_project_settings_file``)."""
    from testuary import settings

    scope = body.get("scope", "global")
    section = body.get("section")
    values = body.get("values") or {}
    remove = body.get("remove") or []
    if ((named is None and scope not in ("global", "project")) or not isinstance(section, str)
            or not isinstance(values, dict) or not isinstance(remove, list)
            or len(values) + len(remove) > 200
            or not all(isinstance(key, str) and isinstance(value, str)
                       for key, value in values.items())
            or not all(isinstance(key, str) for key in remove)):
        raise Refusal("Send a scope (global or project), a section, its values as text and the "
                      "keys to reset.")
    project_dir = known(base, body.get("project_dir")) if scope == "project" else None
    target = named or Path(_settings_files(base, project_dir)[scope])
    if scope == "project":
        try:
            settings._check_project_settings({section: values}, target)
        except settings.ConfigurationError as error:
            field = "testuary_dir" if "testuary_dir" in values else "api_url"
            raise Refusal(str(error), ((field, "set it in the global settings"),)) from None
    elif named is not None:
        try:
            settings._refuse_addresses({section: values}, target)
        except settings.ConfigurationError as error:
            raise Refusal(str(error), (("api_url", "set it in the global settings"),)) from None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        settings.update_file(target, section, values, remove=remove)
    except settings.ConfigurationError as error:
        raise Refusal(str(error)) from None


def problems(refusal: Refusal) -> List[List[str]]:
    """A refusal's problems as the page shows them, each beside its field."""
    return [[field, problem] for field, problem in refusal.problems]


# Directory names for the page's directory browser (a browser cannot hand a
# local path to a page). Only directory names are listed, nothing is changed,
# and a path is checked as text first: an absolute path on a local drive
# (Windows) or below / (Linux and macOS). A network or device path is refused.

MAX_DIRS = 2000


class NotListed(Exception):
    """The path is not a local directory this browser lists."""


def _local(text: str) -> Path:
    """The local absolute path ``text`` names, checked as text."""
    if not text or "\x00" in text:
        raise NotListed("Name a directory.")
    # Two leading separators name a network or device path on Windows, and
    # some systems read // so on Linux too.
    if text[:2] in ("\\\\", "//", "\\/", "/\\"):
        raise NotListed("Network and device paths are not listed.")
    if os.name == "nt":
        path = Path(text)
        if not (len(path.drive) == 2 and path.drive[1] == ":" and path.root):
            raise NotListed("Name a directory by its full path, with its drive.")
        # A drive letter can stand for a network share: only local drives.
        if not _local_drive(path.drive + "\\"):
            raise NotListed("Network and device paths are not listed.")
        return path
    path = Path(text)
    if not path.is_absolute():
        raise NotListed("Name a directory by its full path.")
    return path


def _drive_type(root: str) -> int:
    """Windows' drive type of ``root``: 2 removable, 3 fixed, 4 network, 5
    optical, 6 RAM disk, 0 or 1 unknown."""
    import ctypes

    return ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root))


def _local_drive(root: str) -> bool:
    return _drive_type(root) in (2, 3)


def _drives() -> List[str]:
    """The local fixed and removable drives, as their root paths."""
    import ctypes

    mask = ctypes.windll.kernel32.GetLogicalDrives()
    return [f"{letter}:\\" for index, letter in enumerate(string.ascii_uppercase)
            if mask & (1 << index) and _local_drive(f"{letter}:\\")]


def dir_roots(shortcuts: Dict[str, Path]) -> dict:
    """Where browsing starts: the drives (Windows) or /, and the given
    shortcuts such as the working and the home directory."""
    places = _drives() if os.name == "nt" else ["/"]
    return {"path": None, "parent": None, "crumbs": [],
            "dirs": [{"name": name, "path": name, "project": None} for name in places],
            "shortcuts": [{"name": name, "path": str(path)} for name, path in shortcuts.items()],
            "truncated": False}


def _crumbs(path: Path) -> List[dict]:
    parts = list(path.parents)[::-1] + [path]
    return [{"name": part.name or str(part), "path": str(part)} for part in parts]


def dir_listing(text: str, projects: Optional[Dict[str, Path]] = None,
            shortcuts: Optional[Dict[str, Path]] = None) -> dict:
    """The directories directly in the directory ``text`` names, by name, at most
    MAX_DIRS, marking the directories of ``projects`` (name to directory)."""
    path = _local(text)
    marks = {_results.same_path(shortcut_dir): name for name, shortcut_dir in (projects or {}).items()}
    try:
        with os.scandir(path) as entries:
            names = sorted((entry for entry in entries if _is_dir_entry(entry)),
                           key=lambda entry: entry.name.casefold())
    except FileNotFoundError:
        raise NotListed("No such directory.") from None
    except NotADirectoryError:
        raise NotListed("That is not a directory.") from None
    except OSError:
        raise NotListed("This directory cannot be read.") from None
    dirs = [{"name": entry.name, "path": entry.path, "project": marks.get(_results.same_path(entry.path))}
               for entry in names[:MAX_DIRS]]
    parent = path.parent if path.parent != path else None
    return {"path": str(path), "parent": str(parent) if parent else None,
            "crumbs": _crumbs(path), "dirs": dirs,
            "shortcuts": [{"name": name, "path": str(shortcut_dir)}
                          for name, shortcut_dir in (shortcuts or {}).items()],
            "truncated": len(names) > MAX_DIRS}


def _is_dir_entry(entry: os.DirEntry) -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


#: The most recent projects the list keeps.
MAX_PROJECTS = 50


def _refused(message: str) -> Refusal:
    return Refusal(message, (("project_dir", message),))


#: The server's request threads list, add and forget projects: one at a time
#: reads or writes the list, so none loses another's change and none replaces
#: the file while another reads it, which Windows refuses. add and forget read
#: it while they hold the lock.
_LIST_LOCK = threading.RLock()


def _store() -> Path:
    from testuary.settings import file_locations

    return file_locations().user / "web" / "projects.json"


def recent() -> List[Path]:
    """The recent project directories, latest first. An unreadable list is empty."""
    try:
        with _LIST_LOCK:
            data = json.loads(_store().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    names = data.get("project_dirs") if isinstance(data, dict) else None
    return [Path(name) for name in names or () if isinstance(name, str) and name]


def _save(entries: List[Path]) -> None:
    store = _store()
    store.parent.mkdir(parents=True, exist_ok=True)
    temporary = store.with_name(f"{store.name}.{secrets.token_hex(4)}.tmp")
    stream = open(temporary, "x", encoding="utf-8", newline="\n")
    try:
        with stream:
            stream.write(json.dumps({"project_dirs": [str(path) for path in entries]}, indent=1))
        os.replace(temporary, store)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def local_dir(text: object) -> Path:
    """The existing local directory ``text`` names by its absolute path."""
    if not isinstance(text, str) or not text.strip() or len(text) > 4096 or "\x00" in text:
        raise _refused("Name the project's directory.")
    text = text.strip()
    if not Path(text).is_absolute():
        raise _refused("Name the project's directory by its absolute path.")
    try:
        path = _local(text)
    except NotListed:
        raise _refused("A project's directory must be on a local drive.") from None
    if not path.is_dir():
        raise _refused(f"There is no directory {text}.")
    return path.resolve()


def add(text: object) -> Path:
    """Put the directory first in the recent projects."""
    path = local_dir(text)
    with _LIST_LOCK:
        entries = [path] + [entry for entry in recent() if _results.same_path(entry) != _results.same_path(path)]
        _save(entries[:MAX_PROJECTS])
    return path


#: A GitHub project's HTTPS address, as its page shows it: the owner and the
#: project, with .git or a final slash allowed.
_GITHUB = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?")
#: How long one clone may take.
CLONE_SECONDS = 900


def _git() -> Optional[str]:
    return shutil.which("git")


def _clone_command(address: str, destination: Path) -> Tuple[List[str], Dict[str, str]]:
    """git's clone of ``address`` into ``destination`` and its environment:
    over HTTPS only (no other transport, also not for a submodule, which it
    leaves out), and without asking for a login, which would wait forever."""
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL="https")
    return ([_git() or "git", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always", "clone",
             "--no-recurse-submodules", "--", address, str(destination)], environment)


def _git_clone(address: str, destination: Path) -> subprocess.CompletedProcess:
    command, environment = _clone_command(address, destination)
    return subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL, timeout=CLONE_SECONDS)


def clone(address: object, parent: object) -> Path:
    """Clone a GitHub project's HTTPS address with git into a new directory,
    named after the project, in the existing local directory ``parent``. The
    new directory, which the project setup takes next. Another address, a
    directory that exists already and a computer without git are refused, each
    beside its field."""
    found = _GITHUB.fullmatch(address.strip()) if isinstance(address, str) else None
    if found is None or found.group(2) in (".", ".."):
        message = "Name a project's GitHub HTTPS address, such as https://github.com/owner/project."
        raise Refusal(message, (("url", message),))
    owner, name = found.groups()
    destination = local_dir(parent) / name
    if os.path.lexists(destination):
        message = f"The directory {destination} exists already: choose another directory to clone into."
        raise Refusal(message, (("parent_dir", message),))
    if _git() is None:
        message = ("Cloning needs git, which this computer lacks: install it from git-scm.com, then "
                   "add the project again.")
        raise Refusal(message, (("url", message),))
    url = f"https://github.com/{owner}/{name}.git"
    try:
        done = _git_clone(url, destination)
    except subprocess.TimeoutExpired:
        message = f"git did not finish cloning {url} within {CLONE_SECONDS // 60} minutes."
        raise Refusal(message, (("url", message),)) from None
    if done.returncode != 0:
        said = [line.strip() for line in (done.stderr or "").splitlines() if line.strip()]
        message = f"git could not clone {url}: {said[-1] if said else f'exit status {done.returncode}'}"
        raise Refusal(message, (("url", message),))
    return destination.resolve()


def forget(text: object) -> None:
    """Take the directory off the recent projects. Its files stay."""
    if not isinstance(text, str):
        raise _refused("Name the project's directory.")
    key = _results.same_path(Path(text.strip()).resolve()) if text.strip() else ""
    with _LIST_LOCK:
        _save([entry for entry in recent() if _results.same_path(entry) != key])


def working(base: Path) -> Optional[Path]:
    """The working directory's project, found upward as Testuary finds it, or
    None."""
    from testuary.settings import project_locations

    try:
        locations = project_locations(base)
    except (OSError, ValueError):
        return None
    return locations.project if locations.testuary.is_dir() else None


def below(base: Path) -> List[Path]:
    """When the working directory belongs to no project (a directory that
    holds several projects), the projects directly below it, by name: its
    subdirectories that have a Testuary directory of their own, as Testuary
    finds a project from them. None below a project's directory."""
    from testuary.settings import project_locations

    if working(base) is not None:
        return []
    try:
        # The Testuary directory's name, the same for every project.
        name = project_locations(base).testuary.name
        candidates = sorted((path for path in base.iterdir() if (path / name).is_dir()),
                            key=lambda path: path.name.lower())
    except (OSError, ValueError):
        return []
    found = []
    for path in candidates:
        project = working(path)
        if project is not None and _results.same_path(project.resolve()) == _results.same_path(path.resolve()):
            found.append(path.resolve())
    return found[:MAX_PROJECTS]


def listed(base: Path) -> List[Path]:
    """The listed projects' directories: the working directory's project or the
    projects directly below it, then the recent ones."""
    return [project_dir for project_dir in [working(base), *below(base), *recent()] if project_dir is not None]


def known(base: Path, text: object) -> Path:
    """The directory of a listed project (the working directory's, one below it or
    a recent one). Runs, checks and changes take no other."""
    if not isinstance(text, str) or not text.strip():
        raise _refused("Choose a project.")
    key = _results.same_path(Path(text.strip()).resolve())
    for candidate in listed(base):
        if _results.same_path(candidate.resolve()) == key:
            return candidate.resolve()
    raise _refused("Add this directory as a project first.")


def listing(base: Path) -> dict:
    """The working directory's project or the projects directly below it, then
    the recent ones, each with its directory, name, settings file, settings and
    newest results, and the languages a project can take."""
    from testuary import setup
    from testuary.settings import _read_ini_file

    entries, seen = [], set()
    labels = field_labels()
    first = working(base)
    for project_dir, is_working, is_below in ([(first, True, False)]
                                         + [(path, False, True) for path in below(base)]
                                         + [(path, False, False) for path in recent()]):
        if project_dir is None or _results.same_path(project_dir) in seen:
            continue
        seen.add(_results.same_path(project_dir))
        entry = {"project_dir": str(project_dir), "name": project_dir.name, "working": is_working,
                 "below": is_below, "settings_file": None, "fields": {}, "labels": {},
                 "error": None, "latest": None, "latest_error": None}
        if not project_dir.is_dir():
            entry["error"] = "This directory no longer exists."
            entries.append(entry)
            continue
        try:
            entry["latest"] = _results.latest(project_dir)
        except Exception as error:
            # Shown, never taken for "no results": a broken install or an
            # unreadable summary.
            entry["latest_error"] = f"The project's latest results cannot be read: {error}"
        try:
            project = setup.project(project_dir)
            entry["settings_file"] = str(project.settings_file)
            entry["fields"] = dict(project.fields)
            # The extensions the project's settings file names itself.
            own = _read_ini_file(project.settings_file).get("main", {}).get("extensions")
            if own:
                entry["fields"]["extensions"] = own
            entry["labels"] = {name: labels.get(name, name) for name in entry["fields"]}
        except (setup.SetupError, ValueError):
            entry["error"] = ("Testuary cannot read this project's settings. The check "
                              "names the problem.")
        entries.append(entry)
    return {"projects": entries, "languages": list(setup.languages())}


def source_files(base: Path, text: object) -> dict:
    """A listed project's source files by language as Testuary's setup finds
    them: its language's, or without one every installed language's that
    its directory holds, and the files its settings name."""
    from testuary import setup

    project_dir = known(base, text)
    fields = setup.project(project_dir).fields
    probe = setup.Probe()
    language = str(fields.get("language") or "").strip()
    names = [language] if language else setup.languages()
    found = {name: probe.source_files(project_dir, name) for name in names}
    named = [item.strip() for item in str(fields.get("source_files") or "").split(",") if item.strip()]
    return {"language": language, "files": {name: files for name, files in found.items() if files},
            "project_files": named}


def saved_run(base: Path, text: object, model: object = "", output: object = "") -> dict:
    """What a run of a listed project finds in its run directory, as
    Testuary resolves it for the page's model and directory: whether it holds
    saved work (refused for a new run) and an end-of-run summary, the files
    continuing would finish, and a free directory beside it."""
    from testuary import config
    from testuary import project as project_module
    from testuary.presentation.terminal import COMPLETION_FILE, read_completion
    from testuary.settings import ConfigurationError, resolve_settings

    project_dir = known(base, text)
    overrides: dict = {}
    if isinstance(model, str) and model.strip():
        overrides["api"] = {"model": model.strip()}
    if isinstance(output, str) and output.strip():
        overrides["project"] = {"output_dir": output.strip()}
    try:
        settings = resolve_settings(base_dir=project_dir, overrides=overrides or None)
        with config.use_run_settings(settings):
            target = Path(project_module.output_dir())
    except (ConfigurationError, ValueError, OSError):
        # A setting the run itself refuses: the run says why.
        return {"occupied": False}

    def shown(path: Path) -> str:
        try:
            return path.resolve().relative_to(project_dir.resolve()).as_posix()
        except (OSError, ValueError):
            return str(path)

    try:
        occupied = target.is_dir() and next(target.iterdir(), None) is not None
    except OSError:
        occupied = False
    if not occupied:
        return {"occupied": False, "run_dir": shown(target)}
    unfinished, read = [], False
    summary = target / "results" / COMPLETION_FILE
    if summary.is_file():
        try:
            completion, _provenance = read_completion(summary)
            unfinished, read = [item.path for item in completion.unfinished], True
        except (OSError, ValueError, TypeError):
            unfinished = []
    number = 2
    while (target.parent / f"{target.name}-{number}").exists():
        number += 1
    return {"occupied": True, "run_dir": shown(target), "full": str(target),
            "unfinished": unfinished, "summary": read,
            "fresh": shown(target.parent / f"{target.name}-{number}")}


def marks(base: Path) -> dict:
    """The listed projects by name and directory, for the directory browser."""
    found = {}
    for project_dir in listed(base):
        found.setdefault(project_dir.name, project_dir)
    return found


#: The most answers a step takes, and the longest answer.
MAX_ANSWERS = 100
MAX_ANSWER = 4096
#: The question naming files by their numbers in Testuary's list, and the
#: one naming the directory, here the project's directory itself.
TARGETS = "targets"
DIR = "project.folder"
#: The setup's last question: whether to write the settings it shows.
CONFIRM = "project.confirm"


class _Pending(Exception):
    def __init__(self, question, problem: Optional[str]) -> None:
        super().__init__(question.key)
        self.question = question
        self.problem = problem


def _answers(value: object) -> Dict[str, str]:
    if not isinstance(value, dict) or len(value) > MAX_ANSWERS or not all(
            isinstance(key, str) and isinstance(text, str) and len(text) <= MAX_ANSWER
            for key, text in value.items()):
        raise Refusal("Send the answers as a mapping of question keys to text.")
    return dict(value)


def _pending(said: List[str]) -> List[dict]:
    """The settings the setup is about to write, from the file text it showed:
    each name with its plain label, in the file's order. Empty for other text."""
    import configparser

    lines = "\n".join(said).split("\n")
    start = next((index for index, line in enumerate(lines)
                  if re.fullmatch(r"\[[a-z_]+\]", line.strip())), None)
    if start is None:
        return []
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    try:
        parser.read_string("\n".join(lines[start:]))
    except configparser.Error:
        return []
    labels = field_labels()
    return [{"name": name, "label": labels.get(name, name), "value": value}
            for section in parser.sections() for name, value in parser.items(section)]


def step(body: Mapping[str, object]) -> dict:
    """The setup of ``body["project_dir"]`` with ``body["answers"]``: the next
    question, the setup's refusal, or what it wrote."""
    from testuary import setup

    project_dir = local_dir(body.get("project_dir"))
    answers = _answers(body.get("answers") or {})
    said: List[str] = []
    chosen: Dict[str, str] = {}

    def ask(question) -> str:
        if question.key in chosen:
            # Testuary asks again only when it refused the answer, and said why.
            raise _Pending(question, said[-1].strip() if said else "")
        answer = "." if question.key == DIR else answers.get(question.key)
        if answer is None:
            raise _Pending(question, None)
        chosen[question.key] = answer.strip() or question.default
        return answer

    probe = setup.Probe()
    try:
        result = setup.run_setup(ask, probe, said.append, project_dir, part="project",
                                 announce_run=False)
    except _Pending as pending:
        question = pending.question
        view = {"project_dir": str(project_dir), "said": said, "problem": pending.problem,
                "question": {"key": question.key, "prompt": question.prompt,
                             "default": question.default, "choices": list(question.choices),
                             "reason": question.reason,
                             "open": bool(getattr(question, "open", False))}}
        if question.key == CONFIRM:
            view["pending"] = _pending(said)
        if question.key == TARGETS:
            # The files the question numbers, as Testuary listed them.
            view["files"] = probe.source_files(project_dir, chosen.get("project.language", ""))
        return view
    except setup.SetupError as error:
        return {"project_dir": str(project_dir), "said": said, "refused": str(error),
                "problems": [list(problem) for problem in error.problems]}
    if result.written:
        add(str(project_dir))
    return {"project_dir": str(project_dir), "said": said, "written": list(result.written),
            "missing": list(result.missing)}
