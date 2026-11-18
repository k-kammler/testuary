"""Web interface extensions: modules named in ``[main] extensions`` that
add to the web interface through their top-level name ``testuary_web``,
loaded once at the server's start (docs/extending/web.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

WEB_EXTENSION_API = 1
"""The extension API version this Testuary supports. Version 1 covers the
event file's schema version 3 (``testuary.presentation.EVENT_FILE_SCHEMA``)."""

TOOL_CHECK_LABEL = "Tool check: also start each tool once and try the project in a copy"
"""The label of the tool check option on Testuary's Check page, which the
check report quotes while that page is Testuary's own. /api/state carries
it as ``tool_check_label``, for a replacing Check view that offers
Testuary's tool check."""


@dataclass(frozen=True)
class WebTab:
    """A tab an extension adds: its ``id`` (the address #/<id>), its
    ``title``, its view ``script`` (a file next to the extension's module), its
    ``position``: "first", "last", "before:<tab id>" or "after:<tab id>", and
    the id of the Testuary tab it ``replaces``, the Overview excepted. A
    position may name an added or a moved tab, and a position naming a tab
    no one has places the tab last. A replaced tab leaves the header, and
    its address and Testuary's links to it open the replacing tab."""

    id: str
    title: str
    script: str
    position: str = "last"
    replaces: Optional[str] = None


@dataclass(frozen=True)
class DirType:
    """A kind of directory an extension's result view opens: its ``id`` and
    ``title``, ``detect(named_dir)`` whether a directory is one, ``summary(named_dir)``
    the JSON data its view shows, read from the files the extension wrote,
    and the view ``script``."""

    id: str
    title: str
    detect: Callable[[Path], bool]
    summary: Callable[[Path], Mapping[str, Any]]
    script: str


@dataclass(frozen=True)
class SettingsGroup:
    """Settings an extension adds to the Settings page: its ``id`` and
    ``title``, ``reference(project_dir)`` its sections as
    ``testuary.settings.SectionDescription``, ``values(project_dir)`` each
    setting's value by (section, name), and ``update(project_dir, section,
    values, remove)``, its validated update, which raises ``ValueError``
    with the reason to refuse a change. ``project_dir`` is the listed project the
    page shows, or None for the global settings.

    ``testuary_settings_files(project_dir)``, optional, names the Testuary
    settings files the extension's runs in the listed project ``project_dir``
    read, in the order ``resolve_settings(config_files=...)`` reads them
    and with the extension's own file last, also while it does not exist,
    or None when they name none. The runs read them in place of both the
    global configuration and the project's own settings file. A relative
    path starts from the project directory. The Settings page shows the
    settings as those runs read them and changes the last file, when it
    lies in the project directory.

    ``testuary_models_files(project_dir)``, optional, names the model
    catalogue files those runs read, in the order
    ``resolve_settings(models_files=...)`` reads them and with the
    extension's own file last, also while it does not exist, or None when
    they name none. The Models page shows the models as those runs read
    them and changes the last file, when it lies in the project directory
    or in ``extension_user_dir``, the extension's own user directory.
    ``end_models_import(project_dir)``, optional, makes those runs read the
    last file alone. The Models page calls it after it copied every model
    of the earlier files but one into the last file, to remove a model only
    an earlier file defines, and the function changes the extension's
    settings as its last step.

    After the Settings page or the Models page writes a file of such a list,
    it calls the list's function again, and a function that raises refuses
    the saved file: the file returns to what it held before, and the
    error's message is the reason the page shows."""

    id: str
    title: str
    reference: Callable[[Optional[Path]], Sequence[Any]]
    values: Callable[[Optional[Path]], Mapping[Tuple[str, str], str]]
    update: Callable[[Optional[Path], str, Mapping[str, str], Sequence[str]], None]
    testuary_settings_files: Optional[Callable[[Path], Optional[Sequence[Path]]]] = None
    testuary_models_files: Optional[Callable[[Path], Optional[Sequence[Path]]]] = None
    extension_user_dir: Optional[Path] = None
    end_models_import: Optional[Callable[[Path], None]] = None


@dataclass(frozen=True)
class NestedRun:
    """A Testuary run that a job of an extension's kind started: the id of
    the ``phase`` of the job's plan that started it, the run's event file or
    run directory (``path``, as the page's ``nestedRun`` names a run) and the
    ``title`` its folded part on the job's page shows after "Testuary
    run", or None for none."""

    phase: str
    path: Union[str, Path]
    title: Optional[str] = None


@dataclass(frozen=True)
class JobKind:
    """A job an extension starts: its ``id`` and ``title``, the ``module``
    the job runs as ``python -m`` does, ``command(form)`` the module's
    arguments for the form the page sends (``ValueError`` with the reason
    refuses it), ``resume(arguments)``, the arguments Continue runs, or
    None for a kind without Continue, and ``nested_runs(arguments)``, the
    Testuary runs a job started, each a ``NestedRun`` its page folds into
    its phase, or None for a kind whose jobs start none. Both receive the
    job's arguments without the module. The job adds ``--events PATH``,
    where the module writes Testuary's event file through
    ``testuary.presentation.EventFile``."""

    id: str
    title: str
    module: str
    command: Callable[[Mapping[str, Any]], Sequence[str]]
    resume: Optional[Callable[[Sequence[str]], Sequence[str]]] = None
    nested_runs: Optional[Callable[[Sequence[str]], Iterable[NestedRun]]] = None


@dataclass(frozen=True)
class WebRoute:
    """A read-only route of an extension: it answers GET at
    ``/api/ext/<extension id>/<path>`` with the JSON ``read(named_dir, query)``
    returns. ``named_dir`` is the directory the query's ``dir`` names, resolved
    as the Results view resolves one, or None without ``dir``. ``query``
    holds the query's first values by name. A ``ValueError`` refuses the
    request with its message, and any other error answers with the error."""

    path: str
    read: Callable[[Optional[Path], Mapping[str, str]], Mapping[str, Any]]


@dataclass(frozen=True)
class WebExtension:
    """What an extension module adds to the web interface: the extension
    API version it was written for (``api``), its ``id``, and optionally the
    ``product`` name, ``tabs``, Testuary's tabs' titles by id
    (``tab_titles``), positions for Testuary's tabs other than the Overview
    by id, in the words of a tab's position (``tab_positions``), the ids of
    Testuary's tabs other than the Overview it hides (``hidden_tabs``), the
    address the header's Guide link opens (``guide``): an http or https
    address or the name of an HTML file next to its module, None to hide the
    link, or the empty string for Testuary's guide, ``dir_types``,
    ``settings_groups``, ``job_kinds`` and read-only ``routes``."""

    api: int
    id: str
    product: Optional[str] = None
    tabs: Sequence[WebTab] = ()
    tab_titles: Mapping[str, str] = field(default_factory=dict)
    tab_positions: Mapping[str, str] = field(default_factory=dict)
    hidden_tabs: Sequence[str] = ()
    guide: Optional[str] = ""
    dir_types: Sequence[DirType] = ()
    settings_groups: Sequence[SettingsGroup] = ()
    job_kinds: Sequence[JobKind] = ()
    routes: Sequence[WebRoute] = ()


