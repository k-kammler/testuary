"""The registry of the web interface's extensions and its loader, which
loads the modules ``[main] extensions`` names once at the server's start
and refuses an extension that breaks a rule of docs/extending/web.md.
Private: the server and the job pages read it, an extension reads
testuary.web."""
from __future__ import annotations

import os
import re
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from testuary import _extensions
from testuary.settings import ConfigurationError, extension_items, resolve_settings
from testuary.web.extensions import (WEB_EXTENSION_API, DirType, JobKind, SettingsGroup,
                                     WebExtension, WebRoute)


#: Testuary's tabs' titles by id (the address #/<id>), as the page's header lists them.
TESTUARY_TABS = dict(re.findall(r'data-view="([^"]*)">([^<]*)<', re.search(
    r"<nav[^>]*>(.*?)</nav>", resources.files("testuary.web").joinpath("static", "index.html")
    .read_text(encoding="utf-8"), re.S).group(1)))


#: Addresses an added tab cannot take: Testuary's tabs and its other views.
RESERVED = TESTUARY_TABS.keys() | {"demo", "job", "dir", "ext"}


ID = re.compile(r"[a-z][a-z0-9-]{0,39}")


SCRIPT = re.compile(r"[A-Za-z0-9_-]{1,64}\.m?js")


GUIDE_FILE = re.compile(r"[A-Za-z0-9_-]{1,64}\.html")


GUIDE_ADDRESS = re.compile(r"https?://\S+")


#: A read-only route's path below its extension's prefix.
ROUTE_PATH = re.compile(r"[a-z0-9][a-z0-9-]{0,39}(?:/[a-z0-9][a-z0-9-]{0,39}){0,3}")


#: Ids an extension cannot take: Testuary's own names below /api/ext/.
RESERVED_EXTENSIONS = frozenset({"dir", "settings", "jobs"})


class Registry:
    """The loaded extensions with their modules' directories, what was
    refused with the reason, and the directory a file extension's jobs
    import from, by extension id."""

    def __init__(self, loaded: Sequence[Tuple[WebExtension, Path]] = (),
                 refused: Sequence[Tuple[str, str]] = (),
                 import_dirs: Optional[Dict[str, Path]] = None) -> None:
        self.loaded = list(loaded)
        self.refused = list(refused)
        self.import_dirs = dict(import_dirs or {})

    def _items(self, name: str) -> Dict[str, Tuple[WebExtension, Any]]:
        return {f"{extension.id}/{item.id}": (extension, item)
                for extension, _ in self.loaded for item in getattr(extension, name)}

    def dir_type(self, key: str) -> Optional[DirType]:
        return self._items("dir_types").get(key, (None, None))[1]

    def settings_group(self, key: str) -> Optional[SettingsGroup]:
        return self._items("settings_groups").get(key, (None, None))[1]

    def job_kind(self, key: str) -> Optional[JobKind]:
        return self._items("job_kinds").get(key, (None, None))[1]

    def import_dir(self, key: str) -> Optional[Path]:
        """The directory a job of kind ``key`` imports its module from: a
        file extension's directory for its kinds (EXTENSION/KIND), else
        None."""
        extension_id, _, kind = key.partition("/")
        return self.import_dirs.get(extension_id) if kind else None

    def dir_types(self) -> Dict[str, DirType]:
        return {key: item for key, (_, item) in self._items("dir_types").items()}

    def settings_groups(self) -> Dict[str, SettingsGroup]:
        return {key: item for key, (_, item) in self._items("settings_groups").items()}

    def tab_title(self, tab: str) -> Optional[str]:
        """A Testuary tab's title as the header shows it: its replacing
        tab's, else its new title, else Testuary's own, or None while an
        extension hides the tab."""
        public = self.public()
        if tab in public["hidden_tabs"]:
            return None
        replacing = [item["title"] for item in public["tabs"] if item["replaces"] == tab]
        return replacing[0] if replacing else public["tab_titles"].get(tab, TESTUARY_TABS[tab])

    def own_view(self, tab: str) -> bool:
        """Whether a Testuary tab opens Testuary's own view: no extension
        replaces or hides it."""
        public = self.public()
        return tab not in public["hidden_tabs"] and all(item["replaces"] != tab for item in public["tabs"])

    def guide(self, testuary_guide: str) -> Optional[str]:
        """The address the header's Guide link opens: the first extension's
        guide, a file of its package at its address below /ext/, else
        ``testuary_guide``. None hides the link."""
        given = next((extension for extension, _ in self.loaded if extension.guide != ""), None)
        if given is None:
            return testuary_guide
        if given.guide is None or GUIDE_ADDRESS.fullmatch(given.guide):
            return given.guide
        return f"/ext/{given.id}/{given.guide}"

    def route(self, extension_id: str, path: str) -> Optional[WebRoute]:
        return next((route for extension, _ in self.loaded if extension.id == extension_id
                     for route in extension.routes if route.path == path), None)

    def package_file(self, extension_id: str, name: str) -> Optional[bytes]:
        """A view script or the guide file an extension declared, from its
        module file's directory, or None for one the directory does not hold
        or whose path leads out of it."""
        for extension, module_dir in self.loaded:
            declared = {item.script for item in [*extension.tabs, *extension.dir_types]}
            if extension.guide and GUIDE_FILE.fullmatch(extension.guide):
                declared.add(extension.guide)
            if extension.id == extension_id and name in declared:
                found = (module_dir / name).resolve()
                if not found.is_relative_to(module_dir.resolve()):
                    return None
                return found.read_bytes() if found.is_file() else None
        return None

    def public(self) -> dict:
        """What the page builds its header and links from."""
        product = next((extension.product for extension, _ in self.loaded if extension.product), None)
        return {
            "product": product,
            "tabs": [{"id": tab.id, "title": tab.title, "position": tab.position,
                      "replaces": tab.replaces, "script": f"/ext/{extension.id}/{tab.script}"}
                     for extension, _ in self.loaded for tab in extension.tabs],
            "tab_titles": {tab: title for extension, _ in self.loaded
                           for tab, title in extension.tab_titles.items()},
            "tab_positions": {tab: position for extension, _ in self.loaded
                              for tab, position in extension.tab_positions.items()},
            "hidden_tabs": [tab for extension, _ in self.loaded for tab in extension.hidden_tabs],
            "dir_types": [{"id": key, "title": item.title,
                              "script": f"/ext/{key.split('/')[0]}/{item.script}"}
                             for key, (_, item) in self._items("dir_types").items()],
            "job_kinds": [{"id": key, "title": item.title, "resume": item.resume is not None}
                          for key, (_, item) in self._items("job_kinds").items()],
            "refused": [{"extension": name, "reason": reason} for name, reason in self.refused],
        }


def started(base_dir: Path) -> Registry:
    """The web extensions the settings of ``base_dir`` name in ``[main]
    extensions``, loaded once at testuary-web's start. Settings that cannot
    be read are listed as the problem."""
    try:
        items = extension_items(resolve_settings(base_dir=base_dir))
    except Exception:  # The interface starts whatever its settings hold.
        # No details, as on the Settings page: an error text can quote a
        # configuration value, and the check names the problem.
        return Registry(refused=[("[main] extensions", "Testuary cannot read the settings of this "
                                  "directory, so no extension loaded. The check names the problem")])
    return load(items)


def load(items: Iterable[Tuple[str, str, str]]) -> Registry:
    """The web extensions of ``items``, the rows ``extension_items`` gives:
    each item loaded through ``_extensions.load_item`` and its module's
    ``testuary_web`` read, each resolved file once. An item that cannot
    load is refused with the reason. A module without ``testuary_web`` adds
    nothing to the web interface."""
    found: List[Tuple[str, Any, Path, Optional[Path]]] = []
    refused: List[Tuple[str, str]] = []
    seen = set()
    for item, source, directory in items:
        try:
            extension, module = _extensions.load_item(item, source, directory)
        except ConfigurationError as error:  # One broken item must not stop the interface.
            # The item leads the Overview's line, so the refusal's words that
            # name it again go.
            refused.append((item, str(error).removeprefix(f"[main] extensions: {item}: ")))
            continue
        key = os.path.normcase(extension.file)
        if key in seen or not hasattr(module, "testuary_web"):
            continue
        seen.add(key)
        module_dir = Path(extension.file).parent
        found.append((item, module.testuary_web, module_dir, module_dir if extension.kind == "file" else None))
    return checked(found, refused)


def checked(found: Iterable[Tuple[str, Any, Path, Optional[Path]]],
            refused: Sequence[Tuple[str, str]] = ()) -> Registry:
    """The registry of ``found``, each ``(item, testuary_web, module_dir,
    import_dir)``: one that breaks a rule of docs/extending/web.md is
    refused with the reason, then the conflicts between the rest and their
    single fields."""
    candidates: List[Tuple[WebExtension, Path]] = []
    refused, import_dirs = list(refused), {}
    for item, extension, module_dir, import_dir in found:
        problem = extension_problem(extension)
        if problem:
            refused.append((getattr(extension, "id", item), problem))
            continue
        candidates.append((extension, module_dir))
        if import_dir is not None:
            import_dirs[extension.id] = import_dir
    loaded, conflicts = without_conflicts(candidates)
    return Registry(*single_fields(loaded, refused + conflicts), import_dirs)


def extension_problem(extension: Any) -> Optional[str]:
    """Why an extension cannot load, or None."""
    if not isinstance(extension, WebExtension):
        return "its testuary_web is not a testuary.web.WebExtension"
    if extension.api != WEB_EXTENSION_API:
        return (f"it is written for extension API version {extension.api}, and this Testuary "
                f"supports version {WEB_EXTENSION_API}")
    return name_problem(extension) or header_problem(extension)


def name_problem(extension: WebExtension) -> Optional[str]:
    """Why an extension's ids, route paths or script names refuse it, or None."""
    if not ID.fullmatch(extension.id):
        return "its id is not lowercase letters, digits and dashes, starting with a letter"
    if extension.id in RESERVED_EXTENSIONS:
        return f"its id {extension.id!r} is a name of Testuary's own extension API"
    paths = [route.path for route in extension.routes]
    if len(paths) != len(set(paths)) or not all(ROUTE_PATH.fullmatch(path) for path in paths):
        return "its routes' paths are not distinct lowercase words, digits and dashes joined by /"
    for item in [*extension.tabs, *extension.dir_types, *extension.settings_groups,
                 *extension.job_kinds]:
        if not ID.fullmatch(item.id):
            return f"the id {item.id!r} is not lowercase letters, digits and dashes"
    repeated = repeated_id(extension)
    if repeated:
        return repeated
    for item in [*extension.tabs, *extension.dir_types]:
        if not SCRIPT.fullmatch(item.script):
            return f"the script {item.script!r} is not a .js or .mjs file name of its package"
    return None


def header_problem(extension: WebExtension) -> Optional[str]:
    """Why an extension's changes to the header refuse it, or None: its
    renamed, replaced, moved and hidden tabs and its Guide link."""
    unknown = sorted(set(extension.tab_titles) - TESTUARY_TABS.keys())
    if unknown:
        return f"it renames tabs Testuary does not have: {', '.join(unknown)}"
    replaced = [tab.replaces for tab in extension.tabs if tab.replaces is not None]
    if len(replaced) > len(set(replaced)):
        return "two of its tabs replace the same tab"
    moved, hidden = list(extension.tab_positions), list(extension.hidden_tabs)
    for verb, tabs in (("replaces", replaced), ("moves", moved), ("hides", hidden)):
        if "" in tabs:
            return f"it {verb} the Overview, which stays Testuary's"
        unknown = sorted(set(tabs) - TESTUARY_TABS.keys())
        if unknown:
            return f"it {verb} tabs Testuary does not have: {', '.join(unknown)}"
    changed = [*set(replaced), *moved, *set(hidden)]
    twice = sorted({tab for tab in changed if changed.count(tab) > 1})
    if twice:
        return f"it replaces, moves or hides these tabs more than once: {', '.join(twice)}"
    guide = extension.guide
    if guide and not (GUIDE_FILE.fullmatch(guide) or GUIDE_ADDRESS.fullmatch(guide)):
        return "its guide is neither an http or https address nor the name of an HTML file"
    return None


def repeated_id(extension: WebExtension) -> Optional[str]:
    """Why two of one extension's tabs, directory types, settings groups or job
    kinds have one id, or None."""
    for kind, items in (("tabs", extension.tabs), ("directory types", extension.dir_types),
                        ("settings groups", extension.settings_groups),
                        ("job kinds", extension.job_kinds)):
        ids = [item.id for item in items]
        repeated = sorted({key for key in ids if ids.count(key) > 1})
        if repeated:
            return f"two of its {kind} have the id {repeated[0]!r}"
    return None


def without_conflicts(candidates):
    """The extensions without a shared id, and the refused ones: two
    extensions with one id, adding a tab or directory type of one id, or
    replacing, moving or hiding the same tab, are both refused. A tab taking
    a Testuary tab's address is refused alone."""
    shared = {key: ids for key, ids in owners(candidates).items() if len(ids) > 1}
    clashing = {extension_id for ids in shared.values() for extension_id in ids}
    refused = [(extension_id, f"another extension has the {kind} {name!r}")
               for (kind, name), ids in shared.items() for extension_id in dict.fromkeys(ids)]
    loaded = []
    for extension, module_dir in candidates:
        if extension.id not in clashing:
            taken = [tab for tab in extension.tabs if tab.id in RESERVED]
            refused += [(extension.id, f"the tab {tab.id!r} takes the address of Testuary's own view")
                        for tab in taken]
            kept = tuple(tab for tab in extension.tabs if tab not in taken)
            loaded.append((replace(extension, tabs=kept), module_dir))
    return loaded, refused


def owners(candidates) -> Dict[Tuple[str, str], List[str]]:
    """The extensions that have each id, by kind and id. A tab's and a
    directory type's id name one view, which each extension counts once. Job
    kinds are kept apart by their extension's id."""
    owners: Dict[Tuple[str, str], List[str]] = {}
    for extension, _ in candidates:
        owners.setdefault(("extension", extension.id), []).append(extension.id)
        for view in {item.id for item in [*extension.tabs, *extension.dir_types]}:
            owners.setdefault(("tab or directory type", view), []).append(extension.id)
        placed = {tab.replaces for tab in extension.tabs if tab.replaces is not None}
        for tab in placed | set(extension.tab_positions) | set(extension.hidden_tabs):
            owners.setdefault(("place of the tab", tab), []).append(extension.id)
    return owners


def single_fields(loaded, refused):
    """The first extension's product name and guide, and each tab's first
    rename. A later one is refused for that field."""
    first, titles, kept = {}, set(), []
    for extension, module_dir in loaded:
        changes = {}
        for name, words, unset in (("product", "the product name", None), ("guide", "the guide", "")):
            if getattr(extension, name) in (unset, ""):
                continue
            if name in first:
                refused.append((extension.id, f"{words} is already set by {first[name]}"))
                changes[name] = unset
            else:
                first[name] = extension.id
        renamed = {tab: title for tab, title in extension.tab_titles.items() if tab not in titles}
        if len(renamed) < len(extension.tab_titles):
            refused.append((extension.id, "another extension already renames "
                            + ", ".join(sorted(set(extension.tab_titles) - set(renamed)))))
            changes["tab_titles"] = renamed
        titles |= set(renamed)
        kept.append((replace(extension, **changes) if changes else extension, module_dir))
    return kept, refused
