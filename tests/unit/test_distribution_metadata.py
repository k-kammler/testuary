"""What Testuary's distributions declare about themselves: licenses,
dependencies, classifiers and the files each artifact carries."""
from __future__ import annotations

from pathlib import Path
import re
import shutil

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parents[2]


def _project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_the_license_files_cover_the_third_party_sources_the_demonstrations_ship():
    """Testuary is MIT. The demonstrations ship tenacity's and JSONassert's
    Apache 2.0 sources, and the distribution carries their license texts and
    the notices that explain them."""
    project = _project()
    assert project["license"] == "MIT"
    shipped = {path.relative_to(ROOT).as_posix()
               for path in (ROOT / "src" / "testuary" / "demo" / "data").rglob("LICENSE-*.txt")}
    assert shipped and shipped <= set(project["license-files"])
    assert {"LICENSE", "THIRD_PARTY_NOTICES.md"} <= set(project["license-files"])


def _declared_names():
    import re

    return {re.sub(r"[-_.]+", "-", re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0]).lower()
            for requirement in _project()["dependencies"]}


def test_every_module_level_import_is_a_declared_dependency():
    """A third-party module that Testuary imports at module level belongs to
    a declared dependency, not only to a dependency of a dependency."""
    import ast
    import re
    import sys
    from importlib import metadata

    declared = _declared_names()
    owners = metadata.packages_distributions()
    undeclared = set()
    for path in (ROOT / "src" / "testuary").rglob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in ("testuary", "__future__"):
                    continue
                # A dependency run from a source tree has no installed metadata.
                distributions = owners.get(top) or [top]
                if not {re.sub(r"[-_.]+", "-", d).lower() for d in distributions} & declared:
                    undeclared.add(f"{top} ({path.relative_to(ROOT).as_posix()})")
    assert sorted(undeclared) == []


def test_the_typed_package_and_the_source_distribution_say_what_they_are():
    """py.typed ships, so the classifiers say Typing :: Typed. The sdist
    carries both page generators and the citation file."""
    project = _project()
    assert (ROOT / "src" / "testuary" / "py.typed").is_file()
    assert "Typing :: Typed" in project["classifiers"]
    build = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["hatch"]
    include = set(build["build"]["targets"]["sdist"]["include"])
    assert {"tools/reference_pages.py", "tools/language_page.py", "/CITATION.cff"} <= include


# Every entry point Testuary declares follows
# the entry point grammar as pip's strictest parser reads it. pip on Python
# 3.10 parses entry points with pkg_resources' ``EntryPoint.parse`` and
# refuses to install a package with a value that is no ``module[:attr]``,
# such as a file path.
#: pkg_resources' ``EntryPoint.pattern``, matched against ``name = value``.
_ENTRY_POINT = re.compile(
    r'\s*'
    r'(?P<name>.+?)\s*'
    r'=\s*'
    r'(?P<module>[\w.]+)\s*'
    r'(:\s*(?P<attr>[\w.]+))?\s*'
    r'(?P<extras>\[.*\])?\s*$'
)


def _refused(entries):
    return [f"{group}: {name} = {value}" for group, name, value in entries
            if not _ENTRY_POINT.match(f"{name} = {value}")]


def _declared(project):
    entries = [("console_scripts", name, value)
               for name, value in project.get("scripts", {}).items()]
    for group, members in project.get("entry-points", {}).items():
        entries += [(group, name, value) for name, value in members.items()]
    return entries


def test_testuarys_entry_points_are_ones_every_pip_reads():
    entries = _declared(_project())
    groups = {group for group, _, _ in entries}
    assert "console_scripts" in groups and "langadapter.consumers" not in groups
    assert _refused(entries) == []


def test_langadapter_finds_testuarys_profile_in_its_package(tmp_path):
    """An installed Testuary: LangAdapter's lookup reads the
    langadapter_profile.json in the package's directory, the file Testuary's own
    checks read, without importing Testuary."""
    from langadapter import consumers

    site = tmp_path / "site"
    info = site / "testuary-0.2.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: testuary\nVersion: 0.2.0\n",
                                   encoding="utf-8")
    (info / "RECORD").write_text("testuary/langadapter_profile.json,,\n", encoding="utf-8")
    (site / "testuary").mkdir()
    shutil.copyfile(ROOT / "src" / "testuary" / "langadapter_profile.json",
                    site / "testuary" / "langadapter_profile.json")
    found = consumers.profile("testuary", consumers.Installed(path=[site]))
    assert found.source == "testuary/langadapter_profile.json"
    direct = consumers.read_file(ROOT / "src" / "testuary" / "langadapter_profile.json")
    assert (found.levels, found.needs) == (direct.levels, direct.needs)
