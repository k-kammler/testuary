"""The documentation site's navigation names existing pages and every page.

The site build itself (mkdocs build --strict) needs the docs extra. This
check keeps the navigation, the pages and their includes consistent without it.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from source_tree import requires_source_tree

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _nav_pages(entries):
    for entry in entries:
        if isinstance(entry, str):
            yield entry
        else:
            for value in entry.values():
                yield from ([value] if isinstance(value, str) else _nav_pages(value))


#: Directories of docs/ that are not part of the published site.
EXCLUDED = ("development",)


def _site():
    return yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))


def test_every_navigation_entry_exists_and_every_public_page_is_listed():
    listed = list(_nav_pages(_site()["nav"]))
    assert len(listed) == len(set(listed))
    assert [page for page in listed if not (DOCS / page).is_file()] == []
    public = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.md")
              if not set(EXCLUDED) & set(path.relative_to(DOCS).parts)}
    assert sorted(public - set(listed)) == []


@requires_source_tree
def test_included_files_exist():
    included = [match.group(1) for path in DOCS.rglob("*.md")
                for match in re.finditer(r'^--8<-- "([^"]+)"', path.read_text(encoding="utf-8"), re.M)]
    assert included and [name for name in included if not (ROOT / name).is_file()] == []
