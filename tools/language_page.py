"""The levels of docs/extending/languages.md, from Testuary's profile.

The page holds a region between ``<!-- profile:testuary -->`` and
``<!-- /profile:testuary -->``. This tool fills it from the packaged profile
(``testuary/langadapter_profile.json``), each family linked to its section of
LangAdapter's guide, so the page and the profile a run applies cannot drift
apart. After changing the profile, rewrite the page with

    python tools/language_page.py

``--check`` writes nothing and exits 1 when the page differs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "extending" / "languages.md"
PROFILE = ROOT / "src" / "testuary" / "langadapter_profile.json"
#: LangAdapter's guide on its documentation site, where docs/<path>.md is
#: served at <path>/ with its anchors.
GUIDE = "https://k-kammler.github.io/langadapter/extending/"


def _site(guide: str) -> str:
    """A guide reference ``page.md#anchor`` as its address on the site."""
    page, _, anchor = guide.partition("#")
    return page[:-len(".md")] + "/" + (f"#{anchor}" if anchor else "")


_REGION = re.compile(r"(<!-- profile:testuary -->\n)(.*?)(<!-- /profile:testuary -->)", re.S)
_TIERS = {"required": "required", "optional": "optional", "not used": "not used"}


def family_guides() -> Dict[str, str]:
    """Each LangAdapter family's guide section, as a link target."""
    from langadapter import hooks

    return {family.name: GUIDE + _site(family.guide) for family in hooks.families()}


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render(profile: Mapping, guides: Mapping[str, str]) -> str:
    """The levels of ``profile`` as one table per level."""
    lines = []
    for level in profile["levels"]:
        lines += [f"**Level {level['level']}: {level['title']}**", "",
                  "| Family | Testuary | With it, Testuary | Without it |",
                  "| --- | --- | --- | --- |"]
        for need in level["families"]:
            name = need["family"]
            family = f"[{name}]({guides[name]})" if name in guides else name
            fallback = need.get("fallback", "")
            if need["tier"] == "required":
                fallback = "a run does not start"
            lines.append(f"| {family} | {_TIERS[need['tier']]} "
                         f"| {_cell(need.get('improves', ''))} "
                         f"| {_cell(fallback)} |")
        lines.append("")
    return "\n".join(lines)


def rewrite(text: str, profile: Mapping, guides: Mapping[str, str]) -> str:
    if len(_REGION.findall(text)) != 1:
        raise ValueError("the page needs exactly one profile:testuary region")
    return _REGION.sub(lambda match: match.group(1) + render(profile, guides) + match.group(3),
                       text)


def main(argv: Sequence[str]) -> int:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    text = PAGE.read_text(encoding="utf-8")
    page = rewrite(text, profile, family_guides())
    if "--check" in argv:
        if page != text:
            print("docs/extending/languages.md differs from Testuary's profile: run "
                  "python tools/language_page.py")
            return 1
        return 0
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    print("wrote docs/extending/languages.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
