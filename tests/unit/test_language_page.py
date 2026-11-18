"""The levels on docs/extending/languages.md are Testuary's profile as a run
applies it (tools/language_page.py), each family linked to LangAdapter's
guide."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from source_tree import requires_source_tree

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "language_page.py"
_spec = importlib.util.spec_from_file_location("language_page", _TOOL)
language_page = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(language_page)

PROFILE = {"levels": [
    {"level": 1, "title": "Run", "families": [{"family": "run tests", "tier": "required"}]},
    {"level": 2, "title": "Repair", "families": [
        {"family": "load repair", "tier": "optional", "improves": "keeps a | suite",
         "fallback": "the suite is discarded"}]}]}
GUIDES = {"run tests": "https://example.test/add-a-language.md#run-the-tests"}


def test_each_level_is_a_table_of_its_families():
    text = language_page.render(PROFILE, GUIDES)
    assert "**Level 1: Run**" in text and "**Level 2: Repair**" in text
    assert ("| [run tests](https://example.test/add-a-language.md#run-the-tests) | required "
            "|  | a run does not start |") in text
    assert "| load repair | optional | keeps a \\| suite | the suite is discarded |" in text


def test_the_page_needs_its_region_once():
    with pytest.raises(ValueError, match="exactly one profile:testuary region"):
        language_page.rewrite("no region", PROFILE, GUIDES)
    page = "a\n<!-- profile:testuary -->\nold\n<!-- /profile:testuary -->\nb\n"
    text = language_page.rewrite(page, PROFILE, GUIDES)
    assert "old" not in text and text.startswith("a\n") and text.endswith("b\n")
    assert language_page.rewrite(text, PROFILE, GUIDES) == text


@requires_source_tree
def test_the_page_matches_testuarys_profile():
    pytest.importorskip("langadapter.hooks")
    profile = json.loads(language_page.PROFILE.read_text(encoding="utf-8"))
    page = language_page.PAGE.read_text(encoding="utf-8")
    assert page == language_page.rewrite(page, profile, language_page.family_guides()), (
        "docs/extending/languages.md differs from Testuary's profile: run "
        "python tools/language_page.py")
