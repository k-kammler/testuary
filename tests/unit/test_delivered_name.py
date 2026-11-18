"""A starter suite whose class is not named for the delivered file.

A Java file is named for its public class, and a suite may name another
class (a neighboring class's test). The cleanup renames the class to the
delivered file's name, so every staged copy and the delivered file carry the
same name.
"""
from __future__ import annotations

from testuary import languages
from testuary.languages.generic import GenericBackend
from testuary import starter_salvage
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

TARGET = "src/main/java/org/skyscreamer/jsonassert/JSONCompareResult.java"
NAMED_FOR_A_NEIGHBOR = (
    "package org.skyscreamer.jsonassert;\n\n"
    "import static org.junit.Assert.assertTrue;\n"
    "import org.junit.Test;\n\n"
    "public class JSONCompareTest {\n"
    "    private final JSONCompareTest self = this;\n\n"
    "    @Test\n"
    "    public void passes() {\n"
    "        assertTrue(new JSONCompareResult().passed());\n"
    "    }\n"
    "}\n"
)


def test_java_renames_the_class_to_the_delivered_files_name():
    source, old, refusal = GenericBackend("java", (".java",)).conform_delivered_name(NAMED_FOR_A_NEIGHBOR, TARGET)
    assert (old, refusal) == ("JSONCompareTest", None)
    assert "public class JSONCompareResultTest {" in source
    assert "private final JSONCompareResultTest self" in source
    assert "JSONCompareTest" not in source
    # The rest of the source is unchanged.
    assert source.replace("JSONCompareResultTest", "JSONCompareTest") == NAMED_FOR_A_NEIGHBOR


def test_a_class_already_named_for_its_file_is_left_as_it_is():
    named = NAMED_FOR_A_NEIGHBOR.replace("JSONCompareTest", "JSONCompareResultTest")
    assert GenericBackend("java", (".java",)).conform_delivered_name(named, TARGET) == (named, None, None)


def test_a_class_that_cannot_be_renamed_safely_is_refused():
    clashing = NAMED_FOR_A_NEIGHBOR.replace(
        "    @Test\n", "    void JSONCompareTest() {}\n\n    @Test\n", 1)
    source, old, refusal = GenericBackend("java", (".java",)).conform_delivered_name(clashing, TARGET)
    assert source == clashing and old == "JSONCompareTest" and refusal


def test_other_languages_change_nothing():
    source = "def test_a():\n    assert True\n"
    backend = languages.get_backend("python")
    assert backend.conform_delivered_name(source, "pkg/mod.py") == (source, None, None)


def test_the_starter_cleanup_renames_and_records_it(tmp_path, monkeypatch):
    """The cleanup renames before staging, records the old name and continues
    with the renamed suite."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    backend = languages.get_backend("python")
    renamed = []

    def conform(self, source, target_file_path):
        renamed.append(target_file_path)
        return source.replace("test_old", "test_new"), "Old", None

    monkeypatch.setattr(type(backend), "conform_delivered_name", conform)
    cleanup = {}
    try:
        classification, green = starter_salvage._clean_suite(
            backend, "from calc import add\n\ndef test_old():\n    assert add(1, 1) == 2\n",
            str(project / "calc.py"), str(project), tmp_path / "scratch", timeout=120,
            log=lambda _m: None, cleanup_out=cleanup)
    finally:
        monkeypatch.undo()
    assert renamed == [str(project / "calc.py")]
    assert classification == "all_passing" and "def test_new" in green
    assert cleanup["renamed_class"] == "Old"


def test_a_refused_rename_discards_the_suite_with_its_reason(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    backend = languages.get_backend("python")
    monkeypatch.setattr(type(backend), "conform_delivered_name",
                        lambda self, source, target: (source, "Old", "a method is named Old"))
    cleanup = {}
    try:
        classification, green = starter_salvage._clean_suite(
            backend, "def test_a():\n    assert True\n", str(project / "calc.py"),
            str(project), tmp_path / "scratch", timeout=120, log=lambda _m: None,
            cleanup_out=cleanup)
    finally:
        monkeypatch.undo()
    assert (classification, green) == ("invalid_generated_test", None)
    assert cleanup["disposition_reason"].endswith("a method is named Old")
