"""The edit tool in both Java agents on the real Java backend
(``java_edit_tool.py``, run in a process of its own): single edits, edits
applied in list order, and a list's checks run once on its final result, all
or nothing. Needs a JDK 21 and Maven (``LANGADAPTER_TEST_JAVA_HOMES``,
``LANGADAPTER_TEST_MAVEN``). The Python agents' cases are in
``tests/unit/test_edit_tool.py``."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from java_fixtures import jdk_and_maven

pytestmark = pytest.mark.slow

DRIVER = Path(__file__).with_name("java_edit_tool.py")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    jdk, maven = jdk_and_maven()
    work = tmp_path_factory.mktemp("java-edit")
    done = subprocess.run([sys.executable, str(DRIVER), str(work), str(jdk), maven],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=1800)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return json.loads((work / "report.json").read_text(encoding="utf-8"))


AGENTS = ("coverage", "mutation")


@pytest.mark.parametrize("kind", AGENTS)
def test_a_single_edit_lands_after_one_check(report, kind):
    case = report[f"{kind}/single"]
    assert case["results"][0].startswith("Replaced 1 occurrence in CalcTest.java"), case["results"]
    assert "    void zero() {\n        assertEquals(0, Calc.sign(0));\n    }\n" in case["file"]
    assert case["writes"] == 1 and case["target_parses"] == 1


@pytest.mark.parametrize("kind", AGENTS)
def test_separate_calls_apply_in_call_order(report, kind):
    case = report[f"{kind}/two_calls_apply_in_order"]
    assert [r.startswith("Replaced 1 occurrence") for r in case["results"]] == [True, True]
    assert ("        assertEquals(0, Calc.sign(0));\n        assertEquals(0, Calc.sign(-0));\n"
            in case["file"])
    assert case["writes"] == 2 and case["target_parses"] == 2


@pytest.mark.parametrize("kind", AGENTS)
def test_a_list_lands_a_new_version_before_the_old_one_is_deleted(report, kind):
    """One at a time, the new positive() would be refused as a duplicate."""
    case = report[f"{kind}/list_new_version_then_delete"]
    assert case["results"][0].startswith("Applied all 2 edits in order to CalcTest.java")
    assert case["file"].count("void positive()") == 1 and "Calc.sign(7)" in case["file"]
    assert "Calc.sign(5));" not in case["file"]
    # One write, and the target parser checks the final result once.
    assert case["writes"] == 1 and case["target_parses"] == 1


@pytest.mark.parametrize("kind", AGENTS)
def test_a_list_applies_its_edits_in_order(report, kind):
    """The second edit matches text the first one wrote."""
    case = report[f"{kind}/list_header_then_body"]
    assert case["results"][0].startswith("Applied all 2 edits in order")
    assert "    void zero() {\n        assertEquals(0, Calc.sign(0));\n    }\n" in case["file"]
    assert case["writes"] == 1 and case["target_parses"] == 1


@pytest.mark.parametrize("kind", AGENTS)
def test_a_list_whose_result_does_not_parse_writes_nothing(report, kind):
    case = report[f"{kind}/list_rejects_a_broken_result"]
    text = case["results"][0]
    assert text.startswith("REJECTED: the result of all 2 edits fails a check, so none of the 2 "
                           "edits was applied"), text
    assert "syntactically invalid" in text
    assert case["file"] == report["original"] and case["writes"] == 0


@pytest.mark.parametrize("kind", AGENTS)
def test_a_list_with_an_item_that_misses_writes_nothing(report, kind):
    case = report[f"{kind}/list_rejects_a_missed_item"]
    text = case["results"][0]
    assert text.startswith("REJECTED: edit 2 of 2 failed, so none of the 2 edits was applied")
    assert "as edit 1 of this list\nleft it" in text
    assert case["file"] == report["original"] and case["writes"] == 0
    assert case["target_parses"] == 0


@pytest.mark.parametrize("kind", AGENTS)
def test_a_duplicate_method_is_refused(report, kind):
    case = report[f"{kind}/single_rejects_a_duplicate_method"]
    assert "multiple definitions of CalcTest.negative" in case["results"][0]
    assert case["file"] == report["original"] and case["writes"] == 0
