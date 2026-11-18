"""Real-subprocess check that the per-test evidence names the test that ran
first, whose time holds the run's one-time setup. Marked ``slow``."""
from __future__ import annotations

from pathlib import Path

import pytest

from testuary.languages import GenericBackend
from testuary.reduction import first_test_setup


# Slow, and the code under test reads the run's settings: each test runs in one.
pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("default_run")]

NAMES = ["alpha", "beta", "gamma", "delta"]


def _suite(root: Path, names) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "mod.py").write_text(
        "def f(x):\n    if x > 0:\n        return x + 1\n    return x - 1\n",
        encoding="utf-8")
    # The path is relative to the test file, so a measured copy imports its
    # own module.
    body = ("import os\nimport sys\n"
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))\n"
            "import mod\n\n")
    for name in names:
        body += f"def test_{name}():\n    assert mod.f(1) == 2\n    assert mod.f(-1) == -2\n\n"
    test = root / "tests" / "test_mod.py"
    test.write_text(body, encoding="utf-8")
    return test


@pytest.mark.parametrize("names", [NAMES, NAMES[::-1]], ids=["forward", "reversed"])
def test_the_evidence_names_the_first_test_and_its_cost_drops_the_setup(tmp_path, names):
    test = _suite(tmp_path / "project", names)
    data = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(test)], str(tmp_path / "project"),
        measure_coverage_of=str(tmp_path / "project" / "src" / "mod.py"),
        timeout=120, warmup=True)

    assert data is not None
    assert data.first_test is not None
    assert data.first_test.endswith(f"test_{names[0]}")
    setup = first_test_setup(data.per_test_tokens, data.per_test_time_ms, data.first_test)
    others = [time for test_id, time in data.per_test_time_ms.items()
              if test_id != data.first_test]
    assert setup["cost_ms"] == min(setup["measured_ms"], max(others))
