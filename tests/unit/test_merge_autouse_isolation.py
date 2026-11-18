"""A module-level autouse fixture in one input does not reach another input.

Pytest applies a module-level autouse fixture to every test in its module,
and merging at the phase end puts independent suites into one module. The
merged suite keeps each fixture on the tests of the suite it came from.
These tests check that through the real ``consolidate_phase_end``, the real
Python backend and real pytest runs.
"""
from __future__ import annotations

from testuary import languages
from testuary import project
import pytest
from phase_end_calls import phase_end

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

TARGET = "def shout(text):\n    return text.upper()\n"

# An input whose autouse fixture stubs the module under test.
STUBBING_SEED = (
    "import pytest\n"
    "import calc\n"
    "\n"
    "\n"
    "def fake_shout(text):\n"
    "    return 'STUB'\n"
    "\n"
    "\n"
    "@pytest.fixture(autouse=True)\n"
    "def _stub_shout(monkeypatch):\n"
    "    monkeypatch.setattr(calc, 'shout', fake_shout)\n"
    "\n"
    "\n"
    "def test_stubbed_shout():\n"
    "    assert calc.shout('x') == 'STUB'\n"
)

# An input written against the real function, which the stub would break.
REAL_SEED = (
    "import calc\n"
    "\n"
    "\n"
    "def test_real_upper():\n"
    "    assert calc.shout('abc') == 'ABC'\n"
    "\n"
    "\n"
    "def test_real_empty():\n"
    "    assert calc.shout('') == ''\n"
)


def _merge(tmp_path, monkeypatch, sources):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    target = root / "calc.py"
    target.write_text(TARGET, encoding="utf-8", newline="")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    return phase_end(
        languages.get_backend("python"), sources,
        target_file_path=str(target),
        source_dir=project.resolve_source_dir(str(target)),
        timeout=120, stage_as_delivered=True, log=lambda message: None,
    )


def test_one_inputs_autouse_stub_does_not_break_the_other_inputs(
    tmp_path, monkeypatch,
):
    """Both inputs' tests survive merging unremoved, and the stubbing
    fixture is recorded as localized."""
    phase = _merge(tmp_path, monkeypatch, [STUBBING_SEED, REAL_SEED])

    assert phase.merged_source is not None
    assert phase.collision_resolution["autouse_localized"] == 1
    for name in ("test_stubbed_shout", "test_real_upper", "test_real_empty"):
        assert f"def {name}" in phase.merged_source


def test_the_fixture_still_applies_to_its_own_tests(tmp_path, monkeypatch):
    """Localizing keeps the fixture active: the stubbing input's test
    expects the stub and passes in the merged suite."""
    phase = _merge(tmp_path, monkeypatch, [STUBBING_SEED, REAL_SEED])

    source = phase.merged_source
    assert "@pytest.fixture(autouse=False)" in source
    assert "usefixtures('_stub_shout')" in source
    # The marker sits on the stubbing input's test and on no other.
    assert source.count("usefixtures('_stub_shout')") == 1
    stubbed_at = source.index("def test_stubbed_shout")
    marker_at = source.index("usefixtures('_stub_shout')")
    assert marker_at < stubbed_at
    assert "def test_real_upper" not in source[marker_at:stubbed_at]


def test_a_single_input_is_left_as_written(tmp_path, monkeypatch):
    """A single input keeps its autouse fixture exactly as written."""
    phase = _merge(tmp_path, monkeypatch, [STUBBING_SEED])

    assert "@pytest.fixture(autouse=True)" in phase.merged_source
