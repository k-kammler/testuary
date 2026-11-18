"""Measurement settings have no defaults in code.

A defaulted read would silently run on the fallback after a key is renamed.
So every measurement setting is defined in the bundled ``config.ini``, and
the package reads them only by subscription or ``config.require_config``,
which fail naming the key.
"""
from __future__ import annotations

import configparser
import re
from pathlib import Path

import pytest

import testuary
from testuary import config

#: The package as imported, from the tree or installed.
PACKAGE = Path(testuary.__file__).resolve().parent

#: The measurement settings: anything that changes what a score means.
MEASUREMENT_KEYS = (
    "max_requests_per_phase",
    "suite_count",
    "suite_run_timeout_seconds",
    "goal",
    "goal_metric",
    "minimum",
    "max_measurements_without_improvement",
    "min_improvement_points",
    "reduction_solver",
    "reduction_tie_break",
    "cosmic_ray_operators",
    "pit_mutators",
    "per_mutant_timeout_seconds",
    "per_mutant_timeout_factor",
    "min_per_mutant_timeout_seconds",
    "max_per_mutant_timeout_factor",
    "measurement_timeout_seconds",
)

#: Where measurement settings are read.
SCANNED = (PACKAGE,)


def _bundled_sections() -> dict:
    cp = configparser.ConfigParser()
    cp.read(PACKAGE / "config.ini", encoding="utf-8")
    return {s: dict(cp[s]) for s in cp.sections()}


@pytest.mark.parametrize("key", MEASUREMENT_KEYS)
def test_bundled_config_defines_every_measurement_key(key):
    """The bundled layer defines the key in at least one section."""
    sections = _bundled_sections()
    assert any(key in kv for kv in sections.values()), (
        f"{key!r} is not defined anywhere in the bundled config.ini; "
        f"a strict read of it would fail on every run")


def test_no_defaulted_reads_of_measurement_keys():
    """No scanned source reads a measurement key with ``.get`` and a default."""
    pat = re.compile(
        r"\.get\(\s*['\"](" + "|".join(MEASUREMENT_KEYS) + r")['\"]\s*,",
        re.S)
    offenders = []
    files = []
    for root in SCANNED:
        files.extend((f, root.parent) for f in (
            root.rglob("*.py") if root.is_dir() else [root]))
    for f, base in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{f.relative_to(base)}:{line} ({m.group(1)})")
    assert not offenders, (
        "defaulted reads of measurement keys (use subscription or "
        "config.require_config):\n  " + "\n  ".join(offenders))


def test_require_config_hard_fails_naming_the_key():
    with pytest.raises(KeyError, match="per_mutant_timeout_factor"):
        config.require_config({}, "per_mutant_timeout_factor")
    assert config.require_config({"goal": "90"},
                                "goal") == "90"
