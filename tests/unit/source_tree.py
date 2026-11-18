"""Skip marker for checks that compare repository files with each other: a
test run of the installed package has no ``src/`` tree."""
from __future__ import annotations

from pathlib import Path

import pytest

#: Testuary's source tree in the repository.
SOURCE_TREE = Path(__file__).resolve().parents[2] / "src" / "testuary"

requires_source_tree = pytest.mark.skipif(
    not SOURCE_TREE.is_dir(),
    reason="compares repository files with src/testuary, which a test run of the "
           "installed package does not have")
