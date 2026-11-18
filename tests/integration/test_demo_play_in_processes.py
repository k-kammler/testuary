"""testuary.demo.play plays the installed demonstrations one after the other
in one Python process, as an example script does."""
from __future__ import annotations

import pytest

from testuary import demo

pytestmark = pytest.mark.slow


def test_one_process_plays_two_demonstrations(tmp_path):
    assert demo.play("toy", output_dir=tmp_path / "first", speed="instant") == 0
    assert demo.play("toy", output_dir=tmp_path / "second", speed="instant") == 0
    assert (tmp_path / "second" / "DEMO.json").is_file()
    # The same run directory again is refused, as testuary --demo refuses it.
    assert demo.play("toy", output_dir=tmp_path / "first", speed="instant") == 2
