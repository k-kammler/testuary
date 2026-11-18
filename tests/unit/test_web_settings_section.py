"""[web] configures the browser interface, not a run: it resolves
and is validated like every section and the settings views show it, but
the run records leave it out, and every other section stays in them."""
from __future__ import annotations

import pytest

from testuary import settings


def test_web_settings_resolve_with_their_defaults():
    resolved = settings.RunSettings(sections={"web": {"port": "9000", "access": "KEY"}})
    assert resolved.sections["web"] == {"port": "9000", "access": "key"}
    assert settings.RunSettings().sections["web"] == {"port": "8765", "access": "open"}


@pytest.mark.parametrize("values, refused", [
    ({"port": "65536"}, "port"), ({"port": "-1"}, "port"), ({"port": "eight"}, "port"),
    ({"access": "public"}, "access"), ({"host": "0.0.0.0"}, "host"),
])
def test_web_settings_are_validated_like_any_section(values, refused):
    with pytest.raises(ValueError) as error:
        settings.RunSettings(sections={"web": values})
    assert error.value.setting == ("web", refused)


def test_the_run_records_leave_out_web_and_keep_every_other_section():
    resolved = settings.RunSettings(sections={"web": {"port": "9000"}})
    recorded = resolved.export()["sections"]
    assert "web" not in recorded
    assert set(recorded) == set(resolved.sections) - {"web"}
    assert settings.UNRECORDED_SECTIONS == frozenset({"web"})


def test_the_settings_views_show_web():
    resolved = settings.RunSettings(sections={"web": {"access": "stable_key"}})
    shown = {(item.setting.section, item.setting.name): item.value
             for item in settings.setting_values(resolved, languages=("python",))}
    assert shown[("web", "port")] == "8765" and shown[("web", "access")] == "stable_key"
    (web,) = [section for section in settings.reference(languages=("python",)) if section.name == "web"]
    assert web.title == "Web interface"
    access = next(setting for setting in web.settings if setting.name == "access")
    assert access.kind == "choice" and access.choices == ("open", "key", "stable_key")
