"""Readiness values detach facts without performing an inspection."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess

import pytest

from testuary import project
from testuary.readiness import ReadinessItem, ReadinessLimit, RunReadiness


def report(**fields):
    return RunReadiness(**{
        "scope": "inspection", "status": "verified", "exit_code": 0,
        "source_dir": None, "execution": None, "source_files": ["module.py"],
        "phases": ["starter", "coverage", "mutation", "finishing"],
        "model_alias": None, "provider": None, "model": None, "credentials_configured": None,
        "output_dir": None, "output_mode": "fresh", "limits": [], "items": [], **fields,
    })


@pytest.fixture(autouse=True)
def no_inspection(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(project, "project_runtime", lambda *args, **kwargs: pytest.fail("provisioning"))
    monkeypatch.setattr(Path, "exists", lambda *a, **k: pytest.fail("path inspection"))


def test_reports_detach_nested_sequences_and_json_without_inventing_facts():
    sources, phases = ["one,two.py"], ["starter"]
    items = [ReadinessItem("python.runtime", True, "unverified", selected="chosen/python")]
    limits = [ReadinessLimit("starter", "tokens", None, "tokens", "unknown")]
    value = report(source_files=sources, phases=phases, items=items, limits=limits)
    sources[0], phases[0], items[0], limits[0] = "changed", "changed", None, None
    assert value.source_files == ("one,two.py",) and value.phases == ("starter",)
    exported = value.as_dict()
    exported["items"][0]["status"] = "verified"
    exported["limits"][0]["value"] = 0
    assert value.items[0].status == "unverified" and value.limits[0].value is None
    assert json.loads(json.dumps(value.as_dict())) == value.as_dict()
    assert value.source_dir is value.credentials_configured is None
    assert set(value.as_dict()) == {"scope", "status", "exit_code", "source_dir", "execution",
        "source_files", "phases", "model_alias", "provider", "model", "credentials_configured",
        "output_dir", "output_mode", "limits", "items"}
    for item, field in ((value, "status"), (value.items[0], "status"), (value.limits[0], "value")):
        with pytest.raises(FrozenInstanceError):
            setattr(item, field, "changed")


@pytest.mark.parametrize("status", ["verified", "missing_setup", "unsupported", "ambiguous", "unverified", "failed"])
def test_item_status_preserves_requested_scope_without_computing_readiness(status):
    value = report(scope="verification", status=status, exit_code=1,
                   items=[ReadinessItem("mutation", True, status, reason="actual observation")])
    assert value.status == status and value.exit_code == 1
    assert value.items[0].diagnostic_path is None


@pytest.mark.parametrize("state,value", [("configured", 0), ("configured", 12),
                                          ("unknown", None), ("unlimited", None)])
def test_limits_keep_zero_unknown_and_unlimited_distinct(state, value):
    limit = ReadinessLimit("coverage", "requests", value, "requests", state)
    assert limit.state == state and limit.value == value


@pytest.mark.parametrize("state,value", [("configured", None), ("configured", True),
    ("configured", -1), ("configured", float("nan")), ("configured", float("inf")),
    ("unknown", 0), ("unlimited", 0), ("nonsense", None)])
def test_invalid_limit_refuses(state, value):
    with pytest.raises(ValueError):
        ReadinessLimit("starter", "tokens", value, "tokens", state)


@pytest.mark.parametrize("fields", [
    {"scope": "provisioning"}, {"status": "ready"}, {"exit_code": True}, {"exit_code": 3},
    {"execution": "ambient"}, {"output_mode": "overwrite"}, {"credentials_configured": "authenticated"},
    {"source_dir": {}}, {"source_files": "module.py"}, {"phases": [None]},
    {"items": [{}]}, {"limits": [{}]},
])
def test_invalid_report_contract_refuses(fields):
    with pytest.raises(ValueError):
        report(**fields)


def test_item_rejects_raw_payloads_and_keeps_only_declared_facts():
    item = ReadinessItem("provider.configuration", False, "unverified")
    with pytest.raises(ValueError):
        replace(item, selected={"api_key": "not a report field"})
    with pytest.raises(ValueError):
        replace(item, required=1)
    with pytest.raises(ValueError):
        replace(item, status="authenticated")
