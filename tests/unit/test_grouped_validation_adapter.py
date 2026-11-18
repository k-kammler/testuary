"""Typed validator observations keep completed rejections separate from failures."""
import hashlib
from types import SimpleNamespace

import pytest

from langadapter.sources import SourceObservation, SourceStatus, SourceValidation

from testuary.languages.generic import GenericBackend
from testuary.consolidation import VALIDATION_ATTEMPTS, ValidationUnresolved, validate_judged


@pytest.fixture(params=[lambda: GenericBackend("python", (".py",)), lambda: GenericBackend("java", (".java",))], ids=["python", "java"])
def validation(request, monkeypatch):
    language = request.param()
    calls = []
    group = SimpleNamespace(result=None, supports=lambda name: True)

    def validate(source):
        calls.append(source)
        return group.result

    group.validate_source = validate
    monkeypatch.setattr(language, "project_backend", lambda root=None: SimpleNamespace(sources=group))
    return language, group, calls


@pytest.mark.parametrize("status", ["valid", "syntax_error"])
def test_completed_validation_retains_native_verdict_and_identity(validation, status):
    language, group, calls = validation
    native = SourceValidation(status, "native diagnostic", 3, 7, "native parser", "native digest")
    group.result = SourceObservation(SourceStatus.COMPLETED, native, "native parser")
    result = language.validate_test_source("source bytes", "project")
    assert result.ok is (status == "valid") and result.status == status
    assert result.message == "native diagnostic" and (result.line, result.col) == (3, 7)
    assert (result.validator_identity, result.source_sha256) == ("native parser", "native digest")
    assert calls == ["source bytes"]


@pytest.mark.parametrize("status", [SourceStatus.TIMEOUT, SourceStatus.MISSING_SETUP,
                                   SourceStatus.INFRASTRUCTURE_FAILURE, SourceStatus.UNSUPPORTED])
def test_unavailable_validation_is_not_a_source_rejection(validation, status):
    language, group, calls = validation
    group.result = SourceObservation(status, reason="native validator unavailable")
    source = "complete source bytes"
    result = language.validate_test_source(source, "project")
    assert not result.ok
    assert result.status == ("timeout" if status is SourceStatus.TIMEOUT else "validator_error")
    assert result.message == "native validator unavailable" and (result.line, result.col) == (0, 0)
    assert result.validator_identity == "validator identity unavailable"
    assert result.source_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert calls == [source]


@pytest.mark.parametrize("status", [SourceStatus.TIMEOUT, SourceStatus.MISSING_SETUP,
                                   SourceStatus.INFRASTRUCTURE_FAILURE, SourceStatus.UNSUPPORTED])
def test_actual_consolidation_retries_unavailable_validation_without_ejecting_source(validation, status):
    language, group, calls = validation
    group.result = SourceObservation(status, reason="native validator unavailable")
    with pytest.raises(ValidationUnresolved, match="native validator unavailable"):
        validate_judged(language, "unchanged source", "project")
    assert calls == ["unchanged source"] * VALIDATION_ATTEMPTS


def test_completed_syntax_error_is_judged_once(validation):
    language, group, calls = validation
    group.result = SourceObservation(SourceStatus.COMPLETED, SourceValidation(
        "syntax_error", "bad syntax", 1, 2, "native parser", "digest"))
    result = validate_judged(language, "bad source", "project")
    assert result.status == "syntax_error"
    assert calls == ["bad source"]
