"""A target validator that answers as a test says: the real outcome, or one
that judges no code (the validator timed out or did not start)."""
from __future__ import annotations

from testuary import languages
from testuary.languages import SourceValidation


def patch_validator(monkeypatch, judge):
    """The Python backend's target validator answered by ``judge(source,
    real)``, where ``real()`` is the real outcome."""
    cls = type(languages.get_backend("python"))
    real = cls.validate_test_source

    def validate(self, source, source_dir):
        return judge(source, lambda: real(self, source, source_dir))

    monkeypatch.setattr(cls, "validate_test_source", validate)


def unjudged(status="validator_error"):
    """An outcome that judges no code."""
    return SourceValidation(False, status, "the interpreter did not start",
                            0, 0, "python", "0" * 64)
