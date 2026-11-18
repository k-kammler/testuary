"""Source recovery uses native group results without accepting unavailable work."""
import dataclasses
import sys
from types import SimpleNamespace

import pytest

import langadapter as lb
from langadapter.sources import SourceResult

from testuary.languages.generic import GenericBackend
from testuary.consolidation import ValidationUnresolved


class NativeRecovery(lb.LangAdapter):
    def __init__(self, root):
        super().__init__(lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable)))
        self.requests = []
        self.result = SourceResult("applied", "native repaired source", "fixture-parser",
                                   removed=("native.removed",))

    def prune_unresolvable_imports(self, source):
        self.requests.append(("imports", source))
        return self.result

    def prune_collection_breakers(self, source, *, timeout_s):
        self.requests.append(("collection", source, timeout_s))
        return self.result


@pytest.fixture(params=[lambda: GenericBackend("python", (".py",)), lambda: GenericBackend("java", (".java",))], ids=["python", "java"])
def adapter(request, tmp_path, monkeypatch):
    native = NativeRecovery(tmp_path)
    language = request.param()
    monkeypatch.setattr(language, "project_backend", lambda root=None:
                        SimpleNamespace(sources=native.sources))
    return language, native


@pytest.fixture(params=["prune_unresolvable_imports", "prune_collection_breakers"])
def invoke(request, tmp_path):
    def call(language):
        options = {"timeout": 17} if request.param == "prune_collection_breakers" else {}
        return getattr(language, request.param)("original source", str(tmp_path), **options)
    return call, request.param


def test_recovery_retains_native_rewrite_removals_and_timeout(adapter, invoke):
    language, native = adapter
    call, method = invoke
    assert call(language) == ("native repaired source", ["native.removed"])
    expected = (("collection", "original source", 17) if method == "prune_collection_breakers"
                else ("imports", "original source"))
    assert native.requests == [expected]


@pytest.mark.parametrize("status", ["unchanged", "not_repaired", "parse_error", "refused"])
def test_judged_recovery_results_keep_existing_return_shape(adapter, invoke, status):
    language, native = adapter
    native.result = SourceResult(status, "original source", "fixture-parser", detail="native decision")
    call, _ = invoke
    assert call(language) == ("original source", []) and len(native.requests) == 1


@pytest.mark.parametrize("status", ["missing_setup", "timeout", "infrastructure_failure", "unsupported"])
def test_unavailable_recovery_is_not_an_unchanged_success(adapter, invoke, status):
    language, native = adapter
    native.result = dataclasses.replace(native.result, status=status, detail="native recovery unavailable")
    call, _ = invoke
    with pytest.raises(ValidationUnresolved, match="native recovery unavailable"):
        call(language)
    assert len(native.requests) == 1
