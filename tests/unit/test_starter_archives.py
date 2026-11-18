"""Starter-suite archives: native suffixes, exact source bytes on every
platform, and resume. The tests need no measurement tools."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from testuary import config as config_module
from testuary import models
from testuary import prompts
from testuary import starters
from testuary import starter_salvage
from testuary.settings import RunSettings


SOURCES = {
    "python": "def test_value():\n    assert True\n",
    "java": "public class CalculatorTest { void testValue() {} }\n",
}
SUFFIXES = {"python": ".py", "java": ".java"}


class _ReachedConsolidation(Exception):
    pass


@pytest.fixture
def drive(tmp_path, monkeypatch):
    """Run the real request, persistence and resume path up to merging."""
    backend_calls = []
    model_calls = []
    cleanup_calls = []
    backend = SimpleNamespace()
    root = tmp_path / "subject"
    root.mkdir()
    suites = tmp_path / "suites"
    monkeypatch.setattr(starters.languages, "get_backend", lambda language: backend)
    monkeypatch.setattr(models, "get_api_config", lambda: {
        "model": "fixture", "api_key": "fixture", "api_url": "",
    })
    monkeypatch.setattr(prompts, "resolve_agent_prompts", lambda *args, **kwargs: {
        "system_prompt": "Write tests.", "user_prompt_template": "$source",
    })

    def stop(**kwargs):
        raise _ReachedConsolidation

    monkeypatch.setattr(starters, "_starter_mutation_config", stop)

    def cleanup(backend, source, target, source_dir, scratch, **kwargs):
        cleanup_calls.append(source)
        scratch.mkdir(parents=True, exist_ok=True)
        return "pruned", source

    monkeypatch.setattr(starter_salvage, "_clean_suite", cleanup)
    backend.normalize_shadowed_definitions = lambda source: SimpleNamespace(
        parse_ok=True, changed=False, source=source, removed_bindings=(),
    )
    backend.finalize_test_source = lambda source: source
    backend.validate_test_source = lambda source, source_dir: SimpleNamespace(ok=True, status="valid")
    backend.strip_module_level_skips = lambda source: (source, ())
    # The project context a removal of tests a measurement failed reads.
    backend.project_backend = lambda *args: backend

    def probe(files, source_dir, **kwargs):
        backend_calls.append((files, kwargs))
        return SimpleNamespace(ok=True, timed_out=False, line_coverage=100.0)

    backend.run_test_files = probe

    def run(language, *, reuse=False, generate=None, configuration=None):
        suffix = SUFFIXES[language]
        backend.file_extensions = (suffix,)
        backend.test_file_extensions = lambda: (suffix,)
        backend.prompt_fragments = lambda: {"lang_code_fence": language}
        backend.staged_test_path = lambda directory, stem, source: (
            Path(directory) / ("CalculatorTest.java" if language == "java" else f"{stem}.py")
        )
        target = root / f"Calculator{suffix}"
        target.write_text("target fixture", encoding="utf-8")
        settings = RunSettings(sections={"starter": {
            "suite_count": 1, "attempts_per_suite": 1, "concurrent_requests": 1,
            "suite_run_timeout_seconds": 2,
            **(configuration or {}),
        }}, base_dir=tmp_path)
        real_config = config_module.get_agent_config

        def config(name, parent=None):
            return {**real_config(name, parent=parent), "reuse_suites": reuse}

        def request(messages, **kwargs):
            model_calls.append(kwargs)
            if generate is not None:
                return generate(messages, **kwargs)
            return f"```{language}\n{SOURCES[language]}```", {
                "input_tokens": 10, "output_tokens": 3, "cache_hit_tokens": 2,
            }

        with monkeypatch.context() as patch, config_module.use_run_settings(settings), \
                models.installed(SimpleNamespace(generate_suite=request)):
            patch.setattr(config_module, "get_agent_config", config)
            return starters.run_starter_phase(
                language=language, target_file_path=str(target),
                source_dir=str(root), test_output_dir=str(tmp_path / "output"),
                suites_dir=str(suites), log=lambda message: None,
            )

    return SimpleNamespace(
        run=run, suites=suites, backend=backend, model_calls=model_calls,
        backend_calls=backend_calls, cleanup_calls=cleanup_calls,
    )


def _pool(drive, language, suffix, *, unresolved=False):
    seed = drive.suites / "suite_001"
    seed.mkdir(parents=True)
    source = SOURCES[language]
    (seed / f"original{suffix}").write_bytes(source.encode("utf-8"))
    if not unresolved:
        (seed / f"cleaned{suffix}").write_bytes(source.encode("utf-8"))
    (seed / "original.json").write_text(json.dumps({
        "slot_attempt": 1, "seed": 1, "attempts": 1, "input_tokens": 10, "output_tokens": 3,
        "cache_hit_tokens": 2,
        "extracted": True, "classification": "validation_unresolved" if unresolved else "pruned",
    }), encoding="utf-8")
    (seed / "response.txt").write_text(f"```{language}\n{source}```", encoding="utf-8")
    (drive.suites / starters._GENERATION_MANIFEST).write_text(json.dumps({
        "input_tokens": 10, "output_tokens": 3, "cache_hit_tokens": 2,
        "requests_used": 1, "logical_requests": 1, "n_generated": 1,
        "n_validation_unresolved": int(unresolved), "generation_seconds": 42.0,
        "generation_attempts_per_suite": {"1": 1}, "generation_complete": not unresolved,
    }), encoding="utf-8")
    return seed


@pytest.mark.parametrize("language", ["python", "java"])
def test_fresh_starter_archive_uses_native_suffix(language, drive):
    with pytest.raises(_ReachedConsolidation):
        drive.run(language)
    seed = drive.suites / "suite_001"
    suffix = SUFFIXES[language]
    for stem in ("original", "cleaned"):
        assert (seed / f"{stem}{suffix}").read_text("utf-8") == SOURCES[language]
        if language == "java":
            assert not (seed / f"{stem}.py").exists()
    meta = json.loads((seed / "original.json").read_text("utf-8"))
    assert meta["source_sha256"] == hashlib.sha256(SOURCES[language].encode()).hexdigest()
    assert meta["classification"] == "pruned"
    manifest = json.loads((drive.suites / starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["requests_used"] == manifest["logical_requests"] == 1
    assert manifest["generation_attempts_per_suite"] == {"1": 1}
    assert len(drive.model_calls) == len(drive.cleanup_calls) == 1


@pytest.mark.parametrize("language,suffix", [("python", ".py"), ("java", ".java")])
def test_complete_pool_is_reused_without_requests(language, suffix, drive):
    seed = _pool(drive, language, suffix)
    before = {path.name: path.read_bytes() for path in seed.iterdir()}
    manifest = (drive.suites / starters._GENERATION_MANIFEST).read_bytes()
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    assert {path.name: path.read_bytes() for path in seed.iterdir()} == before
    assert (drive.suites / starters._GENERATION_MANIFEST).read_bytes() == manifest
    assert drive.model_calls == drive.cleanup_calls == drive.backend_calls == []


@pytest.mark.parametrize("language,suffix", [("python", ".py"), ("java", ".java")])
def test_unjudged_suite_is_processed_without_a_new_paid_response(language, suffix, drive):
    seed = _pool(drive, language, suffix, unresolved=True)
    reply = (seed / "response.txt").read_bytes()
    original = (seed / f"original{suffix}").read_bytes()
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    assert (seed / f"original{suffix}").read_bytes() == original
    assert (seed / f"cleaned{suffix}").read_bytes() == SOURCES[language].encode("utf-8")
    assert (seed / "response.txt").read_bytes() == reply
    manifest = json.loads((drive.suites / starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["n_validation_unresolved"] == 0
    assert manifest["requests_used"] == manifest["logical_requests"] == 1
    assert manifest["generation_attempts_per_suite"] == {"1": 1}
    assert drive.model_calls == []
    assert drive.cleanup_calls == [SOURCES[language]]
    before = {path.name: path.read_bytes() for path in seed.iterdir() if path.is_file()}
    manifest_bytes = (drive.suites / starters._GENERATION_MANIFEST).read_bytes()
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    assert {path.name: path.read_bytes() for path in seed.iterdir() if path.is_file()} == before
    assert (drive.suites / starters._GENERATION_MANIFEST).read_bytes() == manifest_bytes
    assert drive.model_calls == []
    assert drive.cleanup_calls == [SOURCES[language]]


@pytest.mark.parametrize("language", ["python", "java"])
def test_paid_unprocessed_response_materializes_native_archive(language, drive):
    seed = drive.suites / "suite_001"
    seed.mkdir(parents=True)
    # The generation's manifest, written before its first request.
    (drive.suites / starters._GENERATION_MANIFEST).write_text(
        json.dumps({"generation_complete": False}), encoding="utf-8")
    (seed / starters._GENERATION_DELIVERY).write_text(json.dumps({
        "slot": 1, "slot_attempt": 1, "seed": 1, "id": "previous:1", "text": f"```{language}\n{SOURCES[language]}```",
        "usage": {"input_tokens": 10, "output_tokens": 3, "cache_hit_tokens": 2},
        "physical_requests": 1,
    }), encoding="utf-8")
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    assert (seed / f"original{SUFFIXES[language]}").read_text("utf-8") == SOURCES[language]
    assert (seed / f"cleaned{SUFFIXES[language]}").read_text("utf-8") == SOURCES[language]
    assert not (seed / starters._GENERATION_DELIVERY).exists()
    assert drive.model_calls == []
    manifest = json.loads((drive.suites / starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["requests_used"] == manifest["logical_requests"] == 1
    assert manifest["generation_attempts_per_suite"] == {"1": 1}


def test_a_java_pool_is_detected_and_loaded_by_its_suffix(tmp_path):
    backend = SimpleNamespace(file_extensions=(".java",), test_file_extensions=lambda: (".java",))
    seed = tmp_path / "suite_001"
    seed.mkdir()
    (seed / "original.java").write_bytes(SOURCES["java"].encode("utf-8"))
    (seed / "cleaned.java").write_bytes(SOURCES["java"].encode("utf-8"))
    (tmp_path / starters._GENERATION_MANIFEST).write_text(
        json.dumps({"n_generated": 1, "generation_complete": True}), encoding="utf-8")
    assert starters._has_persisted_pool(tmp_path)
    backend.normalize_shadowed_definitions = lambda source: SimpleNamespace(parse_ok=True, changed=False, source=source)
    backend.finalize_test_source = lambda source: source
    backend.validate_test_source = lambda *args: SimpleNamespace(ok=True, status="valid")
    result, green = starters.StarterResult(), {}
    assert starters._load_persisted_green(tmp_path, green, result, lambda message: None, backend=backend)
    assert green == {1: SOURCES["java"]}
    assert result.n_generated == 1


def test_a_reused_suite_is_judged_by_the_validators_first_answer(tmp_path):
    """The validator is asked again while it gives no answer, and its first
    answer counts. No answer at all stops the reuse unjudged, not as a
    failed validation, and the stop names the reused suite."""
    from testuary.consolidation import ValidationUnresolved

    seed = tmp_path / "suite_001"
    seed.mkdir()
    (seed / "cleaned.java").write_bytes(SOURCES["java"].encode("utf-8"))
    (tmp_path / starters._GENERATION_MANIFEST).write_text(
        json.dumps({"n_generated": 1, "generation_complete": True}), encoding="utf-8")
    silent = SimpleNamespace(ok=False, status="timeout", message="the validator timed out")
    answers = iter([silent, silent, SimpleNamespace(ok=True, status="ok", message="")])
    backend = SimpleNamespace(
        file_extensions=(".java",), test_file_extensions=lambda: (".java",),
        normalize_shadowed_definitions=lambda source: SimpleNamespace(parse_ok=True, changed=False, source=source),
        finalize_test_source=lambda source: source,
        validate_test_source=lambda *args: next(answers),
    )
    green = {}
    starters._load_persisted_green(tmp_path, green, starters.StarterResult(), lambda message: None, backend=backend)
    assert green == {1: SOURCES["java"]}
    backend.validate_test_source = lambda *args: silent
    with pytest.raises(ValidationUnresolved, match=r"^reused starter suite 1: the target "
                       r"validator did not judge it \(timeout: the validator timed out\)$"):
        starters._load_persisted_green(tmp_path, {}, starters.StarterResult(), lambda message: None, backend=backend)


def test_load_normalization_rewrites_the_archive_as_exact_bytes(tmp_path):
    """A rewrite keeps the source's line endings, also on Windows."""
    seed = tmp_path / "suite_001"
    seed.mkdir()
    (seed / "cleaned.java").write_bytes(b"old source")
    (tmp_path / starters._GENERATION_MANIFEST).write_text("{}", encoding="utf-8")
    backend = SimpleNamespace(
        file_extensions=(".java",), test_file_extensions=lambda: (".java",),
        normalize_shadowed_definitions=lambda source: SimpleNamespace(parse_ok=True, changed=False, source=source, removed_bindings=()),
        finalize_test_source=lambda source: SOURCES["java"],
        validate_test_source=lambda *args: SimpleNamespace(ok=True, status="valid"),
    )
    green = {}
    starters._load_persisted_green(tmp_path, green, starters.StarterResult(), lambda message: None, backend=backend)
    assert (seed / "cleaned.java").read_bytes() == SOURCES["java"].encode("utf-8")
    assert len(list(seed.iterdir())) == 1


def test_a_saved_response_of_a_later_attempt_keeps_the_earlier_attempt(drive):
    """On resume, a saved response for the slot's second attempt moves the
    first attempt's files to attempts/1/ instead of overwriting them."""
    seed = drive.suites / "suite_001"
    seed.mkdir(parents=True)
    broken = "def broken(:\n"
    (seed / "original.py").write_bytes(broken.encode("utf-8"))
    (seed / "response.txt").write_bytes(f"```python\n{broken}```".encode("utf-8"))
    (seed / "validity.json").write_text("{}", encoding="utf-8")
    (seed / "original.json").write_text(json.dumps({
        "slot_attempt": 1, "seed": 1, "attempts": 1, "input_tokens": 10, "output_tokens": 3,
        "extracted": True, "classification": "discarded_syntax",
    }), encoding="utf-8")
    (drive.suites / starters._GENERATION_MANIFEST).write_text(json.dumps({
        "input_tokens": 10, "output_tokens": 3, "cache_hit_tokens": 2,
        "requests_used": 1, "logical_requests": 1, "n_generated": 1,
        "n_discarded_syntax": 1, "generation_seconds": 4.0,
        "generation_attempts_per_suite": {"1": 1}, "generation_complete": False,
    }), encoding="utf-8")
    (seed / starters._GENERATION_DELIVERY).write_text(json.dumps({
        "slot": 1, "slot_attempt": 2, "seed": 1000001, "id": "previous:2",
        "text": f"```python\n{SOURCES['python']}```",
        "usage": {"input_tokens": 10, "output_tokens": 3, "cache_hit_tokens": 2},
        "physical_requests": 1,
    }), encoding="utf-8")
    with pytest.raises(_ReachedConsolidation):
        drive.run("python", reuse=True)
    assert drive.model_calls == []
    earlier = seed / starters._ATTEMPTS_DIR / "1"
    assert (earlier / "original.py").read_bytes() == broken.encode("utf-8")
    assert (earlier / "response.txt").read_bytes() == f"```python\n{broken}```".encode("utf-8")
    assert (earlier / "validity.json").is_file()
    assert json.loads((earlier / "original.json").read_text("utf-8"))["slot_attempt"] == 1
    assert (seed / "original.py").read_text("utf-8") == SOURCES["python"]
    assert (seed / "cleaned.py").read_text("utf-8") == SOURCES["python"]
    assert json.loads((seed / "original.json").read_text("utf-8"))["slot_attempt"] == 2
    assert not (seed / "validity.json").exists()
    assert sorted(path.name for path in (seed / starters._ATTEMPTS_DIR).iterdir()) == ["1"]
