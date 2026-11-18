"""The examples `testuary init --examples` copies out work unchanged: an
added model provider, a language-specific prompt profile and per-language
hints. Nothing here sends a model request.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from testuary import __main__ as cli
from testuary import config
from testuary import prompts


@pytest.fixture
def examples(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli._cmd_init(["--examples"]) == 0
    capsys.readouterr()
    return tmp_path / "testuary-examples"


def test_init_copies_every_example_and_never_replaces_a_file(examples, capsys):
    names = {path.relative_to(examples).as_posix() for path in examples.rglob("*") if path.is_file()}
    assert names == {
        "README.txt", "controls.ini", "hints.json", "language-profile.json", "models.ini",
        "testuary_example_chat.py",
    }
    (examples / "models.ini").write_text("[edited]\n", encoding="utf-8")
    assert cli._cmd_init(["--examples"]) == 0
    assert (examples / "models.ini").read_text(encoding="utf-8") == "[edited]\n"
    assert "skip:" in capsys.readouterr().out


def test_the_examples_resolve_the_added_provider_profile_and_hints(examples, monkeypatch):
    from langchain_openai import ChatOpenAI
    from testuary.models import resolve_phase_model
    from testuary.models import build_resolved_chat_model
    from testuary.languages import get_backend
    from testuary.settings import resolve_settings

    # The provider is the file beside models.ini, with nothing installed.
    facts = SimpleNamespace(test_framework="JUnit 5", timeout_marker="", runtime_version="21")
    monkeypatch.setattr(type(get_backend("java")), "project_backend",
                        lambda self, root: SimpleNamespace(language_facts=lambda: facts))
    settings = resolve_settings(base_dir=examples, home=examples, config_files=["controls.ini"],
                                environment={"LOCAL_MODEL_API_KEY": "example-fixture-key"})
    with config.use_run_settings(settings):
        model = resolve_phase_model(settings, "coverage")
        assert (model.protocol, model.identity.model_alias) == ("added", "example_added")
        client = build_resolved_chat_model(model, api_key="example-fixture-key")
        assert isinstance(client, ChatOpenAI) and client.max_retries == 0 and client.stream_usage
        python = prompts.resolve_agent_prompts("test_coverage", "python", runtime_version="fixture")
        java = prompts.resolve_agent_prompts("test_coverage", "java", runtime_version="fixture")
    assert "Use pytest.raises for expected exceptions." in python["system_prompt"]
    assert "Use assertThrows for expected exceptions." in java["system_prompt"]
    assert "assertThrows" not in python["system_prompt"]
    assert "Prefer plain assert statements over assertion helpers." in python["system_prompt"]
    assert "Name each test method after the behavior it checks." in java["system_prompt"]
