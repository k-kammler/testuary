"""Tests for the new-language command and the file extensions named when a source
file has no supported language."""

from types import SimpleNamespace

import pytest

from testuary import __main__ as cli, languages, orchestrator


def test_new_language_forwards_to_langadapters_scaffold_for_testuary(tmp_path, monkeypatch):
    """testuary new-language is LangAdapter's scaffold written for Testuary:
    the same arguments, with --for testuary added."""
    from langadapter import cli as langadapter_cli

    seen = []
    monkeypatch.setattr(langadapter_cli, "main",
                        lambda argv, prog="langadapter": seen.append(argv) or 0)
    assert cli._cmd_new_language(["lua", "--file-extension", ".lua"]) == 0
    assert cli._cmd_new_language(["lua", "--file-extension", ".lua", "--for", "othertool"]) == 0
    assert seen == [["new-language", "lua", "--file-extension", ".lua", "--for", "testuary"],
                    ["new-language", "lua", "--file-extension", ".lua", "--for", "othertool"]]


def test_source_refusal_uses_public_current_extension_discovery(tmp_path, monkeypatch):
    path = tmp_path / "target.unknown"
    path.write_text("source", encoding="utf-8")
    monkeypatch.setattr(languages, "detect_language", lambda _: "unknown")
    monkeypatch.setattr(languages, "supported_file_extensions", lambda: (".custom", ".py"))
    monkeypatch.setattr(orchestrator.config_module, "project_exec_mode", lambda: "local")
    from testuary.settings import ConfigurationError

    with pytest.raises(ConfigurationError, match="Installed: .custom, .py"):
        orchestrator._source_file_top_files(
            SimpleNamespace(source_files=str(path)), projects_root=str(tmp_path),
        )
