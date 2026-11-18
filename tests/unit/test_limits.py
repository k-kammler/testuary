"""Limits on model-facing text: one registry, one marker, layered values.

Every limit is a [limits] key with a positive integer or ``unlimited``. Zero
and unknown keys are refused. Phase sections, model catalogue entries and
configuration layers resolve like the request controls. The resolved values
reach the resolved-settings record and, where they differ from the defaults,
the generation condition, so a resume with other limits is refused.
"""
from __future__ import annotations

import configparser
import contextvars
import threading

import pytest

from testuary.agent import _limits
from testuary import config as config_module
from testuary.evidence import current_condition, reuse_problems
from testuary.settings import ConfigurationError, resolve_settings

MODELS = {"chosen": {"provider": "openai", "model": "fixture", "api_url": "https://example.test/v1"}}


def _settings(tmp_path, *, models=None, files=(), **overrides):
    paths = []
    for index, sections in enumerate(files):
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_dict(sections)
        path = tmp_path / f"layer{index}.ini"
        with path.open("w", encoding="utf-8") as handle:
            parser.write(handle)
        paths.append(str(path))
    return resolve_settings(config_files=paths, base_dir=tmp_path, environment={},
                            models=MODELS if models is None else models,
                            overrides={"api": {"model": "chosen"}, **overrides})


def test_without_settings_of_its_own_each_phase_has_the_packaged_defaults(tmp_path):
    settings = _settings(tmp_path)
    for phase in (None, "coverage", "mutation"):
        assert _limits.resolve(settings, phase) == _limits.defaults()
    assert set(settings.sections["limits"]) == set(_limits.defaults())


@pytest.mark.parametrize("value", ["0", "-3", "many", "1.5", True])
def test_zero_and_non_integers_are_refused(tmp_path, value):
    with pytest.raises(ValueError, match="positive integer"):
        _settings(tmp_path, limits={"read_chars": value})


def test_unknown_limits_are_refused_with_a_suggestion(tmp_path):
    with pytest.raises(ValueError, match="The nearest known setting is 'read_chars'"):
        _settings(tmp_path, limits={"reed_chars": "5"})
    with pytest.raises(ValueError, match="unknown setting"):
        _settings(tmp_path, limits_mutation={"history_message": "5"})


def test_unlimited_resolves_to_none_for_every_key(tmp_path):
    settings = _settings(tmp_path, limits={name: "Unlimited" for name in _limits.defaults()})
    assert set(_limits.resolve(settings, "mutation").values()) == {None}


def test_phase_sections_override_within_a_layer_and_later_layers_win(tmp_path):
    settings = _settings(
        tmp_path,
        files=[{"limits": {"survivor_diffs": "4", "read_chars": "900"},
                "limits_coverage": {"survivor_diffs": "5"},
                "limits_mutation": {"read_chars": "800"}}],
        limits={"read_chars": "700"},
    )
    coverage, mutation = _limits.resolve(settings, "coverage"), _limits.resolve(settings, "mutation")
    assert coverage["survivor_diffs"] == 5 and mutation["survivor_diffs"] == 5
    # The overrides layer is later than the file, so its [limits] wins over
    # the file's [limits_mutation].
    assert coverage["read_chars"] == 700 and mutation["read_chars"] == 700
    assert _limits.resolve(settings, None)["survivor_diffs"] == 4


def test_a_model_entry_sets_limits_below_the_configuration(tmp_path):
    models = {
        "openai": {"tool_result_chars": "50000"},
        "chosen": {**MODELS["chosen"], "read_chars": "20000", "history_messages": "unlimited"},
    }
    settings = _settings(tmp_path, models=models)
    resolved = _limits.resolve(settings, "coverage")
    assert (resolved["tool_result_chars"], resolved["read_chars"], resolved["history_messages"]) == (
        50000, 20000, None)
    overridden = _settings(tmp_path, models=models, limits={"read_chars": "30000"})
    assert _limits.resolve(overridden, "coverage")["read_chars"] == 30000


def test_the_model_catalogue_reports_invalid_limits():
    from testuary.models import catalogue_problems

    problems = catalogue_problems({"chosen": {**MODELS["chosen"], "read_chars": "0",
                                              "survivor_diffs": "3"}})
    assert problems == ("model catalogue [chosen] read_chars: expected a positive integer, "
                        "or unlimited for no limit",)


def test_value_reads_the_active_settings_and_the_phase_in_scope(tmp_path):
    settings = _settings(tmp_path, limits={"timeout_suspects": "2"},
                         limits_mutation={"timeout_suspects": "unlimited"})
    assert _limits.value("timeout_suspects") == 8
    with config_module.use_run_settings(settings):
        assert _limits.value("timeout_suspects") == 2
        with _limits.phase_scope("mutation"):
            assert _limits.value("timeout_suspects") is None
            # Tool threads start from a copy of the caller's context.
            seen = []
            context = contextvars.copy_context()
            worker = threading.Thread(target=lambda: seen.append(
                context.run(_limits.value, "timeout_suspects")))
            worker.start()
            worker.join()
            assert seen == [None]

        @_limits.in_phase("coverage")
        def entry():
            return _limits.value("timeout_suspects")

        assert entry() == 2


def test_the_resolved_settings_record_holds_every_limit(tmp_path):
    settings = _settings(tmp_path, limits_mutation={"survivor_diffs": "unlimited"})
    with config_module.use_run_settings(settings):
        record = config_module.resolved_test_preset()["limits"]
    assert record["coverage"]["survivor_diffs"] == 10
    assert record["mutation"]["survivor_diffs"] == "unlimited"
    assert set(record["mutation"]) == set(_limits.defaults())


def test_changed_limits_refuse_a_resume_and_defaults_stay_comparable(tmp_path):
    def condition(**overrides):
        settings = _settings(tmp_path, **overrides)
        with config_module.use_run_settings(settings):
            return current_condition("coverage", "python", str(tmp_path), templates={
                "system_prompt": "s", "user_prompt_template": "u", "tool_messages": {}})

    plain, changed = condition(), condition(limits={"read_chars": "1000"})
    # Every limit is recorded, so a resume under other defaults is refused too.
    assert plain.export()["limits"] == {
        name: "unlimited" if default is None else str(default)
        for name, default in _limits.defaults().items()}
    assert changed.export()["limits"]["read_chars"] == "1000"
    unknown = {key: value for key, value in plain.export().items() if key != "limits"}
    assert reuse_problems(unknown, plain) == ("recorded limits are unknown",)
    assert reuse_problems(plain.export(), plain) == ()
    assert reuse_problems(changed.export(), changed) == ()
    assert reuse_problems(plain.export(), changed) == ("limits changed: read_chars",)
    assert reuse_problems(changed.export(), plain) == ("limits changed: read_chars",)
    with pytest.raises(ConfigurationError, match="limits changed: read_chars"):
        from testuary.evidence import require_matching_condition
        require_matching_condition(plain.export(), changed)


def test_a_run_that_recorded_a_limit_this_release_lacks_is_not_resumed(tmp_path):
    """A resume of a run whose recorded limits hold a limit this release
    does not have is refused by the condition check like any changed
    limit."""
    with config_module.use_run_settings(_settings(tmp_path)):
        current = current_condition("coverage", "python", str(tmp_path), templates={
            "system_prompt": "s", "user_prompt_template": "u", "tool_messages": {}})
    earlier = current.export()
    earlier["limits"] = {**earlier["limits"], "no_such_limit": "5"}
    assert reuse_problems(earlier, current) == ("limits changed: no_such_limit",)


def test_cut_text_marks_what_it_leaves_out():
    text = "".join(f"line {i}\n" for i in range(10))
    assert _limits.cut_text(text, None, what="output") == text
    assert _limits.cut_text(text, len(text), what="output") == text
    head = _limits.cut_text(text, 10, what="the output", keep="head")
    assert head == text[:10] + f"\n[... {len(text) - 10} chars of the output omitted ...]"
    tail = _limits.cut_text(text, 9, what="the output", keep="tail")
    assert tail == f"[... {len(text) - 9} chars of the output omitted ...]\n" + text[-9:]
    both = _limits.cut_text(text, 9, what="the output")
    assert both == text[:3] + f"\n[... {len(text) - 9} chars of the output omitted ...]\n" + text[-6:]
    lines = _limits.cut_text(text, 3, what="the log", unit="lines", keep="head")
    assert lines == "line 0\nline 1\nline 2\n[... 7 lines of the log omitted ...]\n"


def test_one_omission_reads_singular():
    assert _limits.marker(1, "items", "further test names") == "[... 1 further test name omitted ...]"
    assert _limits.marker(1, "chars", "the line") == "[... 1 char of the line omitted ...]"
    assert _limits.marker(1, "lines", "big.py", "read on") == "[... 1 line of big.py omitted, read on ...]"
    assert _limits.marker(2, "items", "further test names") == "[... 2 further test names omitted ...]"


def test_cut_items_returns_the_kept_items_and_the_marker():
    assert _limits.cut_items(range(3), None, what="names") == ([0, 1, 2], "")
    assert _limits.cut_items(range(3), 3, what="names") == ([0, 1, 2], "")
    assert _limits.cut_items(range(5), 2, what="test names") == (
        [0, 1], "[... 3 test names omitted ...]")


def test_limits_accept_unlimited_and_a_zero_request_limit_is_refused(tmp_path):
    import sys

    from testuary.agent._base import CodingAgent

    with pytest.raises(ValueError, match="max_requests_per_phase: expected a positive integer, or unlimited"):
        _settings(tmp_path, coverage={"max_requests_per_phase": "0"})
    limits = ("max_requests_per_phase", "max_loop_attempts", "max_input_tokens_per_phase",
              "max_output_tokens_per_phase", "max_total_tokens_per_phase",
              "max_wall_clock_seconds_per_phase")
    settings = _settings(tmp_path, coverage={name: "unlimited" for name in limits})
    with config_module.use_run_settings(settings):
        config = config_module.get_agent_config("test_coverage")
    assert config["max_requests_per_phase"] is None
    assert CodingAgent.graph_attempts(config) == sys.maxsize
    assert [config[name] for name in limits[2:]] == [0, 0, 0, 0.0]
