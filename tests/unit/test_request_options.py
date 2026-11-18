"""Additional request fields have deterministic merging and safe refusals."""
from collections.abc import Mapping
import math
import sys
import traceback

import pytest

from testuary.settings import ConfigurationError


def _merge(*layers, reserved=()):
    from testuary.models import _resolve_request_options
    return _resolve_request_options(layers, reserved=reserved)[0]


def test_options_merge_by_key_with_null_deletion():
    result = _merge(
        ("provider", '{"top_k":20,"min_p":0.1}'),
        ("alias", {"top_k": 30, "new_field": {"enabled": False}}),
        ("config[0]", '{"top_k":null,"min_p":0,"stop_strings":[]}'),
    )
    assert result == {"min_p": 0, "new_field": {"enabled": False}, "stop_strings": ()}


def test_an_empty_setting_clears_options_but_an_empty_object_adds_no_keys():
    assert _merge(("provider", {"top_k": 20}), ("environment", " ")) == {}
    assert _merge(("provider", {"top_k": 20}), ("environment", "{}")) == {"top_k": 20}
    assert _merge(("provider", {"top_k": 20}), ("environment", ""),
                  ("overrides", {"min_p": 0})) == {"min_p": 0}


def test_only_top_level_null_is_a_key_deletion():
    assert _merge(("provider", {"sampling": {"top_k": 20}}),
                  ("alias", {"sampling": {"threshold": None}})) == {"sampling": {"threshold": None}}


def test_merged_options_are_deeply_detached_and_read_only():
    fields = {"sampling": {"choices": [{"enabled": True}]}}
    result = _merge(("alias", fields))
    fields["sampling"]["choices"][0]["enabled"] = False
    assert isinstance(result, Mapping)
    assert result["sampling"]["choices"][0]["enabled"] is True
    with pytest.raises(TypeError):
        result["sampling"]["choices"][0]["enabled"] = False


@pytest.mark.parametrize("value", [None, 0, False, "[]", '"text"', "null", "not-json", []])
def test_options_must_be_an_object_or_an_empty_setting(value):
    with pytest.raises(ConfigurationError, match="alias.*request_options"):
        _merge(("alias", value))


@pytest.mark.parametrize("key", [
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream",
    "stream_options", "temperature", "top_p", "seed", "max_tokens", "max_completion_tokens",
    "n", "reasoning_effort", "options", "think",
])
def test_controlled_request_fields_cannot_be_options_even_when_deleted(key):
    for value in (1, None):
        with pytest.raises(ConfigurationError, match=key):
            _merge(("alias", {key: value}))


@pytest.mark.parametrize("key", ["thinking", "reasoning", "chat_template_kwargs", "base_url"])
def test_protocol_and_dialect_reserved_names_can_be_added(key):
    with pytest.raises(ConfigurationError, match=key):
        _merge(("alias", {key: {}}), reserved={key})


@pytest.mark.parametrize("key", ["api_key", "privateSecret", "password", "Authorization", "access_token"])
@pytest.mark.parametrize("nested", [False, True])
def test_credential_like_names_are_refused_recursively_without_their_values(key, nested):
    value = {key: "private-credential-sentinel"}
    if nested:
        value = {"metadata": [value]}
    with pytest.raises(ConfigurationError, match="credential") as caught:
        _merge(("alias", value))
    assert "private-credential-sentinel" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("value", [
    '{"top_k":1,"top_k":2}', '{"metadata":{"x":1,"x":2}}',
    '{"threshold":NaN}', '{"threshold":Infinity}', '{"threshold":1e999}',
    {"threshold": math.nan}, {"threshold": math.inf}, {1: "value"},
    {"metadata": {False: "value"}}, {"metadata": {"value": object()}},
])
def test_non_json_or_ambiguous_values_are_refused(value):
    with pytest.raises(ConfigurationError, match="request_options"):
        _merge(("alias", value))


def test_overriding_an_invalid_lower_layer_does_not_hide_its_error():
    with pytest.raises(ConfigurationError, match="provider.*credential"):
        _merge(("provider", {"api_key": "private"}), ("alias", ""))


def test_cyclic_programmatic_options_are_refused_as_non_json():
    value = {}
    value["self"] = value
    with pytest.raises(ConfigurationError, match="request_options.*JSON tree"):
        _merge(("alias", value))


def test_excessive_programmatic_nesting_has_an_actionable_refusal():
    value = {"value": 1}
    for _ in range(sys.getrecursionlimit() + 50):
        value = {"nested": value}
    with pytest.raises(ConfigurationError, match="request_options.*(JSON tree|JSON depth)"):
        _merge(("alias", value))
