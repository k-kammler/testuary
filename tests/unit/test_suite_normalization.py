from __future__ import annotations

import pytest

from testuary.languages import get_backend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _backend():
    backend = get_backend("python")
    assert backend is not None
    return backend


def test_module_definition_normalization_keeps_last_binding():
    source = (
        "def test_value():\n"
        "    assert False\n"
        "\n"
        "def test_value():\n"
        "    assert True\n"
    )
    result = _backend().normalize_shadowed_definitions(source)

    assert result.parse_ok
    assert result.changed
    assert result.source == "\ndef test_value():\n    assert True\n"
    assert len(result.removed_bindings) == 1
    removed = result.removed_bindings[0]
    assert removed.scope == ()
    assert removed.name == "test_value"
    assert (removed.removed_start_line, removed.removed_end_line) == (1, 2)
    assert (removed.retained_start_line, removed.retained_end_line) == (4, 5)
    assert result.before_sha256 != result.after_sha256


def test_normalization_removes_the_decorated_definition_as_one_span():
    source = (
        "@pytest.mark.slow\n"
        "@pytest.mark.parametrize('value', [1])\n"
        "def test_value(value):\n"
        "    assert False\n"
        "\n"
        "def test_value():\n"
        "    assert True\n"
    )
    result = _backend().normalize_shadowed_definitions(source)

    assert result.source == "\ndef test_value():\n    assert True\n"
    removed = result.removed_bindings[0]
    assert (removed.removed_start_line, removed.removed_end_line) == (1, 4)


def test_class_method_normalization_keeps_last_method():
    source = (
        "class TestValues:\n"
        "    def test_value(self):\n"
        "        assert False\n"
        "\n"
        "    def test_value(self):\n"
        "        assert True\n"
    )
    result = _backend().normalize_shadowed_definitions(source)

    assert result.parse_ok
    assert result.changed
    assert "assert False" not in result.source
    assert result.source.count("def test_value") == 1
    assert result.removed_bindings[0].scope == ("TestValues",)


def test_same_method_name_in_distinct_classes_is_an_exact_noop():
    source = (
        "class TestA:\n"
        "    def test_value(self):\n"
        "        assert 1\n"
        "\n"
        "class TestB:\n"
        "    def test_value(self):\n"
        "        assert 2\n"
    )
    result = _backend().normalize_shadowed_definitions(source)

    assert result.parse_ok
    assert not result.changed
    assert result.source is source
    assert result.before_sha256 == result.after_sha256
    assert result.removed_bindings == ()


def test_helpers_and_test_classes_are_normalized_too():
    source = (
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def helper():\n"
        "    return 2\n"
        "\n"
        "class TestX:\n"
        "    def test_dead(self):\n"
        "        assert False\n"
        "\n"
        "class TestX:\n"
        "    def test_live(self):\n"
        "        assert helper() == 2\n"
    )
    result = _backend().normalize_shadowed_definitions(source)

    assert "return 1" not in result.source
    assert "test_dead" not in result.source
    assert [item.name for item in result.removed_bindings] == [
        "helper", "TestX",
    ]


def test_three_bindings_remove_all_but_the_last_and_are_idempotent():
    source = (
        "def test_value():\n    assert 1\n\n"
        "def test_value():\n    assert 2\n\n"
        "def test_value():\n    assert 3\n"
    )
    first = _backend().normalize_shadowed_definitions(source)
    second = _backend().normalize_shadowed_definitions(first.source)

    assert len(first.removed_bindings) == 2
    assert first.source.count("def test_value") == 1
    assert "assert 3" in first.source
    assert second.parse_ok
    assert not second.changed
    assert second.source == first.source


def test_parse_failure_is_explicit_and_does_not_rewrite():
    source = "def test_broken(:\n"
    result = _backend().normalize_shadowed_definitions(source)

    assert not result.parse_ok
    assert not result.changed
    assert result.source == source
    assert result.parse_error
    assert result.removed_bindings == ()
    assert result.before_sha256 == result.after_sha256
