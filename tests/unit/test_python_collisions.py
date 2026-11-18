"""Testuary's collision handling over the public backend: independently
generated suites are rewritten free of collisions under the provable-safety
rules, and a suite whose rewrite is unprovable is ejected."""
from __future__ import annotations

import ast

from testuary.languages import get_backend
import pytest
from langadapter.python import source_analysis
from langadapter.sources import CollisionResolution
from testuary.languages.generic import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def resolve_module_collisions(sources, **kwargs):
    return get_backend("python").resolve_test_collisions(sources, **kwargs)


def _names(src: str) -> set:
    tree = ast.parse(src)
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.add(node.name)
    return out


def test_no_collisions_passthrough():
    a = "def test_a(): assert True\n"
    b = "def test_b(): assert True\n"
    res = resolve_module_collisions([a, b], localize_autouse=True)
    assert res.sources == (a, b)
    assert res.ejected == ()
    assert res.renames == ({}, {})


def test_colliding_test_functions_rename_later_sources():
    a = "def test_x():\n    assert 1\n"
    b = "def test_x():\n    assert 2\n"
    c = "def test_x():\n    assert 3\n"
    res = resolve_module_collisions([a, b, c])
    assert res.ejected == ()
    assert "test_x" in _names(res.sources[0])
    assert "test_x_2" in _names(res.sources[1])
    assert "test_x_3" in _names(res.sources[2])
    assert res.renames[1] == {"test_x": "test_x_2"}
    assert res.renames[2] == {"test_x": "test_x_3"}


def test_identical_definitions_dedup_to_first_copy():
    helper = "def make():\n    return 1\n"
    a = helper + "def test_a():\n    assert make() == 1\n"
    b = helper + "def test_b():\n    assert make() == 1\n"
    res = resolve_module_collisions([a, b], localize_autouse=True)
    assert res.ejected == ()
    assert res.deduped[1] == ("make",)
    assert "def make" not in res.sources[1]
    assert "def test_b" in res.sources[1]


def test_identical_assign_statement_dedups():
    a = "x = 1\n\ndef test_a():\n    assert x == 1\n"
    b = "x = 1\n\ndef test_b():\n    assert x == 1\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.deduped[1] == ("x",)
    assert "x = 1" not in res.sources[1]


def test_identical_immutable_scalar_assign_still_dedups():
    """An immutable scalar has no identity to alias, so a byte-identical
    string or float binding deduplicates as an int does."""
    a = 'PREFIX = "ds_"\n\ndef test_a():\n    assert PREFIX == "ds_"\n'
    b = 'PREFIX = "ds_"\n\ndef test_b():\n    assert PREFIX == "ds_"\n'
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.deduped[1] == ("PREFIX",)
    assert 'PREFIX = "ds_"' not in res.sources[1]


def test_identical_mutable_dict_assign_isolates_by_rename():
    """A byte-identical module-level mutable binding is not deduplicated:
    sharing one container would alias state across suites. It is renamed
    with its references, so each suite keeps its own state."""
    a = ("cache = {}\n\n"
         "def test_a():\n    cache['x'] = 1\n    assert cache == {'x': 1}\n")
    b = ("cache = {}\n\n"
         "def test_b():\n    cache['y'] = 2\n    assert cache == {'y': 2}\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.deduped[1] == ()                       # not shared
    assert res.renames[1] == {"cache": "cache_2"}
    assert "cache_2 = {}" in res.sources[1]
    assert "cache_2['y'] = 2" in res.sources[1]
    assert "cache = {}" in res.sources[0]             # first suite keeps the name


def test_identical_mutable_list_assign_isolates_by_rename():
    """A ``_LOG = []`` that two suites' autouse fixtures append to is
    isolated. Shared, each suite's test would read the other's appends and
    fail, though it passes alone."""
    a = "_LOG = []\n\ndef test_a():\n    _LOG.append(1)\n    assert _LOG == [1]\n"
    b = "_LOG = []\n\ndef test_b():\n    _LOG.append(2)\n    assert _LOG == [2]\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames[1] == {"_LOG": "_LOG_2"}
    assert "_LOG_2 = []" in res.sources[1]
    assert "_LOG_2.append(2)" in res.sources[1]


def test_module_autouse_fixtures_are_localized_to_their_source_tests():
    """In the merged module, one suite's autouse fixture does not reach the
    other suites' tests."""
    a = (
        "import pytest\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def reset_options():\n"
        "    yield\n\n"
        "def test_a():\n"
        "    assert True\n"
    )
    b = (
        "import pytest\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def reset_options():\n"
        "    yield from ()\n\n"
        "class TestB:\n"
        "    def test_b(self):\n"
        "        assert True\n"
    )

    res = resolve_module_collisions([a, b], localize_autouse=True)

    assert res.ejected == ()
    assert res.localized_autouse == (
        ("reset_options",), ("reset_options_2",)
    )
    assert "@pytest.fixture(autouse=False)" in res.sources[0]
    assert "@pytest.fixture(autouse=False)" in res.sources[1]
    assert (
        "@pytest.mark.usefixtures('reset_options')\n"
        "def test_a"
    ) in res.sources[0]
    assert (
        "@pytest.mark.usefixtures('reset_options_2')\n"
        "class TestB"
    ) in res.sources[1]


def test_deduped_autouse_fixture_is_requested_by_both_source_test_sets():
    fixture = (
        "@pytest.fixture(autouse=True)\n"
        "def reset_options():\n"
        "    yield\n\n"
    )
    a = "import pytest\n\n" + fixture + "def test_a():\n    assert True\n"
    b = "import pytest\n\n" + fixture + "def test_b():\n    assert True\n"

    res = resolve_module_collisions([a, b], localize_autouse=True)

    assert res.deduped[1] == ("reset_options",)
    assert res.localized_autouse == (
        ("reset_options",), ("reset_options",)
    )
    assert res.sources[0].count("@pytest.fixture(autouse=False)") == 1
    assert "def reset_options" not in res.sources[1]
    assert "@pytest.mark.usefixtures('reset_options')" in res.sources[0]
    assert "@pytest.mark.usefixtures('reset_options')" in res.sources[1]


def test_class_local_autouse_fixture_is_not_widened_or_rewritten():
    source = (
        "import pytest\n\n"
        "class TestScoped:\n"
        "    @pytest.fixture(autouse=True)\n"
        "    def reset(self):\n"
        "        yield\n\n"
        "    def test_value(self):\n"
        "        assert True\n"
    )

    res = resolve_module_collisions([source], localize_autouse=True)

    assert res.sources == (source,)
    assert res.localized_autouse == ()


def test_module_autouse_localization_is_not_the_default():
    source = (
        "import pytest\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def reset_options():\n"
        "    yield\n\n"
        "def test_value():\n"
        "    assert True\n"
    )

    res = resolve_module_collisions([source])

    assert res.sources == (source,)
    assert res.localized_autouse == ()


def test_pytest_reserved_global_still_dedups():
    """``pytestmark`` is read by exact name and applies module-wide, so an
    identical binding shares one copy. pytest would ignore a renamed
    ``pytestmark_2``, and the second suite would lose its mark."""
    a = ("import pytest\n"
         "pytestmark = pytest.mark.slow\n\n"
         "def test_a():\n    assert True\n")
    b = ("import pytest\n"
         "pytestmark = pytest.mark.slow\n\n"
         "def test_b():\n    assert True\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.deduped[1] == ("pytestmark",)
    assert "pytestmark = pytest.mark.slow" not in res.sources[1]


def test_module_binding_shadowing_builtin_renames_to_free_it():
    """A single-suite module-level binding that shadows a builtin is renamed so
    another merged suite calling that builtin unqualified still reaches it. The
    shadowing suite never uses the builtin it hid, so the rewrite stays green."""
    a = "id = 123\n\ndef test_a():\n    assert id == 123\n"
    b = "def test_b():\n    assert id(object()) != 0\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames[0] == {"id": "id_2"}
    assert "id_2 = 123" in res.sources[0]
    assert "assert id_2 == 123" in res.sources[0]
    assert res.sources[1] == b          # the builtin caller is untouched


def test_non_builtin_single_suite_binding_is_left_alone():
    """A module-level name that is not a builtin and lives in one suite has
    nothing to free, so it is not force-renamed."""
    a = "ds = 5\n\ndef test_a():\n    assert ds == 5\n"
    b = "def test_b():\n    assert True\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames == ({}, {})
    assert res.sources == (a, b)


def test_differing_helper_renames_with_reference_rewrite():
    a = "def make():\n    return 1\n\ndef test_a():\n    assert make() == 1\n"
    b = "def make():\n    return 2\n\ndef test_b():\n    assert make() == 2\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames[1] == {"make": "make_2"}
    assert "def make_2():" in res.sources[1]
    assert "assert make_2() == 2" in res.sources[1]
    assert "def make():" in res.sources[0]


def test_rename_survives_multibyte_char_before_identifier():
    """A non-ASCII character before a renamed identifier does not corrupt the
    rewrite. tree-sitter columns are byte offsets: slicing the line string by
    them would shift past every multi-byte character and produce source that
    does not parse, which ejects the whole suite."""
    a = "def helper():\n    return 1\n\ndef test_a():\n    assert helper() == 1\n"
    b = ("def helper():\n    return 2\n\ndef test_b():\n"
         "    msg = 'café'; assert helper() == 2\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()                       # no byte/char shift
    assert res.renames[1] == {"helper": "helper_2"}
    # the rewrite renamed the reference, kept the unicode intact, and re-parses
    assert "assert helper_2() == 2" in res.sources[1]
    assert "café" in res.sources[1]
    ast.parse(res.sources[1])


def test_helper_referencing_a_differing_global_does_not_dedup():
    """A byte-identical helper that reads a module global bound differently
    per suite is renamed, not deduplicated: one shared copy would bind the
    first suite's value, and the other suite's test would fail and be
    stripped."""
    a = "CONFIG = 1\ndef h():\n    return CONFIG\n\ndef test_a():\n    assert h() == 1\n"
    b = "CONFIG = 2\ndef h():\n    return CONFIG\n\ndef test_b():\n    assert h() == 2\n"
    res = resolve_module_collisions([a, b])
    assert "h" not in res.deduped[1]               # renamed, not deduped
    assert res.renames[1].get("h") == "h_2"
    assert res.renames[1].get("CONFIG") == "CONFIG_2"
    merged = "\n\n".join(res.sources)
    ns: dict = {}
    exec(compile(merged, "<merged>", "exec"), ns)
    ns["test_a"]()                                 # both run green
    ns["test_b"]()                                 # fails if h were shared


def test_helper_referencing_an_identical_global_still_dedups():
    """When the referenced global is bound the same in both suites, the
    helper is still deduplicated."""
    a = "CONFIG = 1\ndef h():\n    return CONFIG\n\ndef test_a():\n    assert h() == 1\n"
    b = "CONFIG = 1\ndef h():\n    return CONFIG\n\ndef test_b():\n    assert h() == 1\n"
    res = resolve_module_collisions([a, b])
    assert "h" in res.deduped[1]                    # same binding: deduped


def test_differing_fixture_renames_def_and_consuming_parameters():
    a = ("import pytest\n\n@pytest.fixture\ndef data():\n    return 1\n\n"
         "def test_a(data):\n    assert data == 1\n")
    b = ("import pytest\n\n@pytest.fixture\ndef data():\n    return 2\n\n"
         "def test_b(data):\n    assert data == 2\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert "def data_2():" in res.sources[1]
    assert "def test_b(data_2):" in res.sources[1]
    assert "assert data_2 == 2" in res.sources[1]


def test_name_as_call_keyword_renames_and_leaves_kwarg():
    """A call keyword (``f(name=...)``) addresses the callee's parameter,
    not the module binding, so the uniform rename leaves it untouched and
    the source is renamed, never ejected."""
    a = "def timeout():\n    return 1\n"
    b = ("def timeout():\n    return 2\n\n"
         "def test_b():\n    import requests\n"
         "    requests.get('http://x', timeout=3)\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames[1] == {"timeout": "timeout_2"}
    assert "def timeout_2():" in res.sources[1]
    assert "timeout=3" in res.sources[1]          # the call keyword is left alone


def test_coincidental_string_does_not_block_rename():
    """A name in an ordinary string (a message, a match pattern) is not a
    reference: the def and its references rename, the string stays verbatim,
    and the suite is not ejected."""
    a = "def helper():\n    return 1\n"
    b = ("def helper():\n    return 2\n\n"
         "def test_b():\n"
         "    assert 'helper' in 'a helper message'\n"
         "    assert helper() == 2\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.dropped_defs[1] == ()
    assert res.renames[1] == {"helper": "helper_2"}
    assert "def helper_2():" in res.sources[1]
    assert "assert helper_2() == 2" in res.sources[1]            # ref renamed
    assert "'helper' in 'a helper message'" in res.sources[1]    # string left


def test_reflective_owner_keeps_the_name_and_the_other_is_renamed():
    """A name resolved reflectively by string (getfixturevalue / getattr /
    globals()[...]) cannot be renamed without the string going stale. The
    original name therefore goes to the reflective owner and the renamable
    owner is renamed: both bodies survive, nothing is dropped."""
    a = "def helper():\n    return 1\n"
    b = ("def helper():\n    return 2\n\n"
         "def test_uses_helper(request):\n"
         "    assert request.getfixturevalue('helper')\n\n"
         "def test_independent():\n    assert 1 == 1\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert all(not d for d in res.dropped_defs)             # nothing dropped
    assert res.renames[0].get("helper", "").startswith("helper_")
    assert "def helper(" in res.sources[1]                  # holder keeps it
    assert "getfixturevalue('helper')" in res.sources[1]    # user intact
    assert "def test_independent" in res.sources[1]


def test_monkeypatch_setattr_on_source_module_renames_not_drops():
    """``monkeypatch.setattr(source_module, "x", ...)`` sets an attribute of
    the imported module under test, not a binding of this test module, so a
    colliding binding ``x`` here renames cleanly. Only reflection on this
    module's own namespace blocks a rename. The colliding binding lives in a
    ``try`` block (the ``import x`` fallback idiom), and the rename keeps
    that block valid."""
    a = "import cftime\n\ndef test_a():\n    assert cftime is not None\n"
    b = ("try:\n    import cftime\nexcept ImportError:\n    cftime = None\n\n"
         "def test_b(monkeypatch):\n"
         "    monkeypatch.setattr(some_module, 'cftime', None)\n"
         "    assert cftime is not None\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.dropped_defs[1] == ()
    assert res.renames[1] == {"cftime": "cftime_2"}
    assert "import cftime as cftime_2" in res.sources[1]      # import-in-try renamed
    assert "cftime_2 = None" in res.sources[1]                # except fallback renamed
    assert "assert cftime_2 is not None" in res.sources[1]    # reference renamed
    assert "setattr(some_module, 'cftime', None)" in res.sources[1]  # string left


def test_getattr_owner_keeps_the_name_and_the_other_is_renamed():
    """``getattr(sys.modules[__name__], "x")`` resolves a top-level binding
    of this module by string, so a rename would go stale. The reflective
    owner keeps the name, the plainly bound owner is renamed, and both
    survive."""
    a = "def helper():\n    return 1\n"
    b = ("import sys\n\n"
         "def helper():\n    return 2\n\n"
         "def test_b():\n"
         "    fn = getattr(sys.modules[__name__], 'helper')\n"
         "    assert fn() == 2\n\n"
         "def test_independent():\n    assert 1 == 1\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert all(not d for d in res.dropped_defs)
    assert res.renames[0].get("helper", "").startswith("helper_")
    assert "def helper(" in res.sources[1]
    assert "def test_independent" in res.sources[1]


def test_name_collision_with_coincidental_match_string_renames_cleanly():
    """A name that is also a common word in an unrelated ``match=`` string
    renames cleanly (alias and references) and the string stays. A
    coincidental string is not a use: nothing is ejected or dropped."""
    a = "def variables():\n    return 1\n"
    b = ("import xarray.coding.variables as variables\n\n"
         "def test_b():\n"
         "    with pytest.raises(ValueError, match='variables required'):\n"
         "        variables.bar()\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.dropped_defs[1] == ()
    assert res.renames[1] == {"variables": "variables_2"}
    assert "as variables_2" in res.sources[1]            # import alias renamed
    assert "variables_2.bar()" in res.sources[1]         # reference renamed
    assert "match='variables required'" in res.sources[1]  # string untouched


def test_local_shadow_renames_consistently_not_left_dangling():
    """A uniform rename carries a shadowing local with the binding: every
    occurrence of the identifier becomes the fresh name, so no reference is
    left half-renamed."""
    a = "def data():\n    return 1\n"
    b = ("def data():\n    return 2\n\n"
         "def test_b():\n    data_local = data()\n    assert data_local == 2\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert "def data_2():" in res.sources[1]
    assert "data_local = data_2()" in res.sources[1]
    import ast as _ast
    _ast.parse(res.sources[1])               # the rewrite re-parses


def test_test_name_in_docstring_or_message_still_renames():
    a = "def test_x():\n    assert 1\n"
    b = ('def test_x():\n    """test_x covers the edge."""\n'
         "    assert 2, 'test_x failed'\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert "def test_x_2():" in res.sources[1]


def test_helper_name_in_docstring_only_is_inert_and_renames():
    a = "def make():\n    return 1\n"
    b = ('def make():\n    """make builds the fixture value."""\n'
         "    return 2\n\ndef test_b():\n    assert make() == 2\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames[1] == {"make": "make_2"}


def test_from_import_collision_rewrites_to_alias():
    a = "def sin():\n    return 0\n\ndef test_a():\n    assert sin() == 0\n"
    b = ("from math import sin\n\n"
         "def test_b():\n    assert sin(0) == 0\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert "from math import sin as sin_2" in res.sources[1]
    assert "assert sin_2(0) == 0" in res.sources[1]


def test_dotted_import_owner_keeps_the_name_fixture_survives():
    """A dotted plain import cannot be alias-renamed, so its owner keeps
    the name and the renamable ``def os()`` owner is renamed: the fixture
    that uses ``os.path`` survives with its decorator, and nothing is
    dropped. The rewritten sources stay valid."""
    a = "def os():\n    return 1\n"
    b = ("import os.path\n\n"
         "@pytest.fixture\n"
         "def uses_os():\n"
         "    return os.path.sep\n\n"
         "def test_independent():\n    assert 1 + 1 == 2\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert all(not d for d in res.dropped_defs)
    assert res.renames[0].get("os", "").startswith("os_")
    assert "import os.path" in res.sources[1]
    assert "def uses_os" in res.sources[1]
    assert "@pytest.fixture" in res.sources[1]
    assert "def test_independent" in res.sources[1]
    ast.parse(res.sources[0])
    ast.parse(res.sources[1])


def test_dotted_plain_import_collision_renames_the_other_owner():
    """A dotted plain import cannot be alias-renamed. The original name
    stays with it and the colliding ``def os()`` in the other suite is
    renamed, so the import's dependent test survives instead of being
    dropped with a victim."""
    a = "def os():\n    return 1\n"
    b = ("import os.path\n\n"
         "def test_uses_os():\n    assert os.path.sep\n\n"
         "def test_independent():\n    assert 1 + 1 == 2\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert all(not d for d in res.dropped_defs)
    assert res.renames[0].get("os", "").startswith("os_")
    assert "import os.path" in res.sources[1]
    assert "test_uses_os" in res.sources[1]
    assert "def test_independent" in res.sources[1]


def test_colliding_test_classes_rename():
    a = "class TestFoo:\n    def test_one(self):\n        assert 1\n"
    b = "class TestFoo:\n    def test_two(self):\n        assert 2\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert "class TestFoo_2:" in res.sources[1]


def test_class_method_names_do_not_collide_across_suites():
    a = "class TestA:\n    def test_same(self):\n        assert 1\n"
    b = "class TestB:\n    def test_same(self):\n        assert 2\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames == ({}, {})


def test_syntax_error_source_is_ejected():
    a = "def test_a():\n    assert 1\n"
    b = "def broken(:\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == (1,)
    assert "syntax error" in res.eject_reasons[0]
    assert res.sources[0] == a


def test_constant_collision_renames_uses():
    a = "LIMIT = 5\n\ndef test_a():\n    assert LIMIT == 5\n"
    b = "LIMIT = 9\n\ndef test_b():\n    assert LIMIT == 9\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert "LIMIT_2 = 9" in res.sources[1]
    assert "assert LIMIT_2 == 9" in res.sources[1]


def test_try_except_import_fallback_binding_is_seen():
    a = "np = None\n\ndef test_a():\n    assert np is None\n"
    b = ("try:\n    import numpy as np\nexcept ImportError:\n    np = None\n\n"
         "def test_b():\n    assert np is None or np\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames[1].get("np") == "np_2"
    assert "import numpy as np_2" in res.sources[1]


def test_suffix_skips_taken_names():
    a = "def make():\n    return 1\n"
    b = "def make():\n    return 2\n\ndef make_2():\n    return 22\n"
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames[1]["make"] == "make_3"


def test_rewrites_reparse_and_three_way_pool_is_collision_free():
    srcs = [
        "import pytest\n\nX = 1\n\ndef helper():\n    return X\n\n"
        "def test_a():\n    assert helper() == 1\n",
        "import pytest\n\nX = 2\n\ndef helper():\n    return X * 2\n\n"
        "def test_a():\n    assert helper() == 4\n",
        "X = 1\n\ndef test_a():\n    assert X == 1\n",
    ]
    res = resolve_module_collisions(srcs)
    assert res.ejected == ()
    pool = {}
    for i, src in enumerate(res.sources):
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name not in pool, (i, node.name)
                pool[node.name] = i


def test_same_origin_imports_are_not_collisions():
    a = ("import pytest\nfrom sympy import exp, log\n\n"
         "def test_a():\n    assert exp(0) == 1\n")
    b = ("import pytest\nfrom sympy import exp, sin\n\n"
         "def test_b():\n    assert exp(1) > 2\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames == ({}, {})
    assert res.deduped == ((), ())
    assert res.sources == (a, b)


def test_identical_import_statements_never_dedup():
    a = "import pytest\n\ndef test_a():\n    assert 1\n"
    b = "import pytest\n\ndef test_b():\n    assert 2\n"
    res = resolve_module_collisions([a, b])
    assert "import pytest" in res.sources[1]
    assert res.deduped == ((), ())


def test_different_origin_import_collision_renames():
    a = "from math import sin\n\ndef test_a():\n    assert sin(0) == 0\n"
    b = ("from sympy import sin\n\n"
         "def test_b():\n    assert sin(0) == 0\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert res.renames[1] == {"sin": "sin_2"}
    assert "from sympy import sin as sin_2" in res.sources[1]


def test_within_source_duplicate_class_keeps_only_the_last():
    # A name bound twice in one source is dead code above its last binding:
    # only the last body executes, so the merge keeps exactly that one.
    src = (
        "class TestX:\n"
        "    def test_dead(self):\n"
        "        assert False\n"
        "\n"
        "class TestX:\n"
        "    def test_live(self):\n"
        "        assert True\n"
    )
    res = resolve_module_collisions([src])
    out = res.sources[0]
    assert out.count("class TestX") == 1
    assert "test_live" in out
    assert "test_dead" not in out
    assert "TestX" in res.dropped_defs[0]


def test_shadowed_duplicate_is_not_resurrected_at_the_merge():
    # The second suite defines the same class twice, its live second
    # definition byte-identical to the first suite's. The dead first
    # definition never ran in its own suite, so it does not come back under a
    # fresh name: the live copy deduplicates against suite one.
    live_class = (
        "class TestX:\n"
        "    def test_live_a(self):\n"
        "        assert True\n"
        "    def test_live_b(self):\n"
        "        assert True\n"
    )
    a = live_class + "\n\ndef test_only_a():\n    assert True\n"
    b = (
        "class TestX:\n"
        "    def test_dead(self):\n"
        "        assert False\n"
        "\n\n" + live_class
    )
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    assert "test_dead" not in res.sources[0]
    assert "test_dead" not in res.sources[1]
    assert "TestX_2" not in res.sources[1]          # no resurrection rename
    assert res.renames[1] == {}
    assert "TestX" in res.dropped_defs[1]           # the dead body dropped
    assert "TestX" in res.deduped[1]                # the live one deduped


def test_class_rename_follows_into_expected_literals():
    """A mock class whose __name__ leaks into asserted repr output keeps
    passing after a collision rename: the expected literal in the same
    source is rewritten with the class."""
    a = ("class MockSparse:\n    nnz = 3\n\n"
         "def test_repr():\n"
         "    arr = MockSparse()\n"
         "    assert f\"<{type(arr).__name__}: nnz=3>\" == \"<MockSparse: nnz=3>\"\n")
    b = ("class MockSparse:\n    nnz = 5\n\n"
         "def test_repr_b():\n"
         "    arr = MockSparse()\n"
         "    s = \"<MockSparse: nnz=5>\"\n"
         "    assert f\"<{type(arr).__name__}: nnz=5>\" == s\n")
    res = resolve_module_collisions([a, b])
    assert res.ejected == ()
    renamed = res.renames[1].get("MockSparse")
    assert renamed and renamed != "MockSparse"
    # The second source's literal followed its class. The first is untouched.
    assert f'"<{renamed}: nnz=5>"' in res.sources[1]
    assert '"<MockSparse: nnz=3>"' in res.sources[0]


def test_class_literal_rename_is_word_bounded_and_source_local():
    """A literal naming an external attribute, a longer identifier, or
    another source's class is never rewritten."""
    a = "class Mock:\n    kind = 1\n\ndef test_a():\n    assert Mock\n"
    b = ("class Mock:\n    kind = 2\n\n"
         "def test_b(monkeypatch):\n"
         "    assert \"Mocker\" != \"x\"\n"
         "    assert \"a Mock here\" == \"a Mock here\"\n")
    c = "def test_c():\n    assert \"Mock\" == \"Mock\"\n"
    res = resolve_module_collisions([a, b, c])
    renamed = res.renames[1].get("Mock")
    assert renamed and renamed != "Mock"
    assert '"Mocker"' in res.sources[1]              # longer word untouched
    assert f'"a {renamed} here"' in res.sources[1]   # whole word follows
    assert '"Mock" == "Mock"' in res.sources[2]      # other source untouched


# Several unrenamable owners of one name: the victim follows the evidence.

_REFLECT_A = ("def helper():\n    return 1\n\n"
              "def test_a_low(request):\n"
              "    assert request.getfixturevalue('helper')\n")
_REFLECT_B = ("def helper():\n    return 2\n\n"
              "def test_b_high(request):\n"
              "    assert request.getfixturevalue('helper')\n")


def test_two_unrenamable_owners_keep_the_evidence_heavier_one():
    """Both suites bind the name reflectively, so only one can hold it and
    the other is dropped. The victim is chosen by the measured evidence of
    each owner's static dependents (kills, then branches, then lines), not
    by position: here the second source's dependents carry more kills, so
    the first is dropped despite coming first."""
    evidence = [
        {"test_a_low": ["K:m1", "L:f:1"]},
        {"test_b_high": ["K:m1", "K:m2", "K:m3", "L:f:1"]},
    ]
    res = resolve_module_collisions(
        [_REFLECT_A, _REFLECT_B], evidence=evidence)
    assert res.ejected == ()
    assert "helper" in res.dropped_defs[0]
    assert not res.dropped_defs[1]
    assert "def helper(" not in res.sources[0]
    assert "def helper(" in res.sources[1]
    assert "test_b_high" in res.sources[1]


def test_two_unrenamable_owners_tie_breaks_by_position():
    """Equal evidence on both sides (or none at all) falls back to
    position: the first owner keeps the name."""
    res = resolve_module_collisions([_REFLECT_A, _REFLECT_B])
    assert res.ejected == ()
    assert not res.dropped_defs[0]
    assert "helper" in res.dropped_defs[1]
    assert "def helper(" in res.sources[0]
    assert "def helper(" not in res.sources[1]


def test_unrenamable_victim_drop_consumes_its_attached_comments():
    first = (
        "import pytest\n\n"
        "# rationale for the retained helper\n"
        "def helper():\n    return 1\n\n"
        "def test_a(request):\n"
        "    assert request.getfixturevalue('helper')\n"
    )
    second = (
        "import pytest\n\n"
        "# rationale for the helper that will be dropped\n"
        "def helper():\n    return 2\n\n"
        "def test_b(request):\n"
        "    assert request.getfixturevalue('helper')\n"
    )

    result = resolve_module_collisions([first, second])

    assert "helper" in result.dropped_defs[1]
    assert "retained helper" in result.sources[0]
    assert "will be dropped" not in result.sources[1]
    assert "def helper(" not in result.sources[1]


# Collision selection is consumer policy over the backend's public static
# observations.
#: Each contribution as the chosen owner keeps it. The second one's class
#: is renamed apart from the first one's.
KEPT = (
    "def helper():\n    return 0\n"
    "class TestGroup:\n"
    "    def test_value(self, request):\n"
    "        assert request.getfixturevalue('helper') == 0\n",
    "def helper():\n    return 1\n"
    "class TestGroup_2:\n"
    "    def test_value(self, request):\n"
    "        assert request.getfixturevalue('helper') == 1\n",
)


def _resolution(owner):
    """The expected resolution for these sources: the owner keeps ``helper``
    and its dependent test, and the other contribution loses both."""
    other = 1 - owner
    sources = ["\n", "\n"]
    sources[owner] = KEPT[owner]
    dropped = [(), ()]
    dropped[other] = ("helper",)
    return CollisionResolution(
        sources=tuple(sources), ejected=(), eject_reasons=(),
        renames=({}, {"TestGroup": "TestGroup_2"}), deduped=((), ()),
        dropped_defs=tuple(dropped), localized_autouse=(),
        choices=(("helper", owner, (other,)),), conflicts=())


@pytest.mark.parametrize(("first", "second", "owner"), [
    ({"K:a", "B:1", "B:2", "L:1"}, {"K:a", "K:b"}, 1),
    ({"K:a", "B:1", "L:1", "L:2"}, {"K:a", "B:1", "B:2"}, 1),
    ({"K:a", "B:1", "L:1"}, {"K:a", "B:1", "L:1", "L:2"}, 1),
    ({"K:a", "B:1", "L:1"}, {"K:a", "B:1", "L:1"}, 0),
    (set(), set(), 0),
])
def test_public_consumer_preserves_existing_priority_and_native_records(
        first, second, owner):
    """K records decide first, then B, then L. A tie keeps the earlier
    contribution."""
    sources = [
        f"def helper():\n    return {i}\n"
        "class TestGroup:\n"
        "    def test_value(self, request):\n"
        f"        assert request.getfixturevalue('helper') == {i}\n"
        for i in range(2)
    ]
    evidence = [{"tests.TestGroup.test_value": items} for items in (first, second)]
    assert GenericBackend("python", (".py",)).resolve_test_collisions(sources, evidence=evidence) == (
        _resolution(owner))


def test_consumer_passes_its_preference_to_the_public_operation(monkeypatch):
    calls = []
    sources = ["first", "second"]
    evidence = [{"test_a": {"K:a", "L:1"}, "test_b": {"K:a", "L:1"}},
                {"test_c": {"K:a", "K:b"}}]

    def dependents(source, binding, names):
        calls.append((source, binding, tuple(names)))
        return tuple(names)

    def resolve(received, *, preference, localize_autouse):
        assert list(received) == sources and localize_autouse
        assert preference("helper", (0, 1)) == 1
        return "resolved"

    monkeypatch.setattr(source_analysis, "collision_dependents", dependents)
    monkeypatch.setattr(source_analysis, "resolve_collisions", resolve)
    assert GenericBackend("python", (".py",)).resolve_test_collisions(
        sources, evidence=evidence, localize_autouse=True) == "resolved"
    assert calls == [("first", "helper", ("test_a", "test_b")),
                     ("second", "helper", ("test_c",))]
