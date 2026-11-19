"""Settings parity between the two editing agents.

The mutation section inherits the coverage section, so every shared key
resolves to the same value unless it is in the allow-list below.
"""
from __future__ import annotations

import pytest

from testuary import config

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


#: Shared keys that may differ. full_after_selective_goal is computed from
#: each section's catalogue, and each phase has its own goal keys.
_ALLOWED_DIVERGENT: frozenset = frozenset({
    "full_after_selective_goal",
    "goal", "minimum", "restore_count", "extra_measurements_after_goal",
})


@pytest.fixture()
def resolved_configs(tmp_cwd, in_run):
    in_run()
    return (
        config.get_agent_config("test_coverage"),
        config.get_agent_config("test_mutation", parent="test_coverage"),
    )


def test_shared_config_keys_resolve_identically(resolved_configs):
    cov, mut = resolved_configs
    shared = set(cov) & set(mut)
    assert shared, "the two agent configs share no keys. The parent link broke"
    divergent = {
        key: (cov[key], mut[key])
        for key in sorted(shared)
        if cov[key] != mut[key] and key not in _ALLOWED_DIVERGENT
    }
    assert not divergent, (
        "shared config keys diverge outside the allow-list: "
        f"{divergent}"
    )


def test_allow_list_entries_are_live_shared_keys(resolved_configs):
    """Every allow-list entry names a key both sections carry, so no stale
    entry can hide a divergence."""
    cov, mut = resolved_configs
    shared = set(cov) & set(mut)
    dead = sorted(k for k in _ALLOWED_DIVERGENT if k not in shared)
    assert not dead, f"allow-list entries no longer shared: {dead}"


def test_only_the_mutation_config_names_the_mutation_engine(resolved_configs):
    """The coverage section holds no engine, so its settings name none, and
    the mutation section names Cosmic Ray."""
    cov, mut = resolved_configs
    assert "engine" not in cov and mut["engine"] == "cosmic_ray"
