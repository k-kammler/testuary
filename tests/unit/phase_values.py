"""Test values for a phase's settings mapping (config.PhaseConfiguration)."""
from testuary import config


def set_phase_value(monkeypatch, mapping, key, value):
    """``mapping[key]`` reads ``value`` for the rest of the test in the run
    it was set in, over that run's settings. A run entered later reads its
    own resolved settings."""
    read = mapping._values
    scope = config.active_run_scope()

    def values():
        current = read()
        if config.active_run_scope() is scope:
            current[key] = value
        return current

    monkeypatch.setitem(vars(mapping), "_values", values)


def serve_starter_settings(monkeypatch, values):
    """config.get_agent_config gives ``values`` for the starter phase and the
    run's own settings for every other section, such as the mutation
    phase's settings the starter phase reads."""
    real = config.get_agent_config
    monkeypatch.setattr(config, "get_agent_config", lambda name, parent=None: (
        dict(values) if name == "test_initial" else real(name, parent)))
