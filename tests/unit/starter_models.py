"""The starter generator called as the starter phase calls it."""
from __future__ import annotations

import os

from testuary.models import resolve_phase_model
from testuary.models import StarterModelClients
from testuary import starters
from testuary.settings import RunSettings


def starter_settings(api, **fields):
    """Run settings whose model alias is the model ``api`` names."""
    model = {"provider": "openai", "model": api["model"], "api_url": api["api_url"],
             "api_key": api["api_key"], **fields}
    return RunSettings(environment=dict(os.environ), sections={"api": {"model": "explicit"}},
                       models={"explicit": model})


def starter_model(api, *, temperature=None, timeout=None, max_output_tokens=None, **fields):
    """The starter phase's resolved settings for the model ``api`` names and,
    for a chat protocol, its clients, as keyword arguments of the generator.
    A temperature of None sends none."""
    fields["temperature"] = "" if temperature is None else temperature
    if timeout is not None:
        fields["request_timeout_seconds"] = timeout
    if max_output_tokens is not None:
        fields["max_output_tokens"] = max_output_tokens
    phase = resolve_phase_model(starter_settings(api, **fields), "starter")
    clients = (None if phase.protocol == "ollama-native"
               else StarterModelClients(phase, api_key=api["api_key"]))
    return {"model_settings": phase, "model_clients": clients}


def generate(api, messages, *, seed, temperature=None, timeout=None, max_output_tokens=None,
             **kwargs):
    """One starter generation for the model ``api`` names."""
    return starters._default_generate(
        api, messages, seed=seed, **starter_model(
            api, temperature=temperature, timeout=timeout, max_output_tokens=max_output_tokens),
        **kwargs)
