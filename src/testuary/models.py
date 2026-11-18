"""Model identity, credentials and request settings."""
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING, Protocol, Tuple
from testuary import config as config_module
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import math
import re
from urllib.parse import urlsplit, urlunsplit
from testuary.settings import (
    ConfigurationError, RunSettings, Slot, _catalogue_name, _freeze, _model_credentials, _naming_dir,
    _safe, _unknown, credential_variable, phase_section,
)
from threading import RLock


def get_api_config() -> Dict[str, Any]:
    """The selected model's API configuration as the agents' clients take
    it: the run's model identity (``resolve_model_identity``), its provider
    credential and the output cap of the alias over its provider's
    catalogue section and configuration, None for none.

    Returns:
        Dict with api_url, api_key, model, provider, model_alias and
        max_output_tokens
    """
    settings = config_module.run_settings()
    identity = resolve_model_identity(settings)
    provider_catalogue = settings.models.get(identity.provider, {})
    alias = settings.models.get(identity.model_alias, {})
    fields = {**settings.sections.get(identity.provider, {}), **provider_catalogue, **alias}
    return {
        "provider": identity.provider,
        "model_alias": identity.model_alias,
        "api_url": identity.api_url,
        "api_key": resolve_provider_credential(
            identity.provider, provider_catalogue, alias, config_module.configured_environment()),
        "model": identity.model,
        "max_output_tokens": resolve_output_cap(fields.get("max_output_tokens")),
    }


_CONTROLLED_REQUEST_FIELDS = frozenset({
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls",
    "stream", "stream_options", "temperature", "top_p", "seed", "max_tokens",
    "max_completion_tokens", "max_output_tokens", "n", "reasoning_effort", "options", "think",
})
_CREDENTIAL_SUFFIXES = ("api_key", "secret", "password", "authorization", "token")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _validate_option_tree(value: Any, source: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(f"{source}: request_options keys must be text")
            if key.lower().endswith(_CREDENTIAL_SUFFIXES):
                raise ConfigurationError(f"{source}: request_options field {key!r} is credential-like")
            _validate_option_tree(item, source)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_option_tree(item, source)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigurationError(f"{source}: request_options numbers must be finite")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ConfigurationError(f"{source}: request_options values must be JSON values")


def _resolve_request_options(layers, *, reserved=()):
    controlled = _CONTROLLED_REQUEST_FIELDS.union(reserved)
    result = {}
    sources = {}
    for source, raw in layers:
        if isinstance(raw, str):
            if not raw.strip():
                result.clear()
                sources.clear()
                continue
            try:
                raw = json.loads(raw, object_pairs_hook=_unique_json_object)
            except (ValueError, RecursionError):
                raise ConfigurationError(
                    f"{source}: request_options must be a valid JSON object without duplicate keys"
                ) from None
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"{source}: request_options must be a JSON object or empty")
        try:
            _validate_option_tree(raw, source)
        except RecursionError:
            raise ConfigurationError(f"{source}: request_options must be a finite JSON tree") from None
        for key, value in raw.items():
            if key in controlled:
                raise ConfigurationError(f"{source}: request_options cannot override controlled field {key!r}")
            if value is None:
                result.pop(key, None)
                sources.pop(key, None)
            else:
                result[key] = value
                sources[key] = source
    try:
        return _freeze(result), _freeze(sources)
    except RecursionError:
        raise ConfigurationError("request_options nesting exceeds the supported JSON depth") from None


def resolve_output_cap(*caps: Any) -> int | None:
    """Combine model and phase limits. Absent, empty and zero add no cap."""
    positive = []
    for raw in caps:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        if isinstance(raw, bool) or not isinstance(raw, (str, int)):
            raise ConfigurationError("max_output_tokens must be a nonnegative integer or empty")
        try:
            value = int(raw)
        except ValueError:
            raise ConfigurationError("max_output_tokens must be a nonnegative integer or empty") from None
        if value < 0:
            raise ConfigurationError("max_output_tokens must be a nonnegative integer or empty")
        if value > 0:
            positive.append(value)
    return min(positive) if positive else None


@dataclass(frozen=True, repr=False)
class ModelIdentity:
    provider: str
    model_alias: str
    model: str
    api_url: str
    deployment: str = ""

    def export(self) -> dict[str, str]:
        """Return service identity without URL credentials or request secrets."""
        return _safe({
            "provider": self.provider, "model_alias": self.model_alias,
            "model": self.model, "api_url": self.api_url,
            "deployment": self.deployment,
        })

    def __repr__(self) -> str:
        return f"ModelIdentity({self.export()!r})"


def _text(fields: Mapping[str, Any], key: str, source: str) -> str:
    value = fields.get(key, "")
    if not isinstance(value, str):
        raise ConfigurationError(f"{source}: {key} must be text")
    return value.strip()


#: The providers Testuary configures endpoints and keys for. A catalogue
#: can add its own provider section.
CONFIGURED_PROVIDERS = ("ollama", "openai", "vllm")


def is_provider_section(catalogue: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    """Whether ``name`` is a provider section of the model catalogue: a
    section that names no provider of its own, as an alias does."""
    return name in catalogue and not str(catalogue[name].get("provider") or "").strip()


def resolve_alias_provider(model_alias: str, fields: Mapping[str, Any]) -> str:
    """Every catalogue alias names its provider. The alias name is only a
    label: no provider is inferred from it."""
    provider = _text(fields, "provider", "model alias").lower() if "provider" in fields else ""
    if not provider:
        raise ConfigurationError(
            f"model alias [{model_alias}] needs provider = {', '.join(CONFIGURED_PROVIDERS[:-1])} "
            f"or {CONFIGURED_PROVIDERS[-1]}, or a provider section of models.ini")
    return provider


def selected_model_name(api: Mapping[str, Any]) -> str:
    """The model ``[api] model`` names: an alias of models.ini or a model the
    provider serves. An empty value raises ConfigurationError."""
    selected = _text(api, "model", "[api]")
    if not selected:
        raise ConfigurationError("[api] model must name an alias of models.ini or a model name "
                                 "the provider serves")
    return selected


def resolve_raw_provider(api: Mapping[str, Any], catalogue: str = "models.ini") -> str:
    """A name outside the catalogue never implies a provider. ``catalogue``
    names the models.ini that was searched for the alias."""
    provider = _text(api, "provider", "[api]").lower()
    if not provider:
        selected = _text(api, "model", "[api]")
        raise ConfigurationError(
            f"model {selected!r} is not an alias in {catalogue}, and no [api] provider is set: "
            f"add the alias to that file, or set [api] provider for a model name the provider serves"
        )
    return provider


def resolve_provider_credential(
    provider: str, provider_fields: Mapping[str, Any], alias_fields: Mapping[str, Any],
    environment: Mapping[str, str],
) -> str:
    """Resolve default/named environment keys below explicit catalogue keys.

    Missing explicitly selected variables refuse instead of silently falling
    back. This returns request input only, which must not enter result records.
    """
    variable = ""
    catalogue_keys = []
    for fields, source in ((provider_fields, "provider catalogue"), (alias_fields, "model alias")):
        if "api_key_env" in fields:
            variable = _text(fields, "api_key_env", source)
        if "api_key" in fields:
            catalogue_keys.append(_text(fields, "api_key", source))
    credential = environment.get(credential_variable(provider), "")
    if variable:
        credential = environment.get(variable, "")
        if not credential.strip():
            raise ConfigurationError(f"api_key_env names {variable}, which is not set")
    return catalogue_keys[-1] if catalogue_keys else credential


def _selected_model(api: Mapping[str, Any], catalogue: Mapping[str, Mapping[str, Any]],
                    catalogue_name: str) -> tuple[str, str, Mapping[str, Any]]:
    """[api] model's provider, name and catalogue alias fields (empty for a
    raw API model, whose [api] provider names the provider). An alias names
    its provider. ``catalogue_name`` names the models.ini searched, for the
    refusal of a raw model without a provider."""
    selected = selected_model_name(api)
    alias = catalogue.get(selected)
    if alias is None:
        return resolve_raw_provider(api, catalogue_name), selected, {}
    return resolve_alias_provider(selected, alias), selected, alias


def resolve_model_identity(settings: RunSettings) -> ModelIdentity:
    """Resolve one model and endpoint without reading files or ambient state.

    An alias names its provider, and a raw API model needs an explicit
    provider. Explicit API endpoint/deployment fields override the
    alias, then catalogue provider defaults, then captured configuration.
    """
    api = settings.sections["api"]
    provider, selected, alias = _selected_model(api, settings.models, _catalogue_name(settings))

    provider_config = settings.sections.get(provider, {})
    provider_catalogue = settings.models.get(provider, {})
    model = _text(alias, "model", "model alias") or selected
    endpoint = _text(provider_config, "api_url", "provider configuration")
    deployment = _text(provider_config, "deployment", "provider configuration")
    for fields, source in ((provider_catalogue, "provider catalogue"), (alias, "model alias")):
        if "api_url" in fields:
            endpoint = _text(fields, "api_url", source)
        if "deployment" in fields:
            deployment = _text(fields, "deployment", source)
    if ("api", "api_url") in settings.supplied:
        endpoint = _text(api, "api_url", "[api]")
    if ("api", "deployment") in settings.supplied:
        deployment = _text(api, "deployment", "[api]")
    try:
        parts = urlsplit(endpoint)
        valid = parts.scheme.lower() in {"http", "https"} and bool(parts.hostname)
        parts.port
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError("model endpoint must be an absolute HTTP or HTTPS URL")
    return ModelIdentity(provider, selected, model, endpoint, deployment)


REQUEST_VALUES = frozenset({
    "temperature", "top_p", "seed", "max_output_tokens", "request_timeout_seconds",
    "reasoning", "reasoning_effort", "reasoning_budget_tokens",
    "preserve_reasoning_content", "reasoning_field_names", "request_options",
})
"""The request values ``catalogue_chat_model`` accepts in ``request_values``:
the request fields of a model catalogue entry without ``protocol``, which
belongs to the alias's identity. The refusal of any other key lists them."""
_REQUEST_FIELDS = REQUEST_VALUES | {"protocol"}
_SERVICE_FIELDS = frozenset({"reasoning_dialect", "reasoning_template", "chat_model"})
_INFORMATION_FIELDS = ("price_input_cache_hit", "price_input_cache_miss", "price_output")


def catalogue_problems(models: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """List unknown catalogue fields and invalid limits without including
    their private values. A model entry may set any [limits] key."""
    from testuary.agent import _limits

    limits = _limits.defaults()
    allowed = _REQUEST_FIELDS | _SERVICE_FIELDS | set(_INFORMATION_FIELDS) | set(limits) | {
        "provider", "model", "api_url", "deployment", "api_key", "api_key_env",
    }
    problems = []
    for section, fields in models.items():
        for key in fields:
            if key in limits:
                try:
                    _limits.parse(key, fields[key], source=f"model catalogue [{section}]")
                except ValueError as exc:
                    problems.append(str(exc))
            elif key not in allowed:
                problems.append(str(_unknown(f"model catalogue [{section}]", key, sorted(allowed))))
    return tuple(problems)


_PHASE_SECTIONS = {
    "starter": ("api", phase_section("starter")),
    "coverage": ("api", phase_section("coverage")),
    "mutation": ("api", phase_section("coverage"), phase_section("mutation")),
}


def _request_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if key in _REQUEST_FIELDS}


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        from configparser import ConfigParser
        if value.strip().lower() in ConfigParser.BOOLEAN_STATES:
            return ConfigParser.BOOLEAN_STATES[value.strip().lower()]
    raise ConfigurationError(f"{label} must be true or false")


def _optional_number(value: Any, label: str, *, integer: bool = False) -> float | int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ConfigurationError(f"{label} must be a finite {'integer' if integer else 'number'} or empty")
    try:
        result = int(value) if integer else float(value)
    except (ValueError, OverflowError):
        raise ConfigurationError(f"{label} must be a finite {'integer' if integer else 'number'} or empty") from None
    if (integer and isinstance(value, float)) or (isinstance(result, float) and not math.isfinite(result)):
        raise ConfigurationError(f"{label} must be a finite {'integer' if integer else 'number'} or empty")
    return result


def _optional_text(value: Any, label: str) -> str | None:
    if not isinstance(value, str):
        raise ConfigurationError(f"{label} must be text or empty")
    return value.strip() or None


#: The chat-template keys a vLLM server's template reads, by template, as
#: the published templates read them. ``efforts`` are the effort values the
#: template accepts (None: any text), and ``budget`` the integer range it
#: also accepts as an effort.
#:
#: - qwen: the hybrid Qwen3 template, enable_thinking and no effort.
#: - deepseek: DeepSeek V4 as vLLM renders it, thinking and reasoning_effort
#:   high or max (V4 ships no chat template, and its encoding_dsv4.py accepts
#:   no other effort).
#: - deepseek_v3: the DeepSeek V3.1 and V3.2-Exp templates, thinking and no
#:   effort.
#: - deepseek_v4_1: the DeepSeek V4.1 template, enable_thinking and
#:   reasoning_effort (low, medium, high, max or 1 to 100).
_VLLM_TEMPLATES = {
    "qwen": {"reasoning": "enable_thinking", "effort": None},
    "deepseek": {"reasoning": "thinking", "effort": "reasoning_effort", "efforts": ("high", "max")},
    "deepseek_v3": {"reasoning": "thinking", "effort": None},
    "deepseek_v4_1": {"reasoning": "enable_thinking", "effort": "reasoning_effort",
                      "efforts": ("low", "medium", "high", "max"), "budget": (1, 100)},
}

_TEMPLATE_CHOICES = ", ".join(_VLLM_TEMPLATES)


#: Model names whose published chat templates read the family's keys. Many
#: models of both families have templates that ignore them: QwQ, the
#: DeepSeek-R1 models and their distillations always reason, Qwen2.5 and
#: the Qwen3 Instruct, Thinking and Coder releases have one mode only. Sent
#: to those, a reasoning value would be dropped while the record states it.
_TEMPLATE_NAMES = (
    ("deepseek_v4_1", re.compile(r"deepseek[-_ ]?v4\.1(?![\d])")),
    ("deepseek", re.compile(r"deepseek[-_ ]?v4(?![.\d])")),
    ("deepseek_v3", re.compile(r"deepseek[-_ ]?v3\.[12](?![\d])")),
    # The hybrid Qwen3 models (Qwen3-8B, Qwen3-30B-A3B) read enable_thinking.
    ("qwen", re.compile(r"qwen3-\d+(\.\d+)?b(-a\d+(\.\d+)?b)?(-(fp8|awq|gptq|int4|int8|gguf)[\w-]*)?$")),
)


def _template_effort(template: str, keys: Mapping[str, Any], effort: str) -> str | int:
    """``effort`` as the template reads it, or refused when the template
    would reject it: a budget the template takes as an integer is sent as
    one."""
    allowed = keys.get("efforts")
    if allowed is None or effort in allowed:
        return effort
    low, high = keys.get("budget") or (0, -1)
    if effort.isdigit() and low <= int(effort) <= high:
        return int(effort)
    raise ConfigurationError(
        f"the {template} chat template takes reasoning_effort {', '.join(allowed)}"
        + (f" or {low} to {high}" if high >= low else ""))


def vllm_template(declared: str | None, model: str) -> str | None:
    """The chat-template family of a vLLM-served model: the alias's
    ``reasoning_template``, else the family of a model name whose published
    template reads that family's keys (``_TEMPLATE_NAMES``), else None."""
    if declared is not None:
        return declared
    name = model.lower().rsplit("/", 1)[-1]
    for family, pattern in _TEMPLATE_NAMES:
        if pattern.search(name):
            return family
    return None


def _reasoning_body(protocol: str, dialect: str | None, values: Mapping[str, Any], *,
                    template: str | None = None) -> dict[str, Any]:
    reasoning = values["reasoning"]
    effort = values["reasoning_effort"]
    budget = values["reasoning_budget_tokens"]
    if reasoning == "off" and (effort is not None or budget is not None):
        raise ConfigurationError("reasoning = off cannot be combined with an effort or budget")
    if protocol == "ollama-native":
        if budget is not None:
            raise ConfigurationError("ollama-native cannot express reasoning_budget_tokens")
        return {"think": effort if effort is not None else reasoning == "on"} if (
            effort is not None or reasoning is not None
        ) else {}
    if protocol == "openai-responses":
        if reasoning is not None or budget is not None:
            raise ConfigurationError("openai-responses supports reasoning_effort only")
        return {"reasoning": {"effort": effort}} if effort is not None else {}
    if protocol == "added":
        if any(value is not None for value in (reasoning, effort, budget)):
            raise ConfigurationError("added protocols use request_options for reasoning controls")
        return {}
    if not any(value is not None for value in (reasoning, effort, budget)):
        return {}
    if not dialect:
        raise ConfigurationError("reasoning controls need a reasoning_dialect: deepseek, vllm, openrouter or openai")
    if dialect == "deepseek":
        if effort is not None or budget is not None:
            raise ConfigurationError("deepseek supports reasoning enablement only")
        return {"thinking": {"type": "enabled" if reasoning == "on" else "disabled"}}
    if dialect == "vllm":
        result = {}
        if reasoning is not None or effort is not None:
            # The served model's chat template decides the key names: a key
            # it does not read would be ignored while the record states it.
            if template is None:
                raise ConfigurationError(
                    f"the vllm reasoning_dialect needs reasoning_template on the alias, one of "
                    f"{_TEMPLATE_CHOICES}: the served model's chat template decides the keys")
            keys = _VLLM_TEMPLATES[template]
            if effort is not None and keys["effort"] is None:
                raise ConfigurationError(f"the {template} chat template has no reasoning_effort")
            kwargs = {}
            if reasoning is not None:
                kwargs[keys["reasoning"]] = reasoning == "on"
            if effort is not None:
                kwargs[keys["effort"]] = _template_effort(template, keys, effort)
            result["chat_template_kwargs"] = kwargs
        if budget is not None:
            result["thinking_token_budget"] = budget
        return result
    if dialect == "openrouter":
        if effort is not None and budget is not None:
            raise ConfigurationError("openrouter cannot combine reasoning_effort and reasoning_budget_tokens")
        result = {}
        if reasoning is not None:
            result["enabled"] = reasoning == "on"
        if effort is not None:
            result["effort"] = effort
        if budget is not None:
            result["max_tokens"] = budget
        return {"reasoning": result}
    if reasoning is not None or budget is not None:
        raise ConfigurationError("openai supports reasoning_effort only")
    return {"reasoning_effort": effort}


@dataclass(frozen=True, repr=False)
class PhaseModelSettings:
    """Detached request resolution, without client, credential or ambient reads."""

    identity: ModelIdentity
    #: None for a model handed out to another program (``catalogue_chat_model``).
    phase: str | None
    protocol: str
    reasoning_dialect: str | None
    chat_model: str | None
    values: Mapping[str, Any]
    sources: Mapping[str, Any]
    request_options: Mapping[str, Any]
    reasoning_body: Mapping[str, Any]
    information: Mapping[str, Any] = field(default_factory=dict)
    #: The chat-template family the vllm dialect wrote the keys for.
    reasoning_template: str | None = None
    #: The directory a file named in ``chat_model`` is relative to: the
    #: directory of the file that names it, else the project directory.
    chat_model_dir: str | None = None

    def __post_init__(self) -> None:
        for key in ("values", "sources", "request_options", "reasoning_body", "information"):
            object.__setattr__(self, key, _freeze(getattr(self, key)))

    def export(self) -> dict[str, Any]:
        def detached(value):
            if isinstance(value, Mapping):
                return {key: detached(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [detached(item) for item in value]
            return value

        request_options = detached(self.request_options)
        safe_options = _safe(request_options)
        return {
            **self.identity.export(), "phase": self.phase, "protocol": self.protocol,
            "reasoning_dialect": self.reasoning_dialect,
            "reasoning_template": self.reasoning_template, "chat_model": self.chat_model,
            "values": detached(self.values), "sources": detached(self.sources),
            "request_options": safe_options,
            "request_options_known": safe_options == request_options,
            "reasoning_body": detached(self.reasoning_body),
            "information": detached(self.information),
        }

    def __repr__(self) -> str:
        return f"PhaseModelSettings({self.export()!r})"


def resolve_phase_model(settings: RunSettings, phase: str) -> PhaseModelSettings:
    """Resolve layer-before-section precedence and protocol constraints.

    Model/provider caps and phase caps have independent winning layers. The
    starter phase's seed remains a slot-level input, not a model catalogue setting.
    This resolver does not construct a client.
    """
    if phase not in _PHASE_SECTIONS:
        raise ConfigurationError("model phase must be starter, coverage or mutation")
    return _resolve_model(settings, phase, _PHASE_SECTIONS[phase])


def _resolve_model(settings: RunSettings, phase: str | None, section_names: tuple[str, ...],
                   caller: Mapping[str, Any] | None = None) -> PhaseModelSettings:
    """``resolve_phase_model``'s resolution over ``section_names``. A caller
    outside the phases (``catalogue_chat_model``) gives no phase, the
    ``[api]`` section alone and its own request values ``caller`` as the top
    layer."""
    identity = resolve_model_identity(settings)
    captured = settings.configuration_layers()
    layers: list[tuple[str, Mapping[str, Any], bool]] = []
    for section in section_names:
        defaults = captured[0][1].get(section, {})
        if defaults:
            layers.append((f"defaults:[{section}]", defaults, False))
    for label, fields in (
        (f"provider:[{identity.provider}]", settings.sections.get(identity.provider, {})),
        (f"catalogue:[{identity.provider}]", settings.models.get(identity.provider, {})),
        (f"alias:[{identity.model_alias}]", settings.models.get(identity.model_alias, {})),
    ):
        layers.append((label, fields, True))
    for label, sections in captured:
        if label == "defaults":
            continue
        for section in section_names:
            fields = sections.get(section)
            if fields is not None:
                layers.append((f"{label}:[{section}]", fields, False))
    if caller is not None:
        layers.append(("caller", caller, False))

    raw: dict[str, Any] = {"request_timeout_seconds": 600 if phase == "starter" else 3600}
    sources = {"request_timeout_seconds": "defaults:request_timeout_seconds"}
    caps = {"model_max_output_tokens": None, "phase_max_output_tokens": None}
    options = []
    information = {}
    for label, fields, model_layer in layers:
        normalized = _request_fields(fields)
        if model_layer:
            normalized.update({key: value for key, value in fields.items() if key in _SERVICE_FIELDS})
            information.update({key: fields[key] for key in _INFORMATION_FIELDS if key in fields})
        if "seed" in normalized:
            if model_layer or label.startswith("defaults:") or (phase == "starter" and label.endswith("[starter]")):
                raise ConfigurationError("seed is a setting of the coverage and mutation phases, in [api], "
                                         "[coverage] or [mutation]. The starter phase gives each starter suite "
                                         "its own seed")
            if phase == "starter":
                normalized.pop("seed")
        for key, value in normalized.items():
            if key == "request_options":
                options.append((label, value))
            elif key == "max_output_tokens":
                cap_key = "model_max_output_tokens" if model_layer else "phase_max_output_tokens"
                caps[cap_key] = resolve_output_cap(value)
                sources[cap_key] = label
            else:
                raw[key] = value
                sources[key] = label
    values = {}
    for key in ("temperature", "top_p", "seed", "request_timeout_seconds", "reasoning_budget_tokens"):
        values[key] = _optional_number(raw.get(key), key, integer=key in {"seed", "reasoning_budget_tokens"})
    if values["temperature"] is not None and values["temperature"] < 0:
        raise ConfigurationError("temperature must be nonnegative or empty")
    if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
        raise ConfigurationError("top_p must be between zero and one or empty")
    if values["request_timeout_seconds"] is None or values["request_timeout_seconds"] <= 0:
        raise ConfigurationError("request_timeout_seconds must be positive")
    if values["reasoning_budget_tokens"] is not None and values["reasoning_budget_tokens"] <= 0:
        raise ConfigurationError("reasoning_budget_tokens must be positive; use reasoning = off to disable reasoning")
    for key in ("reasoning", "reasoning_effort"):
        values[key] = _optional_text(raw.get(key, ""), key)
    if values["reasoning_effort"] == "0":
        raise ConfigurationError("reasoning_effort must be provider effort text or empty, not zero")
    if values["reasoning"] not in {None, "on", "off"}:
        raise ConfigurationError("reasoning must be on, off or empty")
    values.update(caps)
    values["max_output_tokens"] = resolve_output_cap(*caps.values())
    values["preserve_reasoning_content"] = _boolean(raw.get("preserve_reasoning_content", False), "preserve_reasoning_content")
    try:
        values["reasoning_field_names"] = field_names(raw.get("reasoning_field_names"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None
    dialect = _optional_text(raw.get("reasoning_dialect", ""), "reasoning_dialect")
    if dialect not in {None, "deepseek", "vllm", "openrouter", "openai"}:
        raise ConfigurationError("reasoning_dialect must be deepseek, vllm, openrouter, openai or empty")
    factory = _optional_text(raw.get("chat_model", ""), "chat_model")
    if factory and not all(factory.rpartition(":")[::2]):
        raise ConfigurationError("chat_model must name a file or module and its attribute, "
                                 "as item:attribute")
    protocol = _optional_text(raw.get("protocol", ""), "protocol")
    if protocol is None:
        protocol = "added" if factory else (
            "ollama-native" if phase == "starter" and urlsplit(identity.api_url).hostname == "ollama.com" else "openai-chat"
        )
        sources["protocol"] = "default:chat_model" if factory else "default:phase/endpoint"
    if protocol not in {"openai-chat", "openai-responses", "ollama-native", "added"}:
        raise ConfigurationError("protocol must be openai-chat, openai-responses, ollama-native or added")
    if (protocol == "added") != bool(factory):
        raise ConfigurationError("chat_model requires the added protocol and vice versa")
    if protocol == "ollama-native" and phase != "starter":
        raise ConfigurationError(
            "ollama-native is supported only for the starter phase" if phase else
            f"model alias [{identity.model_alias}] uses the protocol ollama-native, which serves "
            "only Testuary's starter phase. Another program can use an alias with the protocol "
            "openai-chat, openai-responses or added.")
    template = _optional_text(raw.get("reasoning_template", ""), "reasoning_template")
    if template not in {None, *_VLLM_TEMPLATES}:
        raise ConfigurationError(f"reasoning_template must be empty or one of {_TEMPLATE_CHOICES}")
    if template is not None and dialect != "vllm":
        raise ConfigurationError("reasoning_template applies to the vllm reasoning_dialect only")
    template = vllm_template(template, identity.model) if dialect == "vllm" else None
    body = _reasoning_body(protocol, dialect, values, template=template)
    reservations = {
        "deepseek": ("thinking",),
        "vllm": ("chat_template_kwargs", "thinking_token_budget"),
        "openrouter": ("reasoning",),
        "openai": ("reasoning_effort",),
    }.get(dialect, ()) if protocol == "openai-chat" else (
        ("reasoning",) if protocol == "openai-responses" else ()
    )
    if protocol == "added":
        reservations = ("base_url", "timeout", "max_retries")
    merged_options, option_sources = _resolve_request_options(options, reserved=reservations)
    sources["request_options"] = option_sources
    for key in values:
        sources.setdefault(key, "unset")
    sources["max_output_tokens"] = tuple(
        sources[key] for key, value in caps.items()
        if value is not None and value == values["max_output_tokens"]
    )
    # The model layer that set chat_model names its section: provider:[openai],
    # catalogue:[openai] or alias:[name].
    named = sources.get("chat_model", "")
    directory = _naming_dir(settings, named[named.find("[") + 1:-1], "chat_model",
                            catalogue=not named.startswith("provider:")) if factory else None
    return PhaseModelSettings(identity, phase, protocol, dialect, factory, values, sources,
                              merged_options, body, information, template, directory)


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class StarterModelClients:
    """Clients owned by one starter phase, isolated by slot seed and output token cap."""

    def __init__(self, settings: PhaseModelSettings, *, api_key: str) -> None:
        if settings.phase != "starter" or settings.protocol == "ollama-native":
            raise ConfigurationError("starter chat clients require a non-native starter protocol")
        self._settings = settings
        self._api_key = api_key
        self._clients: dict[tuple[int, int | None], BaseChatModel] = {}
        self._lock = RLock()

    def get(self, *, seed: int, max_output_tokens: int | None) -> BaseChatModel:
        key = (seed, max_output_tokens)
        with self._lock:
            if key not in self._clients:
                request = replace(self._settings, values={
                    **self._settings.values, "seed": seed, "max_output_tokens": max_output_tokens,
                })
                self._clients[key] = build_resolved_chat_model(request, api_key=self._api_key)
            return self._clients[key]


class PreparedModelClients:
    """Consume each preflight phase's clients in one matching target phase."""

    def __init__(self, scope: object) -> None:
        self._scope = scope
        self._prepared: dict[str, tuple[PhaseModelSettings, Any]] = {}
        self._lock = RLock()

    def add(self, settings: PhaseModelSettings, clients: Any) -> None:
        self._prepared[settings.phase] = (settings, clients)

    def __bool__(self) -> bool:
        return bool(self._prepared)

    def take(self, settings: PhaseModelSettings, scope: object | None) -> Any:
        if scope is not self._scope:
            return None
        with self._lock:
            prepared = self._prepared.get(settings.phase)
            if prepared is None or prepared[0] != settings:
                return None
            del self._prepared[settings.phase]
            return prepared[1]


def _mutable(value):
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def native_endpoint(api_url: str) -> str:
    """Keep the configured service base when selecting native Ollama chat."""
    try:
        parts = urlsplit(api_url)
        parts.port
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigurationError("native model endpoint must be an absolute HTTP or HTTPS URL") from None
    path = parts.path.rstrip("/")
    if not path.endswith("/api/chat"):
        if path.endswith("/v1"):
            path = path[:-3]
        path += "/api/chat"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def native_request_body(
    settings: PhaseModelSettings, messages: Sequence[Mapping[str, Any]], *, slot_seed: int,
) -> dict[str, Any]:
    """Build one native starter request without credentials or accounting."""
    if settings.protocol != "ollama-native" or settings.phase != "starter":
        raise ConfigurationError("native requests require the starter ollama-native protocol")
    if isinstance(slot_seed, bool) or not isinstance(slot_seed, int):
        raise ConfigurationError("starter slot_seed must be an integer")
    options = {"seed": slot_seed}
    for key in ("temperature", "top_p"):
        if settings.values[key] is not None:
            options[key] = settings.values[key]
    if settings.values["max_output_tokens"] is not None:
        options["num_predict"] = settings.values["max_output_tokens"]
    return {
        "model": settings.identity.model, "stream": True,
        "messages": _mutable(tuple(messages)), "options": options,
        **_mutable(settings.reasoning_body), **_mutable(settings.request_options),
    }


def resolve_chat_model_factory(model: PhaseModelSettings, secrets: Sequence[str] = ()) -> Any:
    """The factory an added provider's ``chat_model``, ``item:attribute``,
    names in ``model``, without calling it. The item is loaded by
    ``evidence.provider_extension``. A failure is a ConfigurationError that
    names the item and keeps the error's type and message, with ``secrets``
    (the request's credential) removed."""
    import sys

    from testuary.evidence import provider_extension

    chat_model = model.chat_model
    module, _, attribute = chat_model.rpartition(":")
    extension = provider_extension(model, secrets)
    imported, location = sys.modules[extension.module], extension.file
    try:
        factory = getattr(imported, attribute)
    except AttributeError:
        raise ConfigurationError(
            f"chat_model {chat_model}: module {module!r} ({location}) has no attribute {attribute!r}") from None
    if not callable(factory):
        raise ConfigurationError(
            f"chat_model {chat_model}: {attribute!r} in {location} is not callable")
    return factory


def build_resolved_chat_model(settings: PhaseModelSettings, *, api_key: str) -> BaseChatModel:
    """Construct a client only. Orchestration still owns calls and retries."""
    from langchain_core.language_models import BaseChatModel

    if settings.protocol == "ollama-native":
        raise ConfigurationError("ollama-native uses its native stream, not a chat-model client")
    values = settings.values
    sampling = {key: values[key] for key in ("temperature", "top_p", "seed") if values[key] is not None}
    if settings.protocol == "added":
        factory = resolve_chat_model_factory(settings, (api_key,))
        parameters = {
            "model": settings.identity.model, "base_url": settings.identity.api_url,
            "api_key": api_key, "timeout": values["request_timeout_seconds"], "max_retries": 0,
            **sampling, **_mutable(settings.request_options),
        }
        if values["max_output_tokens"] is not None:
            parameters["max_tokens"] = values["max_output_tokens"]
        try:
            model = factory(**parameters)
        except Exception:
            raise ConfigurationError("chat_model construction failed; check its parameters and dependencies") from None
        if not isinstance(model, BaseChatModel):
            raise ConfigurationError("chat_model must construct a LangChain BaseChatModel")
        return model
    from testuary._chat_models import build_chat_model

    body = _mutable(settings.request_options)
    if settings.phase != "starter":
        body["parallel_tool_calls"] = False
    reasoning = _mutable(settings.reasoning_body)
    extra = dict(sampling)
    if settings.phase == "starter":
        extra["stream_usage"] = True
    if settings.protocol == "openai-responses":
        if "seed" in extra:
            # The Responses SDK lacks this keyword. Compatible servers may
            # accept it in the JSON body. A rejection remains terminal.
            body["seed"] = extra.pop("seed")
        extra.update(reasoning)
    elif settings.reasoning_dialect == "openai":
        extra.update(reasoning)
    else:
        body.update(reasoning)
    if body:
        extra["extra_body"] = body
    return build_chat_model(
        model=settings.identity.model, api_url=settings.identity.api_url, api_key=api_key,
        timeout_seconds=values["request_timeout_seconds"], max_tokens=values["max_output_tokens"],
        extra_kwargs=extra, preserve_reasoning_content=values["preserve_reasoning_content"],
        reasoning_field_names=values["reasoning_field_names"],
        use_responses_api=settings.protocol == "openai-responses",
        include_legacy_output_cap=settings.protocol == "openai-chat" and settings.reasoning_dialect != "openai",
    )


_IDENTITY_FIELDS = frozenset({"provider", "model", "api_url", "deployment", "protocol"})


def _caller_values(model: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """``values`` as the caller's layer, or a ConfigurationError naming the
    first key that names the identity, a credential or no request value."""
    for key in values:
        if key in _IDENTITY_FIELDS:
            raise ConfigurationError(
                f"request_values cannot set {key!r}: the model [{model}] and its provider name it.")
        if key == "api_key_env" or key.lower().endswith(_CREDENTIAL_SUFFIXES):
            raise ConfigurationError(
                f"request_values cannot set {key!r}: credentials come from the model catalogue "
                "and the environment.")
        if key not in REQUEST_VALUES:
            raise ConfigurationError(
                f"request_values: {key!r} is no request value. Accepted keys: "
                f"{', '.join(sorted(REQUEST_VALUES))}.")
    return dict(values)


def catalogue_chat_model(model: str, *, base_dir: str | Path, records_dir: str | Path,
                         label: str, request_values: Mapping[str, Any] | None = None,
                         config_files: Sequence[str | Path] | None = None,
                         project_settings: bool = True,
                         models_files: Sequence[str | Path] | None = None,
                         provider: str | None = None,
                         ) -> BaseChatModel:
    """A LangChain chat model for your own program's model calls, for which
    ``model`` and ``provider`` resolve as ``[api] model`` and ``[api]
    provider`` do in a Testuary run in ``base_dir``: an alias of your model
    catalogue with its own provider, else, with ``provider``, a model of
    that provider with its address, protocol, request settings and key, from
    the run's settings files and catalogue, read only by path. ``config_files``,
    ``project_settings`` and ``models_files`` choose the settings files and
    the catalogue as they do for ``testuary.settings.resolve_settings``: an
    empty list reads no configuration file, a list of files reads them in
    place of the global ``config.ini``, ``project_settings`` false reads no
    ``project.ini``, and ``models_files`` names the catalogue files read in
    place of the user directory's ``models.ini``, a later file's section
    replacing the same section of an earlier one. A program whose Testuary
    runs name their settings files and catalogue passes the same choice, so
    that its models read the same files.

    ``request_values`` sets request values above the configured ``[api]``
    values, and no phase section applies. Its keys are the request values
    of models.ini that ``REQUEST_VALUES`` holds: ``temperature``, ``top_p``,
    ``seed``, ``max_output_tokens``, ``request_timeout_seconds``,
    ``reasoning``, ``reasoning_effort``, ``reasoning_budget_tokens``,
    ``preserve_reasoning_content``, ``reasoning_field_names`` and
    ``request_options``. The catalogue's output limits of the model still
    bound the output. The model's tool binding goes through the same retries
    and records.

    Each request is retried under Testuary's bundled retry settings: the
    packaged values of ``[coverage]``'s ``max_consecutive_transient_failures``,
    ``max_transient_failures_per_phase``, ``transient_first_wait_seconds``,
    ``max_transient_wait_seconds``, ``max_transient_hint_seconds`` and
    ``count_retries``, which the [settings reference](settings.md) shows. A
    caller cannot change them. The ceiling of
    ``max_transient_failures_per_phase`` counts over the model's life: once
    it is reached, the first temporary failure of a later request raises
    ``model_requests.TransportExhausted``, and a new model starts over.
    Every request sent is appended to
    ``request_records.jsonl`` in ``records_dir``: a request record with
    the key ``label``. Each created model appends its settings without
    credentials to ``model_settings.jsonl`` there, with ``label``. Both files
    only grow, so a resumed program keeps its earlier records.

    Raises ``ConfigurationError`` before any request for a name the
    catalogue does not define without ``provider``, or a ``provider`` that
    neither Testuary configures nor a provider section defines, naming the
    catalogue files, an absent file of ``models_files``, an
    ``ollama-native`` model, a credential that cannot be resolved, an empty
    ``label``, or a ``request_values`` key that names the identity, a
    credential or no request value. A request raises
    ``model_requests.TransportExhausted`` when a ceiling is reached,
    ``model_requests.TerminalRequestError`` for a terminal refusal and
    ``model_requests.ContextOverflow`` when the prompt does not fit the
    model's context. Their messages name the provider's error type and
    text without the key, and they carry no cause, since a provider's error
    can hold the request and its key. Any other error, such as a request
    record that cannot be written, is raised without the errors chained to
    it. The records hold no key either.
    """
    from testuary import model_requests
    from testuary._model_access import RecordedChatModel
    from testuary.evidence import append_jsonl
    from testuary.settings import resolve_settings

    if not isinstance(label, str) or not label.strip():
        raise ConfigurationError("label must name the caller's stage")
    values = _caller_values(model, request_values or {})
    api = {"model": model} if provider is None else {"model": model, "provider": provider}
    settings = resolve_settings(base_dir=base_dir, config_files=config_files,
                                project_settings=project_settings, models_files=models_files,
                                overrides={"api": api})
    catalogue = _catalogue_name(settings)
    if provider is None and model not in settings.models:
        raise ConfigurationError(
            f"model alias [{model}] is not defined in {catalogue}: add the alias to that file, "
            "or pass provider for a model name the provider serves")
    name = (provider or "").strip().lower()
    if provider is not None and name not in CONFIGURED_PROVIDERS and not is_provider_section(
            settings.models, name):
        raise ConfigurationError(
            f"provider {provider!r} is neither one Testuary configures "
            f"({', '.join(CONFIGURED_PROVIDERS)}) nor a provider section of {catalogue}")
    resolved = _resolve_model(settings, None, ("api",), values)
    identity = resolved.identity
    api_key = resolve_provider_credential(
        identity.provider, settings.models.get(identity.provider, {}),
        settings.models.get(identity.model_alias, {}), settings.environment)
    chat = build_resolved_chat_model(resolved, api_key=api_key)
    records_dir = Path(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    exported = {key: value for key, value in resolved.export().items() if key != "phase"}
    append_jsonl(records_dir / "model_settings.jsonl", {"label": label, **exported})
    bundled = settings.configuration_layers()[0][1].get(phase_section("coverage"), {})
    return RecordedChatModel(
        chat, identity=identity.export(), retry=model_requests.RetrySettings.from_config(bundled),
        records=records_dir / "request_records.jsonl", label=label,
        secrets=(api_key, *_model_credentials(settings)[1]))


class ModelSource(Protocol):
    """What an installed source provides."""

    def generate_suite(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: Optional[float],
        seed: int,
        timeout: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """One suite generation: the reply text and its usage, in the shape the
        starter phase's own generation returns."""

    def agent_chat_model(self, phase: str) -> Any:
        """The chat model an agent phase runs on. ``phase`` is the agent's
        phase label (``coverage`` or ``mutation``)."""


_SOURCE: Slot[ModelSource] = Slot("model source")
install, uninstall, current, installed = (
    _SOURCE.install, _SOURCE.uninstall, _SOURCE.current, _SOURCE.installed)


DEFAULT_FIELDS = ("reasoning_content", "reasoning", "reasoning_details", "thinking")
_RESERVED = frozenset({"role", "content", "name", "tool_calls", "function_call", "refusal", "audio"})


def field_names(value: str | Sequence[str] | None = None) -> tuple[str, ...]:
    """Normalize a configured list without substituting for an explicit empty list."""
    if value is None:
        return DEFAULT_FIELDS
    if isinstance(value, str):
        names = [name.strip() for name in value.split(",") if name.strip()]
    elif isinstance(value, Sequence):
        names = list(value)
    else:
        raise ValueError("reasoning_field_names must be a comma-separated string or a sequence")
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("reasoning_field_names contains an invalid field name")
        if name in _RESERVED:
            raise ValueError(f"reasoning_field_names cannot overwrite the message field {name!r}")
    if len(set(names)) != len(names):
        raise ValueError("reasoning_field_names contains duplicate names")
    return tuple(names)
