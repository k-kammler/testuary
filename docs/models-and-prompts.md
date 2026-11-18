# Models and prompt profiles

Testuary keeps model interaction separate from running each language's tools.
Model settings apply to the starter phase, the coverage phase and the
mutation phase. A prompt profile changes text, not tools, limits, validity
rules or how suites are measured.

`testuary init --examples` copies working examples from the installed package
into `./testuary-examples`: a client factory in one file,
a prompt profile with Python and Java variants, and per-language hints, each
selected by one `controls.ini`. Its `README.txt` lists the steps. Existing
files are never replaced.

## Select a model

Put model aliases in a private `models.ini` in your user directory, which
`testuary setup model` and `testuary init` write. [Where Testuary reads its
files](guide/configuration.md#where-testuary-reads-its-files) says how to
name another user directory.
Selecting `--config` does not change this directory. Every alias states its
service kind with `provider =`: `ollama`, `openai`, `vllm`, or a service
section of its own in `models.ini`. The alias name is only a label, and an
alias without a service kind is refused before any work. A model name used
directly instead of an alias needs `[api] provider`, because Testuary does
not assume one.

An alias that `testuary setup model` or the web interface's Models page
writes for `ollama`, `openai` or `vllm` holds the service address you
confirmed, the packaged one included. It keeps that address when a
`config.ini` sets another one in its service's section, so its key goes only
there. Only `[api] api_url`, which sets the address of every model, still
moves it. To reach such a service through a gateway, give the gateway's
address as the alias's address.

### models.ini

```ini
[local_qwen]
provider = openai
model = qwen3-coder:latest
api_url = http://localhost:11434/v1
api_key_env = LOCAL_MODEL_API_KEY
```

This selects a model service that speaks the OpenAI protocol. It does not
guarantee that a particular server and model accept every setting or support
tool calls. Set the named key variable before running, even when a local
service accepts a placeholder key. Keep keys out of configuration files,
prompt profiles and version control. `testuary tutorial` walks a model alias
on a small project without contacting a model.

### models.ini fields

A section of `models.ini` is a model alias, which `--model` or `[api] model`
selects, or a service section that an alias names with `provider =`. An
alias's field wins over its service section's. Unknown fields are refused
before work on any target starts, with suggestions for the correct spelling.

| Field | Meaning |
| --- | --- |
| `provider` | The kind of model service, as [select a model](#select-a-model) describes. |
| `model` | The model name the service knows. |
| `api_url` | The address of the service's interface. Without it, the service's section of the settings gives it, such as [`[ollama] api_url`](reference/settings.md#ollama). |
| `api_key` | The service's key itself, only in the user directory's `models.ini`. It wins over the variables. |
| `api_key_env` | The environment variable that holds the key. A run refuses a named variable that is not set. Without `api_key` and `api_key_env`, the key comes from `TESTUARY_<PROVIDER>_API_KEY`. |
| `deployment` | A label of the service deployment, for example a gateway's deployment name. It is not sent to the service. It is recorded in the run's model identity, so a resume with another `deployment` counts as changed conditions. A service's section of the settings may set it too, and a value set explicitly in `[api]` wins over every other. |
| `temperature`, `top_p`, `max_output_tokens`, `request_timeout_seconds`, `reasoning`, `reasoning_effort`, `reasoning_budget_tokens`, `protocol`, `preserve_reasoning_content`, `reasoning_field_names`, `request_options` | This model's [request controls](#request-controls). A `seed` here is refused. |
| `reasoning_dialect`, `reasoning_template` | How the reasoning controls are written for this service, as [request controls](#request-controls) describes. |
| `chat_model` | A client factory, as `ITEM:ATTRIBUTE`, which [your own model client](extending/model-provider.md) describes. |
| `price_input_cache_hit`, `price_input_cache_miss`, `price_output` | Numbers copied into the run's record of the resolved model, for cost analysis outside Testuary. Testuary computes nothing from them. |
| a key of [`[limits]`](reference/settings.md#limits) | A limit on what this model is shown, below the settings files, as [limits on what the model sees](guide/configuration.md#limits-on-what-the-model-sees) describes. |

### controls.ini

```ini
[api]
model = local_qwen

[main]
hints_file =
prompt_profile = profile.json

[starter]
temperature = 0
request_timeout_seconds = 900

[coverage]
top_p = 0.9
request_timeout_seconds = 1200
```

In the project's directory, the command is:

```bash
testuary --source-files src/module.py --model local_qwen --config controls.ini --prompt-profile profile.json
```

`--model` wins over `TESTUARY_API_MODEL`, which wins over `[api] model` in
the project's `.testuary/project.ini`, which `testuary setup project` writes.
That wins over `[api] model` in a `--config` file or the global
`config.ini`. `--api-url` or
`--proxy-port` can override the address of the selected service. One model
and one service serve the whole run. The phase sections adjust request
controls, not which model is used.

The service kind and the model name decide which service answers. The address
is recorded with the run, but a resume or reuse does not compare it, so
saved work continues through a local proxy on another port. A changed
address is logged as a warning for each reused phase. An address with a
query part still refuses a resume or reuse, because the query can route to
another deployment and is not recorded.

For request settings, the layers are, from lowest to highest: the shipped
defaults, the service section's defaults, the model alias in `models.ini`, the
settings files in their order, environment variables when they are enabled,
and the options a program or the command line passes. A higher layer wins
over lower layers. Within one layer the phase section wins over `[api]`. The
mutation phase applies the coverage section first, then its own. For example
`TESTUARY_API_TOP_P=0.8` overrides the coverage value above.

Output-token limits set for the model or its service section and for a phase combine
to the smaller one. This differs from the usual rule that the last value
wins. A zero or empty limit means that layer sets no limit. It does not
cancel the other one.

## Request controls

| Control | Meaning |
| --- | --- |
| `temperature`, `top_p` | Empty omits the field, and zero is sent. `top_p` is between 0 and 1. The value is sent as configured whatever the model name. A provider that rejects it fails the request. |
| `seed` | Optional sampling seed for the coverage phase and the mutation phase. A seed in `[starter]`, a service section or a model alias is refused. Each starter suite always uses its own [sampling seed](guide/configuration.md#sampling-seeds-of-the-starter-suites), from `seed_base` or `[main] seed`. |
| `max_output_tokens` | Nonnegative integer or empty. Positive values limit the output. |
| `request_timeout_seconds` | Positive time limit of a model request, not of the project's test runs. Empty and zero are refused. |
| `reasoning` | `on`, `off`, or empty for the provider default. |
| `reasoning_effort` | Provider-defined text or empty, not zero or a boolean. |
| `reasoning_budget_tokens` | Positive integer or empty. Use `reasoning=off`, not zero, to disable reasoning. |
| `protocol` | `openai-chat`, `openai-responses`, `ollama-native` or `added`. |
| `request_options` | JSON object of extra request-body fields. Keys merge across layers, and JSON null removes a lower key. An empty value clears the object. |

An absent setting leaves lower layers in effect. An intentional empty setting
can clear supported optional fields. `request_options` cannot override the
model, the messages, the tools, streaming, the controls in this table or the
keys that Testuary sets itself.

Without further settings, no request carries a temperature, the coverage
and mutation phases send no seed, and the request timeouts are those
[`[coverage]` `request_timeout_seconds`](reference/settings.md#coverage)
states. Without a protocol
selection, the starter suites use the native Ollama protocol for `ollama.com`
and the OpenAI-compatible chat protocol otherwise. The two loops use the
OpenAI-compatible chat protocol. The native Ollama protocol is used only for
starter suites, and its `/api/chat` address is derived from the configured
address.

Reasoning mappings are explicit:

| Protocol and dialect | Supported controls |
| --- | --- |
| `openai-chat` / `deepseek` | Enablement only. |
| `openai-chat` / `vllm` | Enablement and token budget, and effort where the model's chat template reads one, with the notes after the table. |
| `openai-chat` / `openrouter` | Enablement, effort or a token budget. Effort and a token budget cannot be combined. |
| `openai-chat` / `openai` | Effort only. |
| `openai-responses` | Effort only. |
| `ollama-native` | Enablement or effort, no token budget. |

With the `vllm` dialect, enablement and effort go into keys of the served
model's chat template, chosen by `reasoning_template`:

- `qwen` writes `enable_thinking`, for the hybrid Qwen3 models.
- `deepseek_v3` writes `thinking`, for DeepSeek V3.1 and V3.2-Exp.
- `deepseek` writes `thinking` and `reasoning_effort` high or max, for
  DeepSeek V4. That model ships no chat template, and its reference encoding
  accepts no other effort. No vLLM documentation says how a server passes
  `thinking` to it, so check that yours does.
- `deepseek_v4_1` writes `enable_thinking` and `reasoning_effort`, which its
  template takes as low, medium, high, max or 1 to 100, for DeepSeek V4.1.

An effort the template does not read is refused. Without
`reasoning_template`, these models select their template from the model
name. Any other model refuses enablement and effort: the templates of QwQ,
DeepSeek-R1 and its distillations, Qwen2.5 and the separate Qwen3 Instruct and
Thinking releases ignore these keys. Declare `reasoning_template` when the server
renders with another template than the model's own.

Reasoning controls with the OpenAI-compatible chat protocol need
`reasoning_dialect` in the service section or the model alias, and with the
`vllm` dialect `reasoning_template` there too. Unsupported or
contradictory controls are refused before any work, not silently dropped.
Testuary writes each mapping as the table and its notes state, but whether your server
and model accept it is for you to check, for example with one small run. A
request the server rejects ends the
phase at once, with the status `failed_model_request` for the starter suites
or `agent_error` in the coverage or mutation phase. Other targets continue unless
the run stops.
Testuary does not turn reasoning off to make a request succeed.

## Own clients and prompt profiles

A client library other than the OpenAI-compatible one is added as
[your own model client](extending/model-provider.md). Prompt text is changed with a
[prompt profile](extending/prompt-profile.md), and guidance is added with
[hints](extending/hints.md). Reusing earlier results is refused when the
model, the service, the prompts, the
[limits on what the model sees](guide/configuration.md#limits-on-what-the-model-sees),
the measurement settings or the language package that produced them differ
or are unknown. Testuary does not regenerate paid work by itself. Changing
only the model request timeout and the retry settings, or a new key, does
not make earlier results unusable. A changed suite-run or mutation timeout
does.

## Use your models from your own program

Your own program can send its model calls to a model you configured for
Testuary, under the same alias, without configuring it again.
`testuary.models.catalogue_chat_model` returns a LangChain chat model for
an alias of your catalogue, or, when you name its `provider`, for a model
that provider serves, at the provider's address and with its key. It
resolves the model as `[api] model` and `[api] provider` resolve in a
Testuary run in the directory you give, retries temporary failures as
Testuary does, and
appends every request to `request_records.jsonl` and each model's settings
to `model_settings.jsonl` in the directory you name, with your label. It
returns the model, not the key: the model's fields, representation,
records and errors hold no key.

```python
from testuary.models import catalogue_chat_model

model = catalogue_chat_model(
    "local_qwen", base_dir=".", records_dir="records", label="summary",
    request_values={"temperature": 0.2})
reply = model.invoke("Summarize this test report.")
coder = catalogue_chat_model(
    "qwen3-coder:30b", provider="ollama", base_dir=".", records_dir="records",
    label="review")
```

A program whose own Testuary runs read settings files and model catalogue
files of its choosing, through `config_files`, `project_settings` and
`models_files` of `testuary.settings.resolve_settings`, passes the same
values to `catalogue_chat_model`. Its models then read the same files as
its runs.

Such a program lists and changes its catalogue files with
`testuary.setup.models`, `set_model` and `remove_model`, given the same
list as `models_files`. Each model then names the file its section comes
from (`file`), and a change goes to the list's last file, which it creates
when it does not exist. A change to a model an earlier file defines writes
the model's whole section into the last file, where it takes the place of
the earlier one, and a removal takes only a section of the last file away.
No key is written there: a key is refused, and a section whose key an
earlier file holds, such as your own `models.ini`, needs the variable that
holds it (`api_key_env`). An address that holds a credential is not
copied. Every change, to your own `models.ini` too, is refused when a model
a run resolved before would no longer resolve.

The [Python API reference](reference/api.md#models-for-your-own-program)
lists the request values a call can set, the retry settings it uses, the
errors it raises and the records. To test your program without a network,
name an alias in a test catalogue whose client factory returns a
scripted LangChain chat model, as
[your own model client](extending/model-provider.md#trying-the-client-in-a-run) shows. Its requests go
through the same retries and records.

## Inspect and troubleshoot

The resolved records describe the controls that were configured and sent, not
which ones a server honored. The prompt sources used are recorded with each
generation.
Request records keep the model name the service returned only when the
service supplies it. They do not put the requested alias in its place. Unknown usage remains unknown.

[Result files](reference/result-files.md#generation-records-and-the-settings-record)
describes `resolved_settings.json`, the generation origin, the generation
records and the functions that read them.

[Configuration](guide/configuration.md#run-settings) describes
`RunSettings.export()`. Configuration problems
refuse the command with exit status 2. `testuary check` confirms the
configuration, not that the model service or the project's tests work.

A language needs no Testuary code. The words the prompts use for a language,
such as how its tests are declared, come from its LangAdapter adapter, as
[Make a language ready for Testuary](extending/languages.md) describes. These
words are model input: changing them changes what the model is asked.

Request diagnostics redact known captured credentials and the user and query
parts of endpoint addresses. A model that `catalogue_chat_model` returns
raises its errors without the provider's error attached, since that error
can hold the request and its key. Treat
complete external tracebacks and third-party SDK logs as potentially sensitive.
This formatting does not sanitize arbitrary model responses or third-party output.

| Symptom | Check |
| --- | --- |
| An alias or service kind is unknown | Its exact name, its `provider =` and the `models.ini` in the user directory. |
| Reasoning controls are refused | The chosen protocol, dialect and supported combination above. Do not remove controls silently. |
| A setting appears absent in the request | The resolved layer, intentional empty values and protocol support. Zero differs from empty. |
| A client factory cannot be imported or bound | Its installed distribution, factory arguments and `bind_tools` support. Constructors must not send requests. |
| Reuse refuses generation conditions | The original model, service and profile inputs. A changed address alone is a warning, not a refusal. Use a separate run for changed conditions rather than overwriting paid work. |
| A selected profile or hints file is missing | Its path, which starts where [relative paths](guide/configuration.md#relative-paths) says, and its format, before starting a run. |
