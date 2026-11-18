# Your own model client

A model service that speaks the OpenAI chat protocol needs only a model alias
in `models.ini` in your user directory, as the [models guide](../models-and-prompts.md) shows. When
the model needs a different client library, add a client factory: a Python
function, named by `chat_model`, that returns the chat model Testuary sends
its requests through.

## Client factories

The chat model a client factory returns is a LangChain chat model. The
model alias names the factory with
`chat_model = ITEM:factory`. ITEM is a Python file, such as `my_chat.py`, or the name
of an installed module, such as `my_package.chat`. A file needs no
installation. [Relative paths](../guide/configuration.md#relative-paths)
says where a relative file name starts. The text before the last colon is the
item, so a Windows path keeps its drive letter.

`chat_model` selects the protocol `added`, and `protocol = added` states it.
Any other protocol together with `chat_model` is refused. A `chat_model` in a
service's section, or in its variable such as `TESTUARY_OPENAI_CHAT_MODEL`,
applies to every alias of that service.

The `provider =` field names the model service, not the factory. For a
service that is not one of the kinds Testuary configures, `ollama`, `openai`
and `vllm`, name a section of its own as the provider:

```ini
[my_service]
api_url = https://service.example/api

[my_alias]
provider = my_service
model = my-model
api_key_env = MY_SERVICE_API_KEY
chat_model = my_chat.py:factory
```

[The fields of models.ini](../models-and-prompts.md#modelsini-fields)
explains `api_url`, `api_key_env` and the other fields.

Testuary loads the item once and calls the function with these keyword
arguments: `model`, `base_url`, `api_key`, `timeout` in seconds and
`max_retries=0`, then `temperature`, `top_p`, `seed` (the sampling seed of a
starter suite) and `max_tokens` (the output-token limit in effect) when they
are set. `api_key` is an empty string when no key is configured. Each key of
the alias's `request_options` arrives as a keyword argument of its own.
`request_options` cannot set a keyword that Testuary passes itself.

The returned model must:

- be a LangChain `BaseChatModel`,
- stream its answers for the starter suites,
- support `bind_tools` for the coverage phase and the mutation phase, which
  pass only a list of LangChain tools, and return a runnable model,
- accept system messages after the first one, and tool messages,
- report each answer's token counts in `usage_metadata`, in a streamed answer
  too, since Testuary counts the tokens from it,
- send no request while it is created or bound to tools.

Testuary passes `max_retries=0` and keeps it that way, because Testuary itself
retries failed requests and counts them. Reasoning settings for a client
factory's model go in the alias's `request_options`, not in the built-in
reasoning controls.

Each client serves one target and one phase, and each starter suite has its own
client. A request sent again after a connection error reuses the same client.
Do not rely on state shared between clients, or on keywords given when tools
are bound overriding those given when the client is created. Each
repetition's [settings record](../reference/result-files.md#generation-records-and-the-settings-record),
`resolved_settings.json`, holds the factory's file as `extension_file` and
its SHA-256 as `extension_sha256`, and for an installed module the version of
its package as `extension_version`, which Testuary reads from the package's
installed metadata. A resume accepts saved work only while these are
unchanged, and its refusal names the client factory's file that changed. An
installed module goes into the Python environment Testuary runs in, not into
the virtual environment of the project under test.

## What a request carries and expects

A starter request carries no tools. Its answer holds the test file in a
fenced code block. Testuary takes the last block fenced with the language's
tag, such as `python`, else the last block between plain fences. An answer
that ends before the closing fence holds no test file.

A request of the coverage phase or the mutation phase carries the tools of
the current step and the phase's result tool, `emit_coverage_test_result` or
`emit_mutation_test_result`, which every step offers. Each tool carries its
description and its argument schema. The first request offers `read_tests`,
the second the phase's measure tool, `measure_coverage` or
`measure_mutation`. Later requests offer `read_source`, `read_tests` and
`edit_tests`, and after an edit `run_tests` as well. After a passing run of
the edited tests, they offer `read_source`, `read_tests` and the measure
tool, without `edit_tests` and `run_tests`, until the measurement. Only when
that passing run carried a STANDS ALONE note is `edit_tests` offered with the
measure tool, so that the tests the note names can be fixed first. Such a
note names tests that leave the module under test or a library setting
changed, or that fail in reverse order.

`read_tests` and `read_source` take `start_line` and `end_line`.
`edit_tests` takes `old_text` and `new_text`, or `edits`, a list of such
pairs, each with an optional `replace_all`. `run_tests` and the measure tool
take no arguments. The model answers with tool calls. Testuary runs each
call and sends its result back as a tool message. A call of a tool the step
does not offer is answered with a tool message that says what makes the tool
available, and the phase continues. Each request ends with a system message
that states where the phase stands.

The phase's final answer is a call of its result tool. Its required arguments are
`test_file_name`, `explanation` and `targeted_functions` in the coverage
phase, and `mutation_test_file_name`, `explanation` and `targeted_mutants` in
the mutation phase. `test_file_name` and `mutation_test_file_name` hold the
name of the working test file, as the phase's first prompt names it, such as
`test_sign.py`. Testuary ends the phase itself once a measurement
reaches the phase's `goal` and `extra_measurements_after_goal` further
measurements have run, or, with `allow_early_stop`, which is on by default,
after `max_measurements_without_improvement` measurements without
improvement. At the goal, the end waits while the latest passing run carries
a STANDS ALONE note. The tool message of
the ending measurement asks for the result call, and the call is accepted.
An earlier result call is accepted only when the last test run passed, the
current tests are measured, and the goal is met or, with `allow_early_stop`,
the measurements stopped improving. Otherwise the call is refused: a tool
message answers the call, and the reason follows as a human message. With
`allow_early_stop`, once `max_final_answers_without_measurement`
result calls in a row came without a new measurement between them, the phase
ends at the last of them. When Testuary ends the phase at the goal, after the
stall or at this cap, the phase's result is the one Testuary builds from its
own records, not the call's arguments.
[The settings reference](../reference/settings.md#coverage) lists these
settings. A reply without a tool call counts as a final answer that is not
usable: Testuary tells the model why, and the phase continues within its
request limit.

## The example

`testuary init --examples` copies a complete client factory into
`testuary-examples`. Testuary's tests resolve these files unchanged.

The factory, `testuary-examples/testuary_example_chat.py`, wraps the
OpenAI-compatible client:

```python
--8<-- "src/testuary/_templates/examples/testuary_example_chat.py"
```

The alias in `testuary-examples/models.ini` names the factory's file, which
lies beside it:

```ini
--8<-- "src/testuary/_templates/examples/models.ini"
```

Set the key variable the alias names, `LOCAL_MODEL_API_KEY`, even for a local
endpoint that accepts any key. Then run the quick check. `controls.ini`, also
in the examples directory, selects the alias `example_added`, the example
prompt profile and the hints. `--home .` makes Testuary read the examples'
`models.ini` in place of the one in your user directory. Run the command in
`testuary-examples`. `--no-project-settings` keeps a project's settings file
from applying when that directory lies inside a project:

```bash
export LOCAL_MODEL_API_KEY=<key>
testuary check --home . --config controls.ini --no-project-settings --source-dir /path/to/project \
  --python-executable /path/to/project-env/bin/python --source-files package/module.py
```

In PowerShell, set the variable with `$env:LOCAL_MODEL_API_KEY = "<key>"`,
write the command on one line and name the interpreter as
`C:\path\to\project-env\Scripts\python.exe`.

The quick check loads the factory's file, so the file's top-level code runs,
and reports a file or module that cannot be found, fails to load or lacks the
factory. Its default report does not name the factory. With
`--display detailed`, each phase's model settings item reads
`Selected: added`, and its reason says that `the client factory imports`. Neither the
quick check nor `testuary check --verify` calls the factory, creates a client
or contacts the model.

## Trying the client in a run

A run creates the clients. Before any phase runs, it calls the factory for
each phase the run generates in, each phase up to `--stop-after-phase` that
it does not reuse, and binds the tools of the coverage phase and the mutation
phase among them. A client that cannot be created or bound refuses the run
with exit status 2 before any request is sent. The refusal does not show the
factory's own error, since that error can hold the key. Call the factory
yourself in Python, with the same keyword arguments, to see it.

To try the client without a model service, set the alias's `api_url` to a
local server you control, which reaches the factory as `base_url`, or let
the factory return a model with scripted answers. This factory returns a
scripted model for a target `sign.py` whose function `sign(x)` returns `-1`
below zero and `1` otherwise. It answers a starter request with a test of a
positive value. In the coverage phase it chooses its call by the tools a
step offers: it reads the tests, measures, adds a test of a negative value
with one edit, runs the tests and measures again. That measurement reaches
the goal, and its result call is accepted.

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

FENCE = "`" * 3
STARTER = f"{FENCE}python\nfrom sign import sign\n\n\ndef test_positive():\n    assert sign(2) == 1\n{FENCE}"
EDIT = {"old_text": "def test_positive():",
        "new_text": "def test_negative():\n    assert sign(-2) == -1\n\n\ndef test_positive():"}
RESULT = {"test_file_name": "test_sign.py", "explanation": "Covers both signs.",
          "targeted_functions": ["sign"]}
USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class ScriptedChat(BaseChatModel):
    edited: bool = False

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, tools=(), **kwargs):
        offered = {tool.name for tool in tools}
        call = None
        if "run_tests" in offered or "measure_coverage" in offered:
            call = ("run_tests" if "run_tests" in offered else "measure_coverage", {})
        elif "edit_tests" in offered and not self.edited:
            self.edited, call = True, ("edit_tests", EDIT)
        elif offered == {"read_tests", "emit_coverage_test_result"}:
            call = ("read_tests", {})
        elif offered:
            call = ("emit_coverage_test_result", RESULT)
        calls = [{"name": call[0], "args": call[1], "id": f"call_{len(messages)}"}] if call else []
        message = AIMessage("" if call else STARTER, tool_calls=calls, usage_metadata=USAGE)
        return ChatResult(generations=[ChatGeneration(message=message)])


def factory(**parameters):
    return ScriptedChat()
```

It serves the starter phase and the coverage phase only. The example
defines no `_stream`: LangChain's `stream` then yields the whole answer of
`_generate` as one piece. A factory for a different client library
translates the keyword arguments itself.

Save the example as `scripted_chat.py` beside your `models.ini`. The alias
still needs an address, which the scripted factory ignores, and no key
variable:

```ini
[scripted_service]
api_url = http://127.0.0.1:9/scripted

[my_alias]
provider = scripted_service
model = scripted
chat_model = scripted_chat.py:factory
```

A short trial run generates one starter suite and skips the mutation
measurement at the phase ends. Put this in the project's
`.testuary/project.ini`:

```ini
[main]
phase_end_mutation = false

[starter]
suite_count = 1
```

Then run this in the project's directory, with your alias, interpreter and
target:

```bash
testuary generate --model my_alias --stop-after-phase coverage \
  --python-executable /path/to/project-env/bin/python --source-files package/module.py
```

When `[project] source_files` and `[project] python_executable` in the
settings file name the target and the interpreter, the two options can be
left out. The run sends the starter phase's and the coverage phase's
requests through your client and stops after the coverage phase. It ends
with exit status 0 and reports the target as ended after the requested
phase, with `MINIMUMS NOT MET` on its last line. A second trial needs a new
`--output-dir`: a run directory that holds results is refused, and a resume
is refused once the factory's file changed.
