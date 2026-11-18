# A prompt profile

A prompt profile changes the text Testuary sends to the model. It does not
change tools, limits, validity rules or how suites are measured, so two runs that
differ only in their profile measure the same way.
[`testuary tutorial`](../reference/cli.md#testuary-tutorial) walks a prompt
profile on a small project without contacting a model.

## Selecting a profile

Select a profile with `--prompt-profile`, `TESTUARY_MAIN_PROMPT_PROFILE` or
`[main] prompt_profile`, in that order.
[Relative paths](../guide/configuration.md#relative-paths) says where a
relative path starts. The file is read and validated once, when the command
starts, and kept for the whole run. A profile with an empty `phases` object
leaves every prompt unchanged:

```ini
[main]
prompt_profile = profile.json
```

```json
{"format": 1, "name": "unchanged", "phases": {}}
```

## Checking a profile

`testuary check` checks the selected profile, for example
`testuary check --prompt-profile profile.json --model ALIAS --source-files FILE`.
A missing file, an unknown component, section heading or tool message, or
an invalid placeholder refuses the command with exit status 2 before any
work. The refusal of a wrong section heading, tool message name or
placeholder lists the allowed ones: the phase's `## ` headings, such as
`## WORKFLOW` and `## Hard rules`, and its tool messages, as its bundled
prompt names them. The report of a check that passes does not name the
profile, and the check does not show the changed prompts. A check refuses a
run directory that is not empty, so after a run give it a new
`--output-dir`.

A run records the profile under `prompt_profile` in each phase's
`archive/<phase>/generation/record.json`, with a copy of it and the parts
it changed in `changes`
([generation records](../reference/result-files.md#generation-records-and-the-settings-record)).
No file of a run holds the system prompt as sent: the phase's chat log holds
its user prompt and the conversation. `testuary.hints.request_text` returns
the guidance hints add.

## The format

A profile is a JSON file with `"format": 1`, an optional `name` and a `phases`
object. Each of `initial` (the starter phase), `coverage` and `mutation`
can change these components:

- `system_prompt` and `user_prompt_template`, the whole text, with `append` or
  `replace`,
- `system_prompt_sections`, named sections of the system prompt, with `append`,
  `replace` or `remove`, where `remove` takes `true`,
- `tool_messages`, the text of existing tool messages, with `append` or
  `replace`. A profile cannot add tools or change their descriptions.

The text of an operation is a string or a list of lines.
A section is named by its heading line as the bundled prompt spells it,
such as `## Hard rules`. The `initial` phase has no sections and no tool
messages, so only its `system_prompt` and `user_prompt_template` can change.
A `replace` of the whole system prompt cannot be combined with section
operations.

This profile appends one sentence to the system prompt of every phase:

```json
{
  "format": 1,
  "name": "boundary-tests",
  "phases": {
    "initial": {
      "system_prompt": {"append": "Exercise empty, boundary and error inputs."}
    },
    "coverage": {
      "system_prompt": {"append": "Use explicit expected values."}
    },
    "mutation": {
      "system_prompt": {"append": "Keep assertions tied to observable behavior."}
    }
  }
}
```

This profile changes the coverage phase's sections and one of its tool
messages. It appends a sentence to the section `## Hard rules`, removes the
section `## Test libraries available` and appends a sentence to the tool
message `edit_not_found`:

```json
{
  "format": 1,
  "name": "independent-tests",
  "phases": {
    "coverage": {
      "system_prompt_sections": {
        "## Hard rules": {"append": "Keep every test independent of the others."},
        "## Test libraries available": {"remove": true}
      },
      "tool_messages": {
        "edit_not_found": {"append": "Read the test file before the next edit."}
      }
    }
  }
}
```

A `per_language` entry in a phase changes a component for one language. For
that language it replaces the general change to the same component, or to
the same section or tool message, so a language entry must repeat any
general text it keeps. A language that LangAdapter has not registered, built
in or added by an extension, is refused. The
example that `testuary init --examples` copies,
`testuary-examples/language-profile.json`, which Testuary's tests resolve
unchanged, does this for Python and Java:

```json
--8<-- "src/testuary/_templates/examples/language-profile.json"
```

Profile operations run before the template placeholders are filled and before
hints are added. Keep the placeholders the bundled text uses: a profile that
drops one is not refused, and the model then gets the text without that
value. A profile may use the placeholders of the component it changes and
the language's placeholders, such as `$lang_name`. Any other placeholder
refuses the run, and `$$` writes a literal dollar sign.

## Comparing profiles

To compare profiles, use a separate run directory for each profile and keep
every other setting the same. Testuary records which prompts each phase used,
with a copy of the profile and hints files. A resume or reuse with a different
profile is refused rather than silently generating again, and saved model
responses keep the usage recorded when they were made.
