# Hints

Hints add guidance text to the prompts of a phase: for every target, for the
projects of one hint profile, for one language, or for a hint profile and a
language together.
Like a prompt profile, a hint changes model inputs only. Keep hints the same
across runs you compare.
[`testuary tutorial`](../reference/cli.md#testuary-tutorial) walks an
extension's hints on a small project without contacting a model.

## Selecting a hints file

Select a file with `--hints-file` or `[main] hints_file`.
[Relative paths](../guide/configuration.md#relative-paths) says where a
relative path starts. Without a selection, an optional `hints.json` in the
project directory is used when it exists. An empty `hints_file` switches
hints off. `testuary init` writes a starter `hints.json` into the project's
directory. A setting and a file of one hint:

```ini
[main]
hints_file = hints.json
```

```json
{"coverage": {"general": "Name each test after the behavior it checks."}}
```

An extension holds hints in its top-level name `testuary_hints`, in the
format of a hints file. Name the extension's file in `[main] extensions`,
such as `extensions = my_hints.py`:

```python
testuary_hints = {
    "coverage": {"per_language": {"python": "Name each test after the behavior it checks."}}
}
```

[Make a language ready](languages.md#extra-model-hints) shows where a
language's file holds them.

## Checking a hints file

A run and `testuary check` read and check the hints file once when they
start, before any work or cleanup. Changing the file during a run does not
change its prompts. A selected file that is missing refuses the command with
exit status 2, and so does any hints file, selected or not,
that is not valid JSON, names an unknown phase or cell, or holds a cell that
is not text. The report of a check that passes does not name the hints file.
A check refuses a run directory that is not empty, so after a run give it a
new `--output-dir`.
The quick check loads no extension, so `testuary check --verify` and a run
are the first to check an extension's hints.

`testuary.hints.request_text` returns the guidance a phase gets for one
language. Run this in the project directory to see it:

```python
import json

from testuary.hints import request_text

with open("hints.json", encoding="utf-8") as file:
    print(request_text(json.load(file), "python", "coverage"))
```

Its keyword `hint_profile` selects the cells of a hint profile, as
`[project] hint_profile` does.

## The format

A hints file is a JSON object with the phases `initial` (the starter phase),
`coverage` and `mutation`. The starter phase's key is `initial`. Each phase
has up to four cells:

- `general`, for every target,
- `per_profile`, keyed by the project's hint profile (`[project] hint_profile`),
- `per_language`, keyed by language,
- `per_profile_per_language`, keyed by hint profile and then language.

A cell holds a string or a list of strings. The matching cells of a phase are
appended in that order at the end of the phase's system prompt, under
`## Additional guidance`. Empty guidance leaves the prompt unchanged. Keys
that start with an underscore, at the top level or in a phase, are notes.
The hints of the extensions named in `[main] extensions` come first, each
read the same way, then the project's file. A small file:

```json
{
  "initial": {"general": "Cover the public input boundary cases."},
  "coverage": {"per_language": {"python": ["Use explicit expected values."]}}
}
```

The example that `testuary init --examples` copies, `testuary-examples/hints.json`,
gives the coverage phase one hint each for Python and Java. Testuary's tests resolve it unchanged:

```json
--8<-- "src/testuary/_templates/examples/hints.json"
```
