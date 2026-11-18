# Make a language ready for Testuary

Testuary works with every language whose LangAdapter adapter provides the
three families of level 1 below. Testuary itself has no code per language: a
language comes to Testuary as one Python file, which `[main] extensions`
names. Its adapter is built with LangAdapter's guide
[Add a language](https://k-kammler.github.io/langadapter/extending/add-a-language/).

LangAdapter's words, as this page uses them: a family is a group of the
adapter's operations that the guide adds together, such as running tests. A
hook is one method of the adapter that Testuary calls. A fact is a value the
adapter states, such as its file extensions.

## The language file

Start the file with LangAdapter's command, which Testuary passes on:

```text
testuary new-language javascript --file-extension .js --for testuary
```

It writes `javascript.py` in the current directory. Its top-level name
`langadapter_language` names the adapter to LangAdapter, and the hooks
Testuary requires are marked TODO with their levels. Fill them in with
LangAdapter's guide.

Name the file in `[main] extensions`, in the project's settings file
`.testuary/project.ini`, or in the `config.ini` of your user directory for
every project:

```ini
[main]
extensions = javascript.py
```

A relative path starts at the directory of the settings file that holds it:
here the file lies at `.testuary/javascript.py`. Write `../javascript.py` for
a file in the project directory. The table
[Relative paths](../guide/configuration.md#relative-paths) lists where each
setting's relative path starts.

## Check a project

Check a project in your language with Testuary. The quick check runs nothing
in your project and contacts no model, but it checks the model settings as a
run would, so name one of your models with `--model`. `testuary setup model`
adds one to your models.ini. An alias that names a key variable
(`api_key_env`) needs that variable set, with any text: the check does not
try the key. Run it in the project's directory, where it finds the settings
file that names your language's file:

```text
testuary check --model ALIAS --source-dir PATH --source-files FILE
```

The quick check loads no extension, so it lists your file, the target and
each `[project]` value only your language knows, such as its settings or a
`language` the project sets, as ones the tool check checks. Add `--verify` for
the tool check, which loads your file, reports the
[project settings](#project-settings) your adapter declares and whether they
are set, and lets your adapter try the project in a copy, still without a
model request.

## What Testuary needs

Testuary publishes what it needs as a requirement list: each family of
LangAdapter's guide is required, optional or not used. Testuary groups them
into three levels, in the order to add them. Level 1 is enough to run all
four phases.

<!-- profile:testuary -->
**Level 1: Run, coverage, mutation**

| Family | Testuary | With it, Testuary | Without it |
| --- | --- | --- | --- |
| [run tests](https://k-kammler.github.io/langadapter/extending/add-a-language/#run-the-tests) | required |  | a run does not start |
| [coverage](https://k-kammler.github.io/langadapter/extending/add-a-language/#measure-coverage) | required |  | a run does not start |
| [mutation](https://k-kammler.github.io/langadapter/extending/add-a-language/#run-mutation-testing) | required |  | a run does not start |

**Level 2: Load repair**

| Family | Testuary | With it, Testuary | Without it |
| --- | --- | --- | --- |
| [load repair](https://k-kammler.github.io/langadapter/extending/add-a-language/#repair-loading) | optional | keeps generated suites that fail to load, repaired | a generated suite that does not load is discarded |
| [import pruning](https://k-kammler.github.io/langadapter/extending/add-a-language/#remove-failing-imports) | optional | keeps generated suites whose only fault is an import that does not resolve | a generated suite with an unresolvable import does not load and is discarded |

**Level 3: Further improvements**

| Family | Testuary | With it, Testuary | Without it |
| --- | --- | --- | --- |
| [syntax repair](https://k-kammler.github.io/langadapter/extending/add-a-language/#repair-syntax) | optional | keeps suites with a local syntax error, repaired | a suite that does not parse is discarded |
| [skip removal](https://k-kammler.github.io/langadapter/extending/add-a-language/#remove-whole-file-skips) | optional | keeps whole-file skips from hiding the tests of a suite | whole-file skips stay, and a suite without a passing test is rejected |
| [import guidance](https://k-kammler.github.io/langadapter/extending/add-a-language/#import-line) | optional | names in the starter prompt the import line a test needs, or that it needs none | the starter prompt asks the model to follow the language's import conventions |
| [kill attribution](https://k-kammler.github.io/langadapter/extending/add-a-language/#kill-attribution) | optional | reduces suites while keeping every mutant kill | the suite reductions that must keep every kill are skipped |
| [merging](https://k-kammler.github.io/langadapter/extending/add-a-language/#merge-test-files) | optional | combines a phase's candidate suites into one | the better single candidate is kept, by mutation score, then branch coverage, then line coverage, then the timing tie-break |
| [per-test evidence](https://k-kammler.github.io/langadapter/extending/add-a-language/#per-test-evidence) | optional | reduces suites to the tests that add coverage or kills | suite reduction is skipped and each phase chooses among unreduced suites. Each starter suite is scored on its own measurements |
| [instrumentation](https://k-kammler.github.io/langadapter/extending/add-a-language/#prepare-a-suite) | optional | finds the tests whose run warns of leaked state, such as an unawaited coroutine: the starter suites' cleanup removes them, and a loop counts their passing run as failing | such tests are neither removed nor counted as failing |
| [delivered runs](https://k-kammler.github.io/langadapter/extending/add-a-language/#run-as-delivered) | optional | runs a suite the way a user would run it, as the normal run of a language whose runs as delivered take a session seed and for the reverse-order note | the normal run is the language's own test run, and the reverse-order note is off |
| [validity screening](https://k-kammler.github.io/langadapter/extending/add-a-language/#validity-policy) | optional | rejects generated tests that read the implementation | such tests are not rejected |
| [timing](https://k-kammler.github.io/langadapter/extending/add-a-language/#timing) | optional | tells equal suites apart by their run time and records it | the timing tie-break is off, the input order decides, and no run time is recorded |
| [versions](https://k-kammler.github.io/langadapter/extending/add-a-language/#versions) | optional | notes runtime versions and refuses unsupported ones | every runtime version is reported as untested |
| [containers](https://k-kammler.github.io/langadapter/extending/add-a-language/#containers) | optional | runs suites in a container | suites run on this machine only |
| [structural metrics](https://k-kammler.github.io/langadapter/extending/add-a-language/#structural-metrics) | optional | reports sizes and complexity in Finishing | those columns stay empty |
| [code smells](https://k-kammler.github.io/langadapter/extending/add-a-language/#code-smells) | optional | counts lint findings (measure_lint) | those columns stay empty |
<!-- /profile:testuary -->

A required family has nothing under "With it, Testuary": without it a run
does not start.

In level 1, "run tests" means four hooks of LangAdapter's guide:
`run_suite`, which runs a suite, `test_units`, which finds each test in a
test file, `capabilities`, which reports what a project can run, and
`language_facts`, which describes the language: its name, test framework,
code fence and test file patterns, which Testuary's prompts and records
name. Without `capabilities` the project cannot be probed and a run does
not start. The other hooks of that family work by default. The scaffold
marks six hooks of level 1 as TODO: these four, `coverage`, which
`run_suite` serves when a request names a coverage target, and
`open_mutation_session`, the coverage and mutation families' hooks.

Levels belong to a program that uses LangAdapter, here Testuary: another
program groups the families its own way. The check that does not depend on
any program is LangAdapter's conformance check, `check_adapter`, which
tests your adapter against the rules every adapter follows.

An optional family that your adapter does not provide makes Testuary use the
fallback shown above. A run records each fallback it applied, under
`fallbacks` in each repetition's
[`resolved_settings.json`](../reference/result-files.md#generation-records-and-the-settings-record),
so results from runs with and without it are not mixed up. A family Testuary
does not use can still be useful to other programs. LangAdapter's page
[Who needs what](https://k-kammler.github.io/langadapter/extending/who-needs-what/)
explains how requirements add up when a program builds on Testuary.

## Check your adapter for Testuary

LangAdapter's check shows what Testuary still needs from your adapter, level
by level:

```text
langadapter check javascript --extension javascript.py --for testuary
```

It gives each hook and fact one of five status words, present, default
used, missing, failing or not checked. A missing hook names its section of
LangAdapter's guide [Add a language](https://k-kammler.github.io/langadapter/extending/add-a-language/),
such as `add-a-language.md#run-the-tests`. A missing fact comes with one
sentence that says what it holds, and no section. Testuary's levels rank
families, not facts. Of the facts a new scaffold leaves missing, set
`runtime`, as the scaffold's TODO asks: LangAdapter finds the project's
program by that name on PATH unless your adapter writes
`discover_runtime`. Testuary needs no `test_file_patterns`: without them
and without `is_test_file` it tells test files by the test file name
(`test_file_name`). `source_merge_separator` serves only merging, an
optional family of level 3. The check's verdict line is `Ready: no.`, with
exit status 1, while something required is missing, as on a new
scaffold, and `Ready: yes.` or `Ready: everything required is there.`,
with exit status 0, once nothing required is missing. Notes can follow
it, such as the one on the facts and hooks whose default is used.
Without an example the check cannot confirm coverage, which level 1
requires: it never reports level 1 as reached. For an adapter with every
required family it says that everything required is there, and that
coverage is confirmed only by a run on an example. Add `--example` with a
directory that holds a small project of your language, a source file and
its test file: the check then runs your adapter's suite there and confirms
coverage. LangAdapter finds Testuary's requirement list through Testuary's
installed package, so install Testuary in the same environment.

## One run decides

Testuary asks your adapter's normal test run whether a test passes. The
coverage and mutation tools only measure: a test that fails only under a tool
still passes, and the suite gets no number from that tool. Testuary then
names the tests that stopped the tool to the model, while the model can still
change them, and it removes no test from a suite the model worked on. A
starter suite that is not fully measured is generated again within its
attempts, and a phase end chooses only among fully measured suites.

Your adapter's runs of one suite must therefore agree. LangAdapter's
[One run decides](https://k-kammler.github.io/langadapter/extending/add-a-language/#one-run-decides)
says why, with a real case per language, and what that asks of an adapter.
Its conformance check `compare_runs` finds the differences on your example
before a Testuary run meets them.

## Project settings

A language's adapter declares the settings a project gives it, in its
`project_settings`, a tuple of LangAdapter's `ProjectSetting`: each with
its name, its purpose, its kind (`path`, `executable`, `choice` or `text`),
a default and, for a choice, its choices, such as where the language's
runtime is. Another kind is not refused, and its value is not checked. An
empty default means the language finds the value. In the adapter class of
`javascript.py`:

```python
from langadapter.adapter import ProjectSetting

project_settings = (
    ProjectSetting("javascript_runtime", "The program that runs the tests.", "executable"),
    ProjectSetting("javascript_modules", "The module system.", "choice",
                   default="esm", choices=("esm", "commonjs")),
)
```

LangAdapter's [hooks reference](https://k-kammler.github.io/langadapter/reference/hooks/#project_settings)
describes the other fields.
The settings join `[project]`, usually in the project's settings file
`.testuary/project.ini`, each under its name:

```ini
[project]
javascript_runtime = /usr/bin/node
```

Begin each name with the language's name, as the C adapter's settings begin
with `c_`, so it does not meet another language's setting. `testuary setup`
asks them when it writes a project, or asks the questions your adapter
describes with LangAdapter's `setup_questions` hook. `testuary check
--verify` reports each with its purpose and its value, or `Not set.`, and
checks the value by its kind: a choice must be one of its choices, a path
must exist, and an executable must be a file or a program on PATH. Python's
and Java's own settings stay in Testuary's sections.

## A worked example

The example below runs unchanged from the installed packages, in an empty
working directory. It takes the Toy's adapter from LangAdapter's tutorial at
the step where it reaches Testuary's level 1, registers it, writes a small
Toy project and runs Testuary's quick check on it. It registers the adapter
in the program with `la.register`, as a program that uses LangAdapter
directly does: a run reaches the same registration through
`[main] extensions`. The check sends no model request, and the model
selection is a placeholder. Testuary's tests run this example as written.

```python
from importlib import resources
import importlib.util
from pathlib import Path

import langadapter as la
from testuary import orchestrator
from testuary.orchestrator import RunRequest
from testuary.settings import resolve_settings

# The Toy's adapter at step 4 of LangAdapter's tutorial: Testuary's level 1.
steps = resources.files("langadapter.tutorial.example") / "steps"
spec = importlib.util.spec_from_file_location("toy_adapter", steps / "04_mutation.py")
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)

project = Path("toy-project").resolve()
project.mkdir(exist_ok=True)
(project / "calc.toy").write_text("double(x) = x * 2\n", encoding="utf-8")
(project / "test_calc.toy").write_text("test doubles: double(2) == 4\n", encoding="utf-8")

la.register("toy", [".toy"], toy.ToyAdapter)
try:
    settings = resolve_settings(
        base_dir=Path.cwd(), models={},
        overrides={"project": {"source_dir": str(project)},
                   "api": {"model": "example", "provider": "openai",
                           "api_url": "https://example.test/v1"}})
    report = orchestrator.check(RunRequest(source_files=(str(project / "calc.toy"),)),
                                settings=settings)
    print("check:", report.status)
    for item in report.items:
        if ".fallback." in item.key:
            print(" ", item.reason)
finally:
    la.unregister("toy")
```

It prints the check's result and the fallback of each optional family the
Toy does not provide yet:

```text
check: unverified
  The language adapter lacks load repair: operation is not implemented. Without it, a generated suite that does not load is discarded.
  ...
  The language adapter lacks merging: operation is not implemented. Without it, the better single candidate is kept, by mutation score, then branch coverage, then line coverage, then the timing tie-break. This changes Testuary's method.
  ...
```

`unverified` is the quick check's result when it found no problem but tried
nothing yet: the required families wait for the tool check or a run, and the
exit status is 0.

## The first run

Run Testuary on one target with small limits, as
[getting started](../getting-started.md#run-with-small-limits) describes. Run
it in the project's directory, where the run finds the project's settings file
with those limits:

```text
cd PATH
testuary generate --model ALIAS --source-files FILE
```

The results show what each phase measured with your adapter, and
`fallbacks` lists what Testuary did without the optional families.

## Extra model hints

Testuary's prompts describe your language with the words your adapter
declares. The starter suites' prompt names the language, its test
framework, its code fence and its language level. Of your adapter's rules,
it carries only the one-file rule (`single_file_rule`), the import hint and,
when your adapter states one, the validity rule (`validity_rule`). Give the
form of a test in the one-file rule, with one example, as the tutorial's Toy
language does. For advice the general prompts do not give, your language's
file can hold extra hints: the top-level name `testuary_hints`, a mapping in
the format of [Hints](hints.md), which shows such a file and says in what
order Testuary reads them. Put the advice under `per_language` with your
language's name: a `general` cell reaches the targets of every language.
`testuary tutorial` walks an extension's hints on a small project without a
model.

## The tutorial

To see every step on a tiny example language, ending with a Testuary run on
recorded model answers that needs no model access:

```text
langadapter tutorial --for testuary
```
