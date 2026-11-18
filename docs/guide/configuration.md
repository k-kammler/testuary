# Configuration

Testuary reads its settings from the files and variables below, and a run
selects the project and its runtime from them. The [settings
reference](../reference/settings.md) lists every setting with its default,
or with the page that describes it where the package sets none.

## Model configuration

[Models and prompt profiles](../models-and-prompts.md) describes the request
controls, other kinds of model service, custom prompts, and which file wins
when several set the same value.

`testuary init` writes `models.ini` from the packaged template into your
user directory. It holds five example aliases for OpenAI-compatible, Ollama and
vLLM services as comments, so it names no model yet: remove the `#` before
the section and settings of the one that fits your service, and give it
your model name and service address. Give the service's key
through its environment variable or in your private `models.ini`. Do not
commit keys.

[models.ini fields](../models-and-prompts.md#modelsini-fields) lists the
fields of an alias and where its key comes from. A variable ending in
`_API_KEY` is always a key, never a setting, also for a service named like a
settings section, such as `TESTUARY_MAIN_X_API_KEY` for a service `main_x`.
An `api_key` in the run settings files is refused, even an empty placeholder.

The example is a template. The kind of service or a model alias does not
tell you whether a run is free.

## Where Testuary reads its files

Testuary reads its files from two places.

- The user directory holds what belongs to you and this computer: `models.ini`
  with your model aliases and keys, and the global `config.ini` with your
  settings for every project. It is the platform's directory for user
  settings: `%APPDATA%\testuary` on Windows,
  `~/Library/Application Support/testuary` on macOS and `~/.config/testuary`
  on Linux, or `$XDG_CONFIG_HOME/testuary` when that is set. [`--home DIR`](../reference/cli.md#testuary-generate)
  names another directory for one run or check, and the variable
  `TESTUARY_HOME` for every command, `testuary setup`, `testuary init` and
  `testuary-web` included. A `models.ini` or `config.ini` in the directory you
  start Testuary from is not read.
- The project directory holds what belongs to one project: its settings file
  `.testuary/project.ini` and, by default, its results in
  `.testuary/results/`. A run finds the project directory from the directory it
  starts in upward: it is the nearest directory that holds `.testuary/`, or
  the starting directory when none does. The search stops at your home
  directory, which is never a project directory. `[main] testuary_dir` gives
  the `.testuary` directory another name. Set it in the global `config.ini`,
  in a `--config` file or in `TESTUARY_MAIN_TESTUARY_DIR`.

The project's settings file takes the same sections and keys as `config.ini`
and is checked the same way, except that it cannot set
`[main] testuary_dir` or a model's address. `[api] api_url` and the
`api_url` of a provider's section decide where the model's key goes, so
they belong in your global `config.ini`, in `models.ini`, in the
environment or on the command line. Three of its settings describe the
project.
`[project] source_files` lists the targets a run without named targets runs, as
a comma list. `[project] language` is the
project's language. It must be installed, a target named on the command line
must be in it, and empty accepts every installed language. `[api] model` is
the model alias. `testuary setup project` writes the file.

### Relative paths

A relative path starts from the directory the place that names it gives:

| A relative path in | Starts from |
| --- | --- |
| a command-line option, such as `--config`, `--source-dir`, `--output-dir`, `--hints-file` or `--prompt-profile`, and a target named by `testuary FILE` | the directory the command starts in |
| a target named by `--source-files` or `[project] source_files`, and `[project] python_import_root` | the source directory |
| `[main] extensions` in a settings file | the directory of that file, so `extensions = my_tool.py` in `.testuary/project.ini` names `.testuary/my_tool.py` |
| `chat_model` in `models.ini` or in a service's section of a settings file | the directory of that file |
| `[main] extensions` or `chat_model` in an environment variable or a program's overrides | the project directory |
| any other path of Testuary's settings, such as `[main] hints_file`, `[main] prompt_profile` or `[project] output_dir`, in a settings file, an environment variable or a program's overrides | the project directory, even if a program later changes its working directory |
| `config_files` and `models_files` of `testuary.settings.resolve_settings` | its `base_dir`, the project directory |

## Run settings

Settings are kept in INI files. Each layer may set only some values, and a
later layer overrides an earlier one, in this order:

1. the defaults that ship with Testuary
2. the global `config.ini` of the user directory, or the files you pass with
   `--config` in its place
3. the project's settings file, `.testuary/project.ini`
4. environment variables, which the command line always reads. A program
   can leave them out with `resolve_settings(use_environment=False)`.
5. the options on the command line

`--no-project-settings`, or `project_settings=False` in a program, reads no
project's settings file, and the directory the command starts in is then the
project directory.

Repeat `--config` to combine a preset with local settings. Files you pass
with `--config` replace the global `config.ini`. `TESTUARY_CONFIG_FILE`
names one such file, unless a program leaves the environment out.
Misspelled keys, invalid values and a requested settings file that does not
exist are refused before anything runs. A field you leave out is different
from a field you set
to `false`, `0` or an empty value. In a program,
`testuary.settings.setting_values()` names the source of each value:
`default`, the path of a file, `environment` or `overrides`.

`[main] extensions` is the one exception to that order, since each layer
adds its items, as [its entry](../reference/settings.md#main) says. An
extension named in the global `config.ini` therefore reaches every project,
and a project adds its own. A file named twice is loaded at its first place.
No layer removes an earlier layer's item: a run that must not load one names
its settings with `--config`.
`testuary.settings.extension_items()` names each item with its source.
A program's setup names its extension in the user's `config.ini` with
`testuary.settings.add_extension()` and takes it out with
`remove_extension()`.
Starting Testuary or `testuary-web` in a project loads the extension files
that project's settings name, as running the project's tests runs its code.

In the project's directory, or a directory below it, a run of one target:

```bash
testuary generate --config preset.ini --config local.ini --source-files package/module.py
```

`testuary FILE ...`, `testuary check [FILE ...]` and
`testuary generate [FILE ...]` name targets, and so does `--source-files`,
each from the directory [relative paths](#relative-paths) gives. Naming
targets both ways is refused. Without named targets, a run
takes the project's `[project] source_files`, and without those it refuses.

The form without `generate` is the same command. The settings, `--resume`,
`--reuse`, `--runs` and `--run` behave
the same in both forms. `testuary --help` lists the commands, and
`testuary generate --help` lists the options of a run. Without arguments,
either command runs the project's `[project] source_files` in the directory of
a project that has a settings file. Elsewhere it prints its usage and
starts no run.

Your project can stay where it is. A run started elsewhere selects the
source directory and a run directory with options:

```bash
testuary generate --config local.ini --source-dir /path/to/project \
  --source-files package/module.py --output-dir /path/to/testuary-results
```

The source directory, whose files the run tests (the setting `source_dir`),
is the first of these that is set:

1. `--source-dir`
2. `source_dir` in the `[project]` section of a settings file
3. the project directory

`--project-name` and `[project] name` only name the project in results and
records, as `${project}`. The default name is the source directory's name.

[`[project] output_dir`](../reference/settings.md#project), or `--output-dir`
for one run, names the run directory. [The run directory](output.md#the-run-directory) says which
run directories a new run accepts, so a second run with the same model stops
there until you pass `--resume` or `--reuse`, or select another `--output-dir`.
[Relative paths](#relative-paths) says where each
relative path starts. Links and Windows reparse points in the paths of the project are
refused. A run writing below `.testuary/results/`
creates `.testuary/results/.gitignore`, which ignores everything, when it
is missing. Copies of the project leave out the
`.testuary` directory and the files Testuary itself read as configuration
(`models.ini`, the settings files, the prompt profile and the hints file)
when they lie inside the project, so generated tests never find them. All
other project files are copied.

`--python-executable` selects the Python that runs the project's tests, and
Testuary installs nothing into it. `--java-home` selects the project's JDK,
and `--build-java-home` selects a separate JDK for the build tool when one is
needed. These options set the matching fields of the `[project]` section.
Without an interpreter, the
[settings reference](../reference/settings.md#project) says which one a run
takes. In container mode, the paths of the project and its runtime are paths
inside the container, and `--source-dir` must be an absolute POSIX path.
Use `testuary check` to see what your setup provides. Operations that it
does not support or could not try are listed as such in its report.

A program gets the same report without terminal output:

```python
from testuary.orchestrator import RunRequest
from testuary.orchestrator import check
from testuary.settings import resolve_settings

settings = resolve_settings(overrides={
    "api": {"model": "configured_alias"},
    "project": {"source_dir": "/path/to/project", "output_dir": "/path/to/new-results",
                "python_executable": "/path/to/project-env/bin/python"},
})
report = check(RunRequest(("package/module.py",)), settings=settings)
```

Programs can build or resolve settings without starting a run:

```python
from testuary.settings import RunSettings, resolve_settings

settings = RunSettings(
    sections={
        "api": {"model": "local_model"},
        "project": {"source_dir": "/path/to/project", "output_dir": "/path/to/output"},
        "main": {"seed": 0},
    },
    models={"local_model": {
        "provider": "openai", "model": "served-model", "api_url": "http://localhost:11434/v1",
    }},
    base_dir="/path/to/workspace",
)

layered = resolve_settings(
    config_files=["preset.ini", "local.ini"],
    overrides={"main": {"seed": 0}},
    use_environment=False,
    models={},
)
```

`RunSettings` copies its mappings and makes them read-only at every level.
Built directly, it uses the shipped defaults and the models and environment
variables you pass, not local settings files. Without
`environment=dict(os.environ)` it captures no environment, and a run refuses
to start the project's tools. `resolve_settings` reads
`models.ini` from the user directory, `home` when you pass it, unless you pass
`models` or `models_files`. `models_files` names other catalogue files, read
in order in its place, a later file's section replacing the same section of
an earlier one whole, and `models={}` turns off reading a catalogue. A
catalogue file in the project directory, links followed, may not set
`api_url` or `api_key_env`, since they decide where the model's key goes:
set them in `models.ini` or in a file outside the project. A program that
passes its runs other settings or catalogue files checks each one that is
not the user's own with `check_project_settings_file` and
`check_project_catalogue_file`, which refuse such files wherever they lie.
It finds the project directory and its settings file from `base_dir`
upward, the current directory when you pass none, and relative paths of
the settings start from the project directory. `use_environment=False`
turns off the environment variables, `TESTUARY_HOME` among them.
Interpreter and repository paths in a container stay paths inside the
container, not paths on your computer. The local service and model in the
first example are placeholders and do not mean that the setup works.
Building or resolving settings sends no request.

For a project that is already configured, a program builds the run request
directly instead of going through the command-line parser:

```python
from testuary.orchestrator import RunRequest
from testuary.orchestrator import run_pipeline
from testuary.settings import resolve_settings

settings = resolve_settings(config_files=["local.ini"])
request = RunRequest(source_files=("package/module.py",), runs=1)
exit_code = run_pipeline(request, settings=settings)
```

A program that shows the run elsewhere also passes `events=PATH`
([Saving the events to a file](terminal.md#saving-the-events-to-a-file)).
Each source path stays a separate string, so a path may contain commas. The
request selects the targets, the repetitions (`runs` and `run`, as
[`--runs` and `--run`](../reference/cli.md#testuary-generate) select them),
resuming and reuse, and where to stop.
Configuration, models, runtimes and limits belong to `RunSettings`. The other
arguments of the call, such as the parsed command line, must not also carry
configuration options when settings are passed. A run never reloads local settings or
environment variables over this object, and the worker threads that write
the starter suites use the same object. After a run returns, the process's
configuration and logging state are back to what they were before it, so two
runs one after another in one Python process do not affect each other. Two
overlapping invocations in one process are refused. Use separate
processes and separate run directories for parallel runs.

Testuary does not lock or edit the project directory. Only its own working
copies are made writable. Your project's build and test code and other programs
that write files are not sandboxed by this.

`settings.export()` returns a separate copy of the settings in effect, without
secrets and without the `[web]` section, which configures no run. A run's
records are made from it. It is not data for restarting a run. It leaves
out keys and the captured environment variables, and it hides the parts of
addresses that carry keys. Keep the live object and your private files for
running. Do not use the export to rebuild keys.

### Sampling seeds of the starter suites

`seed_base` in `[starter]`, an integer, sets the starter suites' sampling
seeds: the first attempt of suite `i` requests with the seed `seed_base + i`
in the first repetition, `r0`, and with `seed_base + 1000*k + i` in
repetition `rk`, which is why
[`[starter] suite_count`](../reference/settings.md#starter) has an upper
limit. A suite
generated again after a rejected attempt
requests with a seed of its own attempt, adding `(a - 1) * 1000000` for
attempt `a`, so a model service that honors seeds does not return the
rejected answer again. Without `seed_base`, `[main] seed` is the base. The
packaged `config.ini`
names it only in a comment.

### Hints file

Use `--hints-file guidance.json` or `[main] hints_file` to give the model
extra guidance. [Hints](../extending/hints.md) describes the format, the
example and how a hints file is selected and validated.

## Limits on what the model sees

Tool results, test output, surviving mutants, uncovered code and file reads
are cut to named limits before they reach the model. Each limit is a key of
the [`[limits]` section](../reference/settings.md#limits), which says the
values it takes and how the phase sections and a model alias override it. An
alias sets them below the settings files, so a model with a small context can
get smaller limits once. A cut output carries one marker, such as
`[... 1200 chars of the tool result omitted ...]`, and where a tool can show
the rest, the marker says how. The history window, `history_messages`, keeps
the task prompt and drops the oldest messages after it without a marker.

Limits change what the model reads, so a resume or reuse with other limits is refused. LangAdapter
returns complete output up to a safety bound of 4,000,000 characters
per text, marked in the same way if it is ever reached, and these limits
apply after it.

The [settings reference](../reference/settings.md#limits) lists every limit
with what it limits and its default.

## Execution settings

A program that starts Testuary can set `TESTUARY_EXECUTION_ID` to keep the
temporary work of a run in its own directory, `_work/job_<id>`, with unsafe
characters of the ID replaced. Use different IDs for work that runs at the
same time, and the same ID to return to the same directory.
`TESTUARY_MAIN_WORK_DIR_NAMESPACE` sets the directory name directly, with the prefix
`job_explicit_`. The [settings reference](../reference/settings.md#main)
says when `work_dir_job_namespace` gives a run such a directory. Runs at
the same time that could overwrite each other's results or logs need
separate run directories.

The records label the clock their times come from, `clock_domain_id`, so
that times are compared only within one clock: the system's current boot
where it has a boot ID, else the process. `[main] device_name` and
`TESTUARY_CLOCK_DOMAIN_ID` change that label, as the
[settings reference](../reference/settings.md#main) describes.

The [settings reference](../reference/settings.md#timing) describes
`[timing] serialize`, the lock on timing measurements, and its variable
`TESTUARY_TIMING_SERIALIZE`.

`--languages` limits which languages' tools are checked before a run. It does
not select targets, and a target of a language it leaves out refuses the run.

Model requests follow one retry rule in every phase. A temporary failure (a
connection error, a timeout, or HTTP 408, 409, 429 or 5xx) is retried for the
same request after a wait. The wait is the one the service asks for in its
`Retry-After` header, otherwise it doubles from `transient_first_wait_seconds`
up to `max_transient_wait_seconds`.
[`count_retries`](../reference/settings.md#coverage) decides whether such a
retry counts against the phase's request limit. After `max_consecutive_transient_failures` failures in a
row of one request, or `max_transient_failures_per_phase` failures in one
phase, the phase ends with a connection error. Any other failed request, such
as a rejected key, ends the phase at once with the service's error. In the
coverage and mutation phases, so does a request too large for the model's
context, whatever status the service reports, because sending it again cannot
succeed. An attempt of a starter suite counts toward its
`attempts_per_suite` only when an answer arrives.

The coverage and mutation phases also start again from where they stopped when
something other than a model request fails with a temporary error, such as a
tool that raises a connection error. They wait for the error's hint, or 10
seconds doubling up to 60, a wait no setting changes. A test run that times out is not such a failure:
the tool reports it to the model. When a loop rejects the structure of a model
answer, it asks again. Both kinds share `max_loop_attempts`, which bounds
how often the loop starts, the first start included. The packaged defaults set these
values in `[starter]` and `[coverage]`. The mutation phase takes them
from `[coverage]` unless its own `[mutation]` section sets them.
