# Getting started

## Install

Testuary runs on Python 3.10 or newer. It is supported on Windows and
Linux. On macOS it is expected to work, but this release has not run on a
Mac: [Tested environments](reference/tested-environments.md) says what is
known. The project you test can use a different Python version, or a JDK
for Java. You
select it separately.

Install Testuary into a virtual environment of its own, outside the directory of
the project you test, for example by running the commands below in your home
directory. Testuary pins the versions of pytest, coverage, Cosmic Ray and PuLP
that its measurements use, and in a shared environment pip would replace other
versions of these packages. Inside the directory you test, Testuary's own
environment can also count as the project's, as
[`python_executable`](reference/settings.md#project) describes. pip
installs LangAdapter (package `langadapter`) with Testuary, which runs the tests
and tools of each language:

```bash
python3 -m venv testuary-env
source testuary-env/bin/activate
python -m pip install testuary
testuary --version
testuary --help
```

On Windows, create the environment with `python -m venv testuary-env` and
activate it with `testuary-env\Scripts\activate` in cmd or PowerShell. In
PowerShell, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first: the
default policy of Windows PowerShell refuses the activation script, and this
command allows local scripts in that window only. On Debian and Ubuntu,
`python3 -m venv` needs a package, as
[Set up your machine](guide/machine.md#windows-with-wsl) describes.

Testuary comes with a local web interface. Start it in the directory you
work in:

```bash
testuary-web
```

It opens the interface in your browser, or in an SSH login prints the
`ssh -L` line for your own computer. [The web interface](guide/web.md)
describes its pages and, under [Access](guide/web.md#access), how to start it
on a machine you share.

`testuary --version` names the installed versions of Testuary and LangAdapter.

The project you test needs its own working test tools. A Python project needs
pytest, coverage and Cosmic Ray in its virtual environment.
[Tested environments](reference/tested-environments.md) lists the tested
versions, which the commands below install. The quick check lists the
versions it finds. Install them with the project's interpreter:

```bash
/path/to/project-env/bin/python -m pip install pytest==8.1.1 coverage==7.6.8 cosmic-ray==8.4.1
```

For a project on Python 3.14 or 3.15:

```bash
/path/to/project-env/bin/python -m pip install pytest==8.4.2 coverage==7.16.2 cosmic-ray==8.4.1
```

A Java project needs
a JDK and the project's Maven or Gradle build (its wrapper or the tool on the
`PATH`) with its test framework. LangAdapter downloads the JUnit
launcher, JaCoCo and PIT into its cache on first use, unless
`LANGADAPTER_JAVA_TOOLS_OFFLINE` is set to 1. Testuary runs the project's
build and test code as it is. That is not a sandbox, so test only code you
trust.

To see a run of all four phases without a model, a project or any test tools, play back
the [offline demonstration](guide/demonstrations.md):

```bash
testuary --demo --output-dir demo-python
```

## Set up with the wizard

`testuary setup` asks for what a run needs and writes it into the same files
you would edit by hand. Run it in your project's directory:

```bash
testuary setup
```

It has three parts, asked in this order:

1. Model access, in `models.ini` in your user directory: the kind of model
   service, its address,
   the model and an alias for it. Where the service needs a key, the key is
   read without being shown. It goes into `models.ini`, which is then
   readable by you only, or you name an environment variable that holds it.
   The wizard checks access with the service's free list of models. Before
   it asks, it requests that list from Ollama at `http://localhost:11434`,
   waiting at most two seconds. If Ollama on this computer answers, it is
   the suggested choice.
2. This machine, in `config.ini` in your user directory: how many mutants run
   at the same time (about half the CPUs Testuary may use, which a
   container's CPU limit lowers, at most 15) and how many
   concurrent requests the model service gets in the starter phase.
3. The project, in its settings file `.testuary/project.ini` in the
   project directory: the project directory, whose files are tested (the
   current one by default), the language,
   for Python the directory its packages are imported from and the
   interpreter, for Java the JDK and, without a wrapper, the build tool, the
   targets (the files to generate tests for) and the model alias to use. It also creates
   `.testuary/results/.gitignore`, so the results of runs stay out of
   version control.

The user directory holds your models and the settings of this machine for
every project. The
[configuration guide](guide/configuration.md#where-testuary-reads-its-files)
says where it is and describes every file Testuary reads.

Each question shows its default in brackets, and Enter accepts it. Model
access and this machine's parallel work, when already set, are passed with
one line. Every other part, the project included, can be skipped. An existing
section of `models.ini` or `config.ini` is replaced only after you confirm.
In the project's settings file, the settings the wizard asks about take
your answers and every other setting in the file stays. When its input
ends, the wizard stops before it writes the part it is asking about. Parts
already confirmed keep what they wrote. After a project is written, it runs
`testuary check` in the project's directory.

Run one part on its own with `testuary setup model`, `testuary setup
machine` or `testuary setup project [NAME]`, for example to set up a second
project. NAME is the project's name in results and records, by default the
name of its directory.

In the project's directory, or in any directory below it, a run then needs no
options:

```bash
testuary
testuary src/pkg/module.py
testuary check
```

`testuary` generates tests for every target, each source file the project's
settings file lists, with its model alias. Each target goes through all four
phases with its own model requests, and a run of more than one target lists
the targets before the first request. Naming
targets runs only those, and a name starts from the directory you run the
command in. A target outside the project's source directory, or not in its
language, is refused before any work. `--model` and the other options of
`testuary generate` still apply. `testuary check` checks the same run without
starting it.

Without the wizard, a command names the model with `--model`: the packaged
settings choose no model, so a run without one in an option or a settings file is
refused before any work. `[api] model` in the project's settings file, as
the wizard writes it, takes the option's place. `testuary check` without a
model reports it as a problem, with exit status 1, and still looks at the
project and its tools.

## Configure a model

Create the configuration templates:

```bash
testuary init
```

This writes `models.ini` into your user directory and `hints.json` into the
project's directory, names each file it writes and keeps any file that already
exists. In `models.ini`, add a section for your model:

```ini
[ollama_qwen]
provider = ollama
model = qwen3-coder:30b
api_url = http://localhost:11434/v1
```

The section name is the model alias you pass with `--model`. `provider =`
names the kind of model service, `ollama`, `openai` or `vllm`. It also gives
the name of the key variable, here `TESTUARY_OLLAMA_API_KEY`. `model =` is the
model name the service knows, and `api_url =` is the address of the service's
OpenAI-compatible interface, which for a local Ollama ends in `/v1`. The model
must support tool calls. Without `api_url`, an `ollama` alias goes to Ollama
Cloud, the address [`[ollama] api_url`](reference/settings.md#ollama) gives,
which needs a key. For Ollama on your own computer, keep the `api_url` line.

A service that needs a key reads it from that variable, or from another
place [models.ini fields](models-and-prompts.md#modelsini-fields) names. Keep
`models.ini` private and out of version control. A local service without
keys, such as Ollama on your own computer, needs none. At a paid service,
every model request can cost money. The
[settings reference](reference/settings.md) lists every setting.

## Check your setup

Before the first model request, run the quick check. It looks at the same
inputs a run would use:

```bash
testuary check --model ollama_qwen --source-dir /path/to/project \
  --python-executable /path/to/project-env/bin/python \
  --source-files package/module.py --output-dir /path/to/new-results
```

On Windows, the interpreter of a virtual environment is
`<project-env>\Scripts\python.exe`, for example
`--python-executable C:\path\to\project-env\Scripts\python.exe`.

[The check's report](guide/terminal.md#the-checks-report) says what the quick
check looks at. The report goes to standard error and starts with what it
found and how far it looked. A run tries some required parts first, such as
the API key, so the quick check reports no problem only for what it tried.
For a Python project with every tool installed it looks like this (paths
replaced by placeholders):

```text
The quick check found no problem in what it tried. The required parts it left to a run are listed under Not tried yet.
A quick check runs nothing in your project and contacts no model.

Source directory: /path/to/project (on this computer)
Target: package/module.py
Model: qwen3-coder:30b via Ollama
Run directory: /path/to/new-results (a new run)
Phases: Starter phase, Coverage phase, Mutation phase, Finishing
Limits: starter phase up to 4 attempts per suite | coverage phase up to 1000 model requests, no token or time limit | mutation phase up to 1000 model requests, no token or time limit

Found:
  Python: runtime CPython 3.12.10 (/path/to/project-env/bin/python)

Installed but not tried:
  Python: pytest 8.1.1, coverage 7.6.8, cosmic-ray 8.4.1
    run the tool check to try it
  The tool check is testuary check --verify with the same options. It makes no model requests.

Not tried yet:
  the API key: No API key is configured. A service without API keys works, and a service that needs one refuses the first model request.
    If the service needs an API key, set TESTUARY_OLLAMA_API_KEY, or name the key's variable with api_key_env in the [ollama_qwen] section of models.ini.
  Python: running generated tests, listing the mutation tool's operators, running the mutation tool
    run the tool check to try it
  Python: removing single tests from a test file, counting the tests and assertions of a test file
    start a run, which tries it for the first time

Full setup report: the same command with --display detailed
Next step: testuary check --verify with the same options (tries the installed tools, no model requests).
```

The items are grouped by what they mean, as the
[terminal guide](guide/terminal.md#the-checks-report) lists. Each problem
comes with the fix the check reports, and the report always ends with one
next step. Add
`--display detailed` to see every item by its plain name with its key,
status and reason, or `--json` for the report as data on standard output.

The tool check, `--verify`, also starts each tool once and tries your
project in a copy as a run does, without running a test. It contacts no
model:

```bash
testuary check --verify --model ollama_qwen --source-dir /path/to/project \
  --python-executable /path/to/project-env/bin/python \
  --source-files package/module.py --output-dir /path/to/new-results
```

For Python it imports the test tools with the selected interpreter in a copy of
the project, collects the project's tests there without running one, and writes
nothing into the project or its environment. For Java it runs the selected
`java` and `javac`, then, in the copy, resolves the build, compiles and finds
the tests, and starts the measurement tools a run needs. Like a run, it
downloads what the build needs and, unless `LANGADAPTER_JAVA_TOOLS_OFFLINE` is
set, the JUnit launcher, JaCoCo and PIT, within `setup_timeout_seconds`. A
download that fails, or that the variable turns off, is reported as a problem
with the step to take. The first run tries a step that the check had no time
left for, and the build of a project with its own build wrapper, which may
download its build tool. Running the tests and their measurements is a run's
work.
The tool check lists such parts under "not tried yet" and still ends with exit
status 0 when nothing else is open. Where the project's own runs build
offline, with Maven's `-o` or `--offline` in `MAVEN_ARGS` or in
`.mvn/maven.config`, the check builds offline too and reports a needed download
or an offline failure it does not know as a problem, since the first run would
meet it too.
The check's exit statuses:

<!-- exit-statuses: check -->
- Status 0: The check found no problem. Parts that only a real run can try are listed under "not tried yet".
- Status 1: The check found a problem, such as a missing tool or no selected model.
- Status 2: A setting was refused or a choice is ambiguous.
- Status 130: The check was stopped with Ctrl+C. No setup was changed.
<!-- /exit-statuses -->

Results of an earlier run that a run would continue or reuse are confirmed
only by that run. The check lists them under "not tried yet", and they do
not change the exit status.

## Run with small limits

For a first run, keep the model's work small. Put these lines into the
project's settings file, `.testuary/project.ini` in the project's directory,
which the wizard writes or you create:

```ini
[starter]
suite_count = 3
attempts_per_suite = 2

[coverage]
max_requests_per_phase = 20
max_output_tokens_per_phase = 200000
max_wall_clock_seconds_per_phase = 1800

[mutation]
max_requests_per_phase = 20
max_output_tokens_per_phase = 200000
max_wall_clock_seconds_per_phase = 1800
```

Then run the target in the project's directory, where the run finds that file:

```bash
cd /path/to/project
testuary generate --model ollama_qwen \
  --python-executable /path/to/project-env/bin/python \
  --source-files package/module.py
```

A run takes each target through four phases. In the starter phase the model
writes several independent test suites, the starter suites, and Testuary
picks the best result from them. The coverage phase then lets the model improve
that suite so that it covers more lines and branches. The mutation phase does
the same for the mutation score. A mutant is a copy of the code with one small
change, and a test that fails on a mutant kills it. The mutation score is the
percentage of mutants the tests kill. Each loop stops at its goal, at its
limits, or after too many measurements without improvement, as the
[settings reference](reference/settings.md#coverage) sets them for each
loop. At the phase end of each of these three phases,
in its consolidation step, Testuary combines the suites, removes tests
that add nothing and keeps the best suite. The fourth phase, Finishing,
writes the results of the delivered suite: it takes the suite's coverage from
the phase end that handed it on and its mutation result from the phase end
of the mutation phase, so it judges the suite no second time. It measures on
its own only what those did not, such as the mutation score when the
mutation phase was skipped.

These lines allow 3 starter suites with up to 2 attempts per suite, so a
starter suite that fails, for example a reply without tests, is generated
once more. They
allow 20 model requests each for the coverage phase and the mutation phase,
and stop each loop after 200,000 reported output tokens or 30 minutes after
its first request. That is at most 46 model requests per target, not counting
requests sent again after temporary failures such as a lost connection, or
with half the output limit after a starter request did not fit the model's
context. Testuary does
not know what these requests cost. The lines change no goal or measurement,
but with small limits the results are not comparable with those of runs at
the default limits. An earlier run continues only with the same limits. The
[settings reference](reference/settings.md) lists every limit.

Java uses the same command with the project's JDK:

```bash
cd /path/to/java-project
testuary generate --model ollama_qwen --java-home /path/to/jdk \
  --source-files src/main/java/example/Calculator.java
```

The project directory stays where it is. Put quotes around Windows paths that
contain spaces. A new run directory must be missing or empty. Without
`--output-dir`, the run directory is `.testuary/results/<model alias>` in the
project's directory. Inside the project, a run directory must lie in its
`.testuary` directory. To
continue or reuse earlier results, see [results and logs](guide/output.md).

## Follow the run

In an interactive terminal the progress view stays pinned at the bottom. It
shows the whole run, the current target, its phase (Starter phase, Coverage
phase, Mutation phase or Finishing), the step within the phase and the
operations running now, with progress bars where totals are known. A Log row
names the run's plain-text log from the moment it opens. The log of finished
work scrolls above the pinned view. The [terminal guide](guide/terminal.md)
explains each view and the `--display` choices.

## Read the results

When the run ends, the run directory holds:

- `results/tests.csv`, one row per target and repetition with its coverage and
  mutation results.
- `results/<target>/r0/tests/`, the delivered suite of each target, in a
  directory named after the target's path. `r0` is the first repetition. A
  target run more than once with `--runs` has a directory per repetition.
- `logs/run.log`, every log line of the run as plain text.

At the end the run prints a summary with the outcome of each target, its
delivered suite, its scores and its model use. When a target is unfinished, the
summary ends with the command that continues the run.
[Results and logs](guide/output.md) describes the directory, the exit status and
how to continue a run.
