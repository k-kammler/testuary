# Offline demonstrations

The demonstrations run Testuary's four real phases (the starter phase, the
coverage phase, the mutation phase and Finishing) without a model account, a
project checkout, a JDK or measurement tools:

```bash
testuary --demo --output-dir demo-python
testuary --demo --demo-language java --output-dir demo-java
```

`--output-dir` names the demonstration's run directory. Use a new or empty
directory. The command refuses to write over existing files.
Without `--output-dir` it creates a new directory named
`testuary-demo-<example>-<time>` in the current directory.
Without `--demo-language` it plays the Python example. Two more recordings
come with Testuary: `--demo-language toy` plays the closing run of
LangAdapter's tutorial on its small example language, and
`--demo-data tutorial` the run that `testuary tutorial` ends with, recorded
with that tutorial's prompt profile and hints. Every example goes through
the same steps as a normal run. The replies of the model and the answers of the
tools come from a bundled recording, the scenario, instead of a model and real
tools. The run directory and the display options do not change the scenario.

## What the values mean

Both examples are recordings of one real run of this Testuary release: the
Python example on `tenacity/wait.py` of the
[Tenacity](https://github.com/jd/tenacity) project, the Java example on
`JSONCompare.java` of [JSONassert](https://github.com/skyscreamer/JSONassert).
Each run used a real model, DeepSeek V4.1 Flash on Ollama Cloud, and the real
test, coverage and mutation tools. Both runs used the small run's limits of
[getting started](../getting-started.md#run-with-small-limits), one model
request at a time, and higher goals than the defaults: 100% coverage on the
lower of line and branch coverage, and a 100% mutation score, so that each
phase's loop works with the model. Getting started's small run itself keeps
the default goals. All four recordings used a starter temperature of 0.7. The
`toy` and tutorial recordings keep the default goals, and the tutorial's ran on
`tenacity/stop.py`. Every model reply and every result of an operation that started a
process was recorded, and the playback answers with it. Testuary itself
runs as in a normal run: it builds its prompts, makes its decisions and
writes its records again.

A coverage, mutation score, token count or runtime shown in either example is
the recorded run's value, not a new measurement on your computer. The
examples show the workflow and the shape of the output, not what a model
service can handle in general.

The recorded answers name no directory of the computer that recorded them. The
playback shows the directories of your computer in their place, mostly below the
run directory.

## Display

```bash
testuary --demo --display compact
testuary --demo --demo-language java --display detailed --no-color
testuary --demo --demo-language java --display plain --output-dir demo-log
```

Compact is the interactive default. Detailed shows every step in order. Plain
suits redirected logs. The display choice does not change the scenario, its
recorded data or its request limits. The [terminal guide](terminal.md) describes the
views.

## Speed and steps

```bash
testuary --demo
testuary --demo --demo-playback-seconds 60
testuary --demo --demo-speed 0.5
testuary --demo --demo-step --display detailed
testuary --demo --demo-speed instant
```

On an interactive terminal the playback runs at normal speed and takes at
least the seconds [`[demo] playback_seconds`](../reference/settings.md#demo)
gives. The variable `TESTUARY_DEMO_PLAYBACK_SECONDS` or that setting in your
configuration files changes it,
in the order of a run's [settings layers](configuration.md#run-settings), and
`--demo-playback-seconds`, described below, wins over both. The demonstration
reads only this one setting from your configuration and sets the rest aside.
The length is divided evenly over the playback's steps and events: each
observation the display shows waits the same share, so every phase, the final
measurement included, moves at a steady rate, whatever the recorded run took.
That length is a minimum: where the run's own work takes longer, the playback
takes longer and is never faster than that work. When the output is
redirected, in tests and in continuous integration, the playback is instant
unless `--demo-speed` names a speed.

`--demo-playback-seconds SECONDS` sets the length for one playback. `--demo-speed`
takes `instant` or a factor of that length's speed: `2` plays in half the
time and `0.5` in twice the time. `--demo-step` waits for Enter at each
phase start, each model answer and each phase result, for screenshots and
recordings. Only a key pressed while it waits counts, so pressing Enter while
the run works does not skip a later step. It needs an interactive terminal.
Ctrl+C ends a pause at once and stops the demonstration as it stops a run.

The speed changes the presentation only. The answers, results, request counts,
tokens and records are those of instant playback. The clock shows playback
time, marked `[playback time]` in each line and `Playback` in the compact
view, never the time the recorded run took. Every line and score is marked
`[demo]`.

## The setup wizard

```bash
testuary setup --demo
testuary setup --demo --demo-language java --output-dir setup-java
```

This shows [`testuary setup`](../getting-started.md#set-up-with-the-wizard) on
the demo's project and then plays back its run. The command writes the
project into a new run directory and runs the real wizard there. The demo
types the wizard's answers on the screen, at the speed of the playback, and
`--demo-speed`, `--demo-playback-seconds` and `--demo-step` work as above. Nothing
contacts a model or starts a tool. The wizard skips its check of the model
service, since the key's environment variable is not set: the demo sets aside
your environment. For the
Java example the demo creates a `pom.xml` in its project, a JDK directory and a
Maven executable as placeholders. The wizard names them, and nothing builds
or runs them.

The demo uses its run directory as the user directory, so it never touches
yours. The run directory then holds the files the wizard wrote there
(`models.ini`, `config.ini`), the demo's project under `workspace/` with the
project's settings file the wizard wrote in its `.testuary/` directory, and the
playback under `demo/`. Before the playback, the demo
checks that the written project names the recorded run's directory, targets and
model alias, and that the alias names the recorded model at its address. When the wizard asks a question the demo has no answer for, or writes
something else, the demo stops with status 3 and plays nothing. The wizard's
own check of the project's tools is left out, because it would start them.

## Configuration and output

The command prints its fixed scenario configuration and names the settings of a
normal run that it sets aside. It uses its own copy of the example project, the
workspace, and the recorded run's model alias, which it does not contact. It
uses the workspace as the user
directory, so it reads no model key from your user directory and writes nothing
there. The run directory, the example language and the display
choices still apply. During playback it sets aside your `TESTUARY_` and
`LANGADAPTER_` environment variables and those of tracing services such as
LangSmith and Langfuse, and it restores them when the command finishes.

The demonstration writes to the run directory you selected. The bundled
recording and all method settings stay unchanged. Unknown command-line
options are still refused.

The run directory holds:

- `DEMO.json` and `DEMO.md`, which identify the example, its configuration and
  its sources.
- `demo/ledger.jsonl`, which records each answer's class, source and rule.
- `demo/scenario.json.gz`, the copy of the bundled scenario the playback
  answers from.
- `demo/audit.jsonl`, which records any attempted subprocess or network
  connection.
- `run/` with the results and logs of the run, as described in
  [results and logs](output.md), and `workspace/` with the example project.

JSON records and JSON lines logs carry a `testuary_demo` field, and results
tables have a `demo` column. Source, reply and setup bytes are kept as they
are, with a `DEMO_NOTICE.txt` beside them. The licenses and notices of the
bundled sources ship with the package.

When you interrupt the playback, the labeled partial output stays, with a
cancelled outcome. It does not turn into measured results, and the scenario
cannot be resumed in that directory. Keep the partial output and use a new
directory for another playback.

## Exit status and outdated data

<!-- exit-statuses: demo -->
- Status 0: The recorded run was played back completely.
- Status 1: The demonstrated run failed, or the demonstration refused a process or a network connection a library started. What it wrote so far is kept and marked as demonstration output. Its log is under run/logs.
- Status 2: A usage error, such as a run directory that is not empty or a language without demonstration data.
- Status 3: The recorded data do not match this Testuary version. If playback had started, demo/stale.json names the first call that differed.
- Status 130: The demonstration was stopped with Ctrl+C. Start it again in a new or empty run directory.
<!-- /exit-statuses -->

When the demonstrated run fails, the command ends with that run's status.
The command does not rebuild the data and sends no live request in their
place. Report outdated demonstration data
at https://github.com/k-kammler/testuary/issues with the package version and
that record.

## Bundled data

The playback uses only the bundled scenarios, notices and source licenses. It
reads no records of other runs. Each scenario holds one recorded run, and
`manifest.json` beside it names what that run delivered, which a playback
delivers again.
