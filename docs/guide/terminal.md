# Terminal output

Testuary shows its progress in the terminal and keeps it separate from the
results and logs it saves. The display mode changes only what you see. It does
not change generation, measurements, limits or which suite is chosen.

Preview the views with the offline demonstration, which needs no model:

```bash
testuary --demo --display compact
testuary --demo --display detailed
testuary --demo --display plain
testuary --demo --display detailed --no-color
```

For your own project, add the same options to the `generate` command from
[getting started](../getting-started.md).

## Views

Select a view with `--display`. The
[command-line reference](../reference/cli.md#testuary-generate) names its default.

- `auto` uses `compact` in an interactive terminal and `plain` when the output
  is redirected or the terminal has few capabilities.
- `compact` keeps a progress view pinned at the bottom of the terminal.
- `detailed` writes every update as a line of its own, without the pinned
  view. A count that only grows gets a new line at each ten percent, or every
  30 seconds when its total is unknown, and work that waits gets an occasional
  line.
- `plain` writes the same lines without color or cursor movement, for CI and
  log files.
- `debug` is the detailed view with internal identifiers added, and it also
  echoes every log line, for diagnosing a run.

An explicit `compact` or `detailed` also falls back to plain lines outside an
interactive terminal. No full-screen interface is used. Every plain or detailed
record is one line, so paths and commands are never split. The terminal wraps
long lines only on screen. Colors follow `--no-color`, or without it the
`NO_COLOR` environment variable. Progress goes to standard error. The display
does not capture standard output, and a few diagnostic lines still go there, such
as the list of a project's targets before a run and, with `--display debug`,
the path of the structured log.

## The pinned view

The compact view shows the run in layers, from the top down: the whole run, the
current target, its phase, the step within the phase and the operations running
now. This frame comes from the offline Python demonstration during the
coverage phase, in a terminal 100 columns wide (log path shortened):

```text
Testuary | demo playback
Overall  RUNNING
         ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  25%  1 of 4 phases done
Log      demo-python/run/logs/run.log (every step: --display detailed)
Target   tenacity/wait.py
Phase    Coverage phase | phase 2 of 4 | RUNNING
Step     Read tests | RUNNING
 State       Operation                   Progress                                 Playback
 RUNNING     Model reads the tests                                                playback 00:00:03
Model use, Coverage phase: model requests 1 of 20, 3,635 input tokens, 33 output tokens
```

In a demonstration the clock column reads Playback and counts playback time. In
a normal run it reads Elapsed and counts the time the run has taken.

The Overall row counts the phases done of all targets together, and
of every repetition when each target runs more than once (`--runs`), such as
"5 of 8 phases done (2 targets)" or "(2 targets x 3 repetitions)". Here one target
has four phases and the first is done, so it shows 1 of 4. Finishing is the
fourth phase and counts as one, so 100% means that Finishing is done too. Its
row reads "Finishing: measuring the delivered suite and writing the results". In
a run of one target and one repetition, as here, the Overall bar is the only
one: the Target row names the target, and the Phase row its phase, its place
among the phases and its state. With several targets or repetitions, the Target
row counts finished targets and the Phase row the phases of the current target,
each with a bar of its own. The Step row names the step of the phase that is
running, here the coverage phase's Read tests.

The four phases are the starter phase, the coverage phase, the mutation phase
and Finishing, as
[getting started](../getting-started.md#run-with-small-limits) explains. The
starter phase's steps are Generate, Measure suites and Consolidation.
A loop's steps are Read tests, First measurement,
Edit tests, Run tests, Measure and Consolidation. Above the
pinned view, a log scrolls with the start and the end of each run, target, phase
and step and the outcome of every operation. Count updates stay in the pinned
view. Recent failures stay visible while other work continues.

The Log row names the run's plain-text log from the moment it opens, before the
first target starts. Use `--display detailed` to see every update as it
happens. A path too long for the terminal keeps its end, the log's directories
and file name, behind `...`. The run's records keep the whole path.

## Progress bars and estimates

A bar appears wherever Testuary knows how much work there is, except on the
Target and Phase rows of a run of one target and one run, which would repeat
the Overall count. It shows the
percentage, the count and how long that item has run. Percentages are rounded
down, so 100% and a full bar mean that all work is done. Work that is counted
without a known total shows "total unknown" instead of a bar. Mutation
testing counts the mutants run and the caught mutants re-checked, never the
mutants caught, so its count, such as "9 of 37 mutants run", has no
percentage that could read as a score.

Evenly sized measurement work, such as mutants or timing iterations, also shows
an estimate like "about 3 min left" once three items have finished while in
view. Model phases never get an estimate, because the time of a model request
varies too much. The Overall row shows the total time of the run once it ends.

A model request is one question to the model, and the request limit counts
these. A request sent again after a connection error is not counted again. The
count reads "model requests 6 of 8", and the requests sent follow only when
they differ, as in "model requests 6 of 8 (7 sent, 1 sent again after
connection errors)". The detailed view always shows both. A request limit is
a ceiling, not a forecast of how many requests a phase will need, so it never
appears as a bar. A
coverage or mutation goal appears beside the measured value and its basis,
for example "Line coverage: 80% (goal 100%) [measured]". Before the value
is measured, only the goal is shown. Request counts and their limit belong
to the current target and phase and start again with the next one.

## Outcomes

A target that ends without a suite that meets the minimums is a result, not a
failure to execute. It is shown as MINIMUMS NOT MET in a warning color. A target
that failed to execute, for example because it crashed or could not be set
up, stays FAILED in red.

A loop that ends at its request, token or time limit ends as its settings
say, which is no failure either. The request refused at the limit and the
step and work the loop ends name that limit in yellow, and they are not
among the recent failures that stay visible. A failure in the same loop
still is. The end-of-run summary names each loop that a limit stopped before
its goal in a `Warning:` line.

Measured zero is different from unavailable, incomplete, unsupported or failed
data. Progress values and reused results are labeled as such. Model requests,
requests sent, input and output tokens, cache-hit tokens and cost are shown
separately where they are known. A value that is not known is not shown
as zero. When a failed request has unknown usage, the total token
count of the run stays unavailable rather than counting only the requests that
succeeded. Reused mutation results, a reused list of mutants and the model
service's cache-hit tokens are different things and are not merged into one
rate.

Tool rows name what the model does in plain words, such as "Model reads the
tests" or "Tests run", and its outcome, not its arguments or output. The
detailed view adds the tool's own name, such as `read_tests`. While work runs, metrics that are not measured yet are left out. A
finished operation shows the state of each metric. Processing a saved response
is labeled as recorded work, not as a new model request. Phases that are
already complete show their reused results only after their resume checks.

Prompts and model responses are not part of the terminal view. What the saved
logs contain does not depend on the display, and
[results and logs](output.md#logs) describes them. The display hides common credential
patterns, but that does not replace protecting the run directories and your
private configuration.

## The check's report

`testuary check` runs the quick check and writes its report in plain words.
The first line says how many problems it found and how far it looked. The
quick check reads the settings and finds the tools. The tool check,
`--verify`, also starts each tool once and, in a copy of the project, tries its
build where a run builds it and finds its tests without running one, unless the
project has its own build wrapper, which may download its build tool. Both
install nothing, run no test and contact no model. Required parts that only
a run can try do not change the exit status. The check ends with an exit
status other than 0 when it found no problem but a required part that the
check itself should have tried was not tried. The first line then says so, and
the next line says how far the check looked.

Then come the project, the targets, the model, the run directory, the phases and
the request, token and time limits of each phase. The report says whether an
API key is set, but never tries the key. The items follow in groups: found,
installed but not tried, not tried yet, missing or blocking with the fix the
check reported, optional, notes, and the files that hold more detail. Under "not tried yet", an item listed alone
shows the reason the check gave, and on the next line the step to take.

The report ends with one next step:

- With one problem, the fix the check reported.
- With several problems, fixing them and running the same command again.
- After a quick check without problems, `testuary check --verify` with the same
  options.
- After a tool check without problems, a run with `testuary generate`.

When no fix was reported, the next step points to the detailed view instead of
guessing one. A refused setting stops the check before it looks at the project,
and the first line says so. `--display detailed` lists every item by its plain
name with its key, status and reason, also when the output is redirected. Its
status words are the summary's: not tried yet, found (quick check) or works
(tool check), missing, not supported, more than one found and refused.
`--json` writes only the report as data to standard output, with the
status values of its format.

[Getting started](../getting-started.md) shows a complete report.

## Interruptions and problems

A refused configuration is not a failed run. It can happen before the run
directory or any log exists. Correct the reported input, and rely on a saved log
only when the report names its path.

Ctrl+C stops the run through its normal cancellation and restores the terminal.
Work that already finished is kept. A second Ctrl+C does not cut the cleanup
short: the run waits for the measurement or test run in progress to stop and
clean up, and prints "Stopping: the work in flight ends and cleans up first,
then the run ends." While the stopped run closes what it cancelled, up to <!-- cleanup-seconds -->120<!-- /cleanup-seconds -->
seconds and usually a few, further Ctrl+C presses are ignored. The command then prints "Stopped by you."
with what you can do next, for example "Continue with the same command and
--resume.", and ends with status 130. [Results and logs](output.md#the-end-of-a-run) lists every exit status. If the display itself fails, Testuary falls
back to plain lines. A failed measurement or a missing required record stays an
execution error and is never hidden by the display.

## For programs

### Observing a run

Pass `event_sink=callback` to `run_pipeline` to receive each observation as a
frozen `testuary.presentation.DisplayEvent`, including phase transitions after
checkpoints. The callback only observes: its return value does not affect
requests, retries, limits or results. Missing totals and metrics stay unknown
rather than zero. A callback that fails is switched off, and the run continues.

### Saving the events to a file

`--events FILE` writes what the display of a command-line run shows to a
file, for programs that show the run elsewhere. The first line names the
format (`schema_version` 3) and the Testuary version. Each further line is one
JSON object with one record, in the order the display shows them:
`{"event": ...}` holds a `DisplayEvent` with its fields, `{"readiness": ...}`
the setup report before the run, as `RunReadiness.as_dict()` gives it,
`{"plan": ...}` what the run will do, a `RunPlan` with its fields, and
`{"completion": ...}` the end of the run, a `RunCompletion` with its fields.
Credentials are removed from every text, as the display removes them. The
file is written line by line, so a program can follow it while the run goes
on. The directory must exist, and an existing file is replaced. With `--demo`
the file holds the playback's records, with the demonstration's provenance.
If a write fails, the file stops there, one line on standard error says so,
and the run and its display continue.

A program that starts a run with `run_pipeline` names its event file with
`events=PATH`, and Testuary writes the same records into it, without a
command that resumes the run. A relative path starts from the program's
current directory. The path is checked before the run, and a call refused
before its run may start writes no file. A program of your own writes the
same format through `testuary.presentation.EventFile`, with the plan and
completion types it records (the
[API reference](../reference/api.md#event-files)). The format's version is
`testuary.presentation.EVENT_FILE_SCHEMA`.

### Writing through the presenter

When you use `TerminalPresenter` directly, send your own diagnostic output
through its callback so that the pinned view makes room for it:

```python
from testuary.presentation import TerminalPresenter

with TerminalPresenter("detailed") as presenter:
    presenter.write_diagnostic(lambda: print("Preparing selected inputs", flush=True))
```

The callback runs once, on its own stream, with its own text. The compact view
pauses around it and resumes on a new line. Plain and detailed views leave the
write alone. Exceptions from the write propagate, including cancellation. The
presenter does not hide secrets in or save this output. That stays the caller's
logging policy. It does not capture other libraries' output or replace the
standard streams. Use detailed or plain output when your writes cannot go
through this callback.

### Showing a setup report

`present_readiness(report, presenter=presenter)` writes a
`testuary.readiness.RunReadiness` value within the same presenter. It renders
the report's scope, selected inputs, limits and fixes without checking or
starting anything. Statuses and fixes come only from the report. A set API
key is not a tried key, and says nothing about the provider's quota, context
size or cost. Log paths appear only when the report contains them.

Before its first model request, generation shows the report with
`before_run=True`. It then opens with "Setup before the run", in the detailed
view too, without a check's name or an exit status, and advises no further
check. It ends with "Generation starts now" only when there is no problem and
the report's exit status is 0, otherwise with the reported fix or a pointer to
the detailed view. A demonstration's playback adds `playback=True`: it sends no
model request, so the report leaves out the API key and ends with "The
playback starts now".

### Showing the end of a run

Describe each target in a `FileCompletion` from
`testuary.presentation.terminal`. Its `outcome` is one of `complete` (a suite
that meets the minimums), `no_complete_suite` (no suite that meets the
minimums), `failed` (failed to execute), `cancelled` (stopped before
finishing), `already_complete` (a suite that meets the minimums, recorded
earlier) and `stopped` (ended after the phase that `--stop-after-phase`
requested). Add the `reason`, the delivered `suite` paths,
the final `metrics` with their basis, the model `usage`, and the `fallbacks`
and `warnings` of its runs, each a sentence. Set
`resumable=False` for a target that a resume would skip. Collect the
targets in a `RunCompletion` with the `results_directory`, the `log_path`, the
exact `resume_command` that continues the run and `cancelled=True` when the run
was stopped.
`present_completion(completion, presenter=presenter)` writes one headline that
counts the targets with a suite that meets the minimums (including those recorded
earlier), the targets without one, the targets that failed to execute, the targets
stopped before finishing and the targets that ended after the requested phase. One block per target
follows and, when a resume would continue some targets and a command is given,
that command as the next step. Values are
shown as given. Unknown values are left out, never guessed.

### Mutation progress

Where the engine can observe its mutants, which is the case for local Python
runs, LangAdapter's mutation request reports `MutationProgress` records through
its `progress` callback. Show them on the run's clock:

```python
from testuary.presentation.measurement import mutation_progress_event

def show(progress):
    presenter(mutation_progress_event(progress, run_id="run-1", target="src/calc.py",
                                      elapsed_s=run_clock()))
```

The row "Mutation testing" counts the mutants run so far, with a bar
when the total is known and "total unknown" otherwise, and the time estimate
after three mutants. A count is progress, not a result, so it shows no
percentage. No score is shown,
because a mutant's result can still change until mutation testing ends.
The producer ends the `mutation-measurement` operation after each mutation run
with an observation of its outcome. The next run then starts its own count and
clock.

### Warmed Java timing

LangAdapter's warmed suite timing (`tests.benchmark_warmed`, described
in the timing guide of LangAdapter's documentation) reports progress and a result record. Show both with the same
presenter. Here `adapter` is LangAdapter's adapter for the project and `suite`
the suite to time.

```python
from langadapter.runs import WarmedSuiteBenchmarkRequest
from testuary.presentation import TerminalPresenter
from testuary.presentation.measurement import warmed_progress_event, warmed_result_event

with TerminalPresenter() as presenter:
    def show(progress):
        presenter(warmed_progress_event(progress, run_id="timing", target="CounterTest"))

    result = adapter.tests.benchmark_warmed(WarmedSuiteBenchmarkRequest(
        suite, "native_lifecycle", iteration_timeout_s=60, total_timeout_s=900,
        on_progress=show))
    presenter(warmed_result_event(result, run_id="timing", target="CounterTest"))
```

Progress counts finished warmup and measured iterations. An iteration that has
only started is not a sample. Single iteration times are shown as raw values,
never as a summary. The five summaries (mean of fork means, median of fork
medians, smallest and largest sample, standard deviation between forks) appear
only for a successful, complete result, exactly as LangAdapter computed them.
Otherwise the line states the validity and "no timing summary". One fork has no
standard deviation between forks. Durations of zero are marked as limited by
the timer's resolution. No comparison or speedup is claimed.
