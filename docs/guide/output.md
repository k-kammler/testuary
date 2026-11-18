# Results and logs

Every run writes into its run directory, which
[`[project] output_dir`](../reference/settings.md#project) chooses, or
`--output-dir` at the terminal.

## The run directory

A new run needs a run directory that is missing or empty. Testuary refuses a
directory that already holds files and deletes nothing in it, neither earlier
results nor logs nor unrelated files. Inside the project you test, only its
`.testuary/` directory may hold results. A path that passes through a link or
a reparse point, at any level, is refused.

After a run the directory holds:

```text
results/
  tests.csv                 one row per target and repetition with its scores and counts
  summary.json              the summaries of all targets
  completion.json           the end-of-run summary the console printed
  <target>/
    phase_state.json        which phases of each repetition are finished
    summary.json            the summary across the target's repetitions
    r0/                     the first repetition, r1/ the second, and so on
      tests/                the delivered suite and its measurements
      archive/              the suites and records of each phase
      resolved_settings.json  the settings the repetition used
      summary.json          the repetition's final values
      time_limits.json      the time limits the repetition reached, only when it reached one
logs/
  run.log                   every log line of the run, as plain text
  main.jsonl                the run's structured log
  test/<target>/r0/         the logs of each repetition, including the model request records
_work/                      temporary work directory, empty after the run
```

Each target gets a directory named after its path, for example
`src_main_java_example_Calculator` for `src/main/java/example/Calculator.java`.
Each repetition of the target has a directory inside it, `r0` for the first. A
single repetition is `r0`, and `--runs 3` makes `r0`, `r1` and `r2`. The
`archive/` directory keeps the starter suites and the suites of the coverage
phase and the mutation phase, so you can compare them with the delivered suite.
The [result files reference](../reference/result-files.md) lists the columns of
`tests.csv` and the fields of the summaries.

## Several repetitions

[`--runs` and `--run`](../reference/cli.md#testuary-generate) repeat each
target's phases, each repetition in its own directory. `--run` adds a
repetition later or runs a failed one again. Each repetition's
starter suites use their own
[sampling seeds](configuration.md#sampling-seeds-of-the-starter-suites), so a
repetition draws the same seeds whether it runs alone or with others.

## The delivered suite

For a Python target, the delivered suite holds the test file, the
`conftest.py` it was measured with and a `pytest.ini`. The `conftest.py`
makes the project importable from the directory the tests were measured in.
The `pytest.ini` holds only an empty `[pytest]` section. It makes the test
directory pytest's root directory, so that a pytest configuration in a
directory above it does not apply, and every Python target's suite gets it.
The project's own pytest configuration is not part of the delivered suite:
each test run reads an exact copy of it beside the run's copy of the suite.

Every question whether tests pass is answered by one normal test run. For
a Python target, the normal run is the run as delivered: the test files run
one after another, each in its own pytest session, its tests in the order
pytest collects them. Each session hashes strings with the fixed seed 1
(`PYTHONHASHSEED=1`). When it starts, the global random generators of
Python, NumPy and SymPy get the same seed, 1, and so does Hypothesis
unless the project's pytest configuration sets one. A generator that a
test makes itself is not seeded. For this the session imports NumPy and
SymPy where they are installed, which a user's run does not do. To repeat
a Python suite's check as Testuary ran it, run each test file in its own
pytest session with `PYTHONHASHSEED=1` and these generators seeded with 1
when the session starts. For a Java or C file the normal run is the
language's own test run, which takes no seed. Nothing resets state between
tests, as in a user's run, so an accepted suite does not depend on a
cleaner that the user does not have. A test file of the delivered suite can need
its `conftest.py`: keep it with the tests. The starter suites' cleanup removes
the tests that make a suite fail its normal run, since the model never
works on those suites. When no starter suite passes and is fully measured,
the run stops. At the end of a loop, a suite that fails its normal run is
no candidate, no test is removed from it, and the record names the failing
tests. The suite the loop received stays a candidate, so the end of a loop
always has one to choose. A loop does not check the suite it inherits
before it starts: its first measurement shows the model what the suite
does. In the mutation phase, a suite that fails its normal run there, before
any edit, ends the phase as skipped. A pass holds for that run, not for
other test orders, subsets or environments.

A passing test run of a loop can carry a "STANDS ALONE" note for the model.
For Python it names the tests that left the module under test or a
library setting changed. With `report_reverse_order_failures = true` in
`[coverage]` or `[mutation]`, a passing Python run of that loop is also
followed by a run of the file as delivered in reverse order, and the note
names the tests that fail there, or says why the file could not run in
that order. A language whose tests do not run in reverse order, such as
Java, gets no such run and no note. A note decides no test's outcome.
After a note naming tests, the model can change the tests before it
measures, and a loop at its goal ends only after a passing run without
such a note. The loop's request, token and time limits still end it, so a
delivered suite can keep such a test.

Put the delivered suite's test file next to the project's own tests, not inside its package.
A project's rules for its own modules can apply to a file placed there. For
example, xarray requires `__slots__` on every subclass of its types defined in
an `xarray.*` module, and a test module inside the package becomes such a
module.

## The end of a run

At the end, the run prints a summary. A target has a suite that meets the
minimums when every phase handed on a suite and the suite reaches the minimum
coverage and mutation score you configured (the settings `minimum` in
`[coverage]` and `minimum` in `[mutation]`). The summary's first line counts
the targets with a suite that meets the minimums, the targets without one and the
targets that failed to execute. One block per target follows with its outcome and
reason, the delivered suite, its line coverage, branch coverage and mutation score
when the target had one repetition, its model use, and any fallback (a step
skipped because the language lacks it) or warning of its repetitions. When
targets are unfinished and the run can be continued, the summary
ends with the exact command that continues it. The same list of unfinished
targets is in the log.

The exit status says how the command ended:

<!-- exit-statuses: run -->
- Status 0: The run finished. A suite below a configured minimum, a phase that ended without a suite and a target an earlier run in this directory had finished are results, recorded in tests.csv.
- Status 1: A target failed to execute: it crashed, could not be set up, copied, restored or delivered, or some generated starter suites could not be checked, because running them timed out or did not start. The summary and the run's log say why.
- Status 2: A setting or option was refused, or a required tool was missing, before any work started.
- Status 130: The run was stopped with Ctrl+C. Work that already finished is kept, and the run can be continued in its run directory (at the terminal: --resume).
<!-- /exit-statuses -->

The replies of starter suites that the validity check could not judge are
kept and checked again with `--resume`. [Outcomes](terminal.md#outcomes) says
how the terminal shows each target's outcome. A run you interrupt with Ctrl+C
prints "Stopped by you." with what you can do next.

## Continuing a run

To continue an interrupted run, run the same command again with `--resume` and
the same run directory. For each repetition of each target, Testuary skips the
phases that are finished and resumes from the first one that is not. A phase
counts as finished only when its checkpoint, the hash of its recorded results and
the hash of the target's canonical source (its file in the source directory),
which the repetition recorded when the phase started, all match. A later phase is never trusted
past an earlier gap. A repetition that is finished is kept, so `--resume` with
a higher `--runs` count adds only the missing repetitions.

To reuse saved phases of an earlier run instead of running them again, use
`--reuse` with a comma list of phase names: `starter` for the starter
suites, `coverage`, `mutation` and `finishing` for Finishing. Reusing the starter suites,
the coverage phase or the mutation phase keeps their saved results. A phase is
reused with the phases it builds on: `coverage` reuses the starter suites
too, and `mutation` the coverage phase and the starter suites. For example,
`--reuse coverage` runs the mutation phase again on the coverage phase's saved
suite. `finishing` only skips Finishing and reuses no results. A reuse takes
the saved results of the first repetition, `r0`, for every repetition.
`finishing` skips Finishing, which only a run itself makes, so it works with
the first repetition alone. The
[command line reference](../reference/cli.md#testuary-generate) says which
options `--resume` combines with.

A run directory without generation records, or with records in an unknown
format, cannot be resumed or reused. The run refuses before any work, says
so, and asks for a new run in an empty run directory. A resume or reuse
also refuses changed generation conditions, such as another provider,
model, prompts or limits. Another address of the same model service is a
warning, not a refusal, as [models and prompts](../models-and-prompts.md)
describes.

If the temporary work directory cannot be cleaned, the run stops rather than
continuing over leftover work.

## Logs

`logs/run.log` holds every log line of the run as plain text and can be read
while the run is going. The terminal names it from the moment it opens.
Each repetition of a target has its own logs, with the model conversations
of its loops. The [result files reference](../reference/result-files.md#logs)
lists the log files and what each holds.

## Model request records

Every model request sent is recorded with its outcome and whether the
model service reported its token usage: the loops' records are among the
repetition's [logs](../reference/result-files.md#logs), and the starter
suites' records are in their
[archive](../reference/result-files.md#starter-suites). The
[result files reference](../reference/result-files.md#testscsv)
describes the request and token columns of `tests.csv`.
