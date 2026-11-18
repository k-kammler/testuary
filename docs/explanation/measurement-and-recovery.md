# Measurement and recovery

## Reusing measurements of an unchanged suite

At a phase end, Testuary normally measures the suites, reduces them and
selects one. When the coverage phase or the mutation phase ends with exactly
the suite it started from, Testuary can take over the measurements it
inherited instead. This needs the same suite bytes, the same measurement
setup (the version of the target, the set of mutants and the
settings for attributing kills to tests), and complete measurements that
belong to those bytes. Whether they belong is checked per file: the inherited
per-test coverage must carry the hash of the previous phase's delivered suite
itself. That holds when the previous phase handed on its reduced suite, its
combined suite or the suite its loop's model wrote. It does not hold when
the starter phase handed on a single starter suite, whose record keeps no
per-test coverage under the delivered suite's name, and such phases take the
normal path. Reuse covers two cases: the goal was already met when the phase
started, or the model ran but changed nothing. The model's usage is still
reported. The phase archive then holds the files a measuring phase's
phase end writes. Each matrix, such as the kill matrix that records which test
killed which mutant, is a byte-for-byte copy of the inherited phase's final
matrix, the validity records are written for the delivered suite's bytes, and
its `consolidation_record.json` names `inherited` as the chosen suite, with
the source phase and the copied files. Any difference takes the normal path,
including a measurement over a selection of mutants where all mutants are
required. `phase_end_evidence_reuse = false` in `[coverage]` measures an
unchanged suite again.

## Measured and derived mutation scores

Some steps of a run need the mutation score of a suite that is also part of a
larger measurement: each starter suite in the starter phase, and the reduced
suite at every phase end. Testuary can obtain such a score in two
ways.

Measuring runs the mutants against the suite on its own. The score is exact
for that suite, and each suite costs a full mutation run.

Deriving reads the score off a measurement of a larger suite: one pooled run,
a single mutation run of all starter suites together, restricted to the tests
of each suite, or the combined suite's measurement, restricted to the tests
the reduced suite keeps. A
mutant counts as killed for a suite when one of the suite's tests killed it
in the larger run. Deriving saves mutation runs, but its result is not
guaranteed to equal a measurement of the suite on its own. Tests can affect
each other when they run together, through shared state or their order. The
pooled run of the starter suites records every test that kills a mutant, and
so does the combined suite's measurement under
[`cover_attribution`](../reference/settings.md#mutation) `full`. Under
`first` the combined suite's measurement stops at the first
killing test, so the kill is credited to that test only. A timeout counts for
every suite that covers the mutated line, or for every suite when its line
is unknown, and a kill that names no test counts for every suite.

Two settings choose the method:

- `suite_mutation` in `[starter]`, for the individual starter suites.
- `reduced_suite_mutation` in `[main]`, for the reduced suite at each
  phase end.

[Settings](../reference/settings.md) gives their defaults.

For the starter suites, `derive` and `both` run all of them together once.
Python starter suites are pooled whenever one of these is selected, each in a
file that carries its starter suite's number in its name, such as
`test_candidate_suite_001.py`.
Java starter suites usually declare the same test class, so pooling them
renames each suite's top-level test types in the pooled copy. Suites the
language cannot run together are measured alone, and the record says why.

Both take the same three values:

- `measure`: each suite is measured on its own. Exact. In the starter phase
  this costs one mutation run per accepted starter suite.
- `derive`: the scores are derived. This may be faster, without a guarantee
  that the scores equal measurements. Finishing reports no derived score: it
  measures a delivered suite whose score was derived, recording every
  killing test. A derived starter score only ranks the
  starter suites: a starter suite that can be selected is measured on its own
  before it is, so the selection always rests on a measurement. The reported
  starter statistics, such as the mean score of the individual starter
  suites, are then derived, and `starter_suite_mutation_method` reads
  `derive`. When the pooled run
  cannot be staged, its suites do not pass when run together, it gives no
  result, or it did not record every killing test (a Java run that fell back
  to each kill's first killing test), each suite is measured instead, and the
  record names the method used and the reason. No test is removed
  from the pool.
- `both`: each suite is measured, and the derivation is made as well and
  recorded beside the measurement. Measured results decide the selection, the
  reported scores and the mean score of the individual starter suites. The
  derived results are kept for comparison, with every mutant whose derived and
  measured verdicts differ. For the starter suites this costs one pooled run
  more per starter phase. When the pooled run cannot be staged, its suites do
  not pass when run together, it gives no result or it did not record every
  killing test, the derivation is recorded as unavailable with the reason,
  and the measured results still govern. When the measurement of the individual
  suites gives no result, the starter suites have no mutation score, as under
  `measure`. The derivation is still recorded but decides nothing, and the
  record says why the measurement is missing.

The phase archive keeps the records of both methods: for the starter suites
the derivation record, each suite's derived matrix and, under `both`, the
pooled run's matrix, and for a reduction the derived and the measured matrix
side by side, each bound to the hash of the suite it describes, so the two
can be compared mutant by mutant.
[Starter suites](../reference/result-files.md#starter-suites) and
[Repetitions and their summaries](../reference/result-files.md#repetitions-and-their-summaries)
describe these files, and the results table's
`starter_suite_mutation_method` names the method the starter suites used.

### Resuming with changed settings

A run continued with `--resume` keeps the measurement settings it started
with. The resume compares `phase_end_mutation` and `reduced_suite_mutation`,
for the starter suites `suite_mutation` and `failing_test_cleanup_rounds`, the
mutation engine, its mutation operators
(the full catalogue or a selective catalogue) and when it switches between them, how
kills are credited to tests, the timeout rules, the reverse-order note
(`report_reverse_order_failures`) and the suite reduction's settings. When
one of them differs from the run's record, the resume is refused before any
work, and the message names the changed settings. Finishing's settings are
not compared: when they changed, the resume measures the delivered suites again.
The worker count and the distributor are not compared: they change how long
a measurement takes, not its verdicts. To measure with other settings, start
a new run in an empty run directory. A run directory whose records lack these
settings cannot be resumed.

## Tests removed after a failing measurement

A suite can pass its normal run, the run of its tests without a measurement
tool, and still fail under a measurement tool. A test that passes in the
normal build can crash in the mutation tool's build, for example, or fail only
while the coverage tool traces it. At the places below, Testuary removes the
tests the failure names and measures the suite again.

The removal acts only where no model is active:

- In the starter phase: the coverage probe of a new starter suite, each
  starter suite's per-test evidence run and mutation measurement, and a
  starter suite that reaches the best scores, measured once under the
  delivered suite's name before the selection.
- At a phase end: the merged suite's coverage run and mutation measurement,
  and the inherited suite's when it is measured again.
- At a phase end with exactly two candidates, the coverage run and mutation
  measurement of the second one alone: the phase's own suite at a coverage
  or mutation phase end, and the second of exactly two passing starter
  suites at the starter phase's end.

Everywhere else a failing measurement keeps its own handling and removes no
test. This covers the pooled run of the starter suites, the acceptance run of
the chosen starter suite, the reduced suite, the per-test evidence run of a
phase end's candidate or merged suite, which is then scored on its coverage
run, the repair trials, the timing runs, the measurement the mutation phase
starts from, the loops and Finishing.

After a removal, the changed suite runs its normal run again and then every
measurement its later steps read, so that a selection and the numbers it
records describe the suite delivered. No number measured before the removal
is kept for the changed suite, an inherited kill matrix included. When a new
measurement fails and names tests, the next round removes them.
`failing_test_cleanup_rounds` in [`[starter]`](../reference/settings.md#starter)
bounds the rounds for each suite at each place. A name the suite does not
hold is recorded and removes nothing. When a named case of a test cannot be
removed alone, its whole test is removed, and the record says why.

Nothing is removed when the failing measurement names no test, when it reached
its time limit or when the rounds are spent. Nothing is removed either when a
merged suite's coverage matrix cannot be read. The suite then takes the path
of any suite whose measurement fails, and so does a suite that a removal
leaves without a test or whose normal run fails after a removal. A starter
suite at its probe, its evidence run or its own mutation measurement is not
fully measured and leaves the starter suites, as
[Starter suites](../reference/result-files.md#starter-suites) describes. A
starter suite measured under the delivered suite's name and a phase end's
suite are left out of the selection, which chooses only among suites that
pass with complete numbers. The record names the place, the measurement and
the cause. When the mutation tool measures no starter suite at all, nothing is
removed either: every suite stays, and the starter suites have no mutation
score.

Each round writes one entry, with the place, the measurement and the removed
tests. [Starter suites](../reference/result-files.md#starter-suites) and
[Phase records](../reference/result-files.md#phase-records) describe where
the entries, the changed suites and their validity records are kept. A run in
which no measurement fails after a passing normal run removes no test and
writes no entry.

## Mutation measurements cut by their time limit

Every mutation measurement has a time limit, `measurement_timeout_seconds` in
`[mutation]`. A measurement that reaches it
usually ends with some mutants untested: they have no result. Testuary counts each
untested mutant as not killed, so the score is the killed mutants over all
mutants, the untested ones included. Such a score is a lower bound: the
suite's score over every mutant is at least as high.

In the mutation phase, a measurement cut this way gives the lower bound as its
score. The loop judges the best suite so far, whether a measurement improved
and whether the goal is reached by that score, so a cut measurement reaches
`goal` only when the killed mutants alone reach it. The loop's reply
to the model counts the untested mutants in the score's total and says that
the measurement reached its time limit, how many mutants were not tested and
that they count as not killed.

When the mutation measurement of Finishing reaches the time limit, Finishing
is not fully measured and reports no mutation score. Its lower bound is
recorded beside it, as [Phase records](../reference/result-files.md#phase-records)
describes.

Every time limit a run reaches is a warning in the run's summary, which names
the setting and its value: the mutation measurement's
`measurement_timeout_seconds`, a test run's `suite_run_timeout_seconds` and the final
coverage run's `coverage_timeout_seconds`. The end-of-run summary prints
each as a warning line, and the run's log has a line for each. [Repetitions and their summaries](../reference/result-files.md#repetitions-and-their-summaries)
describes the entries.

When measurements reach the limit, raise `measurement_timeout_seconds`. On a
cluster, slow shared storage is a common cause, which
[Node-local storage for the mutation engine](../guide/cluster.md#node-local-storage-for-the-mutation-engine)
addresses.

## Where measurements run

Testuary measures in private copies of the project, never in the project
itself. Each measurement prepares a copy, and a mutation session keeps one
copy for all its measurements. A Python test suite kept
outside the project on the same drive is copied with it, below their shared
directory, so that paths between them keep their shape. Paths in a run's output
are written back as the original project's, so the model and the results name
the files the user knows.

Python runs keep one bytecode store per run and work area. A later copy
reuses the modules an earlier copy compiled, as a run inside the project
reuses its own cache, and the store is removed when the run ends. Preparing
the copies, filling their bytecode store and closing them is reported apart,
as `time_preparation_s`, so the test, coverage and mutation times measure the
work itself.

## Several tool calls in one answer

A model answer can ask for several tools at once, even when the request asks
for one call per answer. The coverage phase and the mutation phase run them one
after another, in the order of the answer, and return the results in that
order, so several edits in one answer each build on the one before.
