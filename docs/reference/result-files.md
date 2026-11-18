# Result files

## Names in the records

Most record keys and `tests.csv` columns use the words of the guides. The
tables below give the meaning of those that do not: `*_turns` counts tool
calls, `*_logical_requests` counts model requests, and the status
`budget_exhausted` means that a request, token or time limit ended the loop.
The phases are `starter`, `coverage`, `mutation` and
`finishing`. The phase directories of a run's archive are `archive/starter`,
`archive/coverage` and `archive/mutation`, and a starter suite's directory is
`archive/starter/suites/suite_NNN/`. A phase-specific column starts with
its phase (`coverage_requests`, `starter_status`). The headline values,
`line_coverage`, `branch_coverage` and `mutation_score`, have no phase: they
describe the delivered suite. Repetitions count from 0: the first repetition's
directory is `r0`, its row in `tests.csv` has `run` 0, and the summaries name
it `r0`. Each target a run works on is numbered by its position in the
selection, from 1 (`target_number`).

Time columns of `tests.csv` end in `_s` (seconds) or `_ms` (milliseconds),
and the summaries name a time `seconds`. Many records hold `t_mono_s`, a
reading of the computer's monotonic clock in seconds, and `clock_domain_id`,
the clock it comes from. Readings with a `start_`, `started_`, `ended_`,
`completed_` or `failed_` prefix are the same pair for that moment. Two
readings compare only within one clock domain. `schema_version` names a
record's format. A demonstration's records also hold `testuary_demo`, which
[Demonstrations](../guide/demonstrations.md) describes.

The records that a user or an analysis reads are described field by field
below. For the other files this page names the file and what it holds:
their fields are working data that this page does not list.

## tests.csv

`results/tests.csv` in the run directory has one row per target and repetition.
It is written as UTF-8 with a byte order mark and Windows line endings, so
spreadsheet programs open it correctly. A demonstration rewrites its table
with plain line endings when it adds its column `demo`. An empty
cell means the value is not known, not zero. A repetition that crashed still
writes its row: the phases it finished carry their values, the crashed
phase's status reads `crashed` with the error in `error`, and the columns
only Finishing writes stay empty. The headline columns:

| Column | Meaning |
| --- | --- |
| `target_number`, `run`, `file_path` | The target's position, the repetition (0 for the one in directory `r0`) and its path |
| `line_coverage`, `branch_coverage` | Final coverage of the delivered suite in percent, as the phase end that handed it on measured it, from Finishing. When Finishing did not run, as after a stop after the coverage phase or a reuse that skips it, the coverage phase's last values, measured on that loop's suite |
| `mutation_score` | The delivered suite's mutation score in percent, described below. When Finishing did not run, as after a stop after the mutation phase, it is the mutation phase's best score, which covers only the selected operators while the phase used the selective catalogue |
| `starter_status`, `coverage_status`, `mutation_status`, `finishing_status` | How each phase ended |
| `test_count_delivered`, `mutation_test_count` | Tests in the delivered suite as Finishing counted them (the coverage phase's suite when Finishing did not run) and in the mutation phase's own test file at the loop's end, before its phase end chose the suite it handed on (`mutation_delivered_tests`) |
| `starter_requests`, `coverage_requests`, `mutation_requests` | Requests sent, the ones sent again included |
| `starter_logical_requests`, `coverage_logical_requests`, `mutation_logical_requests` | Model requests counted against each phase's limit. They are fewer than the requests sent when a request was sent again: in the coverage and mutation phases after a connection error or another temporary failure, as `count_retries` in [`[coverage]`](settings.md#coverage) decides, and in the starter phase when a request was sent again with half the output limit after the model's context was full |
| `*_input_tokens`, `*_output_tokens`, `*_cache_hit_tokens` | Input tokens, output tokens and cache-hit tokens per phase (`starter_input_tokens` and so on), as the model service reported them |
| `*_unknown_usage_requests`, `*_unknown_cache_requests` | Requests for which the model service did not report the token use, or how much of it came from its cache. The token columns miss the use of these requests |
| `duration_s_target` | The repetition's elapsed time in seconds: its own work, its phases and the work between them, a failed attempt included. Empty when the repetition crashed, or when it was stopped or crashed and a resume reused work it had done, a phase it had finished or the starter suites it had saved, since the time of that work is not known |
| `duration_s_file_shared` | The time the target's repetitions share, outside every repetition, such as resume planning and the first copy of the project. The same in each of the target's rows and added to no repetition. For a target with one repetition, it and `duration_s_target` add up to the target's time |
| `time_model_s`, `time_test_s`, `time_coverage_s`, `time_mutation_s`, `time_merge_s`, `time_cover_s`, `time_preparation_s`, `time_other_s` | Where the repetition's time went, in seconds: model requests (in the starter phase, whose requests run in parallel, the time some request is in flight), test runs, coverage and mutation measurement, combining suites, suite reduction, the copies of the project a measurement runs in, and the rest. No second is counted twice. Like `duration_s_target`, they cover the repetition's own work, a failed attempt's time included, and `time_other_s` holds its work between the phases. An empty cell means that work did not happen. All are empty when `duration_s_target` is empty because the time is not known |

The other columns:

| Column | Meaning |
| --- | --- |
| `score` | `mutation_score` when there is one, else 0 when the coverage phase ended with failing tests (`coverage_status` `tests_failed`), else empty |
| `error` | Notes on the row, such as a failure or a coverage phase that stopped after measurements without improvement. A loop that a request, token or time limit ended adds none: its status reads `budget_exhausted` |
| `below_minimum_reason` | Why a suite ended below its configured minimum, a finished result a resume does not continue. Empty otherwise |
| `mutation_switched`, `mutation_selective_final`, `mutation_full_at_switch` | Whether the mutation phase switched from the selective catalogue to the full catalogue, the score over the selective catalogue it had reached and the score over the full catalogue at the switch. Empty without a switch |
| `test_loc_delivered`, `assertion_count_delivered`, `parametrize_count_delivered`, `fixture_count_delivered` | Code lines, assertions, parametrized cases and fixtures of the delivered suite as Finishing counted them (the coverage phase's suite when Finishing did not run) |
| `mutation_test_loc`, `mutation_assertion_count`, `mutation_parametrize_count`, `mutation_fixture_count` | The same of the mutation phase's own test file at the loop's end |
| `test_complexity_sum_delivered`, `test_complexity_max_delivered` | The summed and the largest cyclomatic complexity of the delivered suite |
| `target_loc`, `target_complexity` | Code lines and cyclomatic complexity of the target |
| `source_observations` | The counts behind the columns above as JSON, each with its status, reason and parser |
| `lint_smells_target`, `lint_smells_tests` | Findings of the language's linter (Ruff for Python) in the target and in the tests, with `[finishing] measure_lint = true`. Empty for a language without a linter |
| `suite_timing_median_ms`, `suite_timing_samples_ms`, `suite_timing_run_count`, `suite_timing_warmup_count` | Timed runs of the delivered suite, with `[timing] measure = true`: the median and every sample in milliseconds (separated by semicolons), and the numbers of timed and warm-up runs |
| `suite_timing_requested_cpus`, `suite_timing_allowed_cpus` | The processors the timed runs asked for, and those a process started like them was allowed to use |
| `starter_suites_generated`, `starter_suites_passing` | Starter suites generated, and those whose tests passed |
| `starter_suites_unmeasured_coverage`, `starter_suites_unmeasured_mutation` | Passing starter suites that left without being fully measured: the coverage tool or the mutation tool could not measure them. Each was generated again while its attempts lasted, else left out |
| `starter_merged_tests`, `starter_delivered_tests`, `starter_delivered_from` | Tests in all passing starter suites together and in the suite the phase handed on, and which candidate it chose |
| `starter_baseline_line`, `starter_baseline_branch` | Coverage of the suite the starter phase handed on |
| `coverage_merged_tests`, `coverage_delivered_tests`, `coverage_delivered_from` | The same at the phase end of the coverage phase |
| `mutation_merged_tests`, `mutation_delivered_tests`, `mutation_delivered_from` | The same at the phase end of the mutation phase |
| `coverage_turns`, `mutation_turns` | Tool calls of the coverage phase and the mutation phase. Accepted edits are not counted |
| `starter_duration_s`, `coverage_duration_s`, `mutation_duration_s`, `finishing_duration_s` | Each phase's elapsed time in seconds, of the attempts whose work the row holds: an attempt that crashed, or that failed in a phase other than the starter phase, counts for nothing. `starter_duration_s` is empty when the starter ran again with the suites a stopped or crashed repetition saved, since the time of that work is not known |
| `coverage_agent_duration_s`, `mutation_agent_duration_s` | The time of each loop's model requests and tool calls, without its first measurement and its phase end |
| `finishing_mutation_score_full`, `finishing_mutation_score_selective` | The delivered suite's mutation score over the full catalogue and over the selective catalogue, from one run. In Java, PIT tests a class's mutants in one JVM without running its static initializers again, so an earlier mutant can change a later mutant's verdict: the subset's score can differ from a run with the selective catalogue alone |
| `finishing_mutation_score_lower_bound`, `finishing_untested` | For a mutation measurement in Finishing that left mutants untested: its score with every untested mutant counted as not killed, and how many mutants it left untested. Empty otherwise |
| `finishing_timeout_kills` | Kills that came from the per-mutant time limit. A value above 0 means the score depends on timing |
| `finishing_line_coverage`, `finishing_branch_coverage` | Coverage of the delivered suite from Finishing |
| `starter_original_*`, `coverage_original_*` | Line and branch coverage (`_line`, `_branch`) and the mutation score over the full catalogue and the selective catalogue (`_mutation_full`, `_mutation_selective`) that the starter phase and the coverage phase recorded for the suite each handed on, as `[finishing] phase_scores` selects. Nothing is measured again for them |
| `mutation_original_*` | The same values for the mutation phase: they repeat Finishing's measurement of the delivered suite, whatever `phase_scores` selects |

The [completion summary](../guide/output.md) at the end of a run shows the
headline values of each target.

## Starter suites

The `starter_suite_line_*`, `starter_suite_branch_*` and
`starter_suite_mutation_*` columns summarize the accepted starter suites
(mean, median, standard deviation, minimum and maximum). Each starter suite is
measured on its own for coverage. For mutation,
[`[starter] suite_mutation`](settings.md#starter) chooses how each suite's
score is obtained, and under `both` the columns hold the measured scores. A suite that
passes but that the coverage tool cannot measure, or of which the mutation
tool measures nothing, also after the tests a failure named were removed,
is not fully measured: it leaves the starter suites,
it is generated again within its attempts (`refill_failed_suites`) or left
out, and the remaining suites are measured again without it. Its
record names the classification `coverage_unmeasured` or
`mutation_unmeasured` and the reason. When the mutation tool's check run
before the first mutant stopped the suite because a test failed, the reason
names every stopping test, or says `none named` when the tool names none.
`starter_suite_mutation_method` names the method used: a `derive` or `both`
that could not pool the suites measured them and reads `measure`.
`unavailable` means the per-suite measurement produced no result, so the
columns are empty, and an empty value means no mutation was measured. Under
`both` the derivation record still holds the derived values.
[Measurement and recovery](../explanation/measurement-and-recovery.md)
explains what the methods mean.

Each starter suite's directory in the starter phase's archive,
`suites/suite_NNN/`, holds the files of its latest attempt: the model's reply
(`response.txt`), the code taken from it (`original` with the language's
suffix), its record (`original.json`), the validity record (`validity.json`)
and, for an accepted suite, the cleaned suite. For an accepted suite this is the
accepted attempt. The files of every earlier attempt of the suite, which a later
attempt replaced, are kept in `suites/suite_NNN/attempts/<n>/`, where n is that
attempt's number for its suite. `original.json` records the number as
`slot_attempt`, and each attempt in `suites/_generation.json` records it too.
In a language without a collection step of its own, such as C, an attempt
discarded because its suite does not compile or link records
`collection_failure_reason` in the `cleanup` part of its `original.json`: the
failing run's reason with up to `collection_errors` of the compiler's lines.
For the suite as generated it is the first failing run's text, the text of the
run log's `[starter] collection failed` line. For a suite that fails to
compile after a cleanup round it is that round's run.

When a measurement of a starter suite removed tests, as
[Tests removed after a failing measurement](../explanation/measurement-and-recovery.md#tests-removed-after-a-failing-measurement)
describes, the entries go into `cleanup.measurement_removals` of the
attempt's `original.json`, in the form of the phase end's
[measurement removals](#phase-records), with `place`
`starter.zero_coverage_probe`, `starter.per_suite_evidence` or
`starter.per_suite_mutation`. The cleaned suite is then the suite without
those tests, the suite's `validity_derivations.json` holds its
`measurement_remove` record, and for an accepted suite the slot in
`suites/_generation.json` names the changed suite in
`measurement_removed_source`.

The other files of a starter suite's directory:

| File | Holds |
| --- | --- |
| `cleaned.json` | The accepted suite's per-test data: each test's covered lines, branches and kills as tokens, each test's time, the suite's time and its covered lines and branches of the target. Working data |
| `accepted/<run>/<attempt>/<sha256>` with the language's suffix | The cleaned code an attempt accepted, named by the run, the attempt and the code's SHA-256, which the attempt's entry in `_generation.json` names |
| `unconfirmed` with the language's suffix | A cleaned suite whose last confirmation run gave no answer, kept so that `--resume` checks it again |
| `validity_derivations.json` | The validity derivations of the suite's cleanup, in the format of [the validity records](#validity-records) |

`original.json`, the record of the latest attempt:

| Key | Holds |
| --- | --- |
| `i`, `slot_attempt` | The starter suite's number and the attempt's number for that suite |
| `temperature`, `seed` | The sampling temperature and the seed of the attempt's request |
| `attempts`, `unknown_usage_requests` | How many requests the attempt sent, ones sent again included, and how many of them reported no token use |
| `input_tokens`, `output_tokens`, `cache_hit_tokens`, `cache_miss_tokens`, `unknown_cache_requests` | The attempt's token use as the model service reported it |
| `generation_seconds` | The time of the attempt's request |
| `extracted`, `response_sha256`, `response_chars` | Whether the reply held code, and the reply's SHA-256 and length |
| `source_sha256` | The SHA-256 of the extracted code |
| `normalization` | Whether the code parsed (`parse_ok`, `parse_error`), the rule that removed same-named definitions an earlier one shadows (`rule_version`), whether it changed the code (`changed`, `before_sha256`, `after_sha256`) and what it removed (`removed_bindings`) |
| `cleanup` | The cleanup's outcome: `disposition_basis` and `disposition_reason`, the configured and used cleanup rounds, whether this was the slot's final attempt, the removals of [measurement removals](#phase-records) and the collection failure's reason |
| `validity_record` | The file of the validity record, `validity.json`, when the check wrote one |
| `classification` | The attempt's outcome, such as `all_passing`, `discarded_syntax`, `discarded_collection`, `coverage_unmeasured` or `mutation_unmeasured` |

`suites/_generation.json`, the starter suites' manifest. It is written after
every processed reply, so an interrupted generation keeps what it spent:

| Key | Holds |
| --- | --- |
| `requests_used`, `logical_requests`, `unknown_usage_requests`, `unknown_cache_requests` | Requests sent, model requests counted against the limit, and those without reported token use or cache use |
| `input_tokens`, `output_tokens`, `cache_hit_tokens` | The generation's token use |
| `generation_seconds` | The generation's elapsed time |
| `n_generated` | Starter suites generated |
| `n_discarded_syntax`, `n_discarded_collection`, `n_unsalvageable`, `n_zero_coverage`, `n_validation_unresolved` | Attempts discarded because their code did not parse, did not compile or collect, could not be repaired, covered no line of the target, or whose validity check gave no answer |
| `n_unmeasured_coverage`, `n_unmeasured_mutation` | Passing suites the coverage tool or the mutation tool could not measure |
| `transient_failures` | Temporary failures of requests |
| `processed_deliveries` | The replies already processed, by delivery, so a resume processes and counts each reply once |
| `generation_attempts_per_suite` | The attempts each starter suite spent |
| `generation_complete` | Whether the generation finished |
| `suite_slots` | One entry per requested suite, described below |

`suite_slots` holds `requested_slots` and `slots`, one entry for each starter
suite the run asked for, numbered from 1 (`slot`), also for those that never
started or could not be used. Its `state` is `pending`, `active`, `accepted`,
`rejected` or `unavailable`, and `disposition_basis` says why: code accepted
after cleanup (`accepted_cleaned`), a reply without code (`no_code`), a
rejection after judging (`judged_rejection`), or a check that could not run
(`unavailable`). `terminal` says whether the suite's outcome is final,
`attempts_used` counts its attempts, `classification` and `reason` repeat
its last outcome, `attempts` lists every attempt, `terminal_attempt` names
the attempt that ended it and `accepted_source` the accepted code. The
format's version is 1, and the run records each starter suite's final
outcome there. Each attempt keeps the identifier and the request number of
the model request that produced it. A repair that only checks the code again
updates that attempt and sends no new request. An accepted suite's entry
(`base=starter_suites`) names its starter suite and attempt and the SHA-256 of
the cleaned code it accepted, not of the model's raw reply.

The starter phase's archive holds `starter_pool_attribution.json`
(`testuary.starter_pool_attribution/3`), written only when each suite was
measured on its own. `attribution_mode` and `basis` name how the kills were
attributed, `implementation` how the suites were run, and `union_mutants`
the mutants with a result in any suite. Its `slots` list every requested
starter suite: whether it was `accepted` and `measured`, the `outcome` of
its mutation run, how many mutants it detected (`killed`: killed, timed out
or not viable) among those with a result (`valid`), and the `verdicts`
behind those counts.

`starter_pool_kill_matrix.json` merges the suites' results, in the format of
[the kill matrices](#kill-matrices): a mutant counts as killed when any suite
killed it, and as untested when a suite has no result for it and no suite
killed it. Each mutant holds each suite's verdict in `suite_verdicts`, and
`per_suite` names how the suites were run and each suite's outcome. Under
`derive` it is the pooled run's matrix.

Under `derive` and `both`, `starter_pool_derivation.json`
(`testuary.starter_pool_derivation/1`) records the method requested and used,
why a derivation could not be made, the pooled run's file name per starter
suite, each suite's derived counts and, under `both`, its measured
counts and every mutant whose derived and measured status differ with the
rule that produced the derived status. `starter_pool_derived_kill_matrices.json`
holds each suite's derived matrix, and under `both`
`starter_pool_pooled_kill_matrix.json` holds the pooled run's.

The starter suites' model requests are recorded in
`suites/request_records.jsonl`, one line per request with its outcome and
token use, and `suites/prompt_transport.jsonl`, one line per request with the
roles and SHA-256 of the messages it sent. Both are working data.

## Repetitions and their summaries

[Results and logs](../guide/output.md) shows the run directory's layout, with
a directory for each repetition of a target (`r0`, `r1` and so on).

| File | Holds |
| --- | --- |
| `archive/<phase>/result<suffix>` | The suite the phase end handed on, with the language's suffix |
| `archive/<phase>/selected_kill_matrix.json` | Which tests of that suite kill which mutants, a [kill matrix](#kill-matrices) |
| `archive/<phase>/selected_coverage_matrix.json` | Which lines and branches each test of that suite covers, a [coverage matrix](#coverage-matrices) |
| `archive/<phase>/consolidation_record.json` | The phase end's record, described under [Phase records](#phase-records) |
| `archive/<phase>/merged_kill_matrix.json`, `merged_coverage_matrix.json` | The kill matrix and the coverage matrix of the merged suite |
| `archive/<phase>/reduced_kill_matrix.json`, `reduced_coverage_matrix.json` | The kill matrix and the coverage matrix of the reduced suite. The coverage matrix also names the kept tests (`kept_tests`) and the tests a derived score was restricted to (`derive_kept_tests`) |
| `archive/<phase>/measured_reduced_kill_matrix.json`, `derived_reduced_kill_matrix.json` | The reduced suite's kill matrix as measured, and as derived from the merged suite's matrix for its tests, which [Measurement and recovery](../explanation/measurement-and-recovery.md) explains |
| `archive/<phase>/candidate2_kill_matrix.json`, `candidate2_coverage_matrix.json` | The kill matrix and the coverage matrix of the phase's own new suite, measured alone before the merge |
| `archive/<phase>/validity_merged.json`, `validity_delivered.json`, `validity_derivations.json` | The [validity records](#validity-records) of the merged suite, of the delivered suite and of the steps that derived one suite from another |
| `archive/<phase>/candidate_<label>_removed` with the language's suffix | A candidate that lost tests to a [measurement removal](#phase-records) and was not delivered |
| `archive/coverage/candidate_agent` with the language's suffix | The coverage phase's own suite as its loop left it, before the phase end |
| `archive/mutation/pre_merge/candidate_inherited`, `candidate_agent` with the language's suffix | The mutation phase end's two inputs: the suite the mutation phase received and the mutation phase's own suite |
| `archive/<phase>/generation/record.json` | The phase's [generation record](#generation-records-and-the-settings-record) |
| `tests/` | The delivered suite: its test files and the setup files it was measured with, which [Results and logs](../guide/output.md#the-delivered-suite) describes |
| `tests/kill_matrix.json` | Which tests of the delivered suite kill which mutants, from Finishing's mutation results: the mutation phase end's matrix when Finishing reuses it |
| `tests/coverage_matrix.json` | The delivered suite's coverage from Finishing, in [the coverage matrices' second form](#coverage-matrices) |
| `time_limits.json` | Only when the repetition reached a time limit: one entry each time it reached one, its `phase`, its `setting` in `section.name` form and its `value` in seconds. The `warnings` of `summary.json` are made from it |

A repetition's `summary.json` holds its final values from its row in
`tests.csv`: `line_coverage`, `branch_coverage` and `mutation_score` in
percent, null when not measured, the status of the starter suites, the
coverage phase and the mutation phase, and `fully_measured`, which is true when the
repetition is fully measured: its final line coverage and mutation score were
measured. The target's `summary.json`, in the target's directory beside the
repetition directories, summarizes its repetitions: `runs` names them,
`fully_measured_runs` counts the fully measured ones, and each final value has the
number of repetitions that measured it with their mean, minimum and maximum.
Its `starter_median` holds the mean over the repetitions of the starter
suites' median of each value, before the loops.
A repetition's summary also holds each phase's values for the suite it handed
on with its model use and time (`phases`), the values of Finishing
(`finishing`), the accepted starter suites' statistics (`starter`) and where
the repetition's time went (`time`), each from its column above:

| Summary key | Column |
| --- | --- |
| `phases.<phase>.line_coverage`, `branch_coverage` | `<phase>_original_line`, `<phase>_original_branch` |
| `phases.<phase>.mutation_score`, `mutation_score_selective` | `<phase>_original_mutation_full`, `<phase>_original_mutation_selective` |
| `phases.<phase>.tests` | `<phase>_delivered_tests` |
| `phases.<phase>.logical_requests`, `physical_requests` | `<phase>_logical_requests`, `<phase>_requests` |
| `phases.<phase>.input_tokens`, `output_tokens`, `cached_input_tokens` | `<phase>_input_tokens`, `<phase>_output_tokens`, `<phase>_cache_hit_tokens` |
| `phases.<phase>.turns` | `<phase>_turns` (coverage and mutation) |
| `phases.<phase>.seconds` | `<phase>_duration_s` |
| `finishing.line_coverage`, `branch_coverage` | `finishing_line_coverage`, `finishing_branch_coverage` |
| `finishing.mutation_score`, `mutation_score_selective` | `finishing_mutation_score_full`, `finishing_mutation_score_selective` |
| `finishing.mutation_score_lower_bound`, `mutants_untested`, `seconds` | `finishing_mutation_score_lower_bound`, `finishing_untested`, `finishing_duration_s` |
| `starter.line_coverage`, `branch_coverage`, `mutation_score`, each with `mean`, `median`, `min`, `max` and `std` | `starter_suite_line_*`, `starter_suite_branch_*`, `starter_suite_mutation_*` with the same statistics |
| `time.total`, `model_requests`, `test_runs` | `duration_s_target`, `time_model_s`, `time_test_s` |
| `time.coverage_measurement`, `mutation_measurement` | `time_coverage_s`, `time_mutation_s` |
| `time.merging`, `cover_solver`, `project_copies`, `other` | `time_merge_s`, `time_cover_s`, `time_preparation_s`, `time_other_s` |

`results/summary.json` lists the summaries of every target in the run
directory (`files`), with each value's count, mean, median, lowest, highest
and standard deviation across the targets and the values themselves
(`values`), beside the same for the targets' starter medians
(`starter_median`). A repetition whose mutation measurement in Finishing left
mutants untested has no mutation score: its `finishing` holds the lower bound
(`mutation_score_lower_bound`, the untested mutants counted as not killed) and
the untested mutants (`mutants_untested`). The target's summary states the
lower bound across such repetitions, and the project's summary
the statistics of the lower bounds of the targets without a mutation score
(`mutation_score_lower_bound`).

A repetition's `summary.json` also lists `warnings`: one entry for each phase
and setting whose time limit the repetition reached. Each entry holds `kind`
(`time_limit`), `phase` (`starter`, `coverage`, `mutation` or `finishing`), the
`setting` in `section.name` form, such as
`mutation.measurement_timeout_seconds`, its `value` in seconds (`unit`
is `s`), how often the repetition reached it (`count`) and the sentence the
run's log writes (`text`). The list is empty when the repetition reached no
limit. Raising the named setting gives the next run more time.

After the time limits, the list holds one entry of `kind` `loop_limit` for
each loop (`phase` `coverage` or `mutation`) that one of its limits stopped
before its goal: its request, token or time limit. Its `setting`, such as
`mutation.max_requests_per_phase`, and `value` name the limit from the
loop's stop in `phase_state.json`, `unit` is `s` for the time limit and null
otherwise, and `count` is 1. A stop that names no setting leaves `setting`
and `value` null. The target's `summary.json` counts the repetitions with a
warning in `warning_runs`.

## The end-of-run summary

`results/completion.json`, beside `results/summary.json`, holds the summary
the console prints when a run ends, which
[Results and logs](../guide/output.md#the-end-of-a-run) describes. It is
written in one step at the end of every run, with credentials removed from its
texts, and a resumed run writes it again.
[`read_completion`](api.md#event-files) reads it.

| Key | Holds |
| --- | --- |
| `schema_version` | The format's version |
| `provenance` | Where the values come from, such as `observed` for a run's own measurements and `synthetic` for a demonstration's |
| `completion.files` | One entry per selected target, described below |
| `completion.results_directory`, `log_path` | The run's results directory and its `run.log` |
| `completion.resume_command` | The exact command that continues the run, null when there is nothing to continue |
| `completion.cancelled` | Whether the run was stopped before it finished |

Each entry of `files` holds the target's `path`, its `outcome` (`complete`,
`no_complete_suite`, `failed`, `cancelled`, `already_complete` or
`stopped`) with its `reason`, the delivered suite's files (`suite`), its
`metrics` (each with `name`, `value`, `unit`, `basis` and `goal`), its model
`usage` (requests, tokens, cost and the request limit), whether a resume
continues it (`resumable`), and the sentences of its `fallbacks` and
`warnings`. Its warnings are the sentences of each repetition's
`summary.json` warnings. The console prints each warning as a line
`Warning:` with its sentence.

## Phase records

When a removal ran at a phase end, as
[Tests removed after a failing measurement](../explanation/measurement-and-recovery.md#tests-removed-after-a-failing-measurement)
describes, the phase's `consolidation_record.json` holds
`measurement_removals`, one entry per round: the suite (`place`:
`phase_end.candidate`, `phase_end.merged`, `phase_end.inherited` or, for a
starter suite measured under the delivered name, `starter.candidate`), the
measurement, the round, the removed tests in `removed`, each with its
outcome (`failed` or `error`) where the run gave one, the names the suite
does not hold in `unmatched`, why a named case's whole test went in
`detail`, and, for a failure that removed nothing, why in `unnamed`. The
entries stand in the order the removals ran. A changed candidate that is not
delivered is kept in the archive as `candidate_<label>_removed` with the
test file's suffix, `<label>` the candidate's label in the selection, and a
removal in
the delivered suite's ancestry is a `measurement_remove` record of
`validity_derivations.json`.

### phase_state.json

Each target's `phase_state.json` records which phases of each repetition are
finished:

| Key | Holds |
| --- | --- |
| `target_number`, `stem`, `target_file_path` | The target's number, its file name without the suffix, and its path |
| `starter`, `coverage`, `mutation`, `finishing` | Each phase's entry per repetition, keyed by the repetition's number (`"0"` for `r0`) |
| `pipeline` | Per repetition, the time of its work between its phases, in `segments` |
| `file` | The target's time across its repetitions in this run, the work they share included, in `segments` |

A phase's entry for a repetition holds:

| Key | Holds |
| --- | --- |
| `status` | `in_progress` while the phase runs or after the process stopped in it, `complete`, or `failed` with its `reason` and, when known, `failure_kind` |
| `target_file_path`, `target_source_sha256` | The target's path and the SHA-256 of its raw bytes when the phase started, which [`recorded_source_sha256`](api.md#runs) reads |
| `resume_context` | Where the project ran, which a resume compares |
| `segments` | The phase's attempts, described below |
| `artifacts` | Each file the phase delivered, by its path in the target's directory, with its SHA-256, which a resume checks |
| `metrics` | The phase's values, from which its `tests.csv` columns come. A loop's hold its `status`, such as `budget_exhausted`, or `skipped_inherited_slow` for a mutation phase whose inherited suite ran longer than its time limit before any edit, and in `error` a stopped loop's reason, such as `final_answers_without_measurement` after `max_final_answers_without_measurement` stopped it |
| `generation_condition` | The phase's generation condition, when the phase recorded one |
| `started_t_mono_s`, `completed_t_mono_s`, `failed_t_mono_s` | When the phase started, completed or failed, each with its `clock_domain_id` |

Each entry of `segments` is one attempt: its `outcome` (`ran`, `crashed`,
`abandoned` or `restored`), its `seconds`, its `model_seconds`, its time per
operation (`parts`: `model`, `test`, `coverage`, `mutation`, `merge`,
`cover` and `preparation`) and the time no part claims (`other_seconds`), the
mutation tool's workers (`engine_workers`), whether the starter suites were
reused (`starter_suites_reused`), when it started and ended, and
`discarded` for an attempt that finished but whose work is not kept. A
failed phase's `failure_kind` is `infrastructure`, a model service or tool
problem, for a model service failure, exhausted retries, a crash in the phase
or a mutation tool that gave no usable result, and `method`, no usable suite
within the limits, when the starter suites produced no usable suite or one of
the coverage phase's limits ended it before a usable, measured suite. This
outcome is the method's own result. A phase without that field failed for a
reason the run cannot judge, such as a request too large for the model's
context or a validity check that gave no answer.

### consolidation_record.json

Each phase end writes `consolidation_record.json` in its phase directory of
the archive:

| Key | Holds |
| --- | --- |
| `target_file` | The target |
| `delivered_from`, `winner_source_label` | The candidate the phase end delivered, such as `reduced` or `merged`, and for a single suite delivered as it is, that suite's label |
| `selected_source_sha256`, `execution_name` | The SHA-256 of the delivered suite and its file name |
| `generation_ancestry` | From which inputs the phase built the delivered suite, described under [Generation records](#generation-records) |
| `target_revision`, `measurement_identity` | The SHA-256 of the target's source and the identity of the measurements, which binds them to that source and to the settings they ran with |
| `structural_parser_identity`, `target_validator_identity` | The parser that read the tests' structure and the interpreter or compiler that checked them |
| `merged_source_sha256`, `reduced_source_sha256`, `merged_tests`, `delivered_tests` | The SHA-256 of the merged and the reduced suite, and the tests in the merged and in the delivered suite |
| `reduction_valid` | Whether the reduction produced a passing reduced suite. A reduction may keep every test |
| `repaired_tests`, `evidence_repairs` | The tests a repair added back to the reduction because its evidence missed something, and each repair round |
| `unreduced_fallback`, `fallback_reason`, `fallback_candidates`, `fallback_merged_stats` | When no valid reduced suite could be built: why, the candidates offered instead, and the merged suite's size |
| `selection_audit` | The selection, described below |
| `detail` | A note on the phase end, such as why it built no reduction |
| `derive_measure_delta`, `derive_measure_differences` | Under `reduced_suite_mutation = both`, the derived minus the measured mutation score of the reduced suite and every mutant whose two statuses differ |
| `reduction_solver`, `reduction_solver_metadata` | How the reduction was solved, `exact` or `greedy`, and the solver's settings and times |
| `consolidation_mutation_method` | The value of `reduced_suite_mutation` |
| `artifacts`, `validity_artifacts` | The files the phase end wrote in its directory, and of them the validity records |
| `collision_resolution`, `input_normalizations` | How the merge resolved same-named tests (renamed, deduplicated, dropped) and the shadowed definitions removed from its inputs |
| `mutation_measured`, `shipped_has_matrix` | Whether the merged or the delivered suite has a kill matrix, and whether the delivered one has |
| `measurement_removals` | The [measurement removals](#phase-records), only when one ran |
| `mutation_origin_stops` | Only when the mutation tool's check run before the first mutant stopped a measurement because a test failed without a mutant: each stop with `suite`, the name of the measured suite's directory (such as `merged`), its `reason` and every stopping test in `stopping_tests`. In the starter phase it also lists the stops of the starter suites' own mutation runs, `suite` then naming the starter suite, such as `suite_001`. The run log names each stop on a `[finishing] the mutation tool's origin check stopped` line |
| `validity_policy_version` | The validity policy the delivered suite was checked under |

`selection_audit` holds:

| Key | Holds |
| --- | --- |
| `candidates` | Each candidate with its `label`, whether it is `passing`, its `status`, whether it is `fully_measured`, its `mutation`, `branch` and `line` scores, its `runtime_ms`, `tests`, `source_sha256`, `execution_name`, `status_reason`, `removed_tests` and `repaired_tests`, its `target_revision`, `measurement_identity` and `suite_timing` |
| `selector`, `active_axes` | The scores the selection compared, in order |
| `delivered_from`, `selection_basis`, `top_score_labels` | The chosen candidate, why (`score`, `reduced_preference` or `timing_median`) and the candidates with the best scores |
| `runtime_tie_break_basis`, `runtime_tie_break_candidate_order`, `runtime_tie_break_rounds` | For equal best scores, how the time decided (`median` or `input_order`), the order it took and its timed rounds |
| `input_evidence_missing_measurements`, `delivered_from_missing_from_merged_input` | The inputs without kills, lines or branches on record, and how many of the inputs' kills, lines and branches the delivered suite lacks |
| `reduction_missing_merged_lines`, `reduction_missing_merged_branches`, `reduction_missing_merged_kills` | How many of the merged suite's covered lines, taken branches and kills the reduced suite lacks |
| `reduction_measurement_issue` | Why a built reduction was excluded, when it was |
| `reduced_not_selected`, `merged_stats` | When the merged suite was chosen over a distinct reduction: true, and the merged suite's size |

Under `reduced_suite_mutation = derive` a `reduced`
candidate whose mutation score is derived from the merged suite's kill matrix
for its tests says so in its reason: derived, not measured. Its entry also
holds `mutation_derived` set to true, which no other candidate holds. When
the merged suite's per-test evidence gives other coverage than its coverage
run, in percentages, covered lines or taken arcs, `merged_evidence_difference`
in `selection_audit` records both sides, `measured` and `per_test_evidence`: each
side's `line` and `branch` coverage and the covered lines
(`covered_lines_only`) and taken arcs (`taken_arcs_only`) that only it has.
The merged suite keeps its coverage run's values. At the starter phase, when a
single starter suite is delivered and its acceptance run measures other line
or branch coverage than the evidence it was selected on,
`delivered_coverage_difference` records the sides `selection` and `acceptance`
in the same form, and the acceptance run's values are delivered.

The headline `mutation_score` in `tests.csv` is the delivered suite's mutation
score over all mutants from Finishing, also recorded as
`finishing_mutation_score_full`. `finishing_mutation_measurement` says how
that measurement ended. `complete` means it finished. `incomplete` means it
ended with mutants untested, as when its time limit ran out. `failed` means
it produced no record of which tests kill which mutants (the kill matrix). The
last two report no score, and Finishing is recorded as failed, so a resumed
run measures again. For an incomplete measurement, the counts of the mutants
with a result remain in `finishing_killed`, `finishing_survived` and
`finishing_total`. Such a measurement also records its lower bound, every
untested mutant counted as not killed, in
`finishing_mutation_score_lower_bound`, and its untested mutants in
`finishing_untested`. A later run that reuses the suite and measures no
mutation, as with `[finishing] measure_mutation = false`, keeps the score.

## Generation records and the settings record

A repetition's `resolved_settings.json` holds the settings it used, without
secrets:

| Key | Holds |
| --- | --- |
| `sections` | The settings of every section, as the run resolved them, except those that configure no run (`[web]`). Of the provider sections (`[openai]`, `[ollama]` and `[vllm]`), only that of the provider the run's model goes through, and none when a program supplies the model or the model does not resolve, since another provider's address can name a private host. The model catalogue is not in this file |
| `generated_test_validity` | The language adapter's validity policy (`policy_version`), `none` for an adapter without one |
| `effective` | What the settings alone do not show: the mutation phase's catalogue (`mutation_catalogue`, `full` or `filtered`) and Cosmic Ray's distributor (`cosmic_ray_distributor`) |
| `limits` | The limits of the coverage phase and the mutation phase |
| `model` | One entry per generation phase the run selected, with the resolved values and their sources. A model of a client factory names the factory's file and its SHA-256 (`extension_file`, `extension_sha256`) |
| `fallbacks` | The steps the run skips because the file's language lacks them, only when one applies |
| `absent_models_file` | The last file of a program's `models_files` (of `testuary.settings.resolve_settings`) when it does not exist, which reads as empty |

The `model` entries leave out reused phases, and a model a program supplies
records only its source. A client factory's distribution version is recorded
only when it is known. The file does not describe the ancestry of the
delivered suite. A resume keeps the existing file unchanged. Any other run
writes it afresh, so a run that reuses an earlier run's phases states the
settings its own phases ran under.

The starter suites' `archive/starter/suites/generation_origin.json` and each
phase's `archive/<phase>/generation/record.json` hold the generation
condition, the settings without secrets and links to the copied profile and
hints files. The generation origin's keys are `format` (1), `condition`, the
generation condition, which names the language package the run loaded
(`language_package`) and, where the tests' language differs from the
target's, that language (`test_language`, such as `python` for Python tests
of a C file), `settings`, the sections as in `resolved_settings.json` and,
of the model catalogue, only the run's model alias (`models`),
`prompt_profile` and `hints` (each with its path,
its SHA-256 and its copy), `extensions` (each loaded extension with its file,
SHA-256, distribution and version) and `extension_hints` (each extension's
hints with their SHA-256 and their copy). The copies stand beside the record,
named after it: `<name>.profile.json`, `<name>.hints.json` and
`<name>.extension_hints_<n>.json`. A starter suite reused from an
earlier run carries its origin's profile, hints and extension hints forward.
[Model request records](../guide/output.md#model-request-records) says where
the request records are. These are diagnostic records.
Reusing an earlier suite does not mean it was generated with the current profile.

The mutation tool that ran is the `engine` of `[mutation]` in the `sections`
of `resolved_settings.json` and of these records' `settings`, and the
`engine` of the generation condition's measurement settings, which
`phase_state.json`, the consolidation records and the chat logs copy:
`cosmic_ray` for Python, `pit` for Java and `mull` for C under the shipped
default. The measurement identities, such as `config_sha256`, hash the
mutation settings as configured, so they hold the shipped default
`cosmic_ray` for every language.

### Generation records

Each phase's `consolidation_record.json` holds `generation_ancestry` (format
version 1, `basis=construction_inputs`): from which inputs the phase built
the suite it chose. It names the chosen source, the command and the run
apart from the model request counts, and keeps each input's origin and
parents in the record itself, so later records or profile files cannot
change it. Pruning, combining and suite reduction are construction steps:
they do not say which test the model wrote. A result reused from an earlier
run is named as reused, not as generated again. The settings alone do not
show what generated a chosen suite.

`testuary.generation_evidence.read_suite_slots(manifest, suites_root)`,
`read_cleaned_source(reference, suites_root)` and
`read_ancestry(record, source)` read these records and check them. They
refuse an unknown version, an unsafe or changed reference, a cycle and
contradictory entries with `GenerationEvidenceError`. Missing ancestry stays
unavailable, and an edit whose earlier version is unknown stays partial. The
readers add nothing the records do not hold.

## Matrix and validity formats

### Kill matrices

Every file named `*kill_matrix*.json` above is a kill matrix in LangAdapter's
format, which
[`MutationRun.to_kill_matrix`](https://k-kammler.github.io/langadapter/reference/mutation/#langadapter.mutation.MutationRun.to_kill_matrix)
describes:

| Key | Holds |
| --- | --- |
| `schema_version`, `target_file` | The format's version and the mutated file |
| `mutants` | One row per mutant: the engine's `id`, the `mutant_id` that names the same mutant in every run over the same source, its location (`key`, `line`), its `operator` and `diff`, its `status`, the tests that killed it (`killer`, `killers`, `killing_tests`, and `unmatched_killers` for names the suite does not hold), the kill's `verification` and the engine worker's outcome (`worker_outcome`) |
| `summary` | The counts of the mutants: `total`, `detected`, `survived`, `timeout`, `non_viable`, `untested` and `unresolved` |
| `catalogue_provenance` | The catalogue and operators requested, and the operators and mutants observed |
| `suite`, `execution`, `attribution` | The measured suite's files and SHA-256, where the run executed and how it was supervised, and whether it recorded the first or every killing test |
| `resolution` | How the matrix was obtained, such as `measured`, with its reason |
| `derived` | Only for a derived matrix: what it was derived from and how |
| `per_mutant_timeout` | The per-mutant time limit the run applied, and where it came from |
| `source_sha256`, `execution_name` | The SHA-256 of the measured suite and its file name |
| `t_mono_s`, `clock_domain_id` | When the matrix was written |

A matrix of a run that left mutants untested carries LangAdapter's incomplete
stamp, and it reports no score.

### Coverage matrices

The coverage matrices in a phase's archive hold, for one suite, these keys,
not every matrix all of them:

| Key | Holds |
| --- | --- |
| `schema_version`, `target_file` | The format's version and the target |
| `source_sha256`, `execution_name` | The SHA-256 of the suite and its file name |
| `structural_parser_identity`, `target_validator_identity` | As in `consolidation_record.json` |
| `line`, `branch` | The suite's line and branch coverage in percent |
| `target_total_statements`, `covered_statement_lines` | The target's statements and the lines the suite covers |
| `target_total_branches`, `covered_branch_arcs` | The target's branches and the arcs the suite takes, as `from->to` |
| `per_test_tokens` | Each test's covered lines and branches as tokens, which the reduction reads |
| `per_test_runtime_ms`, `effective_cost_us` | Each test's measured time, and the cost the reduction weighs it with |
| `kept_tests`, `candidate_label` | In `selected_coverage_matrix.json`, the delivered suite's tests and the candidate it came from |

`tests/coverage_matrix.json`, from Finishing, has a second form: `lines`
(each line with its `status`, `covered` or `missing`), `branches` (each arc
`from`, `to` with its `status`, `taken` or `missing`), `summary` (the
covered and missing lines and branches and the target's lines), and the
suite's `source_sha256` and `execution_name`.

### Validity records

`validity.json`, `validity_merged.json` and `validity_delivered.json` each
hold one validity check of a suite under the language's validity policy:

| Key | Holds |
| --- | --- |
| `record_type`, `stage` | `direct` for a check, and where it ran, such as `starter.accepted_suite`, `phase_end.merged` or `phase_end.delivered_assertion` |
| `policy_version`, `target_identity` | The policy and the target it was checked against: its language, module, source directory, source path and SHA-256 |
| `analyzed_source_sha256` | The SHA-256 of the checked suite |
| `findings`, `audit_flags` | What the check found against the policy, and what its screening flagged |
| `removed_tests`, `final_clean_source_sha256` | The tests removed for a finding, and the SHA-256 of the suite without them |

`validity_derivations.json` holds `records`, one per step that made one
suite from another, such as a reduction (`set_cover_remove_restore`) or a
measurement removal (`measurement_remove`): `record_type` `derivation`, its
`stage`, `operation`, `policy_version` and `target_identity`, the SHA-256 of
the suite before (`parent_source_sha256`) and after
(`child_source_sha256`), and the tests removed and restored.

## Logs

`logs/run.log` holds every log line of the run as plain text, and
`logs/main.jsonl` is the run's structured log. Each repetition of a target
has its logs in `logs/test/<target>/r<k>/`:

| File | Holds |
| --- | --- |
| `coverage.jsonl`, `mutation.jsonl`, `finishing.jsonl` | Each phase's structured log, one JSON object per line: its start, each logged line, a finished loop with its conversation, and its end. `main.jsonl` has the same form |
| `coverage_chat.json`, `mutation_chat.json` | Each loop's conversation with the model: the messages and metadata such as the SHA-256 of the tool descriptions and the generation condition |
| `coverage_chat_pre_retry_NNN.json`, `mutation_chat_pre_retry_NNN.json` | The conversation as it stood before a request was sent again after a failure |
| `coverage_usage.jsonl`, `mutation_usage.jsonl` | One line per request sent: its outcome, its token use and whether the model service reported it |
| `coverage_prompt_transport.jsonl`, `mutation_prompt_transport.jsonl` | One line per request: the roles and SHA-256 of the messages it sent and how they were sent |
| `coverage_trace.jsonl`, `mutation_trace.jsonl` | Only with `enable_trace_log = true`: each request, tool choice, final answer check, recovery and reasoning event |
| `generated_test_validity_writes.jsonl` | The validity record of each edit the loop wrote, in the format of [the validity records](#validity-records) |
| `generated_test_validity_rejections.jsonl` | Each edit the validity policy refused, with its findings |
| `coverage_generation_origin.json`, `mutation_generation_origin.json` | Each loop's generation origin, in the form of the [generation records](#generation-records-and-the-settings-record) |

The fields of the logs are working data that this page does not list.
[Results and logs](../guide/output.md#logs) says how to use them.
