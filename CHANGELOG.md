# Changelog

All notable changes are listed here, newest first.

## 0.2.0

First public release. Testuary writes unit tests for Python, Java and C
code with a language model. It runs the tests, measures which lines and
branches they cover and how many mutants (small deliberate bugs) they catch,
and asks the model for more tests wherever something is still missed, until
your goals or limits are reached. It then keeps only the tests that add
coverage or catch an extra mutant, and hands you the finished test suite
with its measurements. Running and measuring the tests is the job of
LangAdapter 0.1.0, a separate package released with it.

### New in this release

- Java projects built with Maven or Gradle and tested with JUnit 4, 5 or 6,
  with JaCoCo for coverage and PIT for mutation.
- C files on Linux, through LangAdapter's C adapter, with gcov for coverage
  and Mull for mutation. Testuary writes their tests in the language the
  adapter names for them: C with utest.h, or Python for a C file built into
  a Python extension. Its prompts and the Results view's place for a
  delivered test follow the tests' language.
- Language support in LangAdapter, with a way to add a language. Every
  language's runtime is found through its adapter in the same way. Without
  a selected interpreter or `python_version`, a Python project runs with
  its only environment, its `.venv` or the one activated in the shell. An
  environment Testuary runs in outside the project is set aside, so the
  project's `.venv` runs the tests. A `[project]` setting that two
  installed languages declare takes the kind, default and host-path rule of
  the first of them by name.
- Commands to set up and check a project before a run: `testuary init`,
  `testuary setup`, `testuary check`, `testuary new-language` and
  `testuary tutorial`.
- Settings in your user directory and in a `.testuary` directory in each
  project, with a reference of every setting.
- Several repetitions of each file with `--runs`, each repetition's results
  in its own directory, and a summary across them.
- Progress views in the terminal, and an end-of-run summary saved with the
  results.
- An offline demonstration that plays back a recorded run without a model:
  `testuary --demo`. It refuses a wrong combination of run options, such as
  `--runs 2 --run 1`, with exit status 2 before the demonstration starts.
- A quickstart directory, `examples/quickstart` in the repository and the
  source distribution: scripts that play the demonstrations and start the
  web interface, and two small example projects to run with your own model.
- Prompt profiles and hints files, which change what the model is told. A
  prompt profile may name every placeholder the prompts fill, among them
  `$lang_test_decl_full`, `$lang_test_collection_rule` and
  `$language_level`.
- A local web interface, started with `testuary-web`. It shows the
  projects, models and settings, tests a model's connection, starts runs
  and checks, follows a run's progress and shows its results and the
  demonstrations. A test file of a delivered suite can be copied from the
  results into its project. A project can be added from its GitHub HTTPS
  address, which git clones into a directory you choose. The `[web]`
  settings choose its port, 8765 by default, and who may use it, every
  program on the computer by default.

What the model sees:

- The file under test is read as UTF-8 with undecodable bytes replaced and
  CRLF or CR read as LF, in a local and a container run alike, so the model
  and the measurement see the same text in both.
- Each loop starts by reading the tests, then gets the current measurement,
  then edits. A tool that is not available yet says what makes it
  available.
- The rules for valid tests are each language's own and name what a test
  may read. Python tests may read a module's `__dict__` and `vars()`.
- A test that the time limit cut off is named, and the model can fix it
  without losing its other changes.
- The tool calls of one answer run one after another, in order, so that
  several edits in one answer all take effect. One edit call can carry a
  list of edits.
- A tool that fails for a temporary reason gets an answer, and the loop
  continues.
- Text that is cut for length is marked, and every such limit is a setting.
- A write outside the loop's test output directory is refused, and the
  refusal the model receives names the field it checks, `test_output_dir`.
- After a passing normal test run, a run under the coverage tool that
  fails, ends without a test result or does not finish is the tool's
  refusal, in both loops. The reply to the loop's measurement names the
  tests that stopped the tool where it can, and it counts as a failed test
  run. Only a normal test run that does not finish is a timeout of the
  suite.
- A passing test run of a Python file can carry a STANDS ALONE note: the
  tests that left the module under test or a library setting changed,
  and with `report_reverse_order_failures = true` in `[coverage]` or
  `[mutation]` the tests that fail when the file runs as delivered in
  reverse order. After a note that names tests, the model can change the
  tests before it measures, and a loop at its goal ends only after a
  passing run without such a note, within its limits.

What is measured:

- By default each starter suite gets its own mutation score from a run of
  that suite alone. `[starter] suite_mutation` can derive the starter
  suites' scores from one run of all of them instead, and
  `[main] reduced_suite_mutation` the reduced suite's score at a phase end
  from the merged suite's run.
- With `per_mutant_timeout_seconds = auto`, the per-mutant time limit for
  C is `per_mutant_timeout_factor` times the mutation tool's own run of the
  passing suite, each test in a process of its own as a mutant's tests
  run. A mutant that no test kills runs all its tests and survives, in a
  large suite too. A fixed value bounds all of a mutant's tests together.
  Logs state a C time limit per mutant.
- One normal test run decides whether tests pass. For Python it is the
  run as delivered: each test file in its own session, with fixed seeds
  set when the session starts and nothing reset between tests.
- Tests are removed only where no model is active. The starter suites'
  cleanup removes failing tests. When a suite passes its normal run and a
  coverage or mutation measurement then fails and names tests, in the
  starter phase or at a phase end, Testuary removes the named tests, runs
  the suite again and records each removal, as
  [Tests removed after a failing measurement](https://k-kammler.github.io/testuary/explanation/measurement-and-recovery/#tests-removed-after-a-failing-measurement)
  describes. A loop removes no test from the suite it inherits, and at the
  end of each phase Testuary chooses among the suites that pass with
  complete numbers.
- A starter suite that the coverage or mutation tool cannot measure first
  loses the tests the failure names. When that does not help, it is
  generated again within its attempts, or left out.
- Finishing, the fourth phase, takes the delivered suite's coverage from
  the end of the phase that measured it and its mutation results from the
  end of the mutation phase, and measures only what those did not. With
  `[finishing] measure_mutation = false`, Finishing records no mutation
  score and judges no mutation minimum, also when the mutation phase
  measured the delivered suite.
- Every measurement runs in a private copy of the project.
- Every mutant has an identifier that stays the same from run to run.
- A Python test's coverage includes its fixtures' setup and teardown. A
  fixture shared by several tests counts for the first of them, and its
  teardown for the last.
- Per-test times are precise, and the reduction does not charge a run's
  one-time setup to its first test.
- Of several suites with equal scores, the end of a phase keeps the
  reduced suite, or else the fastest. When their times do not tell them
  apart, or a test fails in a timing run, the earlier one stays. A tie that
  no candidate's time could break records the input order as its basis,
  and the results page says so.
- Candidate suites are merged by LangAdapter, which checks the merged file
  with the project's own Python or Java tools. When it refuses a pair,
  nothing is merged and the better single candidate is kept. A merge whose
  tools give no answer is an error, not a refusal. A Python merge keeps one
  copy of a top-level statement both suites hold identically, such as a
  repeated docstring or import guard, where one copy cannot change what a
  test sees, and a definition dropped as shared takes its blank lines along.
- When the merged suite's per-test evidence covers other lines or branches
  than its own coverage run, as C's tests can when each runs alone, the
  run's coverage stays the suite's coverage, the reduction works on the
  per-test evidence, and the selection audit records both sides
  (`merged_evidence_difference`). A starter suite delivered alone takes the
  coverage of its run under the delivered name
  (`delivered_coverage_difference`). A candidate whose per-test evidence disagrees with its own coverage run is
  scored on that run, without the evidence, with the disagreement as its
  reason.
- An exact reduction whose time limit runs out before a tier of its
  tie-break records the failure `time_limit_exhausted` with that tier. A
  CBC that cannot run ends the exact reduction with the failure
  `solver_error` and the tier it stopped in, and the phase end writes
  `consolidation_failure.json`. The recorded error names the error type,
  without the solver's message, which can hold paths of your machine.
- A final mutation measurement that did not finish reports no score.
- A mutation measurement cut by its time limit counts its untested mutants
  as not killed, so its score is a lower bound. The mutation phase works
  with that score, and a cut final mutation measurement records it beside
  its missing score.
- When a loop's test run, a mutation measurement or the final coverage run
  reaches its time limit, the run's summary and the end-of-run summary
  carry a warning that names the setting and its value.

How runs behave:

- A temporary failure of the model service costs no model request.
- A run ends with exit status 1 when a file failed.
- A run is refused before any work, as a refused setting is, when its
  required tools are missing or when `--reuse` finds saved work missing for
  a target. It ends with exit status 2, and a program that calls
  `run_pipeline` receives `testuary.settings.ConfigurationError`. For
  missing tools the setup report names what is missing and how
  to fix it, `--events` records that report, and the run's last event reads
  "Setup refused".
- A target outside the source directory is refused before the run, with or
  without `--resume`, as a file no language handles is. A target whose file
  name holds a comma runs as one target. A target's test file is the one
  its name gives (`test_<stem>.py`, `<Stem>Test.java`), so a reused run
  never takes another target's file from the shared tests directory.
- A resumed run reuses its saved starter suites, and a resume with changed
  measurement settings is refused. A resume compares the SHA-256 of a model
  provider's file or a language's file and refuses one that changed, naming
  it. A change to one of Finishing's own settings, such as
  `[finishing] measure_lint`, is no refusal: the resume measures Finishing
  again, with no model request.
- `--resume` judges each repetition's phases against the canonical source
  that repetition's phase recorded when it started: `phase_state.json`
  holds `target_file_path` and `target_source_sha256` in each repetition's
  block of a phase. A repetition made from an earlier source counts as
  unfinished, also after a later repetition ran from the edited source.
- A resume or reuse knows the model service by its provider and model
  name. Another address, such as a local proxy on another port, is a
  warning.
- A resumed or reused run that keeps its earlier `tests.csv` row and
  measures mutation in Finishing again writes each of Finishing's metrics
  columns afresh, empty where the new measurement gave no value, and every
  write with a phase result writes `error` afresh. An earlier attempt's
  lower bound, untested count, kill counts or error text does not stay
  beside the new result in `tests.csv` or `summary.json`. A write that
  measured no mutation, such as a timing pass, keeps the earlier scores.
- A loop that ends at its request, token or time limit names that limit,
  such as TOKEN LIMIT REACHED, an ordinary end, not FAILED.
- The coverage and the mutation phase give a stop the same status. A
  coverage phase that `max_final_answers_without_measurement` stopped, with
  the stop reason `final_answers_without_measurement`, records that it
  stopped before its goal, as the mutation phase does: its suite is green
  and measured, and its target's row holds the mutation phase's results.
  Repeated test timeouts end either phase before its goal when the suite
  it delivers is measured, else with failing tests. A mutation phase whose
  suite runs longer than its time limit before any edit records
  `skipped_inherited_slow` ("skipped: the suite it received runs longer
  than its time limit"), a failed phase with its reason.
- A run whose phase named by `--stop-after-phase` is not reached ends with
  its `Stopped:` line, without a traceback.
- A command refused after its arguments are read leaves an earlier
  `--events` file as it was.

What a run records:

- A mutation run that the mutation tool's origin check stops, because a
  test fails without a mutant, is logged with its stopping tests and
  recorded in the phase end's `consolidation_record.json` as
  `mutation_origin_stops`.
- Each phase's `<phase>_duration_s` counts only the work the row keeps: an
  attempt that crashed, or that failed in a phase other than the starter
  phase, counts for nothing there, while `duration_s_target` and the
  `time_*` columns keep a failed attempt's time. `duration_s_target` and
  every `time_*` column are empty when the repetition crashed, or when a
  resume reused a phase or the saved starter suites of a stopped or crashed
  repetition, and `starter_duration_s` too in the second case, since the
  time of that work is not known.
- `merged_tests` in `consolidation_record.json` and `coverage_merged_tests`
  in `tests.csv` are empty where no merged suite was counted.
  `mutation_merged_tests` and `mutation_delivered_tests` hold 0 for an
  empty count.
- A phase end's `consolidation_record.json` tells a reduced suite that was
  never built (status `absent`) from one that does not pass (`failing`) or
  was not fully measured (`incomplete`), each with its reason. An absent
  one records no tests, removed tests or source hash, and the results page
  shows it as not built. A merged suite whose run wrote no exact coverage
  matrix is recorded as `incomplete` with that reason.
- An inherited phase end's `consolidation_record.json` holds its keys in
  the order of a measured phase end's record, followed by `inherited_from`,
  `inherited_artifacts`, `measured` and `measurement_calls`.
  `reduced_coverage_matrix.json` holds the keys of
  `merged_coverage_matrix.json` in their order, followed by `kept_tests`
  and `derive_kept_tests`, and a time map with no measured time is null.
  `selected_coverage_matrix.json` holds per-test times and solver costs
  only for a merged or reduced winner, and a candidate scored on one
  coverage run records the target's statement and branch totals that run
  measured.
- A phase end's selection audit names the axes the selection compared in
  `selector`, records `runtime_tie_break_candidate_order` as null when no
  timing tie-break ran, and records an input suite that does not pass with
  status `incomplete`. The candidates' times in the audit are their tests'
  measured times summed, or null where a test was not timed.
- A phase end's `collision_resolution` lists in `renamed_names` the names
  the merge's collision rename gave, when it gave any, and `rename_twins`
  counts the copies of a test that the rename kept beside the original.
  `consolidation_record.json` keeps `collision_resolution` when the merged
  suite fails validation.
- A phase end's `reduction_solver_metadata` names CBC by its version
  alone, without the solver's path on your machine. With
  `reduction_solver = greedy` it records the measured and imputed costs
  the reduction weighed, as the exact reduction does.
- The kill matrices a run writes follow LangAdapter's kill-matrix schema
  version 3. A mutant's status is `killed`, `timeout`, `non_viable` or
  `survived`, a per-suite verdict of a suite that measured nothing is
  `untested`, and a summary counts `detected` and `non_viable`.
  `starter_pool_attribution.json` counts each suite's verdicts under these
  words.
- The run log tags the starter phase's lines `[starter]`, the mutation runs
  of the starter suites' pool and of the phase ends `[finishing]`, and
  Finishing's lines `[finishing]` or `[metrics]`.

Setting up a project:

- `testuary setup` and the web interface list the installed languages in
  name order.
- Setup finds a project's source files by each language's own rules. Test
  files, the files a language names as no sources, such as Python's
  `setup.py`, and empty files are left out. For Python a `tests.py` and a
  test file named in another case count as tests, for Java
  `*TestCase.java` and `*IT.java` do.
- `testuary check` reports whether each target is in the project
  directory, as the language's check finds it: verified, or missing setup
  with that check's reason. A run checks the targets itself before any
  work.
- `testuary check --verify` tries the project in a copy as a run does,
  without running a test: for Java it resolves the build without
  downloading, compiles and finds the tests and starts the measurement
  tools a run needs, and for Python it collects the tests and checks where
  they import the project from. The first run tries the build of a project
  with its own build wrapper, which may download its build tool. The report
  names these parts, such as the build of your project and the collection
  of your project's tests.
- The check report gives each operation of a language its own name, such
  as removing single tests from a test file and counting the tests and
  assertions of a test file.
- `testuary check` refuses an unknown phase in `--reuse` or
  `--stop-after-phase` in its own report, with exit status 2, as JSON with
  `--json`.
- A runtime that a project's settings request, such as `python_version`, is
  provisioned by that language's adapter and kept in a directory per
  language in Testuary's cache (`venvs/<language>/<project>`).
  `testuary provision` prepares what every installed language's settings
  request and prints each runtime, or says that none is requested. It
  writes its errors to standard error.

Models and their keys:

- `testuary init` writes the example aliases in `models.ini` as comments,
  so a new catalogue names no model until you add one.
- An alias that `testuary setup model` or the Models page writes for
  `ollama`, `openai` or `vllm` holds the address you confirmed, the
  packaged one included. It keeps that address when a global `config.ini`
  sets a gateway in that service's section, so its key goes only to the
  confirmed address. Only `[api] api_url`, which sets the address of every
  model, moves it. To use the gateway, give its address as the alias's
  address.
- An alias's `chat_model` names a client factory as a Python file or a
  module before its attribute, such as `my_chat.py:factory`. A file is relative to
  the directory of the file that names it and needs no installation, and
  the example `testuary init --examples` copies is one such file.
- A project's settings file may not set `[api] api_url` or the `api_url` of
  a provider's section, which decide where the model's key goes. A run of
  such a project is refused at its start with exit status 2. Set the
  address in your global `config.ini`, in `models.ini`, in the environment
  or on the command line.
- A variable ending in `_API_KEY` is always a model service's key and never
  a setting, so a service may be named like a settings section, such as
  `main_x` with `TESTUARY_MAIN_X_API_KEY`.
- A stored model key goes only to its model's service address, in a Test
  or a run, and no answer of `testuary-web`'s server holds it. The Models
  page says whether a model's key is stored in `models.ini` or comes from
  an environment variable, and Edit changes a stored key.
- When the record of a failed model request cannot be written, the
  traceback in the run log and on the console leaves out the model
  service's error, which can hold the request and its key.
- A `max_output_tokens` that is no nonnegative integer is refused wherever
  it is set, in a phase section or a model alias, before any request, as
  the model's own resolution refuses it. The agents' and the starter
  phase's model address, model name and key come from that same
  resolution.
- Starter suite requests carry no temperature unless you set
  `[starter] temperature`, which is empty by default, so the provider's
  default applies.

Extensions:

- `[main] extensions` names the extensions a run and `testuary-web` load:
  Python files, such as `.testuary/my_extension.py`, and installed modules.
  An extension adds a language with its top-level `langadapter_language`,
  model hints with `testuary_hints` and `testuary-web` pages with
  `testuary_web`. Each settings file adds its items to those of the files
  before it, and a file item is relative to the directory of the settings
  file that names it. `testuary check` finds each item without running it,
  and the tool check loads it. The Settings page edits each file's own
  items and shows the combined list. `testuary.settings.add_extension` and
  `remove_extension` change the items of the user's `config.ini`, and
  `extension_items` names each item with its source. The generation origin
  records the extensions and their hints, and `testuary tutorial` teaches
  the first extension as one file.
- A `testuary.web.WebExtension` adds tabs, result views, settings groups,
  jobs and read-only routes to `testuary-web`. It can set the product name
  and the header's Guide link, rename, move and hide Testuary's tabs, and
  put a tab of its own in the place of one of them. Testuary's links to a
  replaced tab open the extension's, its links to a hidden tab are left
  out, and its pages name a renamed or replaced tab as the header does. The
  extension guide lists every part an extension can change. A job's setup
  report in Testuary's readiness format shows on its page as a check's
  report does. `testuary.web.TOOL_CHECK_LABEL`, which `/api/state` carries
  as `tool_check_label`, is the label of Testuary's tool check for a
  replacing Check page.
- An extension's page shows a Testuary run folded, as Testuary's own pages
  draw it, with `testuaryWeb.nestedRun`: a run's event file as its job's
  page shows it, followed while it is written, or a run directory as
  Results shows it. A job kind names the Testuary runs its jobs start with
  `nested_runs`, each a `testuary.web.NestedRun`, and its job's page folds
  each into its phase.
- An extension's settings group can name the Testuary settings files its
  runs in a project read instead of the global and the project's settings
  files, its own file last (`SettingsGroup.testuary_settings_files`). The
  Settings page offers that list as a choice of the project, shows every
  setting with the file it comes from as those runs read it, and changes
  the last file through Testuary's validated update. A last file outside
  the project directory is read-only.
- A settings group can also name the model catalogue files its runs in a
  project read (`testuary_models_files`), its own user directory
  (`extension_user_dir`) and a function that makes its runs read its own
  file alone (`end_models_import`). The Models page shows those models for
  the project, changes the list's last file, and removes a model an
  earlier file defines by ending the import after a question. After the
  Settings page or the Models page saves a file of a group's list, it asks
  the group again, and a group that raises puts the file back.

Programs of your own:

- `testuary.settings.resolve_settings` takes `models_files`, an ordered
  list of model catalogue files read instead of the user directory's
  `models.ini`, a later file's section overriding the same section of an
  earlier one. A file in the project directory may set neither `api_url`
  nor `api_key_env`, which decide where the model's key goes.
- `testuary.settings.check_project_settings_file` and
  `testuary.settings.check_project_catalogue_file` check one settings or
  model catalogue file that is not your own, as Testuary checks a
  project's: a settings file may not set an `api_url`, and a catalogue file
  neither an `api_url` nor an `api_key_env`. A program that passes such
  files to its runs calls them for each.
- `testuary.models.catalogue_chat_model` gives your own program a LangChain
  chat model for an alias of your model catalogue, or with `provider` for a
  model that provider serves, at the provider's address and with its key.
  It resolves the model as a Testuary run resolves `[api] model` and
  `[api] provider`, retries temporary failures as Testuary does, and
  records each request and each model's settings in a directory you name.
  The model, its records and its errors hold no key. Its `config_files`,
  `project_settings` and `models_files` choose the settings files and the
  model catalogue as for `resolve_settings`, so that a program whose runs
  name their own files gets models that read the same files.
  `testuary.models.REQUEST_VALUES` holds the request values it accepts.
- `testuary.setup.models`, `set_model` and `remove_model` take
  `models_files`, the list of model catalogue files `resolve_settings`
  reads, and change its last file only, writing a model an earlier file
  defines into it whole and no key. Each `ModelEntry` names its `file`. A
  catalogue change is refused when it would leave a model that resolves
  unable to resolve.
- `testuary.readiness.ReadinessLimit` and
  `testuary.presentation.present_readiness` are on the API reference
  beside `RunReadiness`, so a program builds and prints a setup report as
  `testuary check` does.
- `testuary.presentation.EventFile` writes the event file `--events`
  writes, for a program of your own, with `EVENT_FILE_SCHEMA`, `RunPlan`
  and `RunCompletion` (the API reference's "Event files").
  `run_pipeline(..., events=PATH)` writes a program's run into an event
  file as `--events` does, with the setup report, the plan and the end of
  the run, so a page that folds the run shows its phases and its summary.
  The event file has `schema_version` 3. Its plan record names Finishing
  `finishing`, Finishing's operation is `step/finishing`, and the step that
  ends each phase is labeled Consolidation.
- `testuary.generation_evidence.read_suite_slots(manifest, suites_root)`
  and `read_cleaned_source(reference, suites_root)` read the generation
  records.
- `testuary.results.recorded_source_sha256` reads the SHA-256 of a target
  file's raw bytes that a run recorded, so a program can check that an
  earlier run tested the same file before it reuses that run's suite.
- `testuary.presentation.read_completion` reads a run's end-of-run
  summary, `results/completion.json`, as its `RunCompletion` and the
  provenance of its values, so a program reads the file as Testuary's
  console and web interface do.

### Requirements

- Python 3.10 to 3.14 for Testuary, on Windows or Linux. macOS is expected
  to work, but this release has not run on a Mac.
- For a Python project: pytest, coverage and Cosmic Ray in the project's
  environment. Tested with pytest 8.1.1, coverage 7.6.8 and Cosmic Ray
  8.4.1, on Python 3.14 and 3.15 with pytest 8.4.2 and coverage 7.16.2,
  and project Python 3.10 to 3.15. Python 3.9 is expected to work.
  Testuary itself is not tested on Python 3.15, where tree-sitter, which
  LangAdapter uses, has no wheels, so it installs there only with a C
  compiler.
- For a Java project: JDK 11 or newer, tested with 11, 17, 21, 25, 26 and
  27, and the project's Maven or Gradle build. LangAdapter downloads the
  JUnit launcher, JaCoCo and PIT on first use.
- For a C project, on Linux: LangAdapter's extra `langadapter[c]`, the
  project's GCC with its gcov, and Mull with the clang of its LLVM release
  for mutation. Tested with GCC 13.3 on Ubuntu 24.04 under WSL.
- A model service: OpenAI, an OpenAI-compatible service such as vLLM, or
  Ollama.

[Tested environments](https://k-kammler.github.io/testuary/reference/tested-environments/)
lists the tested combinations in detail.

### Known limitations

- The model sees the file under test, not the classes it uses. Tests of
  code that depends on other classes of the project can be harder for it
  to write.
- A test can leave the module under test changed for the tests after it.
  For Python the test run names such a test in a note, and a loop at its
  goal waits for a passing run without one. When the loop's request, token
  or time limit ends it first, the test stays. The measured scores remain
  valid, because they come from running the delivered suite.
- A test that passes in a normal run can fail while coverage or mutation
  is measured, for example a Java test that counts a class's members by
  reflection, since the measurement tools add hidden members. A tool that
  cannot measure a suite gives it no number. When the failure names tests,
  in the starter phase or at a phase end, those tests are removed and the
  suite is measured again. Otherwise the starter suite is generated again,
  a phase end's suite is no candidate, and in the loops the model is told
  and it counts as a failed test run.
- With the default settings, every suite a phase can hand on has run under
  both measurement tools. With `phase_end_mutation = false` the candidates
  at the end of the starter and coverage phases are measured under the
  coverage tool only, the end of the mutation phase still measures
  mutation, and with `reduced_suite_mutation = derive` the reduced suite's
  mutation score is derived from the merged suite's run. The results page
  labels that derived score.
- Adding a language takes a LangAdapter adapter and no Testuary code, as
  [Make a language ready for Testuary](https://k-kammler.github.io/testuary/extending/languages/)
  describes.
- Some Python source operations read the project's code with the Python
  that runs Testuary, so code written for a newer Python than that one can
  be refused.
- Gradle projects were not tested in full runs. A Java project with
  both Maven and Gradle build files is refused.
- Conda, uv, Poetry and Pipenv environments were not tested beyond their
  interpreter.
- On Windows, a project's commands cannot run in a working directory longer
  than 258 characters.

### Documentation

The guides are on the
[documentation site](https://k-kammler.github.io/testuary/),
starting with
[Getting started](https://k-kammler.github.io/testuary/getting-started/).
