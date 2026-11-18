# Settings

This page is generated from the packaged defaults (`config.ini` in the
installed package) and their comments, and from the `[project]` settings
the built-in languages declare. A setting a language declares names
that language. A key that a section accepts without a value in `config.ini`
is listed with "no default" and the page that says what applies when
no layer sets it. A configuration file, the environment
(`TESTUARY_<SECTION>_<KEY>`) or the command line overrides these values,
as the [configuration guide](../guide/configuration.md) describes. An
empty default means unset. The model request controls, such as
`temperature`, `reasoning` and `request_timeout_seconds`, are explained
under [request controls](../models-and-prompts.md#request-controls). A
variable ending in `_API_KEY` is a model service's key, never a setting.

## [api]

The model settings: the model a run uses.

- `model`, default empty
  A model alias of models.ini, or a model name the provider below serves. testuary setup project writes the project's model into its .testuary/project.ini. A run that selects no model in any layer is refused before any work. At the terminal, --model overrides it.

- `provider`, default empty
  The provider of a model name that is no alias: openai, ollama, vllm or a provider section of models.ini. An alias names its own provider.

- `temperature`, `top_p`, `seed`, `max_output_tokens`, `request_timeout_seconds`, `reasoning`, `reasoning_effort`, `reasoning_budget_tokens`, `protocol`, `request_options`, no default
  Request controls, explained under [request controls](../models-and-prompts.md#request-controls).

- `preserve_reasoning_content`, `reasoning_field_names`, no default
  Explained under [`[coverage]`](#coverage).

- `deployment`, no default
  The label of the service deployment, explained under [models.ini fields](../models-and-prompts.md#modelsini-fields).

- `api_url`, no default
  The address of every model, explained under [select a model](../models-and-prompts.md#select-a-model).

## [openai]

OpenAI-compatible endpoint defaults. Per-model settings can override them.

- `api_url`, default `https://api.openai.com/v1`
  The endpoint of the openai provider.

- `seed`, refused by a run
  A run refuses a seed here, as [request controls](../models-and-prompts.md#request-controls) says.

- `temperature`, `top_p`, `max_output_tokens`, `request_timeout_seconds`, `reasoning`, `reasoning_effort`, `reasoning_budget_tokens`, `protocol`, `request_options`, `reasoning_dialect`, `reasoning_template`, no default
  Request controls, explained under [request controls](../models-and-prompts.md#request-controls).

- `preserve_reasoning_content`, `reasoning_field_names`, no default
  Explained under [`[coverage]`](#coverage).

- `chat_model`, no default
  The client factory for this service's aliases, explained in [your own model client](../extending/model-provider.md).

- `deployment`, no default
  The label of the service deployment, explained under [models.ini fields](../models-and-prompts.md#modelsini-fields).

## [ollama]

Ollama endpoint defaults. Keep the provider credential outside this file.

- `api_url`, default `https://ollama.com/v1`
  The endpoint of the ollama provider: Ollama Cloud by default. An Ollama on this machine is http://localhost:11434/v1.

- `seed`, refused by a run
  A run refuses a seed here, as [request controls](../models-and-prompts.md#request-controls) says.

- `temperature`, `top_p`, `max_output_tokens`, `request_timeout_seconds`, `reasoning`, `reasoning_effort`, `reasoning_budget_tokens`, `protocol`, `request_options`, `reasoning_dialect`, `reasoning_template`, no default
  Request controls, explained under [request controls](../models-and-prompts.md#request-controls).

- `preserve_reasoning_content`, `reasoning_field_names`, no default
  Explained under [`[coverage]`](#coverage).

- `chat_model`, no default
  The client factory for this service's aliases, explained in [your own model client](../extending/model-provider.md).

- `deployment`, no default
  The label of the service deployment, explained under [models.ini fields](../models-and-prompts.md#modelsini-fields).

## [vllm]

Local OpenAI-compatible endpoint defaults.

- `api_url`, default `http://localhost:8000/v1`
  The endpoint of the vllm provider.

- `seed`, refused by a run
  A run refuses a seed here, as [request controls](../models-and-prompts.md#request-controls) says.

- `temperature`, `top_p`, `max_output_tokens`, `request_timeout_seconds`, `reasoning`, `reasoning_effort`, `reasoning_budget_tokens`, `protocol`, `request_options`, `reasoning_dialect`, `reasoning_template`, no default
  Request controls, explained under [request controls](../models-and-prompts.md#request-controls).

- `preserve_reasoning_content`, `reasoning_field_names`, no default
  Explained under [`[coverage]`](#coverage).

- `chat_model`, no default
  The client factory for this service's aliases, explained in [your own model client](../extending/model-provider.md).

- `deployment`, no default
  The label of the service deployment, explained under [models.ini fields](../models-and-prompts.md#modelsini-fields).

## [main]

Shared execution, context and end-of-phase settings.

- `prompt_profile`, default empty
  A prompt profile file, which appends to, replaces or removes parts of the bundled prompts (the [prompt profile guide](../extending/prompt-profile.md) describes it). Empty uses the bundled prompts.

- `seed`, default `0`
  The base of the starter suites' sampling seeds, unless [starter] seed_base sets one. The [configuration guide](../guide/configuration.md) gives the formula. The coverage and mutation phases send a seed only when [api], [coverage] or [mutation] sets one.

- `context_rebuild`, default `mode_change`
  When the coverage and mutation phases rebuild the conversation they send: mode_change rebuilds when the agent's set of available tools changes (for example from editing to measuring) and keeps the rebuilt conversation in between. every_turn rebuilds on every turn. A rebuild keeps the task prompt and the most recent messages ([limits] history_messages) and drops earlier reasoning text.

- `context_rebuild_trigger_tokens`, default `0`
  Rebuild also when the conversation reaches this many estimated tokens while the tool set stays the same. Zero disables it.

- `phase_end_mutation`, default `true`
  At the phase ends of the starter and coverage phases, also measure mutation over the full catalogue, so that suite reduction keeps every mutant kill. The mutation phase's phase end measures mutation either way.

- `reduced_suite_mutation`, default `both`
  How the reduced suite's mutation score is obtained: derive restricts the merged suite's kill matrix (which test kills which mutant) to the kept tests, measure measures the reduced suite again, both uses the new measurement and records the derived score as a cross-check. Choosing a suite still requires passing tests and complete measurements. Finishing reports no derived score: it measures a delivered suite whose score was derived.

- `work_dir_job_namespace`, default `auto`
  Whether runs get their own directory for the temporary work tree. never: no own directory. auto: one only when TESTUARY_MAIN_WORK_DIR_NAMESPACE or TESTUARY_EXECUTION_ID names it. always: also without them, named after the process and device_name. Use it when several runs share one run directory. Result and log directories are not separated by this.

- `device_name`, default empty
  A name for this computer in the records, which label the clock their times come from (clock_domain_id), and in the work directory of always above. Empty: the computer's own name appears in no result. TESTUARY_CLOCK_DOMAIN_ID replaces the whole clock label.

- `project_exec`, default `local`
  local runs the project's tests with its interpreter on this machine. container runs them in a running container, which TESTUARY_PROJECT_CONTAINER names.

- `pytest_config`, default `auto`
  auto uses the target project's pytest configuration when present. An explicit path selects a configuration file, and none selects pytest defaults.

- `strict_canonical_git`, default `false`
  Refuse the run, rather than warn, when a target of a project that is a Git checkout differs from the file at its HEAD.

- `hints_file`, default `hints.json`
  The hints file, relative to the project directory. Empty uses no hints. The [hints guide](../extending/hints.md) says when a file is optional and what refuses a run.

- `extensions`, default empty
  Extensions, a comma list of Python files and installed modules, such as my_hints.py or my_tool.extension. A run, the tool check (testuary check --verify), testuary provision and testuary-web load them at their start, each file once. The quick check loads none, and a change of this setting loads nothing. Unlike other settings, each layer adds its items to those of the earlier layers: the global configuration or the --config files, then project.ini, then the environment, then a program's overrides. A file is relative to the directory of the settings file that names it, or to the project directory for the environment and overrides.

- `testuary_dir`, default `.testuary`
  The project's Testuary directory, relative to the project directory: it holds the project's settings file project.ini and its results. A run finds the project from the directory it starts in upward, as the nearest directory holding it. Set it in the global configuration, a file given with --config or the environment, never in project.ini.

## [project]

The project a run uses. The project's settings file .testuary/project.ini usually holds this section, and command-line options override it.

- `name`, default empty
  The project's name in results and records (${project}). Empty uses the source directory's name. At the terminal, --project-name overrides it.

- `source_dir`, default empty
  The source directory, whose files are tested. Empty: the project directory.

- `source_files`, default empty
  The source files a run without named targets runs, separated by commas, relative to the source directory. At the terminal, testuary FILES and --source-files name others.

- `language`, default empty
  The project's language. A target named on the command line must be written in it. Empty accepts the targets of every installed language.

- `setup_timeout_seconds`, default `120`
  Time limit, in seconds, for the whole check of the project's tools (interpreter, test tools, JDK, build tool). It does not limit generation or measurements.

- `setup_probe_timeout_seconds`, default `10`
  Time limit, in seconds, for each single probe of that check, within the remaining total.

- `hint_profile`, default empty
  The name of this project's entries in the hints file's per_profile section. Empty: none.

- `python_import_root`, default empty, declared by `python` and `c`
  The directory below the source directory from which its Python package is imported, for example src. Empty imports from the source directory itself.

- `python_test_environment`, default empty, declared by `python`
  Variables the processes of the Python adapter's own runs get, as NAME=value words, for example PURE_PYTHON=1. Python tests of a C extension do not get them.

- `python_executable`, default empty, declared by `python` and `c`
  The project's Python interpreter. Empty: the environment uv provisions for python_version, else the project's only environment, its .venv or the one activated in the shell. The environment Testuary runs in counts only inside the project. Two environments, or none, refuse the run.

- `python_version`, default empty, declared by `python` and `c`
  A Python version, for example 3.12, for which uv provisions an environment when python_executable is empty.

- `requirements`, default empty, declared by `python` and `c`
  Requirements file for the environment provisioned by python_version.

- `java_home`, default empty, declared by `java`
  A Java project's JDK: its home or its java executable. Blank uses JAVA_HOME, except in a container, where it must be named.

- `build_java_home`, default empty, declared by `java`
  The JDK the build tool runs on when it needs a newer Java than the project's (Gradle 9 runs on Java 17 or newer). Blank uses java_home.

- `maven`, default empty, declared by `java`
  The Maven launcher when the project has no Maven wrapper. Blank uses mvn on the path.

- `gradle`, default empty, declared by `java`
  The Gradle launcher when the project has no Gradle wrapper. Blank uses gradle on the path.

- `runtime_version`, default empty
  The language version the prompts tell the model, for example "Python 3.10.15". Empty: the version of the interpreter that runs the tests.

- `frozen_module_strategy`, default `unfreeze`, declared by `python` and `c`
  unfreeze runs the project's Python with -X frozen_modules=off, so coverage and mutation treat the standard library's frozen modules as ordinary files. skip leaves them frozen.

- `output_dir`, default empty
  The run directory. Empty uses results/<model> in the project's Testuary directory. A relative path resolves from the project directory. The ${model} and ${project} placeholders are filled with file-name-safe values. At the terminal, --output-dir overrides it.

- `c_tests`, default empty, declared by `c`
  The kind of the project's tests: c (utest.h tests), python (Python tests of a target built into a Python extension), programs (the project's own test programs, each judged by its exit status) or ctest (a CMake project's tests as ctest runs them).

- `c_compiler`, default empty, declared by `c`
  The GCC that builds the project's C code and its tests.

- `c_build_kind`, default empty, declared by `c`
  How the project builds: make (its Makefile, run in a copy of the project), cmake (its CMakeLists.txt, built outside its tree), configure, setuptools, or flags (one file, with c_flags and c_link).

- `c_build`, default empty, declared by `c`
  The command that builds the project in its copy, when not plain make. For a CMake project it holds the options of its configure step (-DNAME=VALUE).

- `c_flags`, default empty, declared by `c`
  The compile flags of the target, for a project without a build file.

- `c_link`, default empty, declared by `c`
  The objects, libraries and C sources (also patterns, such as src/*.c) the target links with, for a project without a build file.

- `c_build_cache`, default empty, declared by `c`
  A directory that keeps a configure build for later runs, instead of building once per adapter.

- `c_target`, default empty, declared by `c`
  The C file the tests are for, when a request does not name it. Load repair and import pruning compile against it.

- `c_mull`, default empty, declared by `c`
  Mull's runner (mull-runner-N), whose IR frontend and clang N build the target's mutants.

- `c_test_prelude`, default empty, declared by `c`
  The headers every test file includes first, as the project's own tests include them (such as stdlib.h test/alloc-testing.h), each passed as -include.

- `c_ctest_arguments`, default empty, declared by `c`
  Extra arguments for named ctest tests (c_tests ctest), given after their own. The entries are separated by semicolons, each a test name followed by its arguments, such as fuzz_test {seed}. The value of {seed} is 1, since a C run carries no session seed.

- `c_test_support`, default empty, declared by `c`
  The sources the project's own test programs build with besides the target (c_tests programs), such as test/framework.c test/alloc-testing.c. They are the suite's setup files.

- `c_test_environment`, default empty, declared by `c`
  Variables every process of the C adapter's runs gets, Python tests of the target's extension among them, as NAME=value words such as CK_FORK=no.

- `c_address_space_limit`, default empty, declared by `c`
  The most address space each test process may reserve (RLIMIT_AS, set with prlimit), such as 2G or a number of bytes. Empty: no limit.

## [starter]

The starter phase: generating and repairing the starter suites, and the phase end.

- `suite_mutation`, default `measure`
  How each accepted starter suite's mutation score is obtained: measure, derive or both. The [explanation page Measurement and recovery](../explanation/measurement-and-recovery.md) describes each method and what a derived score can get wrong.

- `suite_count`, default `10`
  Requested number of independent starter suites, at most 1000.

- `refill_failed_suites`, default `true`
  Generate a failed starter suite again, for example after a reply without a test suite, within that suite's own attempts. This does not guarantee suite_count usable suites when the attempts run out. Off, each starter suite has one attempt.

- `attempts_per_suite`, default `4`
  Generation attempts per starter suite, the first included. It applies only while refill_failed_suites is on. A suite does not borrow attempts from another.

- `reuse_suites`, default `false`
  Reuse the starter suites an earlier run in this run directory saved, checked again before use. A missing or empty one is generated again, which sends model requests, and without saved suites every one is. The phase end combines them again. At the terminal, --reuse starter instead takes the starter phase's saved result as it is.

- `cache_warmup`, default `true`
  Send the first request alone before the remaining concurrent work. Provider prefix-cache behavior depends on the endpoint.

- `temperature`, default empty
  Sampling temperature. A blank value omits it from generation requests. Each request carries its own seed from seed_base, which this section may set and which defaults to [main] seed. The [configuration guide](../guide/configuration.md) gives the formula.

- `reasoning_grace_seconds`, default `0`
  Abandon a streamed generation that produces no answer content within this many seconds. Zero disables this content-aware timeout.

- `reasoning`, default empty
  Optional on/off control. Empty leaves the provider's reasoning default. OpenAI-compatible endpoints need a declared reasoning_dialect. An off control cannot be combined with a reasoning effort or token budget.

- `concurrent_requests`, default `10`
  The most starter suite requests sent at the same time.

- `concurrent_requests_ollama`, default `4`
  The most starter suite requests sent at the same time with the ollama provider (Ollama Cloud or an Ollama on this machine), in place of concurrent_requests.

- `suite_run_timeout_seconds`, default `600`
  Wall-clock limit for a suite execution, in seconds.

- `reject_unsound_async_green`, default `true`
  Count an otherwise passing run as failing when it warns of an asynchronous test that did not run properly, such as an unawaited coroutine.

- `failing_test_cleanup_rounds`, default `3`
  How many times a starter suite with failing tests has them removed and runs again before it is generated anew. On the suite's last attempt this goes on while each round removes at least one more failing test. The same number bounds, for each suite at each place, how often a coverage run or mutation measurement that fails the suite after its passing normal run has the tests it names removed before the suite is measured again. The [explanation page Measurement and recovery](../explanation/measurement-and-recovery.md) names the places.

- `reduction_solver`, default `exact`
  Suite reduction: at a phase end, Testuary drops the tests that add no coverage and no mutant kill, keeping the fastest set. exact finds that set with PuLP's CBC solver and requires a proven optimum. greedy is a heuristic you can select instead.

- `reduction_solver_time_limit_seconds`, default `120`
  Time limit of the exact solver, in seconds, for all its steps together.

- `reduction_min_test_time_ms`, default `0.001`
  The smallest run time counted for one test in suite reduction, in milliseconds.

- `reduction_solver_random_seed`, default `1`
  The solver's seed (CBC), separate from the starter suites' seeds.

- `reduction_tie_break`, default `name_order`
  Among sets of tests with the optimal run time and test count: name_order counts exact duplicate tests once, prefers tests earlier in name order and proves the choice unique. solver keeps whichever optimal set CBC returns, which can differ between its builds.

- `max_output_tokens`, default `0`
  The most tokens one model response of this phase may have. When the model's alias also sets a positive limit, the smaller one applies. Zero sets no limit here.

- `max_consecutive_transient_failures`, default `10`
  Temporary failures of one starter suite request in a row before the phase ends. The [configuration guide](../guide/configuration.md) says which failures are temporary and how they are retried.

- `max_transient_failures_per_phase`, default `100`
  Temporary failures in the whole phase before it ends. Zero disables it.

- `transient_first_wait_seconds`, default `10`
  Wait before the first retry without a provider hint, in seconds. It doubles per failure up to the maximum.

- `max_transient_wait_seconds`, default `60`
  The longest such wait, in seconds.

- `max_transient_hint_seconds`, default `3600`
  The longest wait a provider's Retry-After hint may ask for, in seconds.

- `seed`, refused by a run
  A run refuses a seed here, as [request controls](../models-and-prompts.md#request-controls) says.

- `request_timeout_seconds`, `top_p`, `reasoning_effort`, `reasoning_budget_tokens`, `protocol`, `request_options`, no default
  Request controls, explained under [request controls](../models-and-prompts.md#request-controls).

- `preserve_reasoning_content`, `reasoning_field_names`, no default
  Explained under [`[coverage]`](#coverage).

- `seed_base`, no default
  The first sampling seed of the starter suites, explained under [sampling seeds of the starter suites](../guide/configuration.md#sampling-seeds-of-the-starter-suites).

## [coverage]

Coverage phase settings. The mutation phase inherits every key its own section does not set.

- `preserve_reasoning_content`, default `true`
  Send the model's reasoning back with the conversation when the service returns it in a separate field (the names below). It uses private langchain-openai methods, which a later version may change.

- `reasoning_field_names`, default `reasoning_content, reasoning_details, thinking`
  The response fields that carry the reasoning, sent back when present.

- `enable_trace_log`, default `false`
  Write per-request, tool-selection, final answer check, recovery and reasoning events.

- `request_timeout_seconds`, default `3600`
  Time limit, in seconds, for one model request of the coverage phase, and of the mutation phase unless [mutation] sets its own. A provider section, a models.ini alias or [api] can set it for every phase, and a phase section of the same file wins over them. The starter suites' requests use 600 unless [starter], [api], a provider section or the alias sets it.

- `max_consecutive_transient_failures`, default `10`
  Temporary failures of one request in a row before the phase ends. The [configuration guide](../guide/configuration.md) says which failures are temporary and how they are retried.

- `max_transient_failures_per_phase`, default `100`
  Temporary failures in the whole phase before it ends. Zero disables it.

- `transient_first_wait_seconds`, default `10`
  Wait before the first retry without a provider hint, in seconds. It doubles per failure up to the maximum.

- `max_transient_wait_seconds`, default `60`
  The longest such wait, in seconds.

- `max_transient_hint_seconds`, default `3600`
  The longest wait a provider's Retry-After hint may ask for, in seconds.

- `count_retries`, default `false`
  Count every retry of a failed model request against max_requests_per_phase.

- `phase_end_evidence_reuse`, default `true`
  When a phase ends with exactly the suite it started with, measured the same way and with complete measurements on record, it reuses those measurements instead of measuring, reducing and choosing again. False measures, reduces and chooses again.

- `builtin_replacement_guidance`, default `true`
  Ask the model, in the loops' hard rules, to replace a built-in function only in the module under test and to simulate a missing package through sys.modules. Guidance only: no test is rejected for ignoring it. False leaves this guidance out of the prompts.

- `suite_run_timeout_seconds`, default `600`
  Wall-clock limit for a suite execution, in seconds.

- `suite_timeout_stop_after`, default `10`
  Suite timeouts in a row before the phase ends. A passing run resets the count. Nonpositive values disable this stop.

- `reject_unsound_async_green`, default `true`
  Count an otherwise passing run as failing when it warns of an asynchronous test that did not run properly, such as an unawaited coroutine.

- `report_reverse_order_failures`, default `false`
  After a passing test run, also run the file as delivered in reverse order, for a language that offers that order, and tell the model which tests fail there. The note decides no test's outcome, but while tests fail when run in reverse order, the loop does not end at its goal.

- `max_loop_attempts`, default `100`
  Attempts of the loop per phase, the first included, used up by rejected final answers and by reruns after a temporary failure inside the loop, such as a tool that raised. Each rejected final answer counts as a model request. Retries of a model request are separate (the settings above). unlimited ends the attempts only at a failure that is not retried or at a limit.

- `max_requests_per_phase`, default `1000`
  The most model requests this phase may send, a positive number or unlimited. A request whose answer is refused counts. Retries after a temporary failure count only with count_retries.

- `max_input_tokens_per_phase`, default `0`
  The most input tokens all of this phase's requests may have together. Zero or unlimited disables this limit.

- `max_output_tokens_per_phase`, default `0`
  The most output tokens all of this phase's responses may have together. Zero or unlimited disables this limit.

- `max_total_tokens_per_phase`, default `0`
  The most input and output tokens together in this phase. Zero or unlimited disables this limit.

- `max_wall_clock_seconds_per_phase`, default `0`
  The longest time this phase may take, in seconds. Zero or unlimited disables this limit.

- `goal`, default `90`
  Coverage goal in percent, by goal_metric. The loop ends once it is reached (see extra_measurements_after_goal), but not while tests fail when run in reverse order (report_reverse_order_failures).

- `minimum`, default `75`
  A final score below this is marked below its minimum in tests.csv. While the best score is below it, any rise counts as an improvement. complete_tests_below_minimum decides whether the later phases run after it is missed. Zero disables the minimum.

- `complete_tests_below_minimum`, default `true`
  Run the later phases after a phase misses its minimum score. If false, they are skipped. Finishing still records the result.

- `goal_metric`, default `branch`
  line or branch selects that coverage score. both uses the lower of line and branch coverage, so both must reach the goal, and either the higher one. The best passing suite and the measurements without improvement use this score.

- `max_measurements_without_improvement`, default `10`
  Measurements without improvement after which the loop stops. Nonpositive values disable this stop. A share such as 10% is that share of this phase's max_requests_per_phase, rounded up, and is refused while max_requests_per_phase is unlimited.

- `max_final_answers_without_measurement`, default `10`
  Times in a row the model may submit its final answer without a new measurement before the phase stops early. Separate from the measurements without improvement.

- `min_improvement_points`, default `0.5`
  The rise in percentage points that counts as an improvement. Below a configured minimum score, any measured rise counts.

- `restore_count`, default `1`
  Go back to the best passing suite after this many passing measurements that scored lower than it. Equal scores do not count. Nonpositive values disable going back.

- `reduction_solver`, default `exact`
  Suite reduction: at a phase end, Testuary drops the tests that add no coverage and no mutant kill, keeping the fastest set. exact finds that set with PuLP's CBC solver and requires a proven optimum. greedy is a heuristic you can select instead.

- `reduction_solver_time_limit_seconds`, default `120`
  Time limit of the exact solver, in seconds, for all its steps together.

- `reduction_min_test_time_ms`, default `0.001`
  The smallest run time counted for one test in suite reduction, in milliseconds.

- `reduction_solver_random_seed`, default `1`
  The solver's seed (CBC), separate from the starter suites' seeds.

- `reduction_tie_break`, default `name_order`
  Among sets of tests with the optimal run time and test count: name_order counts exact duplicate tests once, prefers tests earlier in name order and proves the choice unique. solver keeps whichever optimal set CBC returns, which can differ between its builds.

- `progress_heartbeat_requests`, default `25`
  Log progress every N model requests. Zero disables this heartbeat.

- `extra_measurements_after_goal`, default `0`
  Extra measurements at the goal after the first one, repeated measurements not counted. Zero stops at the first. Below the goal the count pauses rather than resets.

- `count_repeated_measurements`, default `false`
  Whether a repeated or rejected measurement counts as one without improvement.

- `allow_early_stop`, default `true`
  Let the loop stop once max_measurements_without_improvement is reached.

- `max_failed_test_attempts`, default `10`
  Failing test runs in a row after which the suite is restored to its last passing version.

- `temperature`, `top_p`, `seed`, `max_output_tokens`, `reasoning`, `reasoning_effort`, `reasoning_budget_tokens`, `protocol`, `request_options`, no default
  Request controls, explained under [request controls](../models-and-prompts.md#request-controls).

## [mutation]

Mutation phase settings. The loop edits the coverage phase's delivered suite.

- `goal`, default `80`
  Mutation score goal in percent. The loop ends once it is reached (see extra_measurements_after_goal), but not while tests fail when run in reverse order (report_reverse_order_failures).

- `minimum`, default `65`
  A final mutation score below this is marked below its minimum in tests.csv. While the best score is below it, any rise counts as an improvement. Zero disables the minimum.

- `extra_measurements_after_goal`, default `0`
  Extra measurements at the goal after the first one, repeated measurements not counted. Zero stops at the first. Below the goal the count pauses rather than resets.

- `restore_count`, default `1`
  Go back to the best passing suite after this many passing measurements that scored lower than it. Equal scores do not count. Nonpositive values disable going back.

- `max_measurements_without_improvement`, default `10`
  Measurements without improvement after which the loop stops. Nonpositive values disable this stop. A share such as 10% is that share of this phase's max_requests_per_phase, rounded up, and is refused while max_requests_per_phase is unlimited.

- `min_improvement_points`, default `0.5`
  The rise in percentage points that counts as an improvement. Below a configured minimum score, any measured rise counts.

- `allow_early_stop`, default `true`
  Let the loop stop once max_measurements_without_improvement is reached.

- `count_repeated_measurements`, default `false`
  Whether a repeated or rejected measurement counts as one without improvement.

- `max_requests_per_phase`, default `1000`
  The most model requests this phase may send, a positive number or unlimited. A request whose answer is refused counts. Retries after a temporary failure count only with count_retries.

- `max_input_tokens_per_phase`, default `0`
  The most input tokens all of this phase's requests may have together. Zero or unlimited disables this limit.

- `max_output_tokens_per_phase`, default `0`
  The most output tokens all of this phase's responses may have together. Zero or unlimited disables this limit.

- `max_total_tokens_per_phase`, default `0`
  The most input and output tokens together in this phase. Zero or unlimited disables this limit.

- `max_wall_clock_seconds_per_phase`, default `0`
  The longest time this phase may take, in seconds. Zero or unlimited disables this limit.

- `engine`, default `cosmic_ray`
  The mutation tool. The shipped default cosmic_ray runs the language's own tool: Cosmic Ray for Python, PIT for Java and Mull for C. The values pit, for Java, and mull, for C, select them by name. A tool the target's language does not offer fails when its first mutation measurement starts, with the default phase_end_mutation at the starter phase's measurement after generation. A run's records name the tool that ran.

- `cosmic_ray_distributor`, default `auto`
  local executes mutants sequentially. http uses worker clones and a worker pool. auto follows engine_workers: local for 1 worker, http for more. A Python run with local and more than one worker is refused before it starts. Neither distributor guarantees identical outcomes across resource conditions.

- `engine_workers`, default `1`
  How many workers measure mutants at once: Cosmic Ray's HTTP workers, PIT's threads for Java and parallel mutant runs for C. Cosmic Ray's local distributor runs one. The count is part of a measurement's execution conditions.

- `allow_sequential_fallback`, default `false`
  Allow local execution if the configured HTTP pool cannot start. False preserves the failure instead of silently changing the distributor.

- `per_mutant_timeout_seconds`, default `auto`
  auto derives the timeout from the passing suite's run time, multiplied by the configured factor and bounded by min_per_mutant_timeout_seconds and max_per_mutant_timeout_factor times suite_run_timeout_seconds. For C the mutation tool times the passing suite itself, each test in a process of its own as a mutant's tests run, and applies this rule to that time. A positive integer selects a fixed per-mutant timeout instead. For C it bounds all of a mutant's tests together, and a mutant whose time ends between two of its tests is untested.

- `per_mutant_timeout_factor`, default `10`
  With auto: the per-mutant timeout is this many times the passing suite's run time, for C the mutation tool's own run.

- `min_per_mutant_timeout_seconds`, default `30`
  With auto: the shortest per-mutant timeout, in seconds.

- `max_per_mutant_timeout_factor`, default `1.5`
  With auto: the longest per-mutant timeout, as a factor of suite_run_timeout_seconds.

- `measurement_timeout_seconds`, default `3600`
  The time limit of one mutation measurement, in seconds. A measurement that reaches it is incomplete: its untested mutants count as not killed, so its score is a lower bound.

- `cosmic_ray_progress_interval_s`, default `30`
  Interval for mutation progress observations, in seconds. Zero disables them.

- `cosmic_ray_operators`, default empty
  Python (Cosmic Ray): the mutation operators. Empty selects the full catalogue. wong_mathur_selective selects the arithmetic, relational and logical operator families (AOR, ROR, LCR).

- `pit_mutators`, default empty
  Java (PIT): empty selects PIT's default mutators. A PIT group (STRONGER, ALL) or a comma list of mutator names selects others.

- `cosmic_ray_keep_output`, default `false`
  Include each mutant's captured output in the persisted matrix.

- `cosmic_ray_per_file_adjudication`, default `true`
  For a suite of several test files: when a mutant's killing test cannot be named from the combined run, rerun the files one at a time to name it.

- `full_after_selective_goal`, default `false`
  With loop_catalogue = selective: switch to the full catalogue once the selective score reaches selective_switch_threshold or stops improving.

- `selective_switch_threshold`, default `85`
  The score over the selective catalogue, in percent, at which that switch happens.

- `cover_attribution`, default `full`
  Which killing tests a phase end records: full records every killing test, first stops at the first one (fail-fast), which credits each kill to one test only. full gives suite reduction and the derived scores more choices. It is not a guarantee that new measurements have identical outcomes.

- `loop_catalogue`, default `full`
  The operators the mutation phase measures with: full uses the full catalogue throughout. selective uses the selective catalogue cosmic_ray_operators (Python) or pit_mutators (Java) select.

- `cosmic_ray_session_dir`, default empty
  A directory for Cosmic Ray's session files. Empty uses a temporary directory the engine creates and removes. A directory named here stays yours and is not removed.

- `cosmic_ray_init_timeout_s`, default `1800`
  Time limit for building the list of mutants, in seconds.

- `cosmic_ray_worker_ready_timeout_s`, default `60`
  Readiness timeout for each HTTP worker. Unavailable workers are excluded from the pool.

- `cosmic_ray_verbose`, default `false`
  Enable INFO output from cosmic-ray exec and log its captured stderr tail.

- `cosmic_ray_catalogue_cache_dir`, default `auto`
  Directory of cached lists of mutants. auto resolves under the run directory. Empty disables caching. Cache reuse remains bound to its source and tool identity.

- `cosmic_ray_cache_max_entries`, default `200`
  The most cached lists of mutants. Zero keeps any number.

- `cosmic_ray_cache_max_age_days`, default `0`
  Maximum age of a cached list of mutants, in days. Zero keeps them regardless of age.

- `cosmic_ray_clone_dir`, default `auto`
  The directory for each Cosmic Ray HTTP worker's copy of the project. auto chooses one as the [cluster guide](../guide/cluster.md) describes, and a path sets it.

- `cosmic_ray_clone_sweep_hours`, default `12`
  Remove the worker copies a killed run left once they are older than this many hours. Zero turns the removal off. Set it well above the longest Cosmic Ray session, so that a session running at the same time keeps its copies.

- `cosmic_ray_local_session_dir`, default empty
  The directory in which a session without cosmic_ray_session_dir gets its temporary directory, on this machine or, in container mode, in the container. It must be local, such as /dev/shm or a node's own disk: the session database fails on a network file system. Empty chooses one as the [cluster guide](../guide/cluster.md) describes.

- `suite_run_timeout_seconds`, default `600`
  Wall-clock limit for a suite execution, in seconds.

- `suite_timeout_stop_after`, default `10`
  Suite timeouts in a row before the phase ends. A passing run resets the count. Nonpositive values disable this stop.

- `reject_unsound_async_green`, default `true`
  Count an otherwise passing run as failing when it warns of an asynchronous test that did not run properly, such as an unawaited coroutine.

- `report_reverse_order_failures`, default `false`
  After a passing test run, also run the file as delivered in reverse order, for a language that offers that order, and tell the model which tests fail there. The note decides no test's outcome, but while tests fail when run in reverse order, the loop does not end at its goal.

- Every other key of [`[coverage]`](#coverage) apart from `complete_tests_below_minimum` and `goal_metric`, no default

## [finishing]

Finishing, the fourth phase: the delivered suite is measured, without model requests.

- `measure_lint`, default `false`
  Record the lint findings of the target and its tests (Ruff for Python).

- `phase_scores`, default `full`
  What tests.csv reports for the delivered suites of the starter and coverage phases (their own coverage and mutation columns), read from what those phases recorded: full reports coverage and mutation over the full catalogue, coverage omits mutation, off omits these columns. Nothing is measured again for them. The run's delivered suite is measured as usual.

- `suite_run_timeout_seconds`, default `600`
  Wall-clock limit for a suite execution, in seconds.

- `coverage_timeout_seconds`, default `1800`
  Coverage timeout for the delivered suite, in seconds.

- `measure_mutation`, default `true`
  Measure the delivered suite's mutation score when enabled. Missing required suites or engines remain errors, not empty successful measurements. Off, the mutation minimum is not judged: no score of the delivered suite is measured.

## [timing]

Finishing can time the delivered suite, without instrumentation.

- `measure`, default `false`
  Time the delivered suite. Warmups are discarded and timed repetitions produce the reported median. The process-local lock does not prevent interference from other processes or machines.

- `run_count`, default `5`
  Number of timed repetitions.

- `warmup_count`, default `1`
  Discarded warmup repetitions before timing.

- `cpuset`, default empty
  CPU IDs for container timing. Affinity does not reserve CPUs or remove contention.

- `serialize`, default `true`
  Time one suite at a time in this process. The TESTUARY_TIMING_SERIALIZE variable sets it, as for every setting: true or false (also yes or no, on or off, 1 or 0), in any letter case. Any other value, an empty one included, refuses the run.

## [limits]

Limits on what the model sees. Each value is a positive integer or unlimited. A cut output carries a marker naming what was left out. The coverage phase's and the mutation phase's own limits override these, and the mutation phase takes the coverage phase's first. A models.ini alias or provider entry may set them for one model.

- `tool_result_chars`, default `160000`
  At most this many characters of any tool result, a third kept from its start and the rest from its end.

- `read_chars`, default `160000`
  At most this many characters of one read_source or read_tests call, which then names the next start line.

- `test_tracebacks`, default `3`
  At most this many full failure tracebacks in a test run's output.

- `test_failure_names`, default `8`
  At most this many names of the failures whose tracebacks are left out.

- `test_cause_lines`, default `8`
  At most this many import and collection error lines shown first.

- `test_warning_lines`, default `12`
  At most this many lines of pytest's warnings summary.

- `test_fallback_chars`, default `1200`
  At most this many characters of test output with no recognized section, a third kept from its start.

- `maven_error_lines`, default `12`
  At most this many Maven [ERROR] and [WARNING] lines shown first.

- `failure_group_names`, default `4`
  At most this many test names per group of equal failures.

- `collection_errors`, default `5`
  At most this many error lines of a collection failure.

- `unsound_async_items`, default `10`
  At most this many tests named when a passing run is counted as failing for an asynchronous test warning (reject_unsound_async_green).

- `unsound_async_warnings`, default `6`
  At most this many warnings quoted there when no test can be named.

- `failing_run_output_chars`, default `4000`
  At most this many characters of the end of a failing normal run's output before a mutation measurement.

- `empty_run_output_chars`, default `2000`
  At most this many characters of the end of the output of a coverage run that collected no tests.

- `timeout_suspects`, default `8`
  At most this many tests named as suspects after a test run timed out.

- `run_reason_lines`, default `3`
  At most this many lines of the reason a suite run failed, from its end.

- `leak_names`, default `10`
  At most this many names per kind in the note on a test that leaves shared state changed.

- `survivor_diffs`, default `10`
  At most this many surviving mutants shown with their change.

- `survivor_diff_chars`, default `240`
  At most this many characters of the change of one surviving mutant.

- `uncovered_chars`, default `150000`
  At most this many characters of uncovered source shown inline after a coverage measurement.

- `uncovered_suggestions`, default `8`
  At most this many read_source calls suggested for uncovered code not shown.

- `branch_arcs`, default `60`
  At most this many branches not yet covered listed.

- `branch_snippet_chars`, default `70`
  At most this many characters of the source line of a branch destination.

- `state_marker_chars`, default `250`
  At most this many characters of uncovered ranges in the state line of each model request.

- `edit_echo_lines`, default `30`
  At most this many lines of an inserted block echoed in full after an edit, a longer one keeping its first and last lines, six fewer in all.

- `edit_evidence_lines`, default `5`
  At most this many lines of old_text compared when an edit finds no match.

- `edit_evidence_chars`, default `100`
  At most this many characters of each line of old_text compared when an edit finds no match.

- `edit_list_items`, default `unlimited`
  At most this many edits in one edits list, a longer list refused without writing.

- `edit_list_views`, default `unlimited`
  At most this many changed regions shown in the result of one edits list.

- `syntax_snippet_chars`, default `80`
  At most this many characters of the source quoted in a syntax error of an edit, from its start.

- `recovery_result_chars`, default `200`
  At most this many characters of the last run_tests result in the state block of a retry prompt.

- `rejection_chars`, default `600`
  At most this many characters of the validation error of a rejected final answer.

- `malformed_args_chars`, default `200`
  At most this many characters of the parse error of a tool call whose arguments are not valid JSON.

- `history_messages`, default `30`
  At most this many earlier messages kept when the context is rebuilt, without a marker.

## [limits_coverage]

What the coverage phase's model is shown, over [limits].

- Every key of [`[limits]`](#limits), no default

## [limits_mutation]

What the mutation phase's model is shown, over [limits] and [limits_coverage].

- Every key of [`[limits]`](#limits), no default

## [demo]

The offline demonstration (testuary --demo).

- `playback_seconds`, default `120`
  Length of a paced playback in seconds, at --demo-speed 1. The [demonstrations guide](../guide/demonstrations.md) describes how a playback uses it and what else sets it.

## [web]

The browser interface, testuary-web. These settings configure no run, so a run's records leave them out.

- `port`, default `8765`
  The port on 127.0.0.1 the interface listens on. 0 takes a free one. The [web interface guide](../guide/web.md) describes it with the --port option.

- `access`, default `open`
  Who may use the interface: open, key or stable_key. The [web interface guide's section Access](../guide/web.md#access) says what each one allows.
