# Command line

This page is generated from the command-line parsers. `testuary --help`
and `testuary <command> --help` print the same text. The web interface's
command `testuary-web` is described in the [web interface guide](../guide/web.md#start).

## Commands

Generate and measure tests for selected targets.

- [`testuary generate`](#testuary-generate): Generate test suites for the selected targets with a language model. A run asks the model for several independent test suites (the starter suites) and combines them, improves the result in a coverage phase and a mutation phase, and measures it in Finishing.
- [`testuary check`](#testuary-check): Check the setup without contacting a model or installing anything (the quick check). --verify also starts the tools and tries the project in a copy.
- [`testuary init`](#testuary-init): Copy example files: models.ini to your user directory and hints.json to the project. Existing files are kept.
- [`testuary provision`](#testuary-provision): Install the runtime the settings of the project in this directory ask for (such as a Python version with its requirements, made with uv in Testuary's cache), so that a run need not do it. May download.
- [`testuary new-language`](#testuary-new-language): Write a new language as one file, LANGUAGE.py: its adapter with every fact and hook in the order of the reference page Hooks and facts, and the line that names it to LangAdapter. With --for, the hooks the consumer requires are marked TODO with its levels.
- [`testuary setup`](#testuary-setup): Ask for and write what a run needs: model access in models.ini and what runs in parallel on this machine in config.ini, both in your user directory, and a project's settings in .testuary/project.ini in its directory. Without a part, walk all three, skipping the parts already set.
- [`testuary tutorial`](#testuary-tutorial): Learn Testuary's extension points step by step: a model alias, a prompt profile and an extension's hints, each applied, shown and checked, without contacting a model.

generate (the default in a project) writes and measures tests, and its flags also work without 'generate'. check checks the setup without contacting a model (the quick check). setup asks for model access, what runs in parallel on this machine and a project. init copies example files. provision installs the runtime the project asks for. new-language writes a new language's file. tutorial teaches the extension points offline. Use 'testuary \[FILES\]' in a project to run its \[project\] source_files, or the targets named, and 'testuary --demo' for the offline demonstration.

## testuary

- `--version`
  Print the installed Testuary and LangAdapter versions and exit.

## testuary generate

Generate test suites for the selected targets with a language model. A run asks the model for several independent test suites (the starter suites) and combines them, improves the result in a coverage phase and a mutation phase, and measures it in Finishing.

- `FILE`
  Targets to run, relative to the directory the command starts in. Without them, the project's [project] source_files. --source-files names them relative to the source directory instead.

- `--demo`
  Play back one recorded run offline: every model answer and measurement comes from a recording shipped with Testuary, so it needs no model, network, project or test tools. It uses the recorded settings and lists any of yours it overrides. The run directory, the display options, the event file and the demonstration's language are yours.

- `--demo-language` `LANGUAGE`
  Language of the offline demonstration: python (the default), java, or another language whose demonstration data come with Testuary, such as toy. Requires --demo. Default: `python`.

- `--demo-data` `NAME`
  Demonstration data that come with Testuary, played in place of a language's own, such as a run recorded for a tutorial. Another name is refused with the names Testuary has. Requires --demo, without --demo-language.

- `--demo-speed` `SPEED`
  Pace of the demonstration's playback: auto (paced on an interactive terminal, instant when output is redirected, in tests or in continuous integration), instant, or a factor of the configured length's speed such as 2 (half as long) or 0.5 (twice as long). Pacing changes no value or record. Requires --demo. Default: `auto`.

- `--demo-playback-seconds` `SECONDS`
  Length of a paced playback in seconds, at --demo-speed 1, instead of TESTUARY_DEMO_PLAYBACK_SECONDS or [demo] playback_seconds. The demonstrations guide describes the length. Requires --demo.

- `--demo-step`
  Advance the demonstration's playback one step at a time when Enter is pressed, for screenshots. Requires --demo and an interactive terminal.

- `--display` `auto` | `compact` | `detailed` | `plain` | `debug`
  How the run is shown: auto (compact on a terminal, plain otherwise), compact (one updating screen), plain (one line per change), detailed (also every step and the full setup report), debug (also every log line and internal event names). Default: `auto`.

- `--no-color`
  Disable terminal colors.

- `--events` `FILE`
  Also write the run's progress to FILE as JSON lines, for programs that show a run (the terminal guide describes the records). The directory must exist, and an existing file is replaced. Works with --demo.

- `--config` `CONFIG`
  Read settings from this INI file in place of the config.ini in your user directory. Repeat it for several files: later files win. The configuration guide describes the layers that override it.

- `--no-project-settings`
  Read no project's settings file (.testuary/project.ini), here or in a directory above: for runs whose settings are all given in their own settings files. The working directory is then the project directory.

- `--home` `DIR`
  The user directory that holds models.ini and the global config.ini, in place of TESTUARY_HOME or the platform's user directory.

- `--model` `MODEL`
  Override [api] model: a model alias of models.ini, or a model name [api] provider serves.

- `--proxy-port` `PROXY_PORT`
  Use a model server on this port of this machine: sets [api] api_url to http://localhost:<port>/v1, which wins over the endpoint of the models.ini alias and of the provider section. --api-url wins over it.

- `--api-url` `API_URL`
  Override [api] api_url with a full base URL, for example http://localhost:8010/v1. It wins over --proxy-port and over the endpoint of the models.ini alias and of the provider section.

- `--prompt-profile` `PROMPT_PROFILE`
  A JSON file that changes the prompts sent to the model (examples: testuary init --examples). Read when the command starts.

- `--hints-file` `HINTS_FILE`
  Hints file to read from the working directory, in place of [main] hints_file. The hints guide says when a file is optional and what refuses a run.

- `--reuse-starter-suites`
  Run the starter phase again from the separate starter suites an earlier run saved: check and combine them again, and generate only a missing or empty suite again, which sends model requests (without saved suites, every one). --reuse starter instead keeps the combined result. Sets [starter] reuse_suites.

- `--phase-scores` `full` | `coverage` | `off`
  What tests.csv reports for the delivered suites of the starter and coverage phases, read from what those phases recorded: full reports their line and branch coverage and their mutation score over the full catalogue, coverage omits the mutation score, off omits these columns. Nothing is measured again for them. Overrides [finishing] phase_scores.

- `--project-name` `PROJECT_NAME`
  The project's name in results and records (${project}). Overrides [project] name. Empty uses the source directory's name.

- `--source-dir` `SOURCE_DIR`
  The source directory, whose files are tested. Without it, the project directory: the one whose .testuary directory is found from the current directory upward, else the current directory. A relative path is taken from the current directory. In container mode give the absolute path inside the container.

- `--python-executable` `PYTHON_EXECUTABLE`
  The project's Python interpreter ([project] python_executable). It installs nothing.

- `--java-home` `JAVA_HOME`
  The project's JDK home or java executable ([project] java_home).

- `--build-java-home` `BUILD_JAVA_HOME`
  The build tool's JDK home ([project] build_java_home).

- `--output-dir` `OUTPUT_DIR`
  The run directory: the results of each target and all logs go below it. A relative path is taken from the current directory. Overrides [project] output_dir, which may contain ${model} and ${project} placeholders.

- `--python-import-root` `PYTHON_IMPORT_ROOT`
  The directory from which the project's Python package is imported ([project] python_import_root). pytest still runs in the source directory, with the project's own configuration.

- `--languages` `LANGUAGES`
  Refuse the run if a target is in a language outside this list (installed: c, java, python). Each target's language comes from its file extension. Without this option every installed language is allowed.

- `--targets` `TARGETS`
  Run only these targets of the --source-files list, by position, one after another: one position (`7`), a range (`1-5`), or an open range (`5-` for the fifth to the end, `-5` for the first five). Default: the whole list. Outside container mode each target runs in its own copy of the source directory, so one target's edits cannot change the next one's measurements.

- `--source-files` `SOURCE_FILES`
  Comma list of targets to generate tests for, in the order they run. A relative path is taken from the source directory. Each target must exist and be in an installed language. Without --targets the run covers the whole list.

- `--concurrent-requests` `CONCURRENT_REQUESTS`
  How many model requests for the starter suites run at once (concurrent requests), in place of [starter] concurrent_requests and concurrent_requests_ollama. Fewer put less load on the model server and shorten each request, which helps models that write long answers.

- `--stop-after-phase` `{starter,coverage,mutation,finishing}`
  Stop each target after this phase has saved its result, to try a phase alone: starter, coverage, mutation, finishing. The default runs all four phases. Gives no final scores, and each target is reported as stopped.

- `--reuse` `REUSE`
  Use the saved results of these phases from the earlier run in this run directory instead of running them again: starter, coverage, mutation, finishing. The results guide says what each one reuses. Example: --reuse coverage.

- `--resume`
  Continue an interrupted run in the same run directory: each target and repetition continues from the first phase that did not finish. Finished phases (starter, coverage, mutation, finishing) are reused if their saved results and the target's source are unchanged. Combines only with --reuse starter.

- `--runs` `N`
  Run each target's four phases N times, as repetitions r0 to r(N-1), each from a fresh copy of the project and in its own directory. N must be 1 with --stop-after-phase. Default: `1`.

- `--run` `K`
  Run only repetition K (rK) of each target, the same one --runs makes when N is above K, for example to add a missing repetition.

## testuary check

Check the setup without contacting a model or installing anything (the quick check). --verify also starts the tools and tries the project in a copy.

- `--verify`
  Also start the tools a run needs once and try the project in a copy as a run does, to see that they work (the tool check). Python imports pytest, coverage and the mutation tool with the project's interpreter and collects the tests. Java runs java and javac, resolves the build, compiles and finds the tests, and starts the measurement tools a run needs. The first run tries the build of a project with its own build wrapper, which may download its build tool. The check runs no test, installs nothing and contacts no model. What it cannot try is reported as not tried.

- `--json`
  Print the report as JSON instead of text (API keys removed).

- `FILE`
  Targets to run, relative to the directory the command starts in. Without them, the project's [project] source_files. --source-files names them relative to the source directory instead.

- `--display` `auto` | `compact` | `detailed` | `plain` | `debug`
  How the run is shown: auto (compact on a terminal, plain otherwise), compact (one updating screen), plain (one line per change), detailed (also every step and the full setup report), debug (also every log line and internal event names). Default: `auto`.

- `--no-color`
  Disable terminal colors.

- `--config` `CONFIG`
  Read settings from this INI file in place of the config.ini in your user directory. Repeat it for several files: later files win. The configuration guide describes the layers that override it.

- `--no-project-settings`
  Read no project's settings file (.testuary/project.ini), here or in a directory above: for runs whose settings are all given in their own settings files. The working directory is then the project directory.

- `--home` `DIR`
  The user directory that holds models.ini and the global config.ini, in place of TESTUARY_HOME or the platform's user directory.

- `--model` `MODEL`
  Override [api] model: a model alias of models.ini, or a model name [api] provider serves.

- `--proxy-port` `PROXY_PORT`
  Use a model server on this port of this machine: sets [api] api_url to http://localhost:<port>/v1, which wins over the endpoint of the models.ini alias and of the provider section. --api-url wins over it.

- `--api-url` `API_URL`
  Override [api] api_url with a full base URL, for example http://localhost:8010/v1. It wins over --proxy-port and over the endpoint of the models.ini alias and of the provider section.

- `--prompt-profile` `PROMPT_PROFILE`
  A JSON file that changes the prompts sent to the model (examples: testuary init --examples). Read when the command starts.

- `--hints-file` `HINTS_FILE`
  Hints file to read from the working directory, in place of [main] hints_file. The hints guide says when a file is optional and what refuses a run.

- `--reuse-starter-suites`
  Run the starter phase again from the separate starter suites an earlier run saved: check and combine them again, and generate only a missing or empty suite again, which sends model requests (without saved suites, every one). --reuse starter instead keeps the combined result. Sets [starter] reuse_suites.

- `--phase-scores` `full` | `coverage` | `off`
  What tests.csv reports for the delivered suites of the starter and coverage phases, read from what those phases recorded: full reports their line and branch coverage and their mutation score over the full catalogue, coverage omits the mutation score, off omits these columns. Nothing is measured again for them. Overrides [finishing] phase_scores.

- `--project-name` `PROJECT_NAME`
  The project's name in results and records (${project}). Overrides [project] name. Empty uses the source directory's name.

- `--source-dir` `SOURCE_DIR`
  The source directory, whose files are tested. Without it, the project directory: the one whose .testuary directory is found from the current directory upward, else the current directory. A relative path is taken from the current directory. In container mode give the absolute path inside the container.

- `--python-executable` `PYTHON_EXECUTABLE`
  The project's Python interpreter ([project] python_executable). It installs nothing.

- `--java-home` `JAVA_HOME`
  The project's JDK home or java executable ([project] java_home).

- `--build-java-home` `BUILD_JAVA_HOME`
  The build tool's JDK home ([project] build_java_home).

- `--output-dir` `OUTPUT_DIR`
  The run directory: the results of each target and all logs go below it. A relative path is taken from the current directory. Overrides [project] output_dir, which may contain ${model} and ${project} placeholders.

- `--python-import-root` `PYTHON_IMPORT_ROOT`
  The directory from which the project's Python package is imported ([project] python_import_root). pytest still runs in the source directory, with the project's own configuration.

- `--languages` `LANGUAGES`
  Refuse the run if a target is in a language outside this list (installed: c, java, python). Each target's language comes from its file extension. Without this option every installed language is allowed.

- `--targets` `TARGETS`
  Run only these targets of the --source-files list, by position, one after another: one position (`7`), a range (`1-5`), or an open range (`5-` for the fifth to the end, `-5` for the first five). Default: the whole list. Outside container mode each target runs in its own copy of the source directory, so one target's edits cannot change the next one's measurements.

- `--source-files` `SOURCE_FILES`
  Comma list of targets to generate tests for, in the order they run. A relative path is taken from the source directory. Each target must exist and be in an installed language. Without --targets the run covers the whole list.

- `--concurrent-requests` `CONCURRENT_REQUESTS`
  How many model requests for the starter suites run at once (concurrent requests), in place of [starter] concurrent_requests and concurrent_requests_ollama. Fewer put less load on the model server and shorten each request, which helps models that write long answers.

- `--stop-after-phase` `{starter,coverage,mutation,finishing}`
  Stop each target after this phase has saved its result, to try a phase alone: starter, coverage, mutation, finishing. The default runs all four phases. Gives no final scores, and each target is reported as stopped.

- `--reuse` `REUSE`
  Use the saved results of these phases from the earlier run in this run directory instead of running them again: starter, coverage, mutation, finishing. The results guide says what each one reuses. Example: --reuse coverage.

- `--resume`
  Continue an interrupted run in the same run directory: each target and repetition continues from the first phase that did not finish. Finished phases (starter, coverage, mutation, finishing) are reused if their saved results and the target's source are unchanged. Combines only with --reuse starter.

- `--runs` `N`
  Run each target's four phases N times, as repetitions r0 to r(N-1), each from a fresh copy of the project and in its own directory. N must be 1 with --stop-after-phase. Default: `1`.

- `--run` `K`
  Run only repetition K (rK) of each target, the same one --runs makes when N is above K, for example to add a missing repetition.

## testuary init

Copy example files: models.ini to your user directory and hints.json to the project. Existing files are kept.

- `--examples`
  Copy the extension examples (a client factory, a language-specific prompt profile and per-language hints) into ./testuary-examples instead.

Exit status: 0 when the files are copied or exist already, 1 when a packaged example is missing or a copy fails, 2 when an option is refused.

## testuary provision

Install the runtime the settings of the project in this directory ask for (such as a Python version with its requirements, made with uv in Testuary's cache), so that a run need not do it. May download.

Exit status: 0 when the runtime the settings ask for is ready, or when they ask for none, 1 when the settings cannot be read or the runtime cannot be installed, 2 when an option is refused.

## testuary new-language

Write a new language as one file, LANGUAGE.py: its adapter with every fact and hook in the order of the reference page Hooks and facts, and the line that names it to LangAdapter. With --for, the hooks the consumer requires are marked TODO with its levels.

- `language`
  The language's name, such as lua.

- `--file-extension` `FILE_EXTENSION`
  Its source file extension, such as .lua (required unless --diff or --update-record).

- `--for` `ITEM`
  Write it for a consumer: ITEM names it by its module, or names its requirement profile, a .json file (repeatable).

- `--test-pattern` `PATTERN`
  The test file pattern, such as *_test.lua, with one * for the source file's stem. It gives the adapter's test_file_name (default test_*<file extension>).

- `--runtime` `PROGRAM`
  The program that runs the tests, such as lua. Without it, the adapter marks it TODO.

- `--grammar` `NAME`
  A tree-sitter grammar for the toolkit, such as tree-sitter-lua.

- `--runtime-name` `NAME`
  The runtime's name in version messages (default: the --runtime value).

- `--tested-runtime` `RANGE`
  The tested runtime versions, such as 5.3-5.4. Without it, versions stay undeclared.

- `--expected-runtime` `RANGE`
  Versions expected to work but not tested (needs --tested-runtime).

- `--output-dir` `DIR`
  The directory to write LANGUAGE.py in (default: the current directory). With --diff or --update-record, the shared package's directory.

- `--diff`
  Compare the consumer versions recorded in a shared package's pyproject.toml with what the installed consumers require now, and write nothing.

- `--update-record`
  Record the installed consumers' versions in a shared package's pyproject.toml.

## testuary setup

Ask for and write what a run needs: model access in models.ini and what runs in parallel on this machine in config.ini, both in your user directory, and a project's settings in .testuary/project.ini in its directory. Without a part, walk all three, skipping the parts already set.

- `part` `model` | `machine` | `project`
  Run only this part.

- `name`
  With the project part, the project's name in results and records (default: its directory's name).

testuary setup --demo \[--demo-language java\] \[--output-dir DIR\] plays the setup of the demonstration project with its answers typed visibly, then its run. --demo-speed, --demo-playback-seconds and --demo-step pace the playback as for testuary --demo.

### testuary setup --demo

Show the setup wizard on the demo's project, then play back its run.

- `--demo`
  Show the setup wizard's demonstration.

- `--demo-language` `python` | `java`
  Language of the setup's demonstration: python (the default) or java. Other demonstrations, such as toy, and --demo-data are not accepted here. Default: `python`.

- `--output-dir` `OUTPUT_DIR`
  The run directory: the results of each target and all logs go below it. A relative path is taken from the current directory. Overrides [project] output_dir, which may contain ${model} and ${project} placeholders.

- `--demo-speed` `SPEED`
  Pace of the demonstration's playback: auto (paced on an interactive terminal, instant when output is redirected, in tests or in continuous integration), instant, or a factor of the configured length's speed such as 2 (half as long) or 0.5 (twice as long). Pacing changes no value or record. Requires --demo. Default: `auto`.

- `--demo-playback-seconds` `SECONDS`
  Length of a paced playback in seconds, at --demo-speed 1, instead of TESTUARY_DEMO_PLAYBACK_SECONDS or [demo] playback_seconds. The demonstrations guide describes the length. Requires --demo.

- `--demo-step`
  Advance the demonstration's playback one step at a time when Enter is pressed, for screenshots. Requires --demo and an interactive terminal.

## testuary tutorial

Learn Testuary's extension points step by step: a model alias, a prompt profile and an extension's hints, each applied, shown and checked, without contacting a model.

- `--output-dir` `OUTPUT_DIR`
  A new or empty directory for the tutorial's files. Default: `testuary-tutorial`.

- `--auto`
  Play every step without questions, paced for watching.

- `--speed` `SPEED`
  With --auto: a factor of the pace, 1 by default (2 is twice as fast). Default: `1.0`.

Without --output-dir the files go to ./testuary-tutorial, which must be new or empty.
