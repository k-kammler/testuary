# Code organization

This page follows one `testuary generate` run through the package, from the
command line to Finishing, and names the module that owns each step. Paths
are relative to `src/testuary/`. A module's own docstring has the details.

## From the command line to a run

`testuary` is `__main__.main`. It hands `setup`, `tutorial` and
`new-language` to their own modules (`setup.py`, `tutorial/` and
LangAdapter's command line), and `--demo` to `demo/cli.py`. `init` and
`provision` run in `__main__.py` itself. The parser for `generate` and
`check` is in `__main__.py` as well. A `generate` call opens the terminal
display, resolves the settings and calls `orchestrator.run_pipeline`. A
`check` call goes to `orchestrator.check`, which reports readiness without
starting a run.

Two modules hold the settings:

- `settings.py` reads and validates them. It merges the packaged defaults,
  the settings files, the project's file, the environment and the command
  line, in that order, into one frozen `RunSettings`.
- `config.py` binds the resolved settings to the running process and gives
  each phase its own view of them, such as `get_agent_config` for a phase's
  section.

## The run and its files

`orchestrator.py` owns the run as a whole. Before any model request it checks
the model settings and the run directory, hands the run plan
(`presentation/events.py`) to the display, checks each selected language's
capabilities and tools, and opens the run's logs. Then it works through the
targets one at a time. For each target it:

1. plans from the target's checkpoints what a resumed run can skip
   (`phase_state.py`),
2. makes a prepared copy of the project to work in,
3. runs the four phases (`target_phases.run_target_phases`), with a guard that
   records a crash as the target's outcome instead of ending the run and
   writes the target's row with the phases finished before it,
4. writes the target's row of `results/tests.csv` (`results.py`), copies
   the delivered suite into the run directory and writes the target's repetition
   summaries.

After the last target it writes the project summary, reports the model use
and passes the end-of-run facts to the display.

## The phases

`target_phases.py` runs the four phases of one target in order. Each phase
starts from the suite the previous one handed on. At the phase end of the starter
phase and of each loop, the candidate suites are combined, measured and
reduced, and one suite is chosen as the phase's delivered suite.

| Phase | Code name | Module |
| --- | --- | --- |
| Starter phase | `starter` | `starters.py`, `starter_salvage.py`, `starter_selection.py` |
| Coverage phase | `coverage` | `agent/coverage_agent.py` |
| Mutation phase | `mutation` | `agent/mutation_agent.py` |
| Finishing | `finishing` | `measurement.py` |

The code names are the ones the records use, in `phase_state.json` and in
the archive directories.

- **Starter phase.** `starters.run_starter_phase` sends independent
  generation requests, without tools, and `starter_salvage.py` repairs each
  reply by deterministic means only. `starters.py` measures each starter
  suite's mutation results alone, and `starter_selection.py` stages one
  pooled run of all of them and derives the results from it.
- **Coverage phase** and **mutation phase.** The phase's agent edits the
  inherited suite in a loop of model requests and tool calls, as described
  below. The loop's first measurement shows the model what the suite does.
- **The phase end** (the consolidation step). `consolidation.py`
  combines the candidate suites into one file, measures it and keeps the
  tests that are needed. That suite reduction works on per-test data and is
  solved by `reduction.py`, exactly by default. Where the per-test data
  covers other lines or branches than the combined file's own run, that
  run's coverage stays the combined suite's, a reduced suite is measured
  again in its own run, and the record keeps both sides. The phase end then
  chooses the delivered suite among the reduced suite, the combined suite and
  the phase's inputs by mutation score, then branch coverage, then line
  coverage. Among suites that tie, it keeps the reduced suite when it is one
  of them, else it times the tied suites and keeps the fastest.
  `consolidation.persist_phase_end` writes the phase archive and its
  `consolidation_record.json`. When a loop ends with the suite it started
  from, the inherited measurements can be reused instead (see
  [Measurement and recovery](measurement-and-recovery.md)).
- **Finishing**, the fourth phase, makes no model requests.
  `measurement.measure_combined` reports the delivered suite of the last phase:
  the coverage measured at its phase end (measured here only when no
  phase measured it), complexity, test counts and, when the settings ask for
  them, lint and suite timing. `measurement.run_finishing` adds the
  mutation score, the one measured at its phase end when there is
  one.

`normal_run.py` holds the normal test run, the run that decides whether
tests pass. For a language whose adapter runs suites as delivered,
Python's, it is the run as delivered, with a fixed session seed, and
otherwise the language's test run. The loops, each phase end and the
starter cleanup ask it, and so does the mutation measurement's check that
one test file passes. The check of several files runs them in one session,
as the mutation tool does, since its time sets the per-mutant timeout
wherever Testuary derives it. For C, Testuary passes its rule instead, and
the mutation tool applies it to its own run of the tests, each test in a
process of its own as a mutant's tests run.
`loadable.py` holds the recoveries that make a suite load and the words
of the note on tests that leave shared state changed.

## The agents

`agent/_base.py` holds `CodingAgent`, the agents' base class: the prompts,
the logs, the model use, the file's state and the tools that edit and run
the suite. `CoverageAgent` in `agent/coverage_agent.py` and `MutationAgent`
in `agent/mutation_agent.py` add each loop's own tools and goal.

An agent's `run_phase` builds the loop as a LangGraph graph
(`agent/_langgraph.py`): the model chooses a tool, the tool runs and its result
goes back to the model, and the model's final answer is validated. The loop
goes on until the agent stops or a limit ends it.
Before each model turn, `agent/_context.py` builds the messages the model
gets: it rebuilds the conversation when the agent's set of available tools
changes, or on every
turn with `[main] context_rebuild = every_turn`, and appends the state marker.
`agent/_limits.py` applies the limits on what the model sees, which
config.ini's `[limits]` section sets.
`agent/_helpers.py` holds
the parts that do not depend on LangGraph, such as the loop breaker and the
choice of the best passing suite. The prompt files are in `prompts/`, one
per phase that asks the model.

## Model requests

`models.py` resolves which model a phase uses and builds its client.
`model_requests.py` decides what happens to each request: it classifies the
outcome, retries what can be retried, waits between attempts and records
every attempt. The agents send their requests from `agent/_langgraph.py`, the
starter phase from `starters.py`, and both go through
`model_requests.RequestAccount`.

## Languages

Testuary reaches a language only through a LangAdapter adapter.
`languages/__init__.py` finds the adapter for a file and builds the
context it runs in. `languages/generic.py` wraps the adapter's public
operations in the form the phases call. `languages/mutation.py` runs mutation
measurements through the adapter's mutation sessions, and
`languages/preparation.py` measures in prepared copies.

The adapter runs the tests and the coverage and mutation tools and edits
sources. Testuary decides what to run, what a result means for the method,
what to do when something fails and what the model is told.

## Records

| Record | Written by |
| --- | --- |
| Each target's `phase_state.json` | `phase_state.py` |
| `results/tests.csv` and the summaries | `results.py` |
| Phase archives and `consolidation_record.json` | `consolidation.py` |
| Generation records in the archive | `evidence.py` |
| `logs/run.log` | `_progress.py` |
| The agents' chat, trace and usage logs | `agent/_base.py` |
| `completion.json` | `presentation/terminal.py` |

`generation_evidence.py` writes and reads the records of how each suite was
generated: the model replies, each starter suite's attempts and cleaned
sources, and each delivered suite's ancestry. `exits.py` lists what each exit
status means. The
[result files reference](../reference/result-files.md) describes each file.

## The display

The run reports what it does as observations. `_progress.RunProgress` converts
each one into a `DisplayEvent` (`presentation/events.py`) and hands it to the
display. With an `--events` file, `__main__.py` writes each event there
first, in the order the display shows them. With `run_pipeline`'s `events`,
the run path writes the file itself: `orchestrator.py` opens it once the
run may start and hands it the setup report, the plan and the completion
before a caller's hook.
`presentation/view.py` decides what a display shows of them, and
`presentation/terminal.py` draws it with Rich. A failing display is switched off and the run goes on.

The web interface (`web/`) starts each run as a command of its own with an
`--events` file and reads that file into the same view.

## Other entry points

- `demo/` plays a recorded run through the same code. For one run it
  installs the scenario's model source in `models.py`, its exact suite
  reduction in `reduction.py` and its backend in `languages/`, then starts
  the command line (`demo/cli.py`).
- `setup.py` is the setup walk, `readiness.py` the readiness reports,
  `hints.py` the hints of the extensions the settings name and `project.py` the
  project's directories and executables.
