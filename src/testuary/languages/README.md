# Language adapters

The Python and Java implementations, with the Cosmic Ray and PIT mutation
engines, are in LangAdapter (`langadapter`). Testuary
reaches them through that package's public operations. The mutation runner
and the prepared copies also use its Python helpers in `langadapter.python`
and its `recovery_owners`, which names what a failed copy, command or
cleanup leaves on its exception:

- `LanguageBackend` ([generic.py](generic.py)) is the type of Testuary's
  interface to a language, beside the results its phases read. The
  source-analysis records come from the backend's public
  `langadapter.sources`.
- `GenericBackend` ([generic.py](generic.py)) implements it once, for every
  registered language, Python and Java included: the operations described
  below, runs, coverage, per-test evidence, timing and mutation sessions
  through the backend's public operations,
  and each language's own rules (file names, test ids, request options,
  setup files, run counts, failure listing, merging, the import line)
  through its adapter's hooks. Testuary keeps its phase timers, prompt
  text, dependency-report formatting and the naming of each target's
  test file.
- The package's [__init__.py](__init__.py) translates Testuary's configuration
  into the backend's public `ProjectContext`. For every language, Python and
  Java included, the runtime comes from its adapter's `discover_runtime`
  (`describe_runtime` for the setup) and the tool variables from its
  `context_environment`, over the project settings the adapter declares.
  Python's and Java's settings, such as `python_executable`, `java_home`,
  `build_java_home`, `maven` and `gradle` in `[project]`, are part of
  Testuary's own configuration. Setup files and child processes use the
  backend's `LANGADAPTER_` names.

- [mutation.py](mutation.py) is Testuary's side of the backend's public
  mutation sessions: the session settings from its configuration, the
  distributor and worker rule, the phase timer, and its readings of a run
  (the counts, the in-loop ratio, the survivors' text for the model and the
  kill matrix). It also holds Testuary's kill-matrix reader.

Java container contexts keep the selected module root and guest JDK/build-tool
paths as guest paths. The control Python is separate from the target JDK. Local
Java tool discovery uses the resolved run's PATH. The host PATH is not forwarded
to a guest. An absent captured selection cannot be filled by later environment
changes.

## Responsibilities

The backend owns source validation, execution, coverage, mutation, timing and
source analysis. Testuary's adapter translates its settings and native
observations, while retaining its prompt slots and artifact-role conventions.
The agents and the phase end use that interface without importing the
Python backend directly.

`GenericBackend` shares grouped source-read translation. It is not another
abstract language implementation contract. Native consumer adapters supply
execution and prompt policies. Missing required policies fail explicitly rather
than yielding empty successful results. Important optional policies include:

| Capability | Default |
| --- | --- |
| Shared test setup files | No additional setup files |
| Unsound-test warning detection | No warnings |
| Filename and class-name checks | No additional restriction |
| Module-root resolution | The source directory |
| Per-test reduction data | Unavailable, so reduction is skipped |
| Collision resolution | Each suite stays as it is, and the merge keeps the better single candidate |
| Coverage-matrix export | No sidecar |

The backend supplies structural checks, skeletons, fingerprints and parser
identity. Validation of complete generated tests also uses the target Python
parser, not just structural parsing.

`normalize_shadowed_definitions` reads the backend's public
`sources.normalize_bindings` full static report. Removed bindings, spans and the
language's rule version reach the same phase gates for built-in and registered
languages. A completed parse failure keeps the source as it was and records
`parse_ok=False`. Missing setup or failed analysis has no report and stops the
caller. Static normalization does not certify target-runtime acceptance.

## Languages from other packages

A language LangAdapter registers from another package runs through the
generic backend ([generic.py](generic.py)): its runs, coverage, per-test
evidence, timing and mutation sessions through LangAdapter's public
operations, its file names, test ids and prompt words from its facts
([prompts/__init__.py](../prompts/__init__.py)). Testuary needs no code for it. The
generation condition records the language package's distribution and
version, and a resume on another one is refused.

## Capabilities

Before any phase, a run reads `GenericBackend.capabilities`, the language
adapter's capability report for the project's environment, against Testuary's
requirement list ([__init__.py](__init__.py), `testuary/langadapter_profile.json`).
The list names LangAdapter's capability families in three levels: running
the tests, coverage and mutation are required, and a run stops before its
first phase, with the reason, when one is missing. Every other family is
optional, with the fallback the list states, what a run does without it:
without per-test evidence
the reductions are skipped and each starter suite is scored on its own
measurements, without merging the better single candidate is
kept, and so on. The run announces the fallbacks, records them per target
(`fallbacks` in resolved_settings.json, only when one applies) and shows the
ones that change Testuary's method in its summary. `testuary check` reports
the same.
Test pruning belongs to the required family `run tests`: a language without
it stops before the first phase. `check_dependencies` adds the exact
reduction solver's tools when a phase uses that solver, the same for every
language.

## Mutation

`GenericBackend.mutation_runner` returns Testuary's `MutationRunner` for a
project ([mutation.py](mutation.py)). It opens the language's first native
engine through the backend's public sessions
(`backend.mutations.open_session`): Cosmic Ray for Python, PIT for Java. A
configured `engine` the target's language does not offer is refused. The
shipped default, `cosmic_ray`, stands for each language's own first
engine. A session's run returns a
`MutationMeasurement`: the counts the agent's gates and summary lines use,
the in-loop ratio, the surviving mutants' text and the kill matrix. A final
score comes from the kill matrix, which withholds it for an incomplete run.
A language without a mutation runner cannot run the mutation phase.
An unsuccessful session is closed before a configured sequential fallback is
opened. A cleanup failure retains its backend recovery handle and stops that
fallback. It is not treated as a model or transport failure.

For a pooled session, `mutation_cleanup_pending(error)` distinguishes an
ordinary setup failure from an error carrying recorded cleanup ownership.
Only the ordinary, fully cleaned setup failure may fall back to per-call sessions.
`close_mutation_session(session, primary=error)` preserves an existing body
exception. Without `primary`, a close failure propagates. The session's ordinary
no-argument `close()` closes it too, for replay and registered consumers.
On failure, `mutation_cleanup_handle` retains the same owner and
`mutation_cleanup_error` the close error. These fields help a recovery: they
do not prove that processes stopped. A successful adapter close is idempotent, and a
failed close is retryable and refuses further measurements. Callers must still
propagate the original body exception after closing with `primary`.

A project-backed mutation session prepares one owned copy on open, keeps it and
the same native pool for repeated measurements, and closes both on the opening
thread. Only typed project targets and suite inputs map to the copy. Native
requests, verdicts and identity remain unchanged. Persisted kill matrices retain
the caller's logical target path, while the raw native result retains its actual
measurement path.

Use `with session.worker():` around a whole dispatched tool call, including
its source reads and green/coverage checks, not just the mutation call. It yields
no replacement session. Same-thread nested calls reuse that explicit borrow, and
competing calls refuse rather than queue. Independent checks prepare separate
child copies without restarting the idle mutation pool. `session.run()` also
borrows for its own call. Replay sessions need not implement this optional
worker protocol. Their consumer can use a no-op context instead. A failed close
retains the native session and prepared owner for same-owner recovery, and later
successful close is harmless to subsequent project or session state.

## Per-test data and reduction

`collect_eviction_data` returns test identifiers, coverage and kill tokens,
per-test durations and a suite duration. The default warmup precedes one
instrumented suite run. Durations are measured in the suite context, not by
timing each test separately. Python uses coverage contexts and pytest JUnit
records, together with recorded mutation kill attribution.

An unavailable, failing or timed-out measurement returns no usable reduction
data. A coverage-only matrix has `includes_kills=False`. It cannot establish
mutation-preserving reduction. `prune_tests` removes selected test definitions
from source using the backend's test identifiers. The caller checks the
result before adopting it.

`benchmark_tests` provides suite timing for every language. With `[timing]
serialize` on, the default, Testuary's benchmark lock times one suite at a
time within one process only. Separate processes still require external
coordination to avoid competing timing workloads.

## Combining independently generated suites

The starter phase can select tests from several generated suites.
`resolve_test_collisions` prepares their names before `merge_test_files`
combines them. The Python implementation uses the shared grammar-based
engine through `langadapter.python.source_analysis`:

- Keep one copy of byte-identical colliding definitions.
- Rename other collisions consistently across identifier occurrences,
  including fixture parameters, without rewriting unrelated string literals,
  attribute names or call keywords.
- Drop an unrenamable definition and its dependent statements, recording
  the reason. Reflective references to the current module's own bindings
  require this treatment when identifier rewriting cannot preserve them.
- Reject a whole source if it cannot be parsed or its rewrite fails validation.

The return fields are those of the backend's `CollisionResolution`
(`langadapter.sources`). A backend without that capability
cannot combine the starter suites merely by implementing pairwise
merging.

## Model-visible text

`prompt_fragments()` supplies `$lang_*` values used by the shared phase
prompts. These include the language name, code-fence tag, test framework
and single-file rule. Available dependencies and the execution runtime are
resolved for the target. Every placeholder referenced by a prompt must have
a value.

Phase-specific hints use the shape in
[hints.json](../_templates/hints.json): `general`, `per_profile`,
`per_language` and `per_profile_per_language`, the profile cells keyed by
the project's `hint_profile`. Hints for one phase do not
automatically apply to another. The loader reads the configured file from
the project directory, and a `--hints-file` path from the working directory. An
extension named in `[main] extensions` can hold hints in the same format as
its top-level `testuary_hints`.
They come before the user's file in each prompt and are recorded in the
generation origin. A selected hints file that is missing or invalid
refuses the run. The
bundled template is copied by `testuary init`, not loaded automatically.
Changing prompts, tool
descriptions or hints changes what the model receives.

## Extension interface

A language's execution, analysis and mutation belong in its LangAdapter
adapter, behind its public operations. An extension named in
`[main] extensions` names the adapter with its top-level
`langadapter_language`, and Python code registers it through
`langadapter.register`. Testuary's detection and language-name discovery
follow that public registry, including subsequent replacements or removal.
Editing Testuary's registry is not a registration step. Grouped source
reads need only a backend's public source operations.

A registered language runs at the level its capabilities reach. The
scaffold (`testuary new-language`, LangAdapter's `new-language --for
testuary`) marks what Testuary's profile requires.

### Example: a new language from outside both packages

The worked example, a language registered from outside both packages,
is in the user documentation:
`docs/extending/languages.md`. Testuary's tests run it as written.
