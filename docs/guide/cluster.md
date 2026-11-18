# Running on a cluster

Testuary runs as an ordinary batch job. On a shared cluster a few things need
attention that a workstation hides: how many CPUs the mutation workers get,
where the mutation engine keeps its working files, the limit on open files,
the route of the model's traffic, how many starter requests run at once, and
how to continue after a job is cancelled. Your site's documentation names its
scheduler options, storage
and network rules. The example uses Slurm with placeholders in angle
brackets.

Settings can go into an INI file passed with `--config`, or into environment
variables of the form `TESTUARY_<SECTION>_<KEY>`, for example
`TESTUARY_MUTATION_ENGINE_WORKERS`. The
[configuration guide](configuration.md) explains the layers. Mistakes are
refused before the run starts: an unknown section or key, in a file or as a
variable, and a `TESTUARY_` variable that names no known section, such as a
misspelled `TESTUARY_COVERAG_MAX_REQUESTS_PER_PHASE`. The variables Testuary
reads for their own purpose are not settings and are not refused:
`TESTUARY_<PROVIDER>_API_KEY`, a variable a model alias names with
`api_key_env`, `TESTUARY_HOME`, `TESTUARY_CONFIG_FILE`, `TESTUARY_EXECUTION_ID`,
`TESTUARY_CLOCK_DOMAIN_ID`, `TESTUARY_MAIN_WORK_DIR_NAMESPACE` and the
container inputs below. Run a small job first and check the log
lines named below before you start many.

## Installing

Create a virtual environment on storage every node can read, install Testuary
into it with `python -m pip install testuary`, which brings its pinned
dependencies, and activate it in the job. Check
that `python -c "import testuary; print(testuary.__file__)"`, run from outside
any source checkout, names the installed package. The activated environment
is Testuary's own. Name the project's interpreter with `python_executable` in
`[project]` or `--python-executable`. The
[settings reference](../reference/settings.md#project) says which interpreter
a run takes without one. Keep caches and temporary
directories of the tools out of your home directory if your site limits it.

## CPUs and mutation workers

Mutation testing dominates a run's CPU use. `engine_workers` in
`[mutation]` sets how many mutants run in parallel: Cosmic Ray's worker
processes for Python, PIT's threads for Java. For Python,
[`cosmic_ray_distributor`](../reference/settings.md#mutation) chooses how
Cosmic Ray spreads the mutants over those workers. Each Python worker holds
its own copy of the project. For Java the distributor does not apply.

Give the workers fewer CPUs than the job has, and leave one or two for
Testuary itself and the model client. A layout that has worked: 18 allocated
CPUs, Testuary and its children restricted to 16 of them, 15 workers. Restrict
the processes with `taskset -c` on the Testuary command, so every child
inherits the set. Read the allowed CPUs inside the job, with
`os.sched_getaffinity(0)`, instead of assuming that they are numbered from 0.

The worker count and the CPUs behind it are part of a run's execution
conditions. Per-mutant timeouts derive from the suite's measured runtime, so
fewer CPUs than workers can turn slow runs into timeouts and change a score.
Record the allocation with the results, and keep it the same across runs you
want to compare. Memory is limited for the whole job: the workers, their
project copies and any RAM-backed scratch share it.

## Node-local storage for the mutation engine

For Python, Cosmic Ray copies the project once per worker and records every
mutant's result in a session database. Both are read and written constantly
during a mutation measurement. On a shared network file system this is slow:
every import in every mutant run touches the file system's metadata, and the
database's locking waits on the network. A mutation measurement can then take
many times longer, run into per-mutant timeouts or its time limit
(`measurement_timeout_seconds`), and give a different score, which in turn
changes what the mutation phase does next.

Two settings in `[mutation]` decide where this work goes:

- `cosmic_ray_clone_dir`: where the workers' project copies go.
- `cosmic_ray_local_session_dir`: where the session database is staged while
  the mutation measurement runs.

Left at their defaults, the choice depends on how the project runs. When
Testuary runs the project directly, `auto` puts the copies in `/dev/shm`
(on Linux and other POSIX systems), else in the system's temporary directory,
each only when it is writable with at least 1 GiB free, else in a
`cr_worker_clones` directory in the directory the run started from. An empty
session setting then means the system's temporary directory. When the project
runs in a container (`project_exec = container` in `[main]`), `auto` puts the
copies beside the session in the run's work directory, which can be on shared
storage, and an empty session setting picks `/dev/shm` when it is writable, else
the system's temporary directory.

On many clusters the system's temporary directory (`TMPDIR`) and the work
directory are on shared storage, so set both settings explicitly to a directory
on the node, unique to the job, for example below `/dev/shm`. Each session then
takes its own subdirectory there. A configured directory that cannot be created
or written stops the run, with no fallback to another place. For a Python
project run directly, a session directory that cannot be used stops the run when
it starts, before any model request, and `testuary check --verify` tests it
beforehand. Otherwise the directory is checked before the mutation measurement.
A RAM-backed directory counts against the job's memory. The engine removes a
session's copies when the session ends, and `cosmic_ray_clone_sweep_hours`
reclaims copies that a killed job left behind (`0` switches that off). When the
project runs in a container, the directory must be visible inside the container
too.

The run's log names the choice for each mutation run, on a line from
`[CosmicRayEngine]`. Look for `session SQLite at <directory>` with the reason
for the place, and `15/15 workers ready` (with your worker count), at every
phase that measures mutation, not only the first.

## Running the project's tests in a container

Where a site runs software in containers, the project's tests can run in a
running container while Testuary runs outside it. Six variables and the
source directory in the container select this:

- `TESTUARY_MAIN_PROJECT_EXEC`: `local` or `container`. It sets
  `[main] project_exec`.
- `TESTUARY_PROJECT_CONTAINER`: the name of the running container the tests
  run in. Required with `container`.
- `TESTUARY_PROJECT_CONTAINER_RUNNER`: the container command, `docker` by
  default.
- `TESTUARY_PROJECT_PYTHON_EXECUTABLE`: the project's interpreter, a path inside the
  container with `container`. It sets `[project] python_executable`.
- `LANGADAPTER_HELPER_PYTHON`: the interpreter in the container that
  supervises the project's commands, the project's interpreter by default.
- `TESTUARY_PROJECT_WORK_DIR`: the directory on the host for the run's work
  copies, which the container must also mount. Required with `container`.
- `--source-dir`, or `TESTUARY_PROJECT_SOURCE_DIR`: the source directory
  inside the container, an absolute POSIX path. Required with `container`.

A container run continues with `--resume` when the container command
(`TESTUARY_PROJECT_CONTAINER_RUNNER`), the project's interpreter in the
container (`TESTUARY_PROJECT_PYTHON_EXECUTABLE`) and, for a language such as
Java, its runtime there are the same as when the work was done.
The container's name may differ, so a new container of the same setup in a
later job continues the run. The target and every saved file are still
checked by their hashes. When the command, the interpreter or whether the tests
run in a container changed, the resume does that work again, and the log says
why, for example "the container command or the project's runtime changed".

## Open files

A mutant run starts pytest, which imports the project's libraries and opens
many files for a moment. With many workers the default limit on open files is
reached, test collection fails with "Too many open files", and the failure
reads as a killed mutant. Each mutant's test run raises its soft limit to
65,536, or to the hard limit when that is lower, and never lowers a higher
limit. The log line that names where the mutation session ran ends with
`open-file limit: workers <soft>, tests <limit> (hard <hard>)`: the limit the
workers have, the limit of each test run and the hard limit. Check
`ulimit -Hn` in a job: a low hard limit is a matter for your site. Do not set
an unlimited soft limit yourself. Some programs then close every possible file
descriptor when they start a process, which makes each start slow.

## Java projects

For Java, PIT runs `engine_workers` as its number of threads. Each
Java process takes a quarter of the job's memory limit as its largest heap
unless told otherwise, and Testuary has no setting for the heap of a Java
project's processes. Set one for every Java process of the job, for example
`export JDK_JAVA_OPTIONS=-Xmx1g`. Java then notes "Picked up
JDK_JAVA_OPTIONS" on its error output, which is harmless.

A Java run that Testuary runs directly keeps PIT's files and reports, the
compiled suites and its own copies of the project in the system's temporary
directory. A node-local `TMPDIR` keeps them off shared storage. That is a matter
of speed, not of the results.

On nodes without internet access, Maven needs a local repository that already
holds the project's dependencies, and its offline switch, for example
`export MAVEN_ARGS=-o`. LangAdapter downloads the JUnit launcher,
JaCoCo and PIT into its cache on first use, `~/.cache/langadapter/java`
unless `LANGADAPTER_JAVA_TOOLS` names another directory. Fill that cache
beforehand and set `LANGADAPTER_JAVA_TOOLS_OFFLINE=1` in the job.

With more than one worker, LangAdapter splits the mutants of the class under test into
groups of at most 10 for PIT, so the threads share the work. Each group starts
a Java process with its own heap.

## Model traffic through a proxy

Compute nodes often reach the internet only through a proxy. The model client
follows the standard proxy variables. Set all spellings, because a site's
lowercase variables can take precedence over uppercase ones:

```bash
export ALL_PROXY=<proxy URL> all_proxy=<proxy URL>
export HTTPS_PROXY=<proxy URL> https_proxy=<proxy URL>
export HTTP_PROXY=<proxy URL> http_proxy=<proxy URL>
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
```

A SOCKS proxy works with a `socks5h://host:port` address. The `h` resolves
host names at the proxy, which matters when the node has no name service for
outside addresses. `NO_PROXY` must cover the loopback addresses, because the
Python mutation workers talk to each other over loopback HTTP. Keep the model
key out of job scripts: Testuary reads it from `TESTUARY_<PROVIDER>_API_KEY`
or from a private `models.ini` in the user directory (see the
[configuration guide](configuration.md#where-testuary-reads-its-files)).

## Starter request concurrency

The starter phase asks the model for several independent suites. These
settings in `[starter]` decide how many requests run at once:

- `concurrent_requests`: how many requests the starter phase sends at the
  same time. `concurrent_requests_ollama` takes its place whenever the
  provider is `ollama`, local or hosted. `--concurrent-requests` on the
  command line sets both.
- `cache_warmup`: send the first starter request alone,
  then the rest at once, up to that limit, so a provider can cache the
  shared beginning of the prompt. Whether it does depends on the provider,
  and savings are not guaranteed. When the first suite's validity check
  cannot be resolved, the rest is not requested.

One request followed by up to 24 at once is `cache_warmup = true` with
`concurrent_requests = 24`. Requests one at a time are
`concurrent_requests = 1`. A small fixed number such as 4 is
`concurrent_requests = 4`. Several jobs running at the
same time multiply the load on the provider: count all of them against its
rate limits.

## Time limits

The scheduler's time limit for a job is separate from Testuary's own limits,
such as `suite_run_timeout_seconds` or `measurement_timeout_seconds`. Choose the job's
limit from the elapsed times of comparable runs with headroom. A limit that is
too short wastes the work done so far until you resume, and one that is very
long can delay the job's start.

## Continuing after a cancelled job

Run the same command again with `--resume`, the same `--output-dir` and the
same configuration files. For each repetition of each target, Testuary skips
the phases whose records show them finished, after checking them, and starts
again at the first phase that is not. [Results and logs](output.md#continuing-a-run)
describes the rules. The node-local directories of the cancelled job are gone
and are created anew. Never let two jobs use the same run directory at once.

## Collecting results

Write the run directory to shared storage that outlives the job, and copy
it after the job has ended, not while it still writes. The directory holds the
delivered suites, the phase records and the logs.
[Results and logs](output.md) describes its layout and the exit status. A job
that the scheduler reports as completed can still contain a target without a
suite that meets the minimums: read the summary at the end of the log or the results table
for each target's outcome. Leave out the virtual environment and any scratch
directories when you archive a run.

## Example job script

The script assumes one target of one project per job. It runs in the directory
that holds the settings files, and `--home` names the directory with
`models.ini`. Name the source directory and the run directory by paths
without links: a path that passes through a link, at any level, is
refused. Replace every value in angle brackets.

```bash
#!/bin/bash
#SBATCH --job-name=testuary
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=2G
#SBATCH --time=<hh:mm:ss>
#SBATCH --output=<log directory>/testuary-%j.out
set -euo pipefail

source <environment directory>/bin/activate
cd <configuration directory>

# Node-local scratch for the mutation engine, unique to this job.
SCRATCH="/dev/shm/testuary_$(id -u)/${SLURM_JOB_ID}"
export TESTUARY_MUTATION_COSMIC_RAY_CLONE_DIR="$SCRATCH/clones"
export TESTUARY_MUTATION_COSMIC_RAY_LOCAL_SESSION_DIR="$SCRATCH/sessions"
mkdir -p "$TESTUARY_MUTATION_COSMIC_RAY_CLONE_DIR" \
         "$TESTUARY_MUTATION_COSMIC_RAY_LOCAL_SESSION_DIR"
trap 'rm -rf "$SCRATCH"' EXIT

# Model traffic through the site's proxy, loopback excluded.
export ALL_PROXY=<proxy URL> all_proxy=<proxy URL>
export HTTPS_PROXY="$ALL_PROXY" https_proxy="$ALL_PROXY"
export HTTP_PROXY="$ALL_PROXY" http_proxy="$ALL_PROXY"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1

# 16 of the allowed CPUs for Testuary and its workers.
CPUS=$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:16])))')

taskset -c "$CPUS" testuary generate \
    --home <user directory> --config config.ini --config cluster.ini \
    --source-dir <source directory> --source-files <target> \
    --output-dir <run directory> ${RESUME:+--resume}
```

`config.ini` holds your settings for the run, among them `model` in
`[api]`, an alias of `models.ini`, which has no default. The job starts in
the configuration directory, so the project's own settings file,
`.testuary/project.ini`, is read only when that directory lies inside the
project. Files named with `--config`
replace the global `config.ini` of the user directory, so the script names the
run's own `config.ini` first. `cluster.ini`
comes second, so its machine settings win, with a `cluster.ini` such as:

```ini
[starter]
concurrent_requests = 4

[mutation]
cosmic_ray_distributor = http
engine_workers = 15
```

Submit with `sbatch <script>`. To continue a cancelled job, submit it again
with the same run directory and `RESUME=1` in its environment, for example
`sbatch --export=ALL,RESUME=1 <script>`.
