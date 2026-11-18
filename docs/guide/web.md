# The web interface

The web interface is Testuary's local browser interface. It starts the
same Testuary commands you would type at a terminal and shows what they
report: the quick check and the tool check, a run's progress, its results
and the demonstration. Every value and text on its pages comes from
Testuary, so a page and the console never disagree. The interface is
optional: Testuary works the same without it.

## Install

The web interface comes with Testuary: installing Testuary installs the
`testuary-web` command. It needs nothing else, except git for adding a
project from its GitHub address. The pages are plain HTML, JavaScript and
CSS served from the package, and the server uses the Python standard
library and psutil.

## Start

Open a terminal in your project's directory, or in any directory you want to
work from, and start the interface. Testuary reads your models
(`models.ini`) and global settings (`config.ini`) from your Testuary user
directory, and each project's settings from `.testuary/project.ini` in the
project's directory:

```bash
testuary-web
```

It prints the directory, the installed Testuary and LangAdapter versions and
its address on 127.0.0.1, at the port of
[`[web] port`](../reference/settings.md#web), and opens that address in your
browser. Without an access key it adds that every program on this computer
that reaches the port can use it. Options:

| Option | Meaning |
| --- | --- |
| `--dir DIR` | The directory to work from, instead of the current directory. Its project, if it has one, is listed first. A directory that is no project lists the projects directly below it. Demonstrations go there |
| `--port PORT` | The port on 127.0.0.1, in place of [`[web] port`](../reference/settings.md#web). A port that is taken ends the start with a message that names what holds it: your own interface (then `--replace` stops that one first), an interface of another Testuary user directory or user, or another program (then choose another port) |
| `--access MODE` | Who may use the interface, in place of [`[web] access`](../reference/settings.md#web): `open`, `key` or `stable_key`, described below |
| `--no-browser` | Print the address without opening a browser |
| `--replace` | First stop your interface running on this port, as Ctrl+C stops it, then start |
| `--stop` | Stop your running interfaces, with `--port PORT` only that port's, as Ctrl+C stops them, and start none |
| `--version` | Print the version, which is Testuary's |

`testuary-web` exits with status 0 when the interface is stopped, or when
`--stop` ends every interface it finds, 1 when `--stop` finds one that does
not end, and 2 when the interface cannot start or an option is refused.

### Access

With `open` the interface needs no access key, and its address is
plain. Other web pages in your browser still cannot use it (see Safety),
but every program on this computer that reaches the port can: on a
machine you share with other users, they can start runs with your models
and your model key, and read your results. There, start it with an access
key:

- `--access key`: the printed address carries a key, new at each start,
  after `#token=`. Open the address as it is. The page keeps the key in
  the browser's local storage for the server's address, its port included,
  so new tabs, copied addresses and bookmarks of the same running
  interface work without it. After a new start, open its new address.
- `--access stable_key`: the same, with one key kept in your Testuary user
  directory (`web/access-key`), so the address stays the same from start to
  start. Use it only where nobody else can listen on the port while the
  interface is down: another user who does can take the key from a page
  opened at that address and use it at the next start. Deleting the file
  makes a new key.

`[web] port` and `[web] access` in your `config.ini` set the port and the
access for every start, and the Settings page shows them under Web
interface. As any setting, a project's `.testuary/project.ini` can set
them for starts in its directory, and `TESTUARY_WEB_PORT` and
`TESTUARY_WEB_ACCESS` for starts in that environment.

Ctrl+C in the terminal stops the interface. A job that is still running is
stopped first, the way Ctrl+C stops it at a terminal, and the interface
waits for its cleanup. A second Ctrl+C ends the job at once. One job runs at
a time in the working directory, also across several starts of the interface
there.

An interface running elsewhere, in another terminal or in the background,
stops the same way with `testuary-web --stop`, or with
`testuary-web --stop --port PORT` only the one on that port. Each start
records itself in your Testuary user directory (`web/running/<port>.json`)
with a stop secret, and only a request that carries the secret stops it.
Another program on the port and another user's interface are never
touched. An interface started with another `TESTUARY_HOME` has its record
in that directory: `--stop` and `--replace` find it with the same
`TESTUARY_HOME`. A record whose interface no longer runs is removed.
`--stop` and `--replace` wait until the interface has ended, after its
job's cleanup.

### Over SSH

The interface listens on 127.0.0.1 of the machine it runs on, so a browser
on another computer reaches it only through an SSH tunnel. Started in an
SSH login, `testuary-web` opens no browser on that machine. After the
address it prints the `ssh -L` line for your own computer, for example:

```text
ssh -L 8765:127.0.0.1:8765 you@203.0.113.7
```

Run that line in a terminal on your own computer and keep that login open,
then open the printed address in your own computer's browser. The line
gives the interface's port on both sides, adds `-p` when the SSH server
listens on a port other than 22, and names your login and the address
this SSH connection arrived at. Behind a router or on a cloud machine,
that address can be a private one your computer cannot reach: use the
name you log in with in its place.

The port must be the same on both sides of `ssh -L`, because the
interface answers only requests that name its own port. The port stays the
same from start to start, so one tunnel serves every start: log in with
the `ssh -L` line, run `testuary-web` in that login and open the address it
prints. If the port is taken on either computer, choose another one with
`--port` or `[web] port` and use it in all three places of the `ssh -L` line. A
machine others log in to is a shared one: start the interface there with
`--access key`.

## The pages

Every page fits a window as narrow as a phone's. In a narrow window, these
tables show each row as a block, each value under its column's name: a
check's items and limits, the Jobs list, a run's operations, the Results
page's targets and repetitions, how each phase ended and a phase end's
candidates. In a narrow window, a repetition's row of `tests.csv` shows
each name above its value. Another wide table or a file shown as text
scrolls in its own box. On a job's page, a text box follows new output
while you are at its end. Scrolled up, it keeps your place, and a part you
opened or closed stays so. The Guide link in every page's header opens
this guide for the installed version, in a tab of its own.

### Overview, Projects and Models

The overview shows your projects with their latest results, your models,
the recent jobs and the main actions: start a run, check a project, open
results. A model reads default model when your global settings choose it
(`[api] model` in your `config.ini`, which names none until you set it).
It reads tenacity's model, for example, when tenacity is the project of the
directory you started in and names it (`[api] model` in its
`.testuary/project.ini`, which the project setup writes). Each recent job
names its run directory, its model and its values, as the Jobs page does.
An open overview follows the jobs without a reload: a job shows there as it
starts and ends, and the projects' latest results are drawn again once a
job has ended. A project's latest results are the newest that Testuary
summarized in the project's `.testuary/results`: each target
with its fully measured repetitions and the mean of each value across them.

A project is a directory with its own `.testuary` directory, which holds its
settings file, `project.ini`, and its results.
Projects lists the project of the directory you started in (in a directory that
belongs to none, the projects directly below it) and your recent projects,
which the interface keeps in your Testuary user directory. Add a project by
choosing its directory and answering Testuary's project setup there, one
question at a time, as `testuary setup project` asks them. With a GitHub
address (`https://github.com/owner/project`), git clones the project into a
new directory, named after it, in the directory you choose, and the setup
follows there. It takes HTTPS addresses on github.com only and needs git.
Without git it says so. A directory of that name that exists already is refused.
git clones without the project's submodules, asks for no login and has 15
minutes. The setup offers what it finds. The languages come with the one with
the most of the project's files first. The interpreters are the project's
environments and the Pythons on your PATH, each a choice of its own with its
whole path, and Another path takes an interpreter of yours. The JDKs are the
ones it finds, the source files are checkboxes, and the models are yours, or a
new model added in place. Testuary checks each answer and shows a refused one's
reason beside it. Back returns to the question before. The last step lists the
settings to write by their names, with the file's text folded below, and Write
the settings saves them. An address in the setup's text, such as the settings
reference the file names, is a link. The list then shows the new project. Edit
settings sets its language, its source files, its model and its language's
tools, and links all of the project's settings on the Settings page. Forget
takes a recent project off the list and leaves its directory, settings and
results. The project of the directory you started in and the projects below it,
marked this directory's project and in this directory, stay listed.

Models lists your model aliases as Testuary's setup reads them, with where
each key comes from, and marks the default model and the project's model
as the overview does. A model's badge reads SET UP when the alias has its
key, or needs none at a local address, and NO KEY otherwise. Whether its
service answers, only a connection test tells: Test the connection reaches
the service as a run would, with the same address and key, and asks it for
the models it serves, which generates nothing and costs nothing. The
answer names the address, without credentials, and says whether the
service answered and lists the model. Add, edit and remove an alias there.
Add refuses an alias that exists, with the reason beside the field, and
Edit changes an existing model, opening with its own kind of service and
provider name. Removing a model keeps its provider's section in
`models.ini`, since other models of that provider use its address, and the
confirmation says so. A model's key line says whether a key is stored in
`models.ini` or comes from an environment variable, and never shows the
key. To replace a stored key, Edit the model and type a key in its Key
field. An empty Key field keeps the stored key.

When an installed extension names the model catalogue files its runs in a
project read, the Models for choice offers them for that project beside
Your model catalogue. Each model then shows the file it comes from, its
full path on hover, and a change goes to the last file, which holds key
variables and no key, and no address or key variable at all when it lies
in the project directory. Editing a model an earlier file defines saves a
copy of its section in the last file, where it takes the place of the
earlier one, and removing the copy brings the earlier model back. A new
address for a model of another OpenAI-compatible API also copies its
provider section, shown with its own Remove. A model only an
earlier file defines is removed by ending the import, where the extension
offers it. The page asks first and says that every other model is copied
into the last file, which the runs then read alone, and asks for a key
variable for each key an earlier file holds. It then checks that the runs
read the last file alone, and otherwise puts the file back and says what
holds now. A last file outside the project directory and the extension's
own directory is shown read-only with the reason.

### Settings

Every setting by group (Model, Coverage phase, Mutation phase and so on),
each with its name, its value, where the value comes from and Save, at any
window width: Testuary's default, your global settings, this project's
settings file, the environment or a run's options, told apart by color.
The groups a run usually changes come first and open: the model, the
project, the starter phase and both loops. Each shows its common settings
and every setting a file or the environment sets, and the rest of the
group is one fold away. The other groups follow under More settings. Find
a setting searches every setting by its name, label, purpose or group. A
setting the run does not use, such as another provider's service or
another mutation engine's options, is grayed with the reason. The page
opens on the project of the directory you started in, when there is one, else
on the global settings. Choose all projects to change the global settings,
or one project to change its own. An installed extension can name the
Testuary settings files its own runs in a project read, its own file last.
The project then has a further choice, Testuary settings for the
extension, which shows those settings with the file each value comes from
and changes the last file. Those runs read the files in place of your
global settings and the project's own settings, so a value either of them
sets does not apply to them unless the extension names their file.
Each change goes through Testuary's validated update, which refuses a
value a run would refuse and keeps the file's comments. Save and Reset
keep the page where it is and say what they did, and a refused value reads
"Not saved:" with the reason beside its setting. About this installation,
at the end, names the installed Testuary and LangAdapter versions.

### Run

Starts a run of a project in its directory, as `testuary` runs there. The
project's source files are a checklist: the targets its settings name are
ticked, or every file when they name none. Select all and Deselect all
tick or clear every one, beside how many are ticked. A project whose
settings name no language lists the files of each language it holds, each
language with its own Select all and Deselect all. A target the settings
name that the project's directory does not hold is named above the list, with
what to do. Choose the model, all repetitions (how many per target) or one
repetition only (such as one that did not finish), and the run directory.
Left empty, the model choice names the model the run asks: the project's
model, else the default model. A relative run directory starts at the
project directory, and an empty one takes
[`[project] output_dir`](../reference/settings.md#project). When the run
directory already holds an earlier run, the page offers to continue it
while targets remain unfinished, or to start a new run in a new run
directory. A run sends model requests, which can cost money with a paid
provider. The project's settings can hold small limits for a first run.

### Check

Its button, Run the check (sends nothing to a model), runs the quick
check, `testuary check`, in a project's directory for the targets ticked in its
checklist, with Select all and Deselect all as on the Run page. With "Tool
check: also start each tool once and try the project in a copy" ticked it
runs the tool check (`testuary check --verify`), which also starts each test,
coverage and mutation tool the settings select once and, in a copy of the
project, tries its build where a run builds it and finds its tests without
running one, unless the project has its own build wrapper, which may download
its build tool. When the project's run directory already holds a run, the check
judges the directory the next run from the Run page uses: it continues that
earlier run while targets remain unfinished, else it takes the free directory
beside it.
The page names that directory before the check starts. It shows Testuary's
summary with its next step, the report's facts, items and limits, and the
detailed listing. Where the console's next step names a command, the page
names its own controls: a tool check with its box ticked, the detailed
report below, or a run from the Run page. The check sends no model request
and never shows a key.

### Jobs

Lists the jobs this start of the interface has run, each with its run
directory, its model and each target's values as its end-of-run summary states
them. The list follows the jobs without a reload. A demonstration's model
is the recorded one, and its values are marked as a demonstration's. A
run's page shows its plan as Testuary made it: the four phases (the
starter phase, the coverage phase, the mutation phase and Finishing) with
their steps. The starter phase does Generate and Measure suites. Each loop
does Read tests and First measurement once, then Edit tests, Run tests and
Measure in a cycle. Every phase ends with consolidation, which
combines, reduces and compares the candidate suites, and a loop picks its
suite there. The starter phase picks its suite there too,
after measuring the best candidates under the name they would be
delivered with. Finishing has no steps: while it runs, its box says what
it does. Each step reads
current, finished or upcoming, beside each loop's goal and limits, and the
current step shows how long it has been under way. The starter suites'
first two steps count the suites generated and measured, each loop counts
its model requests against its limit, and a loop that has ended shows its
measured values. A loop that needed none of its steps says why. A loop that
ends at its request, token or time limit names that limit, in amber: its
settings set that end, so it is no failure. Below are the rows the
console's compact view draws, from Testuary's own display rows, their
times moving with the run's clock: the Overall row's phases done, such as
"1 of 4 phases done", the phases counted 1 to 4, each phase end,
and for a failed run the cause in one sentence above the details. When
the cause is a model service that gave no answer after its retries, a link
leads to the connection test on the Models page. What happened in order
follows, and the run's text output folded below it. Paths are shown short.
The full path is on hover, and Copy path copies it.

Stop works as Ctrl+C at a terminal: the work in flight cleans up first, and
work that finished is kept. If a stopped job is still cleaning up after the
time Testuary states for that, Force stop ends it with every process it
started. While a job stops, its header reads STOPPING, and a stopped job's
badge and label share one word, STOPPED or FORCE-STOPPED. The diagram marks
the work under way at the stop as stopped. A run that stops before writing
any results says that it wrote no results, offers no results link, and its
cause panel shows Testuary's own error text. When a run ends, its page
states its outcome in Testuary's words,
such as how many targets have a suite that meets the minimums, and links to
its results. A run that left targets without such a suite reads MINIMUMS NOT
MET, and when Testuary can finish them, Continue the run runs it again with
the same options in the same run directory, keeping the finished work. Other
jobs
say what their exit status means, in Testuary's words. A run whose results
record warnings, such as a time limit reached or a loop stopped at its limit
before its goal, lists them there in
Testuary's words, each with the setting it names and its value.

### Results

Choose a project: its runs are listed, the newest first, from its results
directory (`.testuary/results`) wherever the project lives. Another run
directory, such as a demonstration's, opens by its path.

- The project across its targets: for each value the mean, median, range and
  standard deviation over the targets, a mark per target on a 0 to 100 percent
  track, and beside them each target's starter suites before the loops: the
  median of its starter suites, and with several repetitions the mean of
  the repetitions' medians. A project of one target shows each value in that
  target, without statistics across targets.
- Values per target: each repetition's delivered suite, with several repetitions
  their mean, lowest and highest value, and the target's starter suites as
  above.
- Targets and repetitions: each repetition's final values, how each phase ended and its
  warnings, such as a time limit reached or a loop stopped at its limit
  before its goal. Select
  a target for its details: each phase's values, tests, model requests,
  tokens, tool calls and time, Finishing's values, the starter suites'
  statistics, where the time went, the delivered suite's files and every
  column of the repetition's row in `tests.csv`.
- How each phase end chose its suite, in a target's details: for the
  starter suites and each loop, what it chose and why. That is the best
  score in the order its record names, such as mutation score, then branch
  coverage, then line coverage, or equal scores with the reduced suite
  preferred, or the fastest of equal suites, or, when equal suites could
  not be timed, the input order. Each candidate is listed with its tests,
  coverage, mutation score and outcome: chosen, measured but not chosen,
  passes but not fully measured, does not pass, or not built, the last
  three with their reason. The reduced suite's mutation score, when derived
  from the merged suite's kill matrix rather than measured, reads
  "(derived, not measured)" beside it. A test's output folds behind its
  first line. A
  loop that ended with the suite it received says so, and the starter
  suites' end names the passing suites that left without being fully
  measured.
- Measurements a tool could not make, in a target's details when there were
  any: each measurement that the coverage tool or the mutation tool
  refused in a loop, under that loop's name, with the tests that stopped
  the tool, or a note saying it named no test. The mutation phase's
  measurement also runs the coverage tool, so a refusal by the coverage
  tool can stand under the mutation phase.
- Under a run's suite directory, how to use the delivered suite: copy the
  test file to where the project keeps its tests and run it with the
  project's own test command. Where its tests ran with a setup file beside
  them, the page names it and says that they can need it: copy it into the
  same directory too. For Python this is the `conftest.py` that makes the
  project importable for the tests. Where that directory already has a file
  of that name, the two have to be joined by hand. Each file that goes into
  the project has its copy button, such as Copy test_calc.py into the
  project and Copy conftest.py into the project. The test file's button
  offers the directory that holds most of the project's tests of that
  language, in the delivered suite's package, or else the project's
  directory, and a setup file's button offers its test file's directory. A
  missing directory is made, a directory in `.testuary` is refused, and
  nothing is written over a file. A demonstration's tests belong to its
  recorded project and are not offered.
- The end of the run, as the console printed it, from the run directory's
  `completion.json`, for every run.
- The run's log and every file of the directory, shown as text.

The values and their names are Testuary's, unchanged, from the summaries it
writes (`summary.json` per run, per target and for the project): an empty
value is unknown, and the interface computes no values of its own.

### Demonstration

Plays back one recorded run of this Testuary release, for Python or Java.
Under Speed, Normal speed plays it over the length that Length at normal
speed sets, a minimum, and Instant plays it without pauses. It sends no model
request, runs no test or mutation tool and needs no setup. Each playback
writes its record into a new directory in the interface's working directory,
which the page names, and the Results page opens it. Its stepwise playback, one
step per Enter press, needs a terminal: `testuary --demo --demo-step`.

## Safety

- The interface listens on 127.0.0.1 only and loads nothing from the
  internet: no fonts, scripts or services. During a run, only Testuary's
  own model requests leave the machine. Besides a run, only the model
  setup and a model's connection test reach a model service, and they ask
  it only for the models it serves. Adding a project from its GitHub
  address runs git, which reaches GitHub over HTTPS only.
- Requests must name this server in their Host header, so other web pages
  cannot reach it through another name for 127.0.0.1, and changes must
  come from its own page, so other web pages cannot make them. It sets no
  cookie. Under every access setting these checks hold.
- Without an access key, other programs and other users of the machine
  can use the interface through its port (see Access), so on a computer
  you share with other users, start it with `--access key`. With one, the key
  in the address's `#` part is never sent to a server by the browser. The
  page keeps it in the browser's local storage for this server's address
  and sends it as a header with every request.
- No answer of the server holds a stored model key, and no log, event
  file or address holds one. The key goes only to its model's service
  address, in a Test or a run. The lists come from Testuary's setup and
  key-free export, and the results views never show Testuary's
  configuration files. The progress events Testuary writes for a job have
  keys removed as the console removes them.
- The results views open only directories below the working directory or in a
  listed project, and follow no link or junction.
- The interface's own log in the terminal records actions and exit statuses
  only.

## Extensions

An extension named in `[main] extensions` can add tabs, result views,
settings groups and jobs to the interface and change its product name and
tabs, while Testuary's own pages, settings, access key and jobs stay as they
are. [Extend the web interface](../extending/web.md) lists what an extension
can change and shows how to write one.

## The interface and the console

Every action is a Testuary command, run in the project's directory, and a
job's page shows it, for example
`testuary package/module.py --display plain --no-color --model=ollama_qwen`.
A run writes the
same files as at the console, and results from either can be opened in the
other. The progress rows, the summaries and the exit sentences are
Testuary's own. Among what stays at the console are the setup's machine
part, editing the settings files by hand, the demonstration's stepwise
playback, the `toy` and tutorial demonstrations, `testuary setup --demo`,
`init`, `provision`, `new-language` and `tutorial`. The
pages ask the setup's project part, and the Models page its model part.

## Files the interface writes

The interface writes in these places:

- In the working directory, below `.testuary-web/`: `jobs/<job>/`, each job's
  terminal text (`stdout.txt`, `stderr.txt`) and, for a run or a
  demonstration, Testuary's progress events (`events.jsonl`), and
  `running.json` while a job runs, so that a second start of the interface
  starts no job in the same directory.
- In the working directory, each demonstration's run directory,
  `testuary-demo-<language>-<YYYYMMDD-HHMMSS>/`.
- In Testuary's user directory (`TESTUARY_HOME`, else the platform's user
  directory): `web/projects.json` with the recent projects' directories only,
  `web/access-key` with `--access stable_key`, `models.ini` from the Models
  page, through Testuary's setup, and
  `config.ini` from the Settings page with all projects, through Testuary's
  validated settings update.
- In Testuary's user directory, while an interface runs:
  `web/running/<port>.json` with its process, port, directory, start time and
  stop secret, which `--stop` and `--replace` read. The interface removes
  it when it ends.
- In a project's directory: `.testuary/project.ini`, from adding a project,
  Edit settings and the Settings page for one project, always through
  Testuary's setup or its validated update, and the
  `.testuary/results/.gitignore` that the setup's write creates. A
  delivered suite's test file copied from a run's results goes into the directory of
  the project you choose.
- In the directory you choose for a GitHub address: the project's new directory,
  which git clones.

A run started from a page writes its results where the same run at the
console writes them. That is Testuary's writing, not the interface's.

The results pages read each run's own run directory, its end-of-run
summary `results/completion.json` included, not the job history. The Jobs
page lists only the jobs of the interface's current start. When no job runs,
deleting `.testuary-web/` removes only the stored jobs' terminal text and
event files. A run that ended before writing `results/completion.json` takes
its end-of-run summary from its job's event file, and that summary is then
lost.
