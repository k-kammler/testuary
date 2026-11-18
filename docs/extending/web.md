# Extend the web interface

A Python module named in Testuary's settings can change its web interface
(`testuary-web`). Another product built on Testuary uses it to show its own
pages beside Testuary's, in one interface with one access key and one job
list. Every value a page shows comes from files the extension writes, as
Testuary's pages read Testuary's files.

An extension can change these parts of the interface:

- [the product name](#product-name) the header and the window title show
- [the Guide link](#guide-link) of the header
- [tabs of its own](#added-tabs) at positions it names
- [the titles of Testuary's tabs](#renamed-tabs)
- [Testuary's tabs](#replaced-tabs), each replaced by a tab of its own,
  the Overview excepted
- [the places of Testuary's tabs](#moved-tabs), the Overview excepted
- [which of Testuary's tabs show](#hidden-tabs), the Overview excepted
- [result views](#result-views) for the directories it writes
- [settings groups](#settings-groups) on the Settings page
- [jobs](#job-kinds) it starts, with Testuary's job controls
- [read-only routes](#read-only-routes) its pages read
- [Testuary runs](#testuary-runs-in-a-page) folded into its pages

[What stays fixed](#what-stays-fixed) lists what no extension changes.

## Naming the extension

An extension is a Python module whose top-level name `testuary_web` holds a
`testuary.web.WebExtension`. Name its file, or an installed module, in
`[main] extensions`, for example in the `config.ini` of your user
directory:

```ini
[main]
extensions = notes/notes.py
```

Nothing is installed. When `testuary-web` starts, it loads once the items
that the settings of its working directory name. The Overview lists an item it
cannot load, or whose `testuary_web` is not a `WebExtension`, with the
reason, and the other extensions keep working.
[Configuration](../guide/configuration.md) says how the settings' layers
add their items and what a file item is relative to. A `WebExtension` has
these fields, all optional except the
first two:

- `api`: the extension API version the extension was written for,
  `testuary.web.WEB_EXTENSION_API` (1).
- `id`: the extension's id, lowercase letters, digits and dashes, at most
  40, starting with a letter. It appears in its scripts' addresses and its error texts.
- `product`: the product name the header and the window title show.
- `tabs`: the tabs it adds (`WebTab`).
- `tab_titles`: Testuary's tabs by id with the titles they show instead,
  such as `{"results": "Test results"}`. The ids are `""` (Overview),
  `projects`, `models`, `settings`, `run`, `check`, `jobs` and `results`.
- `tab_positions`: Testuary's tabs by id with the positions they take
  instead, such as `{"results": "after:notes"}`.
- `hidden_tabs`: the ids of Testuary's tabs it hides, such as `("models",)`.
- `guide`: the address the header's Guide link opens, the name of an HTML
  file next to its module, or None to hide the link.
- `dir_types`: the directories its result views open (`DirType`).
- `settings_groups`: its settings on the Settings page (`SettingsGroup`).
- `job_kinds`: the jobs it starts (`JobKind`).
- `routes`: its read-only routes (`WebRoute`).

The [Python API reference](../reference/api.md#web-interface) describes
each type.

## A first example

The smallest extension adds one tab. Its two files stand in a directory
`notes` of your user directory, which the `config.ini` above names.

`notes/notes.py`:

```python
from testuary.web import WebExtension, WebTab

testuary_web = WebExtension(api=1, id="notes", tabs=(WebTab("notes", "Notes", "notes.js"),))
```

`notes/notes.js`:

```javascript
const web = globalThis.testuaryWeb;

web.registerView("notes", () => web.panel(
  web.el("h1", { text: "Notes" }),
  web.el("p", { text: "The notes extension adds this tab." })));
```

To check it, start `testuary-web` and open the Overview: an extension that
failed to load is listed there with the reason. The tab Notes stands last
among the header's tabs. After a change to the Python module, restart `testuary-web`,
since it loads its extensions once when it starts. It reads a view script
from its file each time a page loads it, so a changed script shows after
the page reloads. [The web interface](../guide/web.md) guide says how to
start the interface, choose its port and set who may use it.
[A worked example](#a-worked-example) builds on this one with every part an
extension can add.

## Tabs and names

### Product name

A product name replaces Testuary's name wherever a page names the whole
product: the header, the window title, the Overview's heading, the heading
of a page that failed, a job's progress panel and the footer. Testuary's
own things, such as its projects, runs and settings, keep Testuary's name.
Without a product name, the footer says the values come from Testuary and
its extensions while an extension adds a tab, a directory type or a job
kind or renames a tab, else from Testuary.

### Guide link

The header's Guide link opens Testuary's guide to the web interface. An
extension's `guide` sets what it opens instead: an `http` or `https`
address, such as the product's documentation of its installed version, or
the name of an HTML file next to the extension's Python module, at most 64
letters, digits, underscores and dashes before `.html`, such as
`guide.html`. The interface serves that file at
`/ext/<extension id>/<file name>` with the interface's security rules, which
block inline styles and scripts in the file. A `guide` of None hides the link.

### Added tabs

A `WebTab` has an `id`, the address `#/<id>`, a `title`, its view `script`,
a `position`: `first`, `last` (the default), `before:<tab id>` or
`after:<tab id>`, and optionally the Testuary tab it `replaces`. A position
may name a tab another extension adds or moves, whatever order the
extensions declare them in. A position that names a tab no one has places
the tab last. Two tabs with the same position `first` or `after:<tab id>`
both stand directly there, the one placed later ahead of the other, so a
position that names the other tab, such as `after:notes`, sets their
order. A tab cannot take the address of one of Testuary's views.

### Renamed tabs

`tab_titles` gives Testuary's tabs new titles. A renamed tab's page heading
follows its new title, and Testuary's texts and links name a tab by the
title the header shows, such as Open test results for Results renamed Test
results.

### Replaced tabs

A tab that `replaces` one of Testuary's tabs, any but the Overview, takes
its place. The replaced tab leaves the header, and its address opens the
replacing tab with the address's query. So Testuary's links to it, such as
the Overview's Start a run for `run`, open the replacing tab, and Testuary's
texts name the replacing tab's title. The replacing tab offers what those
links and texts promise: a run page in the place of Run, a check of the
project's setup in the place of Check, settings in the place of Settings.
A position that names the replaced tab places the tab
where it stood, such as `before:run`. While Run is replaced, the Check
page's note on a project's earlier run names the directory the check judges
and leaves out what Testuary's Run page does with it.

Testuary's links pass this query to its tabs' addresses. The parameters
keep their names and meanings within an API version:

- `run`: `project_dir`, a listed project's directory. The page of a run refused
  because its run directory holds an earlier run also passes `model`, the
  run's model alias, and `output_dir`, the run's run directory.
- `check`: `project_dir`, a listed project's directory, from the project's Check
  the setup. The Overview's Check a project passes none.
- `settings`: `project_dir`, a listed project's directory.
- `results`: `run_dir`, the directory to open, from a project's latest results,
  a run directory on the Jobs page, a job page's Open the results and
  `resultsLink`.
- `projects`, `models` and `jobs`: none.

While Check is replaced, the check's report names the tool check as
Testuary's and no box of the Check page, which the replacing page need not
have. A replacing page that offers Testuary's tool check labels it with
`/api/state`'s `tool_check_label`, the value of
`testuary.web.TOOL_CHECK_LABEL`, so it follows Testuary's label.

### Moved tabs

`tab_positions` moves Testuary's tabs, any but the Overview, each to a
position in the words of an [added tab's](#added-tabs) `position`. A moved
tab keeps its address and its view. A position that names a replaced tab
places the tab where the replaced tab stood, so a product that replaces
Run with its own tab and wants Results after it names its own tab, such as
`{"results": "after:notes"}`. An extension cannot move a tab it replaces,
since that tab leaves the header.

### Hidden tabs

`hidden_tabs` takes Testuary's tabs, any but the Overview, out of the
header. A hidden tab's address opens the Overview. Testuary's links to a
hidden tab are left out with the words that belong to them, so the rest
reads as whole sentences: while Run is hidden, the Overview offers no
Start a run and the Jobs page reads "No job started yet in this session.
Play the demonstration." No text names a hidden tab's page. While Run is
hidden, the check's next step reads "start a run", and while Check is
hidden, the check's report names Testuary's tool check and no box of the
Check page. `resultsLink` returns null while Results is hidden, which `el`
and a view leave out. A
position that names a hidden tab places the tab where the hidden tab
stood. An extension cannot hide a tab it replaces or moves.

## View scripts

A view script is a JavaScript module next to the extension's Python
module, named with at most 64 letters, digits, underscores and dashes
before `.js` or `.mjs`. The interface serves the scripts the extension declares under
`/ext/<extension id>/` and loads a script the first time its tab opens.
The script registers its view with `testuaryWeb.registerView(id, view)`,
where `id` is the tab's or directory type's id and `view(query)` returns the
page's parts: nodes, lists of nodes and null for a part left out. It may
be `async`. `query` is the address's query as `URLSearchParams`.

`testuaryWeb` offers the page's building blocks:

- `el(tag, properties, ...children)`: an element. `text` sets its text,
  `class` its classes, `style` its style properties from an object, an
  `on...` property a listener, and any other property the attribute of that
  name. Text is always set as text, never as HTML.
- `call(path, body)`: the interface's API, with the access key. Without a
  body it reads, with one it changes. A refusal throws an error with the
  server's message.
- `panel(...children)` and `table(headings, rows)`: the page's boxes.
- `jobLink(job)` and `resultsLink(dir, text)`: links to a job's page and
  to Testuary's Results of a directory. Without a text, the results link
  reads Open the results, with the Results tab's title as the header shows
  it, such as Open the test results. While Results is hidden, the results
  link is null.
- `tabTitle(id)`: a tab's title as the header shows it, so a link names a
  renamed or replaced tab as the header does.
- `nestedRun(path, title)`: a Testuary run, folded under its title and
  drawn by Testuary ([Testuary runs in a page](#testuary-runs-in-a-page)).
- `pathText(full, short)`: a path with a Copy path button.

A view that offers a choice of project or model reads Testuary's lists
with `call`. These fields keep their names and meanings within an API
version: `/api/projects` holds `projects`, each with `name`, `project_dir`,
`working` (the working directory's project) and `error` (None, or why
Testuary cannot read the project), and `/api/models` holds `models`, each
with `alias`, `provider`, `model` and `usable` (whether a run can use the
alias), with `default_model` and `project_model`.
`/api/models?project_dir=DIR&scope=EXTENSION/GROUP` reads the list of model
catalogue files a settings group names for the listed project `DIR`, as
the group's runs read it: it holds `models` with the same four fields of
each model and `error` (None, or why Testuary cannot read the list). A
project that is not listed, or a group without a list for it, gives
`error` with no `models`. The query's two parameters keep their names
and meanings within an API version too. The routes' other fields belong
to Testuary's pages and can change.

A script that fails, or a view that throws, shows its error in its own tab,
and every other tab works.

## Result views

A `DirType` has an `id`, a `title`, `detect(named_dir)`, which tells
whether a directory is one, `summary(named_dir)`, the JSON data its view shows,
read from the files the extension wrote, and the view `script`. The view
reads the summary from `/api/ext/dir?type=<extension id>/<id>&dir=<directory>`
and computes nothing of its own. The address
`#/dir?type=<extension id>/<id>&dir=<directory>` shows a directory in its view,
and Testuary's Results page links a directory to each directory type that
detects it. A Testuary run directory can also be one of the extension's
directories: Results and the extension's view each show what they read
there. A `summary` that fails, or
returns what JSON cannot hold, answers with its error as a route does.

## Settings groups

A `SettingsGroup` has an `id`, a `title`, `reference(project_dir)`, its sections
as `testuary.settings.SectionDescription` with their
`SettingDescription` rows, `values(project_dir)`, each setting's value by
`(section, name)`, `update(project_dir, section, values, remove)`, its
validated update, and optionally `extension_user_dir`, the extension's own
user directory, described below with the files it may change. A
`SettingDescription`'s kind is `boolean`, `choice`, `number` or `text`. `project_dir` is the project the Settings page shows, or None
for the global settings. The page shows the group as it shows Testuary's
sections, and a change goes only through `update`. An `update` that raises
`ValueError` refuses the change: the page shows the message and Testuary
writes nothing, so `update` checks the values before it writes its file.

A group's optional `testuary_settings_files(project_dir)` names the
Testuary settings files the extension's runs in the listed project
`project_dir` read, in the order they are read and with the extension's own
file last, also while it does not exist, or returns None for a project
whose runs name none. A relative path starts from the project directory. The Settings page then
offers the list as a choice for the project, "Testuary settings for" the
group's title. It shows every section of Testuary's settings as those runs
read them, each value with the file it comes from, and changes the last
file only, through Testuary's validated update. That update refuses an
`api_url` in the last file, as runs refuse one in a settings file in the
project directory. A value an earlier file
sets is changed by setting it in the last file, and Reset takes it out
again. The earlier files are read as the runs read them and never
written, and one that does not exist or is a directory shows as the choice's
error. A last file outside the project directory, also one reached through a
link, is shown read-only with the reason, and Testuary's global
`config.ini` as the last file links to the global settings, which change
it. A last file that is a directory is refused. A function that raises shows
its error in that choice, and the other choices keep working.

The list replaces both the global `config.ini` and the project's own
`.testuary/project.ini` for the extension's runs in that project. The page
shows Testuary's defaults, the list's files in order and the environment,
and neither the global settings nor the project's settings apply unless
the list names them. The model catalogue stays `models.ini` in the user
directory unless the group names its own list (below). A run of the
extension reads the same settings, and its own `overrides` after them, with
`testuary.settings.resolve_settings(config_files=files,
project_settings=False, base_dir=directory)`, where `files` leaves out the
last file while it does not exist.
[Configuration](../guide/configuration.md) describes `project_settings=False`.
An empty list reads neither the global `config.ini`
nor the file `TESTUARY_CONFIG_FILE` names, which only `config_files=None`
reads. The first change on the page creates the last file.

A group's optional `testuary_models_files(project_dir)` names the model
catalogue files the extension's runs in the listed project read, in the
order `resolve_settings(models_files=...)` reads them and with the
extension's own file last, also while it does not exist, or returns None.
The Models page then offers the list as a choice, "models for" the group's
title, and shows the models as those runs read them, each with the file
its section comes from and no key. A change goes to the last file only,
and only while it lies, links followed, in the project directory or in the
group's `extension_user_dir`, the extension's own user directory. Elsewhere
the list is read-only with the reason, and the user directory's
`models.ini` as the last file links to the page's own choice, which
changes it. A change to a model or a provider an earlier file defines
writes its whole section into the last file, from the earlier section's
keys, where it takes the place of the earlier one. Remove takes a section
of the last file away. The last file holds key variables (`api_key_env`)
and no key: a key is refused, a section whose key an earlier file holds
needs a variable, and a section whose address holds a credential is not
copied. A catalogue file in the project directory may set neither
`api_url` nor `api_key_env`, as runs refuse it, so a list whose sections
need them keeps its last file in the extension's user directory.

A group's optional `end_models_import(project_dir)` makes the extension's
runs in that project read the last file alone. Without it, a model only an
earlier file defines can be changed but not removed. With it, Remove asks
first. On yes, every other model is copied into the last file, with a key
variable for each key an earlier file holds, the group is asked as after
any save, the function runs, and the list must then name the last file
alone. Otherwise the last file is put back, and the page says which files the
runs read now.
The page cannot undo what the function changed before it failed, so the
function changes the extension's settings as its last step.

After the Settings page or the Models page writes a file of a group's
list, it calls the list's function again. A function that raises refuses
the saved file: the file returns to the bytes it held, or is removed when
the save created it, and the page shows the error's message. An extension
checks its own rules for its files there.

## Job kinds

A `JobKind` has an `id`, a `title`, the `module` the job runs as
`python -m` runs it, `command(form)`, the module's arguments for the form
the extension's view sends, and `resume(arguments)`, the arguments Continue
runs, or None for a kind without Continue. A view starts a job with
`call("/api/ext/jobs", {kind: "<extension id>/<id>", ...})` and opens its
page at `#/job/<job id>`. A `command` that raises `ValueError` refuses the
form with its message. For an extension that is a file named in
`[main] extensions`, the job runs with that file's directory on its import
path, so a job module beside the file needs no installation.

The interface adds `--events PATH` to the arguments. The module writes its
run there through `testuary.presentation.EventFile`, with a `RunPlan` and
its `DisplayEvent` observations, so the job page draws its plan and
progress as it draws a run's. The page matches each observation to the
plan by its phase and operation: a phase's own with `phase/run-0` (the
first run), a step's with `step/<step>`, and the plan's last step with
`step/finishing` in the phase `finishing`, labeled as the plan labels it.
The job page shows no repetition for an extension's job, and its progress
panel carries the product name, else the job kind's title. A
`RunCompletion`, written with `events.completion(completion)`, adds a run's
end-of-run summary, in a run's words about
targets and their suites, so a job writes one only when its files are
targets with suites. The job runs one at a time with Testuary's jobs. Stop
ends it as Ctrl+C ends a run, so the module ends on `KeyboardInterrupt`,
and Force stop ends it with every process it started.

An observation's `run_id`, `target`, `phase` and `operation` tell
concurrent work apart. None may be empty, they show only at the debug
level, and the page matches `phase` and `operation` to the plan as above.
The worked example's run-level observations name the phase `pipeline` and
the run `r0` for this reason. These fields take fixed words:

| Field | Words |
| --- | --- |
| `level` of a `DisplayEvent` | `run`, `target`, `phase`, `step` or `operation`: where the observation stands in the page's layers. None takes it from the identity |
| `state` of a `DisplayEvent` | `running`, `waiting` (such as before a request is sent again), `retrying` (a request sent again after a temporary failure), `succeeded`, `failed`, `cancelled` (stopped), `refused` and `limit_reached` (stopped at a limit its settings set, an ordinary end and no failure) |
| `state` of a `PlannedPhase` or `PlannedStep` | `planned`, `off` for one this run does not reach, such as after a stop after an earlier phase, and `reused` for one taken from saved results |

A job that checks a setup, such as the extension's own requirements, writes
a `RunReadiness` with `events.readiness(report)`, its items as
`ReadinessItem` and its limits as `ReadinessLimit`. Its page shows the
report as the page of Testuary's check shows its own: what the check
found, the items and the limits. The report's `scope` sets its first line in
Testuary's words: `inspection` says a quick check, which runs
nothing in the project and contacts no model, and `verification` a tool
check, which also starts tools. On the extension's command line,
`testuary.presentation.present_readiness` prints the report as `testuary
check` prints its own.

A later plan replaces the earlier one. A job whose first phase runs once
before its files are known writes a plan without files, then a plan with
the files in which that phase is reused. The page draws each phase and
step in its newest observed state for the current file, else in the state
the newest plan gives it.

## Testuary runs in a page

`nestedRun(path, title)` returns a folded part that shows a Testuary run
as Testuary's own pages show it. Its summary reads "Testuary run",
followed by `title` when one is given, so the run's phases stay apart
from the page's own. `path` names the run's event file or its run
directory as Results names a directory: relative to the working directory, or
absolute inside it or a listed project's directory, with no link on the
way. The part reads the run when it is opened, and while it stays open:

- An event file, such as the one `testuary --events FILE` writes, shows
  the run's progress as a job's page shows it: its phases and steps, its
  progress, its end-of-run summary and what happened, in order. While the
  run writes the file, the part follows it.
- A run directory or a demonstration's directory shows its results as the
  Results page shows them.

Testuary draws the run, so a page that folds one shows it as the
installed Testuary draws it. A path the part cannot read shows the reason
inside the part, and Testuary's configuration files are never read. A job
that runs Testuary in its own process names the run's event file with
`testuary.orchestrator.run_pipeline`'s `events`, which records the plan
and the end-of-run summary too.

A job kind names the Testuary runs its jobs start with
`nested_runs(arguments)`, which returns a `testuary.web.NestedRun` for
each: `phase`, the id of the phase of the job's plan that started it,
`path`, its event file or run directory, and `title`. `arguments` are the
job's arguments as `resume` receives them. The job's page folds each run
into its phase in the phases and steps, and a run whose phase the plan
does not name after the phases. The page asks again whenever a phase's
state changes and when the job ends, so a run that a later phase starts
appears there. A `nested_runs` that fails shows its error in their place,
and the rest of the page works.

## Read-only routes

A `WebRoute` has a `path` below the extension's prefix and
`read(named_dir, query)`, which returns JSON. The route answers GET at
`/api/ext/<extension id>/<path>`, with the interface's access key, so a
view calls it with `call`. `named_dir` is the directory the query's `dir` names,
resolved as the Results view resolves a directory: inside the working directory
or a listed project's directory, following no links. Without `dir` it is None.
`query` holds the query's first value of each name. The route gets
nothing else of the server. A path with `..`, a `dir` outside those
directories, any method but GET and a request without the key are refused
before it runs. Testuary does not stop the function from writing, so the
function only reads. A `ValueError` refuses the request with its message.
Any other error, and an answer JSON cannot hold, answers with the error,
and the other routes and pages keep working. A path is at most four parts
joined by `/`, each 1 to 40 lowercase letters, digits and dashes, starting
with a letter or digit. The extension ids `dir`, `settings` and `jobs`
are Testuary's own names below `/api/ext/`.

## What stays fixed

An extension cannot change:

- the access key, the host check, the local listener and the stop secret,
- the job runner's one job at a time, Stop and Force stop,
- Testuary's event file format,
- Testuary's own pages, settings and validated settings update,
- Testuary's own routes,
- the view of a Testuary tab, other than by replacing or hiding the whole
  tab,
- the Overview's place and view,
- how Testuary draws a nested run.

## Compatibility

- The interface imports each named module and uses an extension whose `api`
  it supports. It lists the others on the Overview with the reason and uses
  nothing of them.
- A tab's id and a directory type's id name one view. Two extensions with the
  same id, or adding a tab or directory type of one id, or replacing, moving
  or hiding the same tab, are both refused. One extension may give its tab
  and its directory type one id, and two extensions may each add a job kind
  of one id, since a job kind is named with its extension's id. An
  extension that gives two of its tabs, directory types, settings groups or
  job kinds one id is refused.
  An extension that replaces, moves or hides the Overview or a tab
  Testuary does not have, whose tabs replace one tab twice, that replaces,
  moves or hides one tab more than once, or whose guide is neither an
  `http` or `https` address nor an HTML file's name, is refused. One extension
  at most sets the product name, one sets the guide, and one renames each tab.
  A later one is refused for that field.
- An extension that fails to load is listed on the Overview with its
  error.
- A later Testuary keeps every field of an API version it supports. A change
  that removes or changes a field raises the version number. Version 1 covers the event file's schema version 3
  (`testuary.presentation.EVENT_FILE_SCHEMA`).

## A worked example

The extension `notes/notes.py` adds a tab Notes in the place of Run and
moves Results after it. Its job writes a directory's `notes.json`, its
directory type opens such a directory, its settings group sets how many notes the job writes and
names the Testuary settings file of a project's notes runs,
`.notes/testuary.ini`, and its route `/api/ext/notes/count?dir=DIR`
answers a directory's number of notes. Its second job checks a notes directory
and reports in Testuary's readiness format. Its third job plays
Testuary's demonstration into a notes directory and names it as a nested
run, and the Notes view folds the demonstration's directory.

The directory `notes` in your user directory holds the extension's five
files, and the user directory's `config.ini` names the extension:

```ini
[main]
extensions = notes/notes.py
```

`notes/notes.py`:

```python
import configparser
import json
from pathlib import Path

from testuary.settings import SectionDescription, SettingDescription, file_locations
from testuary.web import DirType, JobKind, NestedRun, SettingsGroup, WebExtension, WebRoute, WebTab

SUMMARY = "notes.json"
DEMO_EVENTS = "demo-events.jsonl"
COUNT = SettingDescription("notes", "count", "How many notes Collect notes writes, from 1 to 100.",
                           "number", "3")


def is_notes(notes_dir: Path) -> bool:
    return (notes_dir / SUMMARY).is_file()


def summary(notes_dir: Path) -> dict:
    return json.loads((notes_dir / SUMMARY).read_text(encoding="utf-8"))


def _settings_file() -> Path:
    return file_locations().user / "notes.ini"


def reference(project_dir):
    return [SectionDescription("notes", "The notes the Collect notes job writes.", (COUNT,))]


def values(project_dir) -> dict:
    parser = configparser.ConfigParser()
    parser.read(_settings_file(), encoding="utf-8")
    return {("notes", "count"): parser.get("notes", "count", fallback=COUNT.default)}


def update(project_dir, section, changes, remove) -> None:
    count = changes.get("count", COUNT.default)
    if section != "notes" or not count.isdigit() or not 1 <= int(count) <= 100:
        raise ValueError("count: expected a whole number from 1 to 100")
    parser = configparser.ConfigParser()
    parser.read(_settings_file(), encoding="utf-8")
    parser["notes"] = {} if "count" in remove else {"count": count}
    with open(_settings_file(), "w", encoding="utf-8") as stream:
        parser.write(stream)


def testuary_settings_files(project_dir: Path) -> list:
    return [Path(".notes") / "testuary.ini"]


def command(form) -> list:
    notes_dir = form.get("dir")
    if not isinstance(notes_dir, str) or not notes_dir.strip():
        raise ValueError("Name the directory the notes go to.")
    return ["--dir", notes_dir, "--count", values(None)[("notes", "count")]]


def check_command(form) -> list:
    return command({"dir": form.get("dir")})


def demo_command(form) -> list:
    arguments = command(form)[:2]
    if (Path(arguments[1]) / "demo").exists():
        raise ValueError("This notes directory already holds a demonstration.")
    return arguments


def demo_runs(arguments) -> list:
    return [NestedRun("play", Path(arguments[1]) / DEMO_EVENTS, "Testuary's demonstration")]


def count(notes_dir, query) -> dict:
    if notes_dir is None or not is_notes(notes_dir):
        raise ValueError("Name a notes directory with dir.")
    return {"count": summary(notes_dir)["count"]}


testuary_web = WebExtension(
    api=1, id="notes",
    tabs=(WebTab("notes", "Notes", "notes.js", position="before:run", replaces="run"),),
    tab_positions={"results": "after:notes"},
    dir_types=(DirType("notes", "Notes", is_notes, summary, "notes.js"),),
    settings_groups=(SettingsGroup("notes", "Notes", reference, values, update,
                                   testuary_settings_files),),
    routes=(WebRoute("count", count),),
    job_kinds=(JobKind("collect", "Collect notes", "notes_job", command),
               JobKind("check", "Check notes", "notes_check", check_command),
               JobKind("demo", "Play the demonstration", "notes_demo", demo_command,
                       nested_runs=demo_runs)))
```

The example keeps its settings in Testuary's user directory,
`testuary.settings.file_locations().user`. An extension chooses its own place.

`notes/notes_job.py`, the Collect notes job:

```python
import argparse
import json
import time
from pathlib import Path

from testuary.presentation import DisplayEvent, EventFile, RunPlan
from testuary.presentation.events import PlannedPhase, PlannedStep


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="collect-notes")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--events", required=True)
    args = parser.parse_args(argv)
    notes_dir = Path(args.dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def event(state, message, operation="run", level="run", phase="collect", label=""):
        return DisplayEvent(run_id="r0", target="notes", phase="pipeline" if level == "run" else phase,
                            operation=operation, state=state, elapsed_s=time.monotonic() - started,
                            message=message, level=level, label=label)

    with EventFile(args.events) as events:
        events.plan(RunPlan(files=("notes",), runs=("r0",),
                            phases=(PlannedPhase("collect", "Collect notes",
                                                 steps=(PlannedStep("write", "Write notes"),)),),
                            finishing=PlannedStep("summary", "Write the summary")))
        events(event("running", "Collecting notes"))
        events(event("running", "Collect notes", operation="phase/run-0", level="phase"))
        events(event("running", "Writing notes", operation="step/write", level="step"))
        notes = [f"Note {number}" for number in range(1, args.count + 1)]
        (notes_dir / "notes.json").write_text(json.dumps({"count": len(notes), "notes": notes}),
                                           encoding="utf-8")
        events(event("succeeded", "Notes written", operation="step/write", level="step"))
        events(event("succeeded", "Collect notes", operation="phase/run-0", level="phase"))
        events(event("succeeded", "Summary written", operation="step/finishing", level="step",
                     phase="finishing", label="Write the summary"))
        events(event("succeeded", "Collected notes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`notes/notes_check.py`, the Check notes job:

```python
import argparse
from pathlib import Path

from testuary.presentation import EventFile
from testuary.readiness import ReadinessItem, ReadinessLimit, RunReadiness


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="check-notes")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--events", required=True)
    args = parser.parse_args(argv)
    notes_dir = Path(args.dir)
    ready = not notes_dir.exists() or notes_dir.is_dir()
    item = ReadinessItem("notes_dir", required=True, status="verified" if ready else "missing_setup",
                         selected=str(notes_dir), reason="" if ready else "The path names a file.",
                         action=None if ready else "Name a directory for the notes.")
    report = RunReadiness(
        scope="inspection", status=item.status, exit_code=0 if ready else 1, source_dir=str(notes_dir),
        execution=None, source_files=(), phases=("collect",), model_alias=None, provider=None,
        model=None, credentials_configured=None, output_dir=str(notes_dir), output_mode="fresh",
        limits=(ReadinessLimit("collect", "count", args.count, "notes", "configured"),), items=(item,))
    with EventFile(args.events) as events:
        events.readiness(report)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
```

`notes/notes_demo.py`, the Play the demonstration job:

```python
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from notes import DEMO_EVENTS
from testuary.presentation import DisplayEvent, EventFile, RunPlan
from testuary.presentation.events import PlannedPhase, PlannedStep


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="play-demonstration")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args(argv)
    notes_dir = Path(args.dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    # The demonstration's event file exists from the start, so the job's
    # page can fold it at once.
    (notes_dir / DEMO_EVENTS).write_text("", encoding="utf-8")
    started = time.monotonic()

    def event(state, message, operation="run", level="run", phase="play", label=""):
        return DisplayEvent(run_id="r0", target="notes", phase="pipeline" if level == "run" else phase,
                            operation=operation, state=state, elapsed_s=time.monotonic() - started,
                            message=message, level=level, label=label)

    with EventFile(args.events) as events:
        events.plan(RunPlan(files=("notes",), runs=("r0",),
                            phases=(PlannedPhase("play", "Play the demonstration"),),
                            finishing=PlannedStep("note", "Write a note")))
        events(event("running", "Playing the demonstration"))
        events(event("running", "Play the demonstration", operation="phase/run-0", level="phase"))
        status = subprocess.run([sys.executable, "-m", "testuary", "--demo", "--demo-speed", "1",
                                 "--output-dir", str(notes_dir / "demo"), "--events", str(notes_dir / DEMO_EVENTS),
                                 "--display", "plain", "--no-color"], stdin=subprocess.DEVNULL).returncode
        state = "succeeded" if status == 0 else "failed"
        events(event(state, "Play the demonstration", operation="phase/run-0", level="phase"))
        note = f"Testuary's demonstration ended with exit status {status}."
        (notes_dir / "notes.json").write_text(json.dumps({"count": 1, "notes": [note], "demo": "demo"}),
                                           encoding="utf-8")
        events(event("succeeded", "Note written", operation="step/finishing", level="step",
                     phase="finishing", label="Write a note"))
        events(event(state, "Played the demonstration" if status == 0 else note))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
```

The demonstration plays at its configured length, `[demo]
playback_seconds`, and writes into `demo` below the notes directory, which
holds one demonstration.

`notes/notes.js`, the view of the tab and of the directory type:

```javascript
const web = globalThis.testuaryWeb;

web.registerView("notes", async (query) => {
  const dir = query.get("dir");
  const parts = [web.el("h1", { text: "Notes" })];
  const input = web.el("input", { type: "text", id: "notes-dir", value: dir || "" });
  const said = web.el("p", { class: "error", role: "alert" });
  // A button that starts a job of the kind for the directory and opens its page.
  const starter = (kind, text, style) => {
    const button = web.el("button", { type: "button", class: style, text });
    button.addEventListener("click", async () => {
      try {
        const job = await web.call("/api/ext/jobs", { kind, dir: input.value });
        location.hash = `#/job/${job.id}`;
      } catch (error) { said.textContent = error.message; }
    });
    return button;
  };
  parts.push(web.panel(web.el("label", { for: "notes-dir", text: "Notes directory" }), input,
    starter("notes/collect", "Collect notes", "primary"), starter("notes/check", "Check the directory", ""),
    starter("notes/demo", "Play the demonstration", ""), said));
  if (dir) {
    const found = await web.call(`/api/ext/dir?type=notes/notes&dir=${encodeURIComponent(dir)}`);
    parts.push(web.el("h2", { text: `${found.summary.count} notes` }),
      web.table(["Note"], found.summary.notes.map((note) => [note])),
      web.panel(web.resultsLink(dir, `Open this directory in ${web.tabTitle("results")}`)));
    // The demonstration Play the demonstration played, drawn by Testuary.
    if (found.summary.demo) parts.push(web.nestedRun(`${dir}/${found.summary.demo}`, "Testuary's demonstration"));
  }
  return parts;
});
```

With `notes/notes.py` named in the settings, `testuary-web` shows the Notes
tab where Run stood with Results after it, the Overview's Start a run opens
Notes, and the Settings page shows the Notes group. Collect notes runs as
a job, and the Results page links its directory to the Notes view. Check the
directory runs Check notes, whose job page shows its report. Play the
demonstration's job page folds Testuary's demonstration into its phase
while it plays, and the Notes view of its directory folds the
demonstration's results.
