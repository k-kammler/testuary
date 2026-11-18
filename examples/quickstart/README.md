# Quickstart

The commands below assume Testuary is installed in an activated virtual
environment, as [getting started](https://k-kammler.github.io/testuary/getting-started/)
describes:

```bash
python -m pip install testuary
```

Run them in this directory.

## Demonstrations: no model, no cost

Each demonstration plays back one recorded run of this release. It sends no
model request and needs no key and no setup. The Python one plays
`tenacity/wait.py`, the Java one JSONassert's `JSONCompare.java`, the Toy one
the small example language of LangAdapter's tutorial, and the tutorial one
the run that Testuary's tutorial ends with.

At the console:

```bash
testuary --demo --output-dir demos/python
testuary --demo --demo-language java --output-dir demos/java
testuary --demo --demo-language toy --output-dir demos/toy
testuary --demo --demo-data tutorial --output-dir demos/tutorial
```

Each command needs a new or empty run directory. Add `--demo-speed instant`
to see a run at once. Two scripts play all four at instant speed, each run
of a script into its own new directory below `demos`, named by its start time,
on Linux:

```bash
sh run_demos.sh
```

and on Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demos.ps1
```

From Python, `run_demos.py` does the same through `testuary.demo.play`:

```bash
python run_demos.py
```

## The web interface

Start it in the directory you work in:

```bash
testuary-web
```

`start_web.py` starts it from Python, through `testuary.web.serve`:

```bash
python start_web.py
```

Either opens the interface in your browser for the directory you start it in,
unless you add `--no-browser`.
[Access](https://k-kammler.github.io/testuary/guide/web/#access) says how to
start it on a shared computer. Ctrl+C stops it.

## Your own model on real files: needs your model, costs requests

The two example projects hold one small file each, with the settings of
getting started's small run in their `.testuary/project.ini`: at most 46
model requests per target, as [getting started](https://k-kammler.github.io/testuary/getting-started/#run-with-small-limits)
counts them. First set up your model once:

```bash
testuary setup model
```

The setup asks for the service, the model and its key, and names the model
with an alias. Use that alias below in place of `YOUR_ALIAS`.

The Python example, `python-project/shipping.py`, has no dependencies, so the
Python of your Testuary environment runs its tests: it holds pytest,
coverage and Cosmic Ray. Name it with `--python-executable`. On Linux:

```bash
cd python-project
testuary generate --model YOUR_ALIAS --python-executable "$(command -v python)"
```

On Windows, in PowerShell:

```powershell
cd python-project
testuary generate --model YOUR_ALIAS --python-executable (Get-Command python).Source
```

The Java example, `java-project`, is a Maven project with one class,
`src/main/java/example/Shipping.java`. It needs a JDK 11 or newer, named by
`JAVA_HOME`, and Maven on the path:

```bash
cd java-project
testuary generate --model YOUR_ALIAS
```

Each run writes its results into the project's `.testuary/results` directory,
in a directory named after the model alias.
[Results and logs](https://k-kammler.github.io/testuary/guide/output/)
explains them.
