# Testuary

[![PyPI version](https://img.shields.io/pypi/v/testuary?color=green)](https://pypi.org/project/testuary/)
[![Python versions](https://img.shields.io/pypi/pyversions/testuary?color=green)](https://pypi.org/project/testuary/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/k-kammler/testuary/blob/main/LICENSE)
[![Documentation](https://github.com/k-kammler/testuary/actions/workflows/docs.yml/badge.svg)](https://k-kammler.github.io/testuary/)

Testuary writes unit tests for Python, Java and C code with a language model.
It runs the tests, measures which lines and branches they cover and how many
mutants (small deliberate bugs) they catch, and asks the model for more tests
wherever something is still missed, until your goals or limits are reached.
It then keeps only the tests that add coverage or catch an extra mutant, and
hands you the finished test suite with its measurements.

Testuary is built to be extended: your own model service, prompt profile,
hints for your project or a new language each come as one file, or a few
lines in one. Running and measuring the tests is the job of
[LangAdapter](https://github.com/k-kammler/langadapter), a separate package
that installs with Testuary.

## Supported systems

| Projects in | Windows | Linux | macOS |
| --- | --- | --- | --- |
| Python | yes | yes | expected to work |
| Java | yes | yes | expected to work |
| C | in WSL | yes | not yet supported |

Testuary was tested on Windows 11, Ubuntu 24.04 under WSL and AlmaLinux 8.10.
This release has not run on a Mac. On Windows, C projects run with Testuary
installed inside [WSL](https://k-kammler.github.io/testuary/guide/machine/#windows-with-wsl).
[Tested environments](https://k-kammler.github.io/testuary/reference/tested-environments/)
says what is known.

## Install

Testuary runs on Python 3.10 or newer. Install Testuary into a virtual
environment of its own, outside the directory of the project you test, as
[getting started](https://k-kammler.github.io/testuary/getting-started/#install)
explains:

```bash
python3 -m venv testuary-env
source testuary-env/bin/activate
python -m pip install testuary
```

On Windows, create the environment with `python -m venv testuary-env` and
activate it with `testuary-env\Scripts\activate`. On Debian and Ubuntu,
`python3 -m venv` needs the package `python3-venv`
(`sudo apt install python3-venv`), as [Set up your machine](https://k-kammler.github.io/testuary/guide/machine/)
describes.

Testing a C file needs LangAdapter's extra for C
(`python -m pip install "langadapter[c]"`) and the compiler tools that
[LangAdapter's C guide](https://k-kammler.github.io/langadapter/guide/c/)
lists.

Testuary comes with a local web interface. Start it in the directory you
work in:

```bash
testuary-web
```

It opens the interface in your browser, or in an SSH login prints the
`ssh -L` line for your own computer.
[The web interface](https://k-kammler.github.io/testuary/guide/web/) describes
its pages and, under [Access](https://k-kammler.github.io/testuary/guide/web/#access),
how to start it on a machine you share.

The project you test needs its own test tools: pytest, coverage and Cosmic Ray
for Python, or a JDK and the project's Maven or Gradle build for Java.
[Tested environments](https://k-kammler.github.io/testuary/reference/tested-environments/)
lists the tested versions. For Java,
LangAdapter downloads the JUnit launcher, JaCoCo and PIT into its cache on
first use, unless `LANGADAPTER_JAVA_TOOLS_OFFLINE` is set to 1.

## Example

Create the configuration templates in your project's directory:

```bash
cd /path/to/project
testuary init
```

This writes `models.ini` into your user directory and `hints.json` into the
project's directory, and names each file it writes. Then add your model to that
`models.ini`. The section name is the alias you pass with
`--model`, and `provider =` names the kind of model service, `ollama`,
`openai` or `vllm`:

```ini
[ollama_qwen]
provider = ollama
model = qwen3-coder:30b
api_url = http://localhost:11434/v1
```

Run the quick check in the project's directory.
[The check's report](https://k-kammler.github.io/testuary/guide/terminal/#the-checks-report)
says what it looks at and what it leaves alone:

```bash
testuary check --model ollama_qwen --python-executable /path/to/project-env/bin/python --source-files package/module.py
```

Generate a suite for one target. For a first run, a few lines in the project's
settings file keep it small, as
[getting started](https://k-kammler.github.io/testuary/getting-started/#run-with-small-limits)
shows. Run it in the project's directory too, where the run reads that file:

```bash
testuary generate --model ollama_qwen --python-executable /path/to/project-env/bin/python --source-files package/module.py
```

The results go to `.testuary/results/ollama_qwen` in the project's directory.

At a paid service, model requests can cost money. To see a run of all four phases without
a model, play back the offline demonstration with `testuary --demo --output-dir demo-python`.

The [quickstart directory](https://github.com/k-kammler/testuary/tree/v0.2.0/examples/quickstart)
of the repository, also in the source distribution, holds these first steps
as scripts and commands: the demonstrations, the web interface started from
the console or from Python, and two small projects, one Python file and one
Java class, to run with your own model.

## Documentation

- [Getting started](https://k-kammler.github.io/testuary/getting-started/): install, check your setup, run
  with small limits and read the results.
- [Terminal output](https://k-kammler.github.io/testuary/guide/terminal/): the progress views and the
  check's report.
- [Results and logs](https://k-kammler.github.io/testuary/guide/output/): the run directory, the delivered
  suite, the exit status and how to continue a run.
- [Offline demonstrations](https://k-kammler.github.io/testuary/guide/demonstrations/): recorded runs without
  a model.
- [Web interface](https://k-kammler.github.io/testuary/guide/web/): `testuary-web`, a local browser
  interface for projects, models, settings, runs and results.
- [Configuration](https://k-kammler.github.io/testuary/guide/configuration/): models, settings and hints.
- [Running on a cluster](https://k-kammler.github.io/testuary/guide/cluster/): mutation workers, node-local
  scratch, proxies and resuming batch jobs.
- [Models and prompt profiles](https://k-kammler.github.io/testuary/models-and-prompts/)
- [Extension points](https://k-kammler.github.io/testuary/extending/): every way to
  extend Testuary, with its guide.
- [Make a language ready for Testuary](https://k-kammler.github.io/testuary/extending/languages/)
- [Measurement and recovery](https://k-kammler.github.io/testuary/explanation/measurement-and-recovery/)
- [Result files](https://k-kammler.github.io/testuary/reference/result-files/)
- [Contributing](https://k-kammler.github.io/testuary/contributing/)
- [Changelog](https://github.com/k-kammler/testuary/blob/v0.2.0/CHANGELOG.md)

## License

Testuary is licensed under the
[MIT license](https://github.com/k-kammler/testuary/blob/v0.2.0/LICENSE).
A few files from other projects, in the demonstration data, keep their own
licenses, and so do the installed dependencies.
[Third-party material](https://github.com/k-kammler/testuary/blob/v0.2.0/THIRD_PARTY_NOTICES.md)
lists them.
