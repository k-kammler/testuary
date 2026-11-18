# Set up your machine

This page prepares a computer for Testuary: the tools of the languages you
test, Linux under WSL on Windows, and your model keys.
[Getting started](../getting-started.md) installs Testuary and runs it for
the first time.

## Language tools

Testuary runs the tests and tools of each language through LangAdapter.
[LangAdapter's documentation](https://k-kammler.github.io/langadapter/getting-started/)
says what each language needs. The
[quick check](../getting-started.md#check-your-setup) finds the tools a run
would use before the first model request.

## Windows with WSL

Testuary runs on Windows itself. Testing a C file needs Linux, which WSL
(Windows Subsystem for Linux) provides on Windows. Testuary is tested with
Ubuntu 24.04 under WSL ([tested environments](../reference/tested-environments.md)).

Install Ubuntu 24.04 from a Windows terminal:

```text
wsl --install -d Ubuntu-24.04
```

Open Ubuntu and run the commands of this guide there. Ubuntu 24.04 names
its Python `python3` and needs the package `python3-venv` for virtual
environments:

```bash
sudo apt update
sudo apt install python3-venv
```

Then install Testuary as [getting started](../getting-started.md#install)
shows, with `python3 -m venv` in place of `python -m venv`. In the
activated environment, `python` is the environment's Python.

### Keep projects in the Linux file system

Keep your projects and Testuary's environment below your Linux home
directory (`~`), not below `/mnt/c`. WSL reaches the Windows drives below
`/mnt` through a file system that is much slower for the many small file
operations of builds, test runs and mutants. Windows programs, such as an
editor, reach the Linux files at `\\wsl.localhost\Ubuntu-24.04\`.

### Your user directory and keys in WSL

Testuary in WSL reads its user directory in Linux, `~/.config/testuary`,
not the one on Windows
([where Testuary reads its files](configuration.md#where-testuary-reads-its-files)).
Set up your models there with `testuary setup model`, also when Testuary on
Windows has them already.

Environment variables set on Windows are not seen in WSL. Set a key's
variable in Ubuntu, or let the wizard keep the key in `models.ini`:

```bash
export TESTUARY_OPENAI_API_KEY="your key"
```

### The web interface from the Windows browser

Start the web interface in Ubuntu without a browser, and open the address
it prints in your Windows browser:

```bash
testuary-web --no-browser
```

It prints its address on 127.0.0.1, at the port of
[`[web] port`](../reference/settings.md#web). With WSL's default settings, a
port that a Linux program opens on 127.0.0.1 is reachable from Windows at
the same address. Without `--no-browser`, Testuary looks for a browser in
Linux, prints an error such as `gio: ... Operation not supported`, and the
interface keeps running.

A Windows program that already holds the same port answers that address in
place of the interface in WSL, and the start in WSL reports no conflict.
If `testuary-web` also runs on Windows, give the two interfaces different
ports with `--port`.

## Keys checklist

- Keep each model key in `models.ini` in your user directory or in an
  environment variable, in one of the
  [forms the model catalogue accepts](../models-and-prompts.md#modelsini-fields).
- Keep keys out of your projects. Testuary refuses a key in its settings
  files and in any model catalogue other than your own `models.ini`.
- Write the catalogue with [the setup wizard](../getting-started.md#set-up-with-the-wizard),
  `testuary setup model`, or on the web interface's
  [Models page](web.md#overview-projects-and-models).
- On a computer you share with other users, start the web interface with
  `testuary-web --access key` ([access](web.md#access)).
