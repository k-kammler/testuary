# Tested environments

Testuary was tested in the combinations below. The Release column names the
newest Testuary version tested in each combination, 0.2.1 or 0.2.0. Other
combinations may work, but they were not tested. How projects are run in each
language, and in containers, comes from LangAdapter, whose
[tested environments](https://k-kammler.github.io/langadapter/reference/tested-environments/)
list the language tools in more detail.

## Where Testuary runs

| What | Tested | Release |
| --- | --- | --- |
| Operating systems | Windows 11, Ubuntu 24.04 under WSL | 0.2.1 |
| Operating systems | AlmaLinux 8.10 | 0.2.0 |
| Python for Testuary itself | 3.10, 3.11, 3.12, 3.13 and 3.14, installed from the built wheels into a new virtual environment | 0.2.1 |
| Testuary's own test suite | Python 3.12 and 3.14 on Windows 11 and on Ubuntu 24.04 under WSL | 0.2.1 |
| Testuary's own test suite | Python 3.10, 3.11 and 3.13 on Windows 11 and on Ubuntu 24.04 under WSL | 0.2.0 |

## Projects

| Language | Tested | Release |
| --- | --- | --- |
| Python | pytest 8.1.1, coverage 7.6.8 and Cosmic Ray 8.4.1 in the project's virtual environment (on Python 3.14 and 3.15 pytest 8.4.2, coverage 7.16.2 and Cosmic Ray 8.4.1), project Python 3.12 in full runs | 0.2.0 |
| Java | Maven 3.9 and a project's Maven wrapper, JUnit 4.13.2 and JUnit 5, JDK 17 and JDK 21 in full runs | 0.2.0 |
| C | utest.h test programs, gcov through gcovr and Mull, with GCC 13.3 on Ubuntu 24.04 under WSL. [LangAdapter's C guide](https://k-kammler.github.io/langadapter/guide/c/) lists what a project needs | 0.2.1 |

Full runs, all four phases from the starter phase to Finishing, were tested
with Testuary 0.2.0 for a Python project on Ubuntu 24.04, and for a Python and
a Java project on AlmaLinux 8.10.

## Support ranges

The quick check compares a project's runtime with LangAdapter's
[support ranges](https://k-kammler.github.io/langadapter/reference/tested-environments/#support-ranges).
When
Testuary runs on an older Python than the project, the quick check notes
it: some source operations read the project's code with Testuary's
Python. A source in newer syntax ends the work on it with a reading error
that names both Pythons and the step to take: run Testuary, the program
that uses LangAdapter, on the project's Python or newer.

## Model services

| What | Tested | Release |
| --- | --- | --- |
| Whole runs with real model requests | Ollama Cloud | 0.2.0 |
| Request protocols | OpenAI-compatible chat, OpenAI Responses and Ollama, against local loopback services in the test suite | 0.2.1 |

## macOS

This release has not run on a Mac. macOS keeps the system's temporary
directory below the link `/var`. If a run on a Mac fails with a message that
names `/var`, set `TMPDIR` to the directory's physical path:

```text
export TMPDIR="$(cd "$TMPDIR" && pwd -P)/"
```

Where it was tested, this release needs no such setting: LangAdapter
puts its temporary directories below the physical path, and its collection
check resolves the directories Testuary hands it. That is tested on Linux with
a temporary directory behind a symbolic link, a mutation session without
`/dev/shm` included, but not on a Mac. No Java project has run on a Mac.

LangAdapter runs a project's command on macOS in its own process group:
stopping it ends the group, but a process that leaves the group (a new
session, a daemon) is neither ended nor seen. Cleanup is therefore not
confirmed: the first command warns, and LangAdapter's observations record
the supervision `process_group_unconfirmed`.

Testuary refuses a link in the path of the project or of the run
directory. On macOS `/tmp` and `/var` are links, so keep both elsewhere, such
as in your home directory. macOS volumes usually ignore letter case in paths,
while LangAdapter compares paths there case-sensitively, so name files with
their exact spelling. A macOS JDK bundle (`name.jdk`) works when you select
the bundle, its `Contents/Home` or its `java`.

## Not tested

- Project environments prepared by Conda, Poetry, Pipenv or your own uv. A
  project interpreter from such an environment is used like any other
  interpreter, but the environments' own activation is not part of the
  tests. The environment Testuary provisions through uv for
  `[project] python_version` was tested with Testuary 0.2.0: CPython
  3.12.13 on Ubuntu 24.04 under WSL, from the built wheel, with a suite run,
  coverage and a Cosmic Ray session in it.
- Gradle projects in full runs, and the Gradle wrapper.
- Testuary itself on Python 3.15. LangAdapter's parser library,
  tree-sitter 0.25, has no wheels for 3.15, so Testuary installs there
  only with a C compiler. A project on Python 3.15 is tested by Testuary
  on Python 3.14 or older, so a source that uses syntax new in 3.15 ends
  the work on it with a reading error.
- POSIX systems other than macOS. LangAdapter runs commands there as on
  macOS, in a process group without confirmed cleanup.

## Not supported

- Project runtimes outside LangAdapter's support ranges. A project built for
  Java 8 runs on JDK 11 or newer.
- Java builds other than Maven and Gradle.
- A Java project with both Maven and Gradle build files. The run is
  refused, and the quick check says to remove the build files of the build
  system the project does not use.
- Test frameworks other than pytest for Python, JUnit for Java and, for C,
  the test kinds LangAdapter's C guide lists. A TestNG
  suite without JUnit is refused by name.
- Running a project's commands on Windows in a working directory longer than
  258 characters. Windows refuses it even with long paths enabled.
  LangAdapter keeps its copy of the project within that length where it
  can. Otherwise the run is refused with the length and what to change.
