# Contributing

## Structure

| Path | Responsibility |
| --- | --- |
| `src/testuary/` | Generation, repair, phase orchestration, measurement, reduction, and result persistence |
| `tests/` | Package regression tests, with separate real-measurement tests |

The package contains the four phases (the starter phase, the coverage phase,
the mutation phase and Finishing), with repair, the choice of a suite at each
phase end and suite reduction.
[Code organization](explanation/code-organization.md) follows one run
through the modules. How a language plugs in is described in
[Make a language ready for Testuary](extending/languages.md).

## Setup

For work on the source code, check out the LangAdapter repository next to this
one and install both in editable mode in a virtual environment with Python
3.10 or newer:

```bash
python -m pip install -e ../langadapter
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest -q
python -m pytest tests/integration -m slow -q
```

The first runs every test not marked `slow`: the unit tests, integration tests
and web tests. The unit tests check single parts in isolation. Tests marked
`slow` run real measurements in subprocesses and need the measurement tools
installed. When you measure run times, run nothing else heavy on the machine at
that time.

## Documentation

The site is built from `docs/` with MkDocs. Install the `docs` extra and build
it strictly:

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
```

The command-line and settings reference pages are generated from the
argument parsers and the packaged `config.ini`. After changing an option, a
help text, a default, a setting or an exit status, rewrite them with
`python tools/reference_pages.py`. It also writes the exit statuses in the
guides. The level
tables of [Make a language ready for Testuary](extending/languages.md) come
from Testuary's profile: after changing the profile, rewrite them with
`python tools/language_page.py`. A test fails while either is out of date.

Public text is ASCII only, with no dashes, ellipses or emoji, no semicolons
in prose, and no internal labels such as review codes, role names, commit
hashes or decision dates. This holds for the README, the documentation
pages, the shipped examples and the package's docstrings and comments.
Comments say why the code does what it does.

