# Third-party material

## Testuary code

The project's original code is licensed under the MIT license in [LICENSE](LICENSE).
That license does not replace licenses for third-party source, test code,
datasets or separately installed dependencies.

## Demonstration data

`testuary --demo` plays data in `src/testuary/demo/data`, included in the
source and wheel distributions. Each demonstration was recorded in one real
run of this Testuary release.

- The Python demonstration holds `tenacity/wait.py` and
  `tenacity/__init__.py`, from [tenacity](https://github.com/jd/tenacity) at
  commit `8be01b1daf2010566fd936a5c2214efefa43edff`, unmodified, under the
  Apache License 2.0 in `LICENSE-tenacity.txt`. Tenacity distributes no
  NOTICE file.
- The tutorial's closing run (`--demo-data tutorial`) holds
  `tenacity/stop.py` and `tenacity/__init__.py` from the same tenacity
  commit, unmodified, under the Apache License 2.0 in its own
  `LICENSE-tenacity.txt`.
- The Java demonstration holds
  `src/main/java/org/skyscreamer/jsonassert/JSONCompare.java`, from
  [JSONassert](https://github.com/skyscreamer/JSONassert) at commit
  `e81c16c59ce0860f97a65d871589ab2337370c4b`, unmodified, under the Apache
  License 2.0 in `LICENSE-jsonassert.txt`. JSONassert distributes no NOTICE
  file.

The `NOTICE.md` in each demonstration's data directory describes the rest of
its content. Testuary's MIT license does not relicense these files or the
recorded material.

## Dependencies and execution environments

The Python dependencies declared in `pyproject.toml` are installed
separately by the package installer, not bundled. They retain their own
licenses and notices. In particular, Testuary uses Cosmic Ray, pytest, coverage.py, PuLP/CBC, LangGraph,
LangChain and tree-sitter through their installed interfaces.

Cosmic Ray requires qprompt, which requires
[iterfzf](https://github.com/dahlia/iterfzf) (GPL-3.0-or-later). Cosmic Ray
also requires [yattag](https://www.yattag.org) (LGPL). Installing Testuary
installs them as well. Testuary does not include, modify or import them,
and its MIT license does not apply to them.

Container images, target repository checkouts, model services and dependency
wheelhouses are not bundled in the source or wheel distributions. Their
respective licenses and terms apply independently. The project MIT license
does not grant rights to redistribute those external artifacts.
