"""The extensions ``[main] extensions`` names, loaded through LangAdapter's
``load_extension``, which registers what LangAdapter reads in each module.
A run, the tool check, testuary provision and testuary-web load them at
their start, the quick check none, and the run's settings later read the
loaded modules without importing anything."""
from __future__ import annotations

import os
import sys
from typing import Any, Iterable

from langadapter import Extension, InvalidRequest, RegistrationConflict, load_extension

from testuary.settings import ConfigurationError

#: Each item a load took, by its item and directory, with its Extension and
#: module.
_TAKEN: dict[tuple[str, str], tuple[Extension, Any]] = {}

Row = tuple[str, str, str]


def place(source: str) -> str:
    """Where an item of the source ``source`` is named, for a refusal: the
    path of a file, the variable or the program's overrides."""
    return {"environment": "the variable TESTUARY_MAIN_EXTENSIONS",
            "overrides": "the program's overrides"}.get(source, source)


def load_item(item: str, source: str, directory: str) -> tuple[Extension, Any]:
    """The item ``item`` of ``extension_items`` loaded, with its module. An
    item that cannot be resolved or imported raises ConfigurationError
    naming the item, the reason and where it is named."""
    try:
        extension = load_extension(item, relative_to=directory)
    except (InvalidRequest, RegistrationConflict) as exc:
        reason = exc.message if isinstance(exc, InvalidRequest) else f"{item}: {exc}"
        raise ConfigurationError(f"[main] extensions: {reason}. It is named in "
                                 f"{place(source)}") from None
    _TAKEN[(item, directory)] = extension, sys.modules[extension.module]
    return _TAKEN[(item, directory)]


def load(rows: Iterable[Row]) -> tuple[tuple[str, str, Extension, Any], ...]:
    """The rows of ``extension_items`` loaded through ``load_item``, as
    ``loaded`` returns them. The first item that cannot load raises."""
    rows = tuple(rows)
    for row in rows:
        load_item(*row)
    return loaded(rows)


def loaded(rows: Iterable[Row]) -> tuple[tuple[str, str, Extension, Any], ...]:
    """The rows a load took, without importing, as ``(item, source,
    extension, module)`` in their order. A resolved file named again is
    listed once, at its first place."""
    found, seen = [], set()
    for item, source, directory in rows:
        taken = _TAKEN.get((item, directory))
        if taken is not None and os.path.normcase(taken[0].file) not in seen:
            seen.add(os.path.normcase(taken[0].file))
            found.append((item, source, *taken))
    return tuple(found)
