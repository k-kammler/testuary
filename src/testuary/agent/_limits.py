"""Limits on what the model sees. The packaged config.ini's ``[limits]``
section names each limit and sets its default.

Every cut of model-facing text goes through ``cut_text`` or ``cut_items``
with one marker, ``[... N <unit> of <what> omitted ...]``. Each limit is a
key of the ``[limits]`` configuration section, overridable per phase in
``[limits_coverage]`` and ``[limits_mutation]`` (the mutation phase applies
the coverage section first) and per model in a ``models.ini`` alias or
provider entry. A value is a positive integer or ``unlimited``.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

UNLIMITED = "unlimited"


@functools.lru_cache(maxsize=None)
def _packaged() -> Mapping[str, str]:
    """Each limit's default as the packaged config.ini writes it, in its order."""
    from testuary import settings

    return settings._read_ini_file(settings._bundled_data_path("config.ini"))["limits"]


#: The sections each phase reads, lowest first.
PHASE_SECTIONS: Mapping[Optional[str], Tuple[str, ...]] = {
    None: ("limits",),
    "starter": ("limits",),
    "coverage": ("limits", "limits_coverage"),
    "mutation": ("limits", "limits_coverage", "limits_mutation"),
}

_PHASE: contextvars.ContextVar = contextvars.ContextVar("testuary_limits_phase", default=None)


def parse(name: str, value: Any, *, source: str = "settings") -> str:
    """Normalize one limit value: a positive integer or ``unlimited``. Zero is
    refused, so a limit never hides an output by accident."""
    label = f"{source} {name}"
    if name not in _packaged():
        raise ValueError(f"{label}: not a limit")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label}: expected a positive integer or {UNLIMITED}")
    text = str(value).strip().lower()
    if text == UNLIMITED:
        return UNLIMITED
    try:
        number = int(text)
    except ValueError:
        raise ValueError(f"{label}: expected a positive integer or {UNLIMITED}") from None
    if number <= 0:
        raise ValueError(f"{label}: expected a positive integer, or {UNLIMITED} for no limit")
    return str(number)


def _as_limit(text: str) -> Optional[int]:
    return None if text == UNLIMITED else int(text)


def resolve(settings: Any, phase: Optional[str] = None) -> dict[str, Optional[int]]:
    """Each limit's value for ``phase``, None for unlimited. Layers apply in
    input order: the packaged defaults, then the selected model's provider
    and alias entries in ``models.ini``, then each configuration layer. In
    one layer a phase section wins over ``[limits]``."""
    from testuary import models

    sections = PHASE_SECTIONS.get(phase, ("limits",))
    captured = settings.configuration_layers()
    values = {name: UNLIMITED if number is None else str(number)
              for name, number in defaults().items()}
    for section in sections:
        values.update(captured[0][1].get(section, {}))
    try:
        identity = models.resolve_model_identity(settings)
    except Exception:
        identity = None
    if identity is not None:
        for entry in (settings.models.get(identity.provider, {}),
                      settings.models.get(identity.model_alias, {})):
            values.update({key: parse(key, value, source="model catalogue")
                           for key, value in entry.items() if key in _packaged()})
    for _, layer in captured[1:]:
        for section in sections:
            values.update(layer.get(section, {}))
    return {name: _as_limit(value) for name, value in values.items()}


def defaults() -> dict[str, Optional[int]]:
    """Each limit's packaged default, None for unlimited."""
    return {name: _as_limit(parse(name, text, source="config.ini"))
            for name, text in _packaged().items()}


@contextlib.contextmanager
def phase_scope(phase: Optional[str]) -> Iterator[None]:
    """Make ``phase`` the phase whose limits ``value`` reads, in this thread
    and in the tool threads it starts."""
    token = _PHASE.set(phase)
    try:
        yield
    finally:
        _PHASE.reset(token)


def in_phase(phase: str):
    """Decorate a phase entry point so everything it renders for the model
    reads that phase's limits."""
    import functools

    def wrap(function):
        @functools.wraps(function)
        def scoped(*args, **kwargs):
            with phase_scope(phase):
                return function(*args, **kwargs)
        return scoped
    return wrap


def recorded(settings: Any, phase: Optional[str]) -> dict[str, str]:
    """Every resolved limit of ``phase``, as a generation condition records
    it. All of them, so a resume on a release with other defaults is
    refused as well."""
    return {name: (UNLIMITED if number is None else str(number))
            for name, number in resolve(settings, phase).items()}


def export(phase: Optional[str]) -> dict[str, Any]:
    """Every limit of ``phase`` for the resolved-settings record, unlimited
    written as such."""
    from testuary import config

    settings = config.active_settings()
    resolved = defaults() if settings is None else resolve(settings, phase)
    return {name: (UNLIMITED if number is None else number) for name, number in resolved.items()}


def value(name: str) -> Optional[int]:
    """The current value of one limit, None for unlimited: from the active run
    settings and the phase in scope, else the packaged default."""
    from testuary import config

    if name not in _packaged():
        raise KeyError(name)
    settings = config.active_settings()
    if settings is None:
        return defaults()[name]
    return resolve(settings, _PHASE.get())[name]


def marker(count: int, unit: str, what: str, more: str = "") -> str:
    """The one marker of an omission: ``[... 3 chars of the line omitted ...]``,
    or for counted things ``[... 3 test names omitted ...]``. ``more`` says how
    to see the rest, only where a tool can show it."""
    tail = f", {more}" if more else ""
    if count == 1:
        # One item of a plural noun, or one char or line.
        if unit == "items" and what.endswith("s"):
            what = what[:-1]
        unit = unit[:-1]
    counted = what if unit in ("items", "item") else f"{unit} of {what}"
    return f"[... {count} {counted} omitted{tail} ...]"


def cut_text(text: str, limit: Optional[int], *, what: str, keep: str = "head_tail",
             unit: str = "chars", sep: str = "\n") -> str:
    """``text`` within ``limit`` characters or lines, the omission marked.
    ``keep`` is ``head``, ``tail`` or ``head_tail`` (a third from the start,
    the rest from the end). ``sep`` joins text and marker, a space for text
    shown inline. None is unlimited."""
    if limit is None or not text:
        return text
    if unit == "lines":
        parts = text.splitlines(keepends=True)
        if len(parts) <= limit:
            return text
        omitted = len(parts) - limit
        note = marker(omitted, "lines", what) + "\n"
        if keep == "head":
            return "".join(parts[:limit]) + note
        if keep == "tail":
            return note + "".join(parts[-limit:])
        head = limit // 3
        return "".join(parts[:head]) + note + "".join(parts[len(parts) - (limit - head):])
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    note = marker(omitted, "chars", what)
    if keep == "head":
        return text[:limit] + sep + note
    if keep == "tail":
        return note + sep + text[-limit:]
    head = limit // 3
    return text[:head] + sep + note + sep + text[len(text) - (limit - head):]


def cut_items(items: Sequence[Any], limit: Optional[int], *, what: str) -> Tuple[list, str]:
    """The first ``limit`` items and the marker for the rest, or "" when
    nothing is left out. ``what`` names the left-out items in the plural.
    None is unlimited."""
    items = list(items)
    if limit is None or len(items) <= limit:
        return items, ""
    return items[:limit], marker(len(items) - limit, "items", what)


def cut_lines(lines: Sequence[str], limit: Optional[int], *, what: str) -> list:
    """``lines`` within ``limit``, the marker as the last line when some are
    left out."""
    kept, note = cut_items(lines, limit, what=what)
    return kept + [note] if note else kept
