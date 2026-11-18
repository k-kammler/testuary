"""The hints of the extensions the settings name, for other programs.

An extension, a Python file or an installed module that ``[main]
extensions`` names, may define ``testuary_hints``, a mapping in the format
of a project's hints file. ``extension_hints`` loads the extensions and
captures each mapping now, as a run does at its start, and ``request_text``
gives the guidance one extension's hints or a hints file add to a phase's
model request. A run puts the extensions' guidance first, in the order of
the combined list, then the project's hints file.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from testuary import _extensions, config
from testuary._phase_keys import PHASES
from testuary.settings import extension_items

if TYPE_CHECKING:
    from testuary.settings import RunSettings

__all__ = ["PHASES", "ExtensionHints", "extension_hints", "request_text"]


@dataclass(frozen=True)
class ExtensionHints:
    """An extension's hints as one command captured them: the item that
    names the extension, the extension's resolved file and its
    ``testuary_hints`` as JSON bytes."""

    item: str
    file: str
    contents: bytes


def extension_hints(settings: RunSettings) -> Tuple[ExtensionHints, ...]:
    """The hints of the extensions ``settings`` name, in the combined list's
    order, captured now. Each mapping is written once as JSON and validated
    as a hints file. An extension that cannot load, a mapping JSON cannot
    hold and invalid hints raise ``testuary.settings.ConfigurationError``
    naming the item and where it is named."""
    return _captured(_extensions.load(extension_items(settings)))


def _read_extension_hints(settings: RunSettings) -> Tuple[ExtensionHints, ...]:
    """The hints of the extensions ``settings`` name that the command loaded
    at its start, captured as ``extension_hints`` captures them, with
    nothing imported: the quick check and testuary-web's pages also enter a
    run's settings."""
    return _captured(_extensions.loaded(extension_items(settings)))


def _plain(value: Any) -> dict:
    """A mapping that is not a dict, such as a read-only view, as JSON
    writes a dict. Any other value JSON cannot hold is refused."""
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _captured(loaded) -> Tuple[ExtensionHints, ...]:
    """The ``testuary_hints`` of the ``loaded`` extensions that define them."""
    from testuary.settings import ConfigurationError

    found = []
    for item, source, extension, module in loaded:
        value = getattr(module, "testuary_hints", None)
        if value is None:
            continue
        named = f"It is named in {_extensions.place(source)}"
        try:
            contents = json.dumps(value, indent=2, allow_nan=False, default=_plain).encode("ascii")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ConfigurationError(f"[main] extensions: {item}: testuary_hints is not JSON: "
                                     f"{exc}. {named}") from None
        problems = validate_hints(json.loads(contents), label=f"{item} testuary_hints")
        if problems:
            raise ConfigurationError("invalid hints:\n" + "\n".join(problems) + f"\n{named}")
        found.append(ExtensionHints(item, extension.file, contents))
    return tuple(found)


def request_text(hints: Union[ExtensionHints, Mapping[str, Any]], language: str, phase: str,
                 *, hint_profile: str = "") -> str:
    """The guidance ``hints`` adds to ``phase``'s model request for
    ``language``, empty when it adds none. ``hints`` is an extension's
    hints from ``extension_hints`` or a hints file's parsed content.
    ``phase`` is initial, coverage or mutation, and ``hint_profile`` selects
    the hints file's profile cells as ``[project] hint_profile`` does."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}: one of {', '.join(PHASES)}")
    if isinstance(hints, ExtensionHints):
        hints = json.loads(hints.contents)
    return _resolved_hints(hints, hint_profile, language, phase)


def _hint_string(value: Any) -> str:
    """A cell as hint text: ``None``, ``[]`` and ``""`` give ``""``, and a
    list of strings is joined by newlines, so a cell can be one string or a
    list of lines."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(line) for line in value).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def hints_filename() -> str:
    """The run's hints file name (``[main] hints_file``), relative to the project
    directory. Empty selects none. The bundled starter is never read
    in its place."""
    return str(config.load_config("main").get("hints_file", "hints.json")).strip()


def load_hints() -> Dict[str, Any]:
    """Return detached hints captured at invocation entry, else the run's
    file read and validated now. Only an absent optional default or an
    explicitly empty selection means no hints. Bad input refuses."""
    run = config.current_run()
    return json.loads(run.hints if run.hints is not None else _read_hint_input())


def _read_hint_input() -> bytes:
    from testuary.settings import ConfigurationError

    filename = hints_filename()
    if not filename:
        return b"{}"
    selected = ("main", "hints_file") in config.run_settings().supplied or filename != "hints.json"
    path = config.configuration_base_dir() / filename
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        if selected:
            raise ConfigurationError(f"selected hints file is absent: {filename}") from None
        return b"{}"
    except OSError:
        raise ConfigurationError(f"cannot read hints file: {filename}") from None
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeError):
        raise ConfigurationError(f"{filename}: hints file must contain valid JSON") from None
    problems = validate_hints(payload)
    if problems:
        raise ConfigurationError("invalid hints:\n" + "\n".join(problems))
    config.current_run().hints_path = path
    return content


def hint_profile() -> str:
    """The hints file's profile this project uses (``[project] hint_profile``).
    Empty uses none."""
    return config.get_project_config().get("hint_profile", "")


# The four cells a phase block may hold (see resolve_hints). Any other key
# inside a phase block is a typo no resolver reads.
HINT_CELLS = (
    "general", "per_profile", "per_language", "per_profile_per_language",
)


def validate_hints(hints: Dict[str, Any], *, label: Optional[str] = None) -> List[str]:
    """Return all input problems, naming cells but not their contents.

    Keys beginning with an underscore are notes. The loader refuses any
    reported problem before work starts. Valid cell text renders unchanged.
    ``label`` names the file in the problems (the user's hints file by
    default).
    """
    out: List[str] = []
    name = label or hints_filename()
    if not isinstance(hints, dict):
        return [f"{name}: hints file must contain a JSON object"]
    valid_phases = ", ".join(PHASES)
    valid_cells = ", ".join(HINT_CELLS)
    not_text = "is not text (a string or a list of strings)"

    def text(value):
        return isinstance(value, str) or (isinstance(value, list) and all(isinstance(line, str) for line in value))

    for phase, block in hints.items():
        if isinstance(phase, str) and phase.startswith("_"):
            continue
        if phase not in PHASES:
            out.append(
                f"{name}: unknown phase {phase!r} "
                f"(valid phases: {valid_phases})"
            )
            continue
        if not isinstance(block, dict):
            out.append(
                f"{name}: phase {phase!r} must be an object of cells "
                f"({valid_cells})"
            )
            continue
        for cell, value in block.items():
            if isinstance(cell, str) and cell.startswith("_"):
                continue
            if cell not in HINT_CELLS:
                out.append(
                    f"{name}: unknown cell {phase}.{cell!r} "
                    f"(valid cells: {valid_cells})"
                )
            elif cell == "general":
                if not text(value):
                    out.append(f"{name}: {phase}.general {not_text}")
            elif not isinstance(value, dict):
                out.append(f"{name}: {phase}.{cell} must be an object")
            elif cell == "per_profile_per_language":
                for profile, languages in value.items():
                    if not isinstance(languages, dict):
                        out.append(f"{name}: {phase}.{cell}[{profile}] must "
                                   f"be an object of languages")
                        continue
                    for language, content in languages.items():
                        if not text(content):
                            out.append(f"{name}: {phase}.{cell}[{profile}]"
                                       f"[{language}] {not_text}")
            else:
                for key, content in value.items():
                    if not text(content):
                        out.append(f"{name}: {phase}.{cell}[{key}] {not_text}")
    return out


def resolve_hints(
    hint_profile: Optional[str],
    language: Optional[str],
    *,
    phase: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
) -> str:
    """The guidance of the phase block ``phase`` for ``(hint_profile,
    language)``: its non-empty cells ``general``,
    ``per_profile[hint_profile]``, ``per_language[language]`` and
    ``per_profile_per_language[hint_profile][language]``, in that order,
    joined by a blank line. ``""`` when all are empty, so the caller leaves
    out the ``## Additional guidance`` heading, and when ``phase`` is None.

    ``hints`` is for tests. Otherwise the extensions' hints the run captured
    come first, in the combined list's order, each walked the same way, then
    the configured file. An empty ``hint_profile`` or ``language`` skips its
    cells.
    """
    if hints is not None:
        return _resolved_hints(hints, hint_profile, language, phase)
    blocks = [_resolved_hints(json.loads(item.contents), hint_profile, language, phase)
              for item in _captured_extension_hints()]
    blocks.append(_resolved_hints(load_hints(), hint_profile, language, phase))
    return "\n\n".join(block for block in blocks if block)


def _resolved_hints(h: Any, hint_profile: Optional[str], language: Optional[str],
                    phase: Optional[str]) -> str:
    """One hints file's four cells for the phase, as ``resolve_hints``
    describes."""
    if not isinstance(h, dict) or not phase:
        return ""
    phase_block = h.get(phase)
    if not isinstance(phase_block, dict):
        return ""
    parts: List[str] = []

    general = _hint_string(phase_block.get("general"))
    if general:
        parts.append(general)

    if hint_profile:
        per_profile = phase_block.get("per_profile") or {}
        if isinstance(per_profile, dict):
            profile_hint = _hint_string(per_profile.get(hint_profile))
            if profile_hint:
                parts.append(profile_hint)

    if language:
        per_language = phase_block.get("per_language") or {}
        if isinstance(per_language, dict):
            language_hint = _hint_string(per_language.get(language))
            if language_hint:
                parts.append(language_hint)

    if hint_profile and language:
        per_pl = phase_block.get("per_profile_per_language") or {}
        if isinstance(per_pl, dict):
            profile_section = per_pl.get(hint_profile) or {}
            if isinstance(profile_section, dict):
                cross_hint = _hint_string(profile_section.get(language))
                if cross_hint:
                    parts.append(cross_hint)

    return "\n\n".join(parts)


def _captured_extension_hints() -> Tuple[ExtensionHints, ...]:
    """The extensions' hints the active run captured at its start, none
    outside a run."""
    run = config.active_run()
    return run.extension_hints if run is not None else ()


def captured_hints_path() -> Optional[Path]:
    """The hints file the active run captured at its start, or None outside a
    run or when it names none."""
    run = config.active_run()
    return run.hints_path if run is not None else None
