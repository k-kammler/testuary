"""Scenario entries, the oracle that answers calls, the stop, and the parts
of the calls' keys.

A scenario is the list of answers one run of the pipeline asks for: every
model response and every measurement result, each with its class (real for
an answer recorded in a run), its source and the rule or method that
produced it. A recorder (the repository's tools/record_demo.py, which also
writes the file) records it from one real run, and the demonstration
plays it back. Both go through an oracle: the recorder's asks the real model
and tools and records each answer, the playback's looks it up and stops when
it is missing.

Calls are keyed by what they measure, never by where it lies on disk: test
files by name and content hash, sources by hash, together with the options
that change the answer. The same bytes measured twice are two entries,
answered in the order they were recorded.

A model request is identified by the hashes of its messages, as the
transport records compute them, so a request that differs from the
scenario's is a call the scenario does not hold and stops the run.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.messages import BaseMessage

from testuary.agent import _langgraph as lg
from testuary import evidence as validity
from testuary import languages
from testuary.generation_evidence import source_sha256

SCHEMA_VERSION = 2
# A run's calls fall into segments that end where the next agent phase
# builds its model: the starter phase with its phase end, the coverage phase
# with its phase end, and the mutation phase with its phase end and
# Finishing. The segment names are recorded in the data.
SEGMENTS = ("starter", "coverage", "mutation")



class ScenarioStop(BaseException):
    """The package asked for an answer the scenario does not hold.

    A ``BaseException`` so that no phase converts it into a failed request, a
    retry or a recorded phase failure: those paths catch ``Exception``."""

    def __init__(self, call: "Call", reason: str) -> None:
        super().__init__(f"{call.describe()}: {reason}")
        self.call = call
        self.reason = reason


def sha256_bytes(data: bytes) -> str:
    """The SHA-256 of ``data`` as hexadecimal text."""
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    """``value`` as JSON text with sorted keys, without spaces between
    items and with its non-ASCII characters kept."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


# Machine paths in recorded answers. A recording replaces the paths of the
# machine it ran on with placeholders: the demonstration's run directory, the
# project's interpreter, its site-packages directory, its environment, the JDK,
# the temporary directory and the home directory. The playback puts the paths of the
# machine it runs on in their place, with that machine's separators, so a
# playback names only directories of the person running it. Model requests are
# keyed with the placeholders put back, so the keys do not depend on where a
# recording or a playback ran.

#: The placeholder of each machine path, most specific first.
PATH_PLACEHOLDERS = ("out", "site", "python", "env", "jdk", "tmp", "home")
_PATH_TOKEN = "{{testuary:{name}}}"
# Characters that continue a path after a directory: anything up to white
# space, a quote, a bracket or a separator of a list or a line number.
_PATH_REST = r"(?P<rest>(?:(?:\\\\|\\|/)[^\s\\/\"'`<>|*?:,;()\[\]{}=]*)*)"
_ACTIVE_PATHS: Optional["PathMap"] = None


def path_token(name: str) -> str:
    """The placeholder that stands for the machine path ``name``, one of
    ``PATH_PLACEHOLDERS``, in recorded answers."""
    return _PATH_TOKEN.format(name=name)


def _path_variants(path: str) -> List[str]:
    """The forms one path takes in text: as given, with forward slashes, and
    on Windows with doubled backslashes as a Python repr shows them."""
    forms = {path, path.replace("\\", "/")}
    if "\\" in path:
        forms.add(path.replace("\\", "\\\\"))
    return sorted((form for form in forms if form), key=len, reverse=True)


class PathMap:
    """The machine paths of one run under their placeholder names."""

    def __init__(self, paths: Mapping[str, Optional[str]]) -> None:
        self.paths: Dict[str, str] = {}
        for name in PATH_PLACEHOLDERS:
            value = paths.get(name)
            if value:
                self.paths[name] = os.path.normpath(str(value))
        pairs: List[Tuple[str, str]] = []
        for name, path in self.paths.items():
            for form in _path_variants(path):
                pairs.append((form, name))
        # Longest first, so a directory inside another is matched as itself.
        pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
        self._pairs = pairs
        self._neutral = (
            re.compile("(?:" + "|".join(f"(?P<p{i}>{re.escape(form)})"
                                        for i, (form, _) in enumerate(pairs)) + r")(?![\w\-])"
                       + _PATH_REST)
            if pairs else None)
        self._local = re.compile(
            "(?:" + "|".join(re.escape(path_token(name)) for name in PATH_PLACEHOLDERS) + ")"
            + _PATH_REST)

    def neutral(self, text: str) -> str:
        """``text`` with this machine's paths as placeholders, separators
        after a placeholder as forward slashes."""
        if self._neutral is None or not text:
            return text

        def replace(match: "re.Match[str]") -> str:
            for index, (_, name) in enumerate(self._pairs):
                if match.group(f"p{index}") is not None:
                    rest = match.group("rest").replace("\\\\", "/").replace("\\", "/")
                    return path_token(name) + rest
            return match.group(0)

        return self._neutral.sub(replace, text)

    def local(self, text: str) -> str:
        """``text`` with placeholders as this machine's paths, separators
        after them as this machine's."""
        if not text or "{testuary:" not in text:
            return text

        def replace(match: "re.Match[str]") -> str:
            whole = match.group(0)
            name = whole[len("{testuary:"):whole.index("}")]
            base = self.paths.get(name, path_token(name))
            rest = match.group("rest")
            if os.sep != "/":
                rest = rest.replace("/", os.sep)
            return base + rest

        return self._local.sub(replace, text)


def _walk_strings(value: Any, change) -> Any:
    if isinstance(value, str):
        return change(value)
    if isinstance(value, dict):
        return {key: _walk_strings(item, change) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_walk_strings(item, change) for item in value]
    return value


def install_paths(paths: Optional[PathMap]) -> Optional[PathMap]:
    """Make ``paths`` the run's path map and return the one it replaced."""
    global _ACTIVE_PATHS
    previous, _ACTIVE_PATHS = _ACTIVE_PATHS, paths
    return previous


def active_paths() -> Optional[PathMap]:
    """The run's path map that ``install_paths`` made current, or None."""
    return _ACTIVE_PATHS


def neutral_paths(value: Any) -> Any:
    """``value`` with the active run's paths as placeholders, in every
    string it holds."""
    paths = _ACTIVE_PATHS
    return value if paths is None else _walk_strings(value, paths.neutral)


def local_paths(value: Any) -> Any:
    """``value`` with placeholders as the active run's paths, in every
    string it holds."""
    paths = _ACTIVE_PATHS
    return value if paths is None else _walk_strings(value, paths.local)


@dataclass(frozen=True)
class Call:
    """One question the pipeline asks: its kind, the key that identifies what
    is measured or requested, and context the builder derives from (paths,
    the raw inputs). Only ``kind`` and ``key`` take part in matching."""

    kind: str
    key: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def key_text(self) -> str:
        return canonical(self.key)

    def describe(self) -> str:
        return f"{self.kind} {self.key_text()[:400]}"


@dataclass
class Entry:
    """One recorded answer: its class, source and rule, and for an answer
    whose fields differ in class (a synthetic reply with its recorded
    usage), the class, source and rule of each such field in ``parts``."""

    segment: str
    kind: str
    key: Dict[str, Any]
    answer: Any
    value_class: str
    source: str
    rule: str
    parts: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def as_json(self) -> Dict[str, Any]:
        data = {
            "segment": self.segment, "kind": self.kind, "key": self.key,
            "answer": self.answer, "class": self.value_class,
            "source": self.source, "rule": self.rule,
        }
        if self.parts:
            data["parts"] = self.parts
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Entry":
        return cls(
            segment=str(data["segment"]), kind=str(data["kind"]),
            key=dict(data["key"]), answer=data["answer"],
            value_class=str(data["class"]), source=str(data["source"]),
            rule=str(data["rule"]),
            parts={str(k): dict(v) for k, v in (data.get("parts") or {}).items()},
        )


@dataclass
class Scenario:
    """The entries of one run and what they were built from."""

    entries: List[Entry]
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Scenario":
        with gzip.open(path, "rb") as handle:
            document = json.loads(handle.read().decode("utf-8"))
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"scenario schema {document.get('schema_version')!r} is not "
                f"{SCHEMA_VERSION}"
            )
        blobs = document.get("blobs") or {}

        def unpack(value: Any) -> Any:
            if isinstance(value, dict):
                if set(value) == {"$blob"}:
                    return blobs[value["$blob"]]
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value

        entries = [Entry.from_json(unpack(item)) for item in document["entries"]]
        return cls(entries=entries, meta=dict(document.get("meta") or {}))


class Oracle:
    """Answers the pipeline's calls and tracks the segment the run is in.

    ``on_answer`` receives every answered entry, in the order answered, for
    the ledger and the console."""

    def __init__(self, *, on_answer: Optional[Callable[[Entry], None]] = None):
        self._lock = threading.RLock()
        self._segment = SEGMENTS[0]
        self._on_answer = on_answer

    @property
    def segment(self) -> str:
        return self._segment

    def enter_segment(self, name: str) -> None:
        if name not in SEGMENTS:
            raise ValueError(f"unknown segment {name!r}")
        with self._lock:
            if name == self._segment:
                return
            if SEGMENTS.index(name) < SEGMENTS.index(self._segment):
                raise ScenarioStop(
                    Call("segment", {"from": self._segment, "to": name}),
                    "the run went back to an earlier phase",
                )
            # A segment the run passed over is closed too, so the answers of
            # a skipped phase are checked.
            for passed in SEGMENTS[SEGMENTS.index(self._segment):SEGMENTS.index(name)]:
                self._close_segment(passed)
            self._segment = name

    def finish(self) -> None:
        """The run is over: its segment and every later one are closed."""
        with self._lock:
            for remaining in SEGMENTS[SEGMENTS.index(self._segment):]:
                self._close_segment(remaining)

    def answer(self, call: Call) -> Any:
        entry = self._answer_entry(call)
        if self._on_answer is not None:
            self._on_answer(entry)
        return self._returned(entry)

    def _returned(self, entry: Entry) -> Any:
        """What the pipeline receives for ``entry``."""
        return entry.answer

    # Subclasses supply these two.
    def _answer_entry(self, call: Call) -> Entry:
        raise NotImplementedError

    def _close_segment(self, segment: str) -> None:
        raise NotImplementedError


class PlaybackOracle(Oracle):
    """Answers only what the scenario holds, for the segment the run is in,
    in the order the scenario holds it. A call it cannot answer, and a
    segment that ends with answers unused, stop the run."""

    def __init__(self, scenario: Scenario, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._queues: Dict[Tuple[str, str, str], Deque[Entry]] = {}
        for entry in scenario.entries:
            key = (entry.segment, entry.kind, canonical(entry.key))
            self._queues.setdefault(key, deque()).append(entry)

    def _answer_entry(self, call: Call) -> Entry:
        with self._lock:
            queue = self._queues.get(
                (self._segment, call.kind, call.key_text()))
            if not queue:
                raise ScenarioStop(
                    call,
                    f"the scenario holds no answer for this call in the "
                    f"{self._segment} segment. It must be regenerated for "
                    f"this package version",
                )
            return queue.popleft()

    def _returned(self, entry: Entry) -> Any:
        # A recorded answer names machine paths as placeholders: the
        # pipeline receives the paths of this playback in their place.
        return local_paths(entry.answer)

    def _close_segment(self, segment: str) -> None:
        unused = [
            (kind, key_text)
            for (seg, kind, key_text), queue in self._queues.items()
            if seg == segment and queue
        ]
        if unused:
            kind, key_text = sorted(unused)[0]
            raise ScenarioStop(
                Call(kind, json.loads(key_text)),
                f"{len(unused)} answer(s) of the {segment} segment were "
                f"never asked for. The scenario must be regenerated for this "
                f"package version",
            )


def _neutral_bytes(data: bytes) -> bytes:
    """``data`` with the run's machine paths as placeholders, when the run
    has a path map and the data are text."""
    active = active_paths()
    if active is None:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return active.neutral(text).encode("utf-8")


def content_sha256(path: Path) -> str:
    """The hash of a file's content with CRLF read as LF and machine paths as
    placeholders. The package writes text files in text mode, which ends lines
    with CRLF on Windows and LF elsewhere, and hashes the text it wrote.
    Hashing the normalized bytes keeps a scenario the same on every platform
    and in every directory."""
    return sha256_bytes(_neutral_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n")))


def _collected(name: str, language: str = "python") -> bool:
    """Whether a run of ``language``'s tests in a directory measures the file
    ``name``: Java's every source (helper classes compile with the tests),
    pytest's setup files, and any language's test files by its backend's own
    rule."""
    if language == "java":
        return name.endswith(".java")
    if language == "python" and name in ("conftest.py", "pytest.ini"):
        return True
    return languages.get_backend(language).is_test_file(Path(name))


def tree_files(root: Path, *, language: str = "python") -> List[List[str]]:
    """The files a run of the directory ``root`` measures, as ``[relative
    posix path, sha256]``, sorted. Other files the package keeps beside the
    tests (sidecars, backups) take no part in the run and are left out."""
    out: List[List[str]] = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or not _collected(path.name, language):
            continue
        rel = path.relative_to(root)
        if any(part in ("__pycache__", ".pytest_cache") for part in rel.parts):
            continue
        out.append([rel.as_posix(), content_sha256(path)])
    return out


# Fields a kill matrix carries about when it was captured, not what it holds.
_CAPTURE_TIME_FIELDS = ("t_mono_s", "clock_domain_id")


def kill_matrix_entry(path: Path) -> List[str]:
    """A kill matrix file as ``[name, sha256]`` of its content without the
    capture time, so the same matrix captured twice has the same key."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if k not in _CAPTURE_TIME_FIELDS}
    return [Path(path).name, source_sha256(canonical(neutral_paths(data)))]


def file_entry(path: Path) -> List[str]:
    """A file as ``[name, content sha256]``."""
    return [Path(path).name, content_sha256(Path(path))]


def suite_files(test_paths: Iterable[str], *, language: str = "python") -> Dict[str, Any]:
    """The measured bytes of a run over explicit test paths: each path (a
    file, or a directory the runner collects), with Python's adjacent setup
    files where applicable."""
    files: List[Any] = []
    setup: set = set()
    for raw in test_paths:
        path = Path(raw)
        if path.is_dir():
            files.append(["dir", tree_files(path, language=language)])
            continue
        files.append(file_entry(path))
        for name in (() if language == "java" else ("conftest.py", "pytest.ini")):
            sibling = path.parent / name
            if sibling.is_file() and sibling != path:
                setup.add(tuple(file_entry(sibling)))
    return {"files": files, "setup": [list(item) for item in sorted(setup)]}


def message_hashes(messages: Sequence[Mapping[str, Any]]) -> List[str]:
    """The transport record's hashes of outbound message dicts, with the
    machine paths a request names as placeholders, so the
    key is the same wherever the run took place."""
    record = validity.prompt_transport_record(
        phase="", request_idx=0, messages=neutral_paths(list(messages)),
        prompt_template_version="", provider_transport="",
    )
    return list(record["message_sha256"])


def agent_message_hashes(messages: Sequence[BaseMessage]) -> List[str]:
    """The transport record's hashes of an agent request's messages."""
    return message_hashes(
        [lg._message_transport_payload(message) for message in messages]
    )
