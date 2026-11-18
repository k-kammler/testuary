"""Versioned records of how test suites were generated: the paid model
replies, each starter suite slot's attempts, the cleaned sources and each
selected suite's construction lineage. Measurements, scores and request
accounting are recorded elsewhere."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import uuid


class GenerationEvidenceError(ValueError):
    """Generation evidence is missing, contradictory or cannot be resolved."""


_STATES = {"pending", "active", "accepted", "rejected", "unavailable"}
_BASES = {"accepted_cleaned", "no_code", "judged_rejection", "unavailable"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"[0-9a-f]{32}\Z")


def source_sha256(source: str) -> str:
    """The SHA-256 hex digest of the text's UTF-8 encoding, not of file bytes.
    A lone surrogate, which a model can send, is hashed by its own code unit
    (surrogatepass), so a text without one keeps its hash and no text hashes
    like another that holds the surrogate's escape."""
    return hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()


def _copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise GenerationEvidenceError("generation evidence is not finite acyclic JSON") from None


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def paid_attempt(delivery_id: str | None) -> dict | None:
    """The run, model request index and ID of a delivery ID of the form
    ``<run>:<model request index>``, or None for any other ID."""
    if not isinstance(delivery_id, str):
        return None
    run, separator, index = delivery_id.partition(":")
    if not separator or not _RUN.fullmatch(run) or not index.isdecimal():
        return None
    try:
        number = int(index)
    except ValueError:
        return None
    if str(number) != index:
        return None
    return {"run": run, "logical_request_idx": number, "delivery_id": delivery_id}


def _check_attempt(value):
    if (not isinstance(value, dict) or not _integer(value.get("logical_request_idx"))
            or paid_attempt(value.get("delivery_id")) != value):
        raise GenerationEvidenceError("unavailable paid attempt identity")


def read_paid_inputs(delivery):
    """The ``origin`` and ``run_idx`` an unprocessed paid reply recorded for its
    construction, both None when missing, invalid or not of the starter phase."""
    unknown = {"origin": None, "run_idx": None}
    try:
        item = delivery.get("construction_inputs")
        if (not isinstance(item, dict) or type(item.get("version")) is not int or item["version"] != 1
                or item.get("attempt_id") != paid_attempt(delivery.get("id"))
                or item.get("attempt_id") is None or not _integer(item.get("run_idx"), minimum=0)):
            return unknown
        _check_attempt(item["attempt_id"])
        origin = item.get("origin")
        _check_origin(origin)
        if origin is not None and origin["condition"]["phase"] != "starter":
            return unknown
        return {"origin": _copy(origin), "run_idx": item["run_idx"]}
    except (ValueError, AttributeError, TypeError, RecursionError):
        return unknown


def _reference_parts(reference: dict) -> tuple[str, ...]:
    path = reference.get("path")
    if reference.get("base") != "starter_suites" or not isinstance(path, str):
        raise GenerationEvidenceError("unknown source reference base")
    parsed = PurePosixPath(path)
    if (not path or "\\" in path or ":" in path or parsed.is_absolute()
            or any(part in {".", "..", ""} for part in path.split("/"))):
        raise GenerationEvidenceError("unsafe source reference")
    return parsed.parts


def _reference_path(root: Path, reference: dict) -> Path:
    parts = _reference_parts(reference)
    base = root.resolve()
    target = (base / Path(*parts)).resolve()
    if not target.is_relative_to(base):
        raise GenerationEvidenceError("source reference escapes its base")
    return target


def _check_reference(reference, *, slot, identity, suites_root=None):
    if (not isinstance(reference, dict) or not _integer(reference.get("slot"), minimum=1)
            or reference["slot"] != slot
            or reference.get("attempt_id") != identity
            or not isinstance(reference.get("source_sha256"), str)
            or not _SHA.fullmatch(reference["source_sha256"])):
        raise GenerationEvidenceError("cleaned source is not bound to its paid slot")
    _reference_parts(reference)
    if suites_root is not None:
        read_cleaned_source(reference, suites_root)


def read_cleaned_source(reference: dict, suites_root: Path | str) -> str:
    """The cleaned source a reference names, checked against its recorded
    hash, so a file overwritten since raises."""
    if not isinstance(reference, dict) or not _SHA.fullmatch(str(reference.get("source_sha256", ""))):
        raise GenerationEvidenceError("invalid cleaned source identity")
    _check_attempt(reference.get("attempt_id"))
    if not _integer(reference.get("slot"), minimum=1):
        raise GenerationEvidenceError("invalid cleaned source slot")
    path = _reference_path(Path(suites_root), reference)
    try:
        source = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError):
        raise GenerationEvidenceError("cleaned source is unavailable") from None
    if source_sha256(source) != reference["source_sha256"]:
        raise GenerationEvidenceError("cleaned source identity changed")
    return source


def read_suite_slots(manifest: dict, suites_root: Path | str | None = None) -> dict:
    """The manifest's ``suite_slots`` section, checked, as a detached copy.
    With ``suites_root`` the accepted references are verified too. A missing
    or unknown section raises: a disposition is never inferred from counts
    or original.json."""
    section = manifest.get("suite_slots") if isinstance(manifest, dict) else None
    if (not isinstance(section, dict) or type(section.get("version")) is not int
            or section["version"] != 1):
        raise GenerationEvidenceError("suite slot evidence version is unavailable")
    count = section.get("requested_slots")
    slots = section.get("slots")
    if not _integer(count, minimum=1) or not isinstance(slots, list) or len(slots) != count:
        raise GenerationEvidenceError("invalid requested slot population")
    seen, paid = set(), set()
    for slot in slots:
        if not isinstance(slot, dict):
            raise GenerationEvidenceError("invalid suite slot")
        number = slot.get("slot")
        if not _integer(number, minimum=1) or number > count or number in seen:
            raise GenerationEvidenceError("duplicate or out-of-range suite slot")
        seen.add(number)
        state, basis = slot.get("state"), slot.get("disposition_basis")
        if (not isinstance(state, str) or state not in _STATES
                or not isinstance(basis, str) or basis not in _BASES or type(slot.get("terminal")) is not bool
                or not _integer(slot.get("attempts_used"))):
            raise GenerationEvidenceError("invalid suite disposition")
        if ((state == "accepted") != (basis == "accepted_cleaned")
                or (state == "rejected") != (basis in {"no_code", "judged_rejection"})):
            raise GenerationEvidenceError("contradictory suite disposition")
        if state in {"pending", "active"} and slot["terminal"]:
            raise GenerationEvidenceError("unfinished suite marked terminal")
        attempts = slot.get("attempts")
        if not isinstance(attempts, list) or len(attempts) > slot["attempts_used"]:
            raise GenerationEvidenceError("invalid processed attempts")
        keys = set()
        by_id = {}
        for attempt in attempts:
            _check_attempt(attempt.get("attempt_id") if isinstance(attempt, dict) else None)
            key = attempt["attempt_id"]["delivery_id"]
            attempt_basis = attempt.get("disposition_basis")
            if key in paid or not isinstance(attempt_basis, str) or attempt_basis not in _BASES:
                raise GenerationEvidenceError("invalid processed attempt disposition")
            keys.add(key)
            paid.add(key)
            by_id[key] = attempt
            original = attempt.get("original_source_sha256")
            if original is not None and (not isinstance(original, str) or not _SHA.fullmatch(original)):
                raise GenerationEvidenceError("invalid original source identity")
            accepted = attempt.get("accepted_source")
            _check_origin(attempt.get("generation_origin"))
            run_idx = attempt.get("generation_run_idx")
            if run_idx is not None and not _integer(run_idx, minimum=0):
                raise GenerationEvidenceError("invalid suite generation repeat")
            if "slot_attempt" in attempt and not _integer(attempt["slot_attempt"], minimum=1):
                raise GenerationEvidenceError("invalid slot_attempt number")
            if "seed" in attempt and type(attempt["seed"]) is not int:
                raise GenerationEvidenceError("invalid attempt seed")
            construction = attempt.get("construction")
            if construction is not None:
                if not isinstance(construction, dict) or attempt_basis != "accepted_cleaned":
                    raise GenerationEvidenceError("unaccepted attempt claims construction")
                read_ancestry({"generation_ancestry": construction,
                               "selected_source_sha256": (accepted or {}).get("source_sha256")})
                if construction.get("suite_attempt") != {"slot": number, "attempt_id": attempt["attempt_id"]}:
                    raise GenerationEvidenceError("suite construction has a different paid binding")
                if (construction.get("origin") != attempt.get("generation_origin")
                        or construction.get("run_idx") != run_idx):
                    raise GenerationEvidenceError("suite construction has different generation inputs")
            if attempt["disposition_basis"] == "accepted_cleaned":
                if original is None:
                    raise GenerationEvidenceError("accepted attempt has no original source identity")
                _check_reference(accepted, slot=number, identity=attempt["attempt_id"],
                                 suites_root=suites_root)
            elif accepted is not None:
                raise GenerationEvidenceError("unaccepted attempt claims cleaned source")
            if attempt["disposition_basis"] == "no_code" and original is not None:
                raise GenerationEvidenceError("no-code attempt claims extracted source")
        terminal = slot.get("terminal_attempt")
        if terminal is not None:
            _check_attempt(terminal)
            if terminal["delivery_id"] not in keys:
                raise GenerationEvidenceError("terminal attempt was not processed")
            current = by_id[terminal["delivery_id"]]
            if any(slot.get(name) != current.get(name) for name in (
                    "disposition_basis", "classification", "accepted_source")):
                raise GenerationEvidenceError("slot disagrees with its current attempt")
        elif basis != "unavailable":
            raise GenerationEvidenceError("judged slot has no paid attempt")
        reference = slot.get("accepted_source")
        if basis == "accepted_cleaned":
            _check_reference(reference, slot=number, identity=terminal, suites_root=suites_root)
        elif reference is not None:
            raise GenerationEvidenceError("unaccepted slot claims cleaned source")
        if "measurement_removed_source" in slot:
            if basis != "accepted_cleaned":
                raise GenerationEvidenceError("unaccepted slot claims a changed source")
            _check_reference(slot["measurement_removed_source"], slot=number, identity=terminal,
                             suites_root=suites_root)
    return _copy(section)


class SuiteSlots:
    """The starter suite slots of one generation and each attempt's
    disposition, written into the starter suites' manifest."""

    def __init__(self, suites_root: Path, count: int, *, previous: dict | None = None):
        if not _integer(count, minimum=1):
            raise GenerationEvidenceError("invalid requested slot population")
        self.root = suites_root
        self.count = count
        self.slots = [self._empty(i) for i in range(1, count + 1)]
        if previous is not None:
            try:
                section = read_suite_slots(previous, suites_root)
                if section["requested_slots"] != count:
                    raise GenerationEvidenceError("requested slot count changed")
            except GenerationEvidenceError:
                for slot in self.slots:
                    slot.update(state="unavailable", reason="prior slot evidence unavailable")
            else:
                self.slots = sorted(section["slots"], key=lambda item: item["slot"])

    @staticmethod
    def _empty(i):
        return {"slot": i, "state": "pending", "terminal": False,
                "attempts_used": 0, "disposition_basis": "unavailable",
                "classification": None, "reason": "not started",
                "terminal_attempt": None, "accepted_source": None, "attempts": []}

    def _slot(self, i):
        if not _integer(i, minimum=1) or i > self.count:
            raise GenerationEvidenceError("suite slot is outside requested population")
        return self.slots[i - 1]

    def unavailable(self, i, classification, reason, *, active=False):
        slot = self._slot(i)
        slot.update(
            state="active" if active else "unavailable", terminal=False,
            disposition_basis="unavailable", classification=classification,
            reason=reason, terminal_attempt=None, accepted_source=None,
        )
        slot.pop("measurement_removed_source", None)

    def _retain(self, i, identity, source, suffix):
        """A reference to ``source``, retained as exact UTF-8 text under slot
        ``i``'s ``accepted/`` beside attempt ``identity``'s other sources."""
        if not isinstance(source, str) or not re.fullmatch(r"\.[A-Za-z0-9]+", suffix):
            raise GenerationEvidenceError("accepted cleaned source is unavailable")
        digest = source_sha256(source)
        relative = (f"suite_{i:03d}/accepted/{identity['run']}/"
                    f"{identity['logical_request_idx']}/{digest}{suffix}")
        reference = {"base": "starter_suites", "path": relative, "source_sha256": digest,
                     "slot": i, "attempt_id": identity}
        path = _reference_path(self.root, reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if read_cleaned_source(reference, self.root) != source:
                raise GenerationEvidenceError("retained cleaned source changed")
        else:
            # Exact UTF-8 text, not host-dependent newline translation.
            with path.open("xb") as stream:
                stream.write(source.encode("utf-8"))
        return reference

    def record(self, i, delivery_id, *, classification, basis, original_source=None,
               cleaned_source=None, suffix=".py", reason="", origin=None, run_idx=None,
               slot_attempt=None, seed=None):
        """Record a processed attempt: ``slot_attempt`` is its number in the slot
        (naming its directory under ``attempts/`` once a later attempt replaces its
        files), ``seed`` its request's sampling seed. Recorded again, it keeps
        its origin and run index, and its number and seed unless given."""
        slot = self._slot(i)
        identity = paid_attempt(delivery_id)
        if identity is None or basis not in _BASES:
            self.unavailable(i, classification, "paid attempt identity or judgement unavailable")
            return
        reference = None
        if basis == "accepted_cleaned":
            if not isinstance(original_source, str):
                raise GenerationEvidenceError("accepted cleaned source is unavailable")
            reference = self._retain(i, identity, cleaned_source, suffix)
        attempt = {"attempt_id": identity, "classification": classification,
                   "disposition_basis": basis, "reason": reason,
                   "original_source_sha256": source_sha256(original_source) if original_source is not None else None,
                   "accepted_source": reference,
                   "generation_origin": _copy(origin), "generation_run_idx": run_idx}
        if slot_attempt is not None:
            attempt["slot_attempt"] = slot_attempt
        if seed is not None:
            attempt["seed"] = seed
        old = next((item for item in slot["attempts"] if item["attempt_id"] == identity), None)
        if old is None:
            slot["attempts"].append(attempt)
        else:
            origin, run_idx = old.get("generation_origin"), old.get("generation_run_idx")
            old.update(attempt)
            attempt = old
            attempt.update(generation_origin=origin, generation_run_idx=run_idx)
        if basis == "accepted_cleaned":
            attempt["construction"] = construction(
                cleaned_source, phase="starter", invocation=identity["run"],
                run_idx=run_idx, operation="accepted_suite", origin=origin,
                suite_attempt={"slot": i, "attempt_id": identity},
            )
        else:
            attempt.pop("construction", None)
        state = ("accepted" if basis == "accepted_cleaned" else
                 "rejected" if basis in {"no_code", "judged_rejection"} else "unavailable")
        slot.update(state=state, terminal=False, disposition_basis=basis,
                    classification=classification, reason=reason,
                    terminal_attempt=identity, accepted_source=reference)
        slot.pop("measurement_removed_source", None)

    def measurement_removed(self, i, source, *, suffix):
        """Record slot ``i``'s accepted suite after a measurement's removal of
        tests: ``source`` is retained under ``accepted/`` and named in the slot
        key ``measurement_removed_source``. The slot and its attempt keep their
        accepted reference and construction, and ``construction_for`` gives
        the changed suite a ``pruned`` node with the attempt's as parent. A
        slot that is not accepted, such as a reused suite whose evidence is
        unavailable, keeps no such record."""
        slot = self._slot(i)
        if slot["state"] == "accepted":
            slot["measurement_removed_source"] = self._retain(
                i, slot["accepted_source"]["attempt_id"], source, suffix)

    def withdraw(self, i, classification, reason):
        """Withdraw slot ``i``'s accepted suite, which a measurement could not
        measure: its terminal attempt keeps no accepted source, and the slot
        is free for another attempt."""
        slot = self._slot(i)
        for attempt in slot["attempts"]:
            if attempt["attempt_id"] == slot["terminal_attempt"]:
                attempt.update(classification=classification, disposition_basis="unavailable",
                               reason=reason, accepted_source=None)
                attempt.pop("construction", None)
        slot.update(state="unavailable", disposition_basis="unavailable",
                    classification=classification, reason=reason, accepted_source=None)
        slot.pop("measurement_removed_source", None)

    def revalidation_id(self, i, source):
        slot = self._slot(i)
        identity = slot.get("terminal_attempt")
        for attempt in slot["attempts"]:
            if (attempt["attempt_id"] == identity
                    and attempt["classification"] == "validation_unresolved"
                    and attempt["original_source_sha256"] == source_sha256(source)):
                return identity["delivery_id"]
        return None

    def reconcile_reused_sources(self, green):
        """Mark unavailable each accepted slot whose candidate in ``green`` is
        missing or differs from its accepted cleaned source, or from its
        ``measurement_removed_source`` where a removal changed the suite.
        Return whether any was."""
        changed = False
        for slot in self.slots:
            if slot["state"] == "accepted":
                previous = read_cleaned_source(
                    slot.get("measurement_removed_source") or slot["accepted_source"], self.root)
                if green.get(slot["slot"]) != previous:
                    self.unavailable(slot["slot"], "reused_source_changed",
                                     "reused candidate differs from accepted cleaned source")
                    changed = True
        return changed

    def finish(self, reason="generation stopped"):
        for slot in self.slots:
            if slot["state"] in {"pending", "active"}:
                self.unavailable(slot["slot"], "not_completed", reason)
            slot["terminal"] = True

    def snapshot(self, attempts_used):
        result = {"version": 1, "requested_slots": self.count, "slots": _copy(self.slots)}
        for slot in result["slots"]:
            slot["attempts_used"] = attempts_used.get(slot["slot"], 0)
        # Validate the structure before publishing. References are retained already.
        return read_suite_slots({"suite_slots": result}, self.root)

    def construction_for(self, i, source, scope):
        """The construction of slot ``i``'s selected starter suite: its terminal
        attempt's, a ``pruned`` node with it as parent when the source changed
        since, or an unavailable node when it has none. A slot without a terminal
        attempt (marked unavailable, such as a reused suite whose bytes
        changed) uses its last attempt."""
        slot = self._slot(i)
        terminal = slot.get("terminal_attempt")
        if terminal is None:
            attempt = slot["attempts"][-1] if slot["attempts"] else None
        else:
            attempt = next((item for item in slot["attempts"] if item["attempt_id"] == terminal), None)
        node = attempt.get("construction") if attempt else None
        if node is None:
            return construction(source, **scope, operation="unavailable",
                                reason="suite construction inputs unavailable")
        if node["selected_source_sha256"] == source_sha256(source):
            return _copy(node)
        return construction(source, **scope, operation="pruned", parents=[node])


def construction_scope(phase, run_idx=None, invocation=None):
    """The scope of one construction call: its phase, its run and an
    invocation ID, a new one unless given."""
    return {"phase": phase, "run_idx": run_idx,
            "invocation": invocation if invocation is not None else uuid.uuid4().hex}


def capture_origin(path: Path | str | None) -> dict | None:
    """The safe generation inputs recorded at ``path``, copied inline before
    their latest files change, or None when unavailable or unsafe."""
    if path is None:
        return None
    from testuary.settings import _safe

    path = Path(path)
    try:
        recorded = json.loads(path.read_bytes())
        if type(recorded.get("format")) is not int or recorded["format"] != 1:
            return None
        condition = recorded.get("condition")
        if not isinstance(condition, dict) or condition.get("format") != 1:
            return None
        safe_condition = _copy(_safe(condition))
        inputs = {key: _inline(path, recorded.get(key)) for key in ("prompt_profile", "hints")}
        if any(recorded.get(key) is not None and inputs[key] is None for key in inputs):
            return None
        # The extensions' hints, which a run records only when it captured some.
        extension_hints = recorded.get("extension_hints")
        if extension_hints is not None:
            inputs["extension_hints"] = [{"item": item.get("item"), **(_inline(path, item) or {})}
                                         for item in extension_hints]
            if not all("contents" in item for item in inputs["extension_hints"]):
                return None
        result = {"version": 1, "condition": safe_condition, "inputs": inputs,
                  "identity_known": safe_condition == condition}
        _check_origin(result)
        return result
    except (OSError, ValueError, UnicodeError, AttributeError, TypeError, RecursionError):
        return None


def _inline(path: Path, item) -> dict | None:
    """The captured input ``item`` of the origin at ``path``, read from its
    copy beside it and checked against its SHA-256, inline. None when it
    is absent, unsafe or changed."""
    from testuary.settings import _safe

    if item is None:
        return None
    name = item.get("copy") if isinstance(item, dict) else None
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or any(char in name for char in ("/", "\\", ":"))):
        return None
    target = (path.parent / name).resolve()
    if not target.is_relative_to(path.parent.resolve()):
        return None
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != item.get("sha256"):
        return None
    contents = data.decode("utf-8")
    parsed = json.loads(contents)
    if not isinstance(parsed, dict) or _copy(_safe(parsed)) != parsed:
        return None
    return {"base": "inline", "contents": contents, "sha256": item["sha256"]}


def _check_origin(origin):
    if origin is None:
        return False
    from testuary.settings import _safe

    if (not isinstance(origin, dict) or type(origin.get("version")) is not int
            or origin["version"] != 1 or type(origin.get("identity_known")) is not bool):
        raise GenerationEvidenceError("generation origin version is unavailable")
    condition = origin.get("condition")
    if (not isinstance(condition, dict) or type(condition.get("format")) is not int
            or condition["format"] != 1 or not isinstance(condition.get("phase"), str)
            or condition["phase"] not in {"starter", "coverage", "mutation"}
            or not isinstance(condition.get("language"), str) or not condition["language"]
            or not isinstance(condition.get("model"), dict) or not isinstance(condition.get("request"), dict)
            or not _SHA.fullmatch(str(condition.get("prompt_composition", "")))
            or _copy(_safe(condition)) != condition):
        raise GenerationEvidenceError("generation origin condition is unavailable or unsafe")
    inputs = origin.get("inputs")
    # Evidence recorded before extension hints holds the first two only.
    if not isinstance(inputs, dict) or set(inputs) - {"extension_hints"} != {"prompt_profile", "hints"}:
        raise GenerationEvidenceError("captured generation inputs are unavailable")
    extension_hints = inputs.get("extension_hints", [])
    if not isinstance(extension_hints, list) or not all(
            isinstance(item, dict) and isinstance(item.get("item"), str) for item in extension_hints):
        raise GenerationEvidenceError("captured extension hints are unavailable")
    for item in [inputs["prompt_profile"], inputs["hints"],
                 *({key: value for key, value in entry.items() if key != "item"}
                   for entry in extension_hints)]:
        if item is None:
            continue
        if (not isinstance(item, dict) or item.get("base") != "inline"
                or not isinstance(item.get("contents"), str)
                or source_sha256(item["contents"]) != item.get("sha256")):
            raise GenerationEvidenceError("captured generation input changed")
        try:
            parsed = json.loads(item["contents"])
        except (ValueError, RecursionError):
            raise GenerationEvidenceError("captured generation input is not JSON") from None
        if not isinstance(parsed, dict) or _copy(_safe(parsed)) != parsed:
            raise GenerationEvidenceError("captured generation input is unsafe")
    policy = condition.get("language_policy")
    if policy is not None and not isinstance(policy, dict):
        raise GenerationEvidenceError("generation language policy is unavailable")
    return (origin["identity_known"] and condition.get("service_identity_known") is True
            and condition.get("request_options_known") is True
            and (policy or {}).get("identity_known", True) is True)


_OPERATIONS = {"accepted_suite", "selected", "merge", "reduced", "pruned", "edited", "inherited", "unavailable"}


def _ancestry_status(node):
    known_scope = (isinstance(node.get("invocation"), str)
                   and _RUN.fullmatch(node["invocation"]) is not None
                   and _integer(node.get("run_idx"), minimum=0))
    parents = node["parents"]
    origin = node.get("origin")
    known_origin = _check_origin(origin)
    if origin is not None and origin["condition"]["phase"] != node["phase"]:
        raise GenerationEvidenceError("origin phase does not match its construction")
    if node["operation"] == "unavailable" or node["selected_source_sha256"] is None:
        return "unavailable"
    if node["operation"] in {"accepted_suite", "edited"}:
        complete = known_origin and (node["operation"] == "accepted_suite" or bool(parents))
    elif node["operation"] in {"selected", "inherited"}:
        if len(parents) > 1:
            raise GenerationEvidenceError("source-preserving construction has multiple parents")
        parent_digest = parents[0]["selected_source_sha256"] if parents else None
        if parent_digest is not None and parent_digest != node["selected_source_sha256"]:
            raise GenerationEvidenceError("source-preserving construction changed its parent source")
        complete = len(parents) == 1 and parent_digest is not None and origin is None
    else:
        complete = bool(parents) and origin is None
    return ("complete" if known_scope and complete
            and all(parent["status"] == "complete" for parent in parents) else "partial")


def read_ancestry(record: dict, source: str | None = None) -> dict:
    """The record's construction graph (``generation_ancestry``), checked
    and detached. It must belong to the record's selected source, and
    ``source`` when given. Anything unreadable or contradictory raises."""
    try:
        node = _copy(record.get("generation_ancestry"))
        selected = record.get("selected_source_sha256")
        def check(item):
            if (not isinstance(item, dict) or type(item.get("version")) is not int or item["version"] != 1
                    or item.get("basis") != "construction_inputs"
                    or (item.get("phase") not in {"starter", "coverage", "mutation"}
                        and not (item.get("phase") is None and item.get("operation") == "unavailable"))
                    or not isinstance(item.get("operation"), str) or item["operation"] not in _OPERATIONS
                    or not isinstance(item.get("parents"), list)):
                raise GenerationEvidenceError("selected construction version is unavailable")
            digest = item.get("selected_source_sha256")
            if (item.get("invocation") is not None
                    and (not isinstance(item["invocation"], str) or not _RUN.fullmatch(item["invocation"]))
                    or item.get("run_idx") is not None and not _integer(item["run_idx"], minimum=0)):
                raise GenerationEvidenceError("invalid construction invocation or repeat")
            if digest is not None and (not isinstance(digest, str) or not _SHA.fullmatch(digest)):
                raise GenerationEvidenceError("invalid construction source identity")
            for parent in item["parents"]:
                check(parent)
            binding = item.get("suite_attempt")
            if item["operation"] == "accepted_suite" and binding is None:
                raise GenerationEvidenceError("accepted suite has no paid construction binding")
            if binding is not None:
                if item["operation"] != "accepted_suite" or not _integer(binding.get("slot"), minimum=1):
                    raise GenerationEvidenceError("invalid suite construction binding")
                _check_attempt(binding.get("attempt_id"))
                if binding["attempt_id"]["run"] != item.get("invocation"):
                    raise GenerationEvidenceError("suite construction invocation does not match paid attempt")
            if item.get("status") != _ancestry_status(item):
                raise GenerationEvidenceError("contradictory construction availability")
        check(node)
        if node["selected_source_sha256"] != selected or (source is not None and source_sha256(source) != selected):
            raise GenerationEvidenceError("selected construction is bound to different source")
        return node
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError, RecursionError):
        raise GenerationEvidenceError("selected construction is unavailable or contradictory") from None


def construction(source, *, phase, invocation=None, run_idx=None, operation="selected",
                 parents=(), origin=None, suite_attempt=None, reason=""):
    """One construction node: the inputs that went into ``source``, not who
    wrote its tests."""
    node = {"version": 1, "basis": "construction_inputs", "phase": phase,
            "invocation": invocation, "run_idx": run_idx,
            "selected_source_sha256": source_sha256(source) if source is not None else None,
            "operation": operation, "parents": _copy(list(parents)), "origin": _copy(origin),
            "reason": reason}
    if suite_attempt is not None:
        node["suite_attempt"] = _copy(suite_attempt)
    node["status"] = _ancestry_status(node)
    return read_ancestry({"generation_ancestry": node, "selected_source_sha256": node["selected_source_sha256"]})


def predecessor(record, *, phase, source=None, expected_sha256=None):
    """The construction graph a predecessor record holds for ``source``, or
    an explicitly unavailable node when it holds none that fits."""
    digest = source_sha256(source) if source is not None else expected_sha256
    try:
        node = read_ancestry(record, source)
        if node["selected_source_sha256"] != digest:
            raise GenerationEvidenceError("predecessor source changed")
        return node
    except GenerationEvidenceError:
        node = construction(source, phase=phase, operation="unavailable", reason="predecessor ancestry unavailable")
        if source is None and isinstance(digest, str) and _SHA.fullmatch(digest):
            node["selected_source_sha256"] = digest
        return node


def select_construction(source, *, scope, winner, inputs):
    """The construction of the selected suite ``winner``: its own input's
    lineage, or every input's for a merged or reduced suite, not every
    candidate it was compared with."""
    if source is None:
        return construction(None, **scope, operation="unavailable", reason="no suite selected")
    selected = inputs.get(winner)
    if selected is not None:
        operation = "selected" if selected["selected_source_sha256"] == source_sha256(source) else "pruned"
        parents = [selected]
    elif winner in {"merged", "merged_unreduced", "reduced"}:
        operation = "reduced" if winner == "reduced" else "merge"
        parents = list(inputs.values())
    else:
        operation, parents = "unavailable", []
    return construction(source, **scope, operation=operation, parents=parents,
                        reason="winner construction inputs unavailable" if not parents else "")
