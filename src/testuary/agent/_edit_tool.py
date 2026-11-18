"""The edit tool of the coding agents: ``EditTool.edit_file`` edits the
agent's own file by content match, with its matching, gates and echo.
CodingAgent takes it as a mixin. It uses the agent's log, tool messages,
target path, sandbox guard and session."""
from __future__ import annotations

import re as _re
from pathlib import Path
from typing import Any, List, Optional

from typing_extensions import NotRequired, TypedDict

from testuary.agent import _limits
from testuary.agent import _helpers as agent_helpers


class EditItem(TypedDict):
    """One edit of an ``edits`` list: ``old_text``, ``new_text`` and an
    optional ``replace_all``, as for a single edit."""

    old_text: str
    new_text: str
    replace_all: NotRequired[bool]


class _EditMatch:
    """One edit matched and applied in memory (``_match_edit``)."""

    __slots__ = ("content", "old_norm", "new_norm", "occurrences",
                 "used_loose_match", "loose_indent_diff", "landed_at", "starts")

    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


_NUMBERED_LINE = _re.compile(r"(\d+)\t")


def strip_line_number_prefixes(text: str) -> str:
    """Remove the line numbers of the read and edit views (``number_line``)
    from pasted ``text``.

    Only text in which every line carries that form, with consecutive
    numbers, is cleaned. An empty source line shows as ``N<TAB>``, and a paste
    can lose that tab or the whole line: a line that is exactly the expected
    number, or an empty line, counts as that empty line. Any other text is
    returned unchanged, so code whose lines start with numbers, such as a dict
    literal with integer keys, is kept. The edit tool decides when to apply it
    (``CodingAgent.edit_file``).
    """
    if not text:
        return text
    lines = text.split("\n")
    trailing = text.endswith("\n")
    if trailing:
        lines.pop()
    expected = None
    cleaned = []
    for line in lines:
        match = _NUMBERED_LINE.match(line)
        if match is None:
            if expected is not None and line == str(expected):
                # An empty source line whose tab the paste dropped.
                cleaned.append("")
                expected += 1
                continue
            if line.strip():
                return text
            cleaned.append(line)
            if expected is not None:
                expected += 1
            continue
        number = int(match.group(1))
        if expected is not None and number != expected:
            return text
        expected = number + 1
        cleaned.append(line[match.end():])
    if expected is None:
        return text
    return "\n".join(cleaned) + ("\n" if trailing else "")


def strip_markdown_code_fences(text: str) -> str:
    """Strip a leading and trailing markdown code fence from ``text``.

    Models often wrap the edit tool's texts in a markdown fence even when
    the schema asks for plain source. The wrapped text would miss the file
    or be written with its fences, so this takes the body without a
    rejection. The opening fence is the first non-blank line when it starts
    with three backticks (with or without a language tag), the closing one
    the last non-blank line when it is exactly three backticks. Without a
    closing fence only the opening line is removed. Inner backticks stay,
    and text without a fence is returned unchanged.
    """
    if not text:
        return text
    lines = text.splitlines()
    # Find first non-blank line
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return text
    first = lines[i].strip()
    if not first.startswith("```"):
        return text
    # Find the matching closing fence: walk backwards from end, skip
    # trailing blank lines, expect a line that is exactly ``` (after
    # strip).
    j = len(lines) - 1
    while j > i and not lines[j].strip():
        j -= 1
    if j <= i:
        return text
    last = lines[j].strip()
    if last != "```":
        # Opening fence but no closing one: still strip the opener.
        return "\n".join(lines[i + 1:])
    # Strip both fences
    return "\n".join(lines[i + 1:j])


def whitespace_loose_match(needle: str, haystack: str):
    """Find ``needle`` in ``haystack`` line by line, ignoring leading
    whitespace: the edit tool's fallback when an exact match fails on a
    uniform indentation difference (``old_text`` pasted with 4 spaces where
    the file uses 8).

    Returns ``(actual_block, indent_diff, offset)`` for exactly one match:
      - ``actual_block``: the file's text of the matched region, with its
        real whitespace.
      - ``indent_diff``: the leading-whitespace difference of the first
        line (file minus needle), to apply to every line of ``new_text``.
      - ``offset``: where ``actual_block`` starts in ``haystack``. Its text
        can also occur earlier inside another line, so the caller replaces
        at this offset, not at the first occurrence.

    None for no match or several. The caller normalizes CRLF first: both
    inputs and ``actual_block`` use ``\\n`` endings, and lines end at
    ``\\n`` only, as the read and edit views number them (a form feed or a
    lone carriage return stays inside its line).
    """
    if not needle or not haystack:
        return None
    needle_lines = needle.split("\n")
    if needle.endswith("\n"):
        needle_lines.pop()
    needle_stripped = [ln.lstrip() for ln in needle_lines]
    haystack_lines = [ln + "\n" for ln in haystack.split("\n")]
    haystack_lines[-1] = haystack_lines[-1][:-1]
    if not haystack_lines[-1]:
        haystack_lines.pop()
    n = len(needle_lines)
    if n > len(haystack_lines):
        return None
    stripped = [ln.rstrip("\n").lstrip() for ln in haystack_lines]
    matches = []
    offset = 0
    for i in range(len(haystack_lines) - n + 1):
        if stripped[i:i + n] == needle_stripped:
            if matches:
                return None
            matches.append((i, offset))
        offset += len(haystack_lines[i])
    if not matches:
        return None
    (first, offset), = matches
    actual_block = "".join(haystack_lines[first:first + n])
    # Drop the last line's newline if needle has none: only that one, so a
    # needle ending in a blank line keeps the newline before it.
    if not needle.endswith("\n") and actual_block.endswith("\n"):
        actual_block = actual_block[:-1]
    needle_first = needle_lines[0]
    actual_first_line = haystack_lines[first].rstrip("\n")
    needle_lead = len(needle_first) - len(needle_first.lstrip())
    actual_lead = len(actual_first_line) - len(actual_first_line.lstrip())
    indent_diff = actual_lead - needle_lead
    return actual_block, indent_diff, offset


def apply_indent_diff(text: str, diff: int) -> str:
    """Add or remove leading whitespace on every non-blank line.

    Positive ``diff``: prepend ``diff`` spaces to each non-blank line.
    Negative ``diff``: strip ``abs(diff)`` leading spaces from each
    non-blank line that has them. Blank lines are left alone.

    Used together with ``whitespace_loose_match`` in the edit body so
    that ``new_text`` written with the model's indentation is rewritten
    to match the file's actual indentation when a loose match is used.
    """
    if not text or diff == 0:
        return text
    lines = text.split("\n")
    out = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        if diff > 0:
            out.append(" " * diff + line)
        else:
            rm = abs(diff)
            if line.startswith(" " * rm):
                out.append(line[rm:])
            else:
                out.append(line)
    return "\n".join(out)


class EditTool:
    """The edit tool, mixed into CodingAgent."""

    def edit_file(
        self, old_text: str = "", new_text: Optional[str] = None, replace_all: bool = False,
        edits: Optional[List[EditItem]] = None,
    ) -> str:
        """Edit this agent's file by content match (no line numbers). Three
        operations from one tool:

        - Insert: anchor ``old_text`` on an existing unique line and put that
          same line plus the new code in ``new_text``. To add a block, anchor on
          the line above or below where it goes and include it in ``new_text``.
        - Replace: ``old_text`` is the exact block to swap for ``new_text``.
        - Delete: pass "" as ``new_text`` to remove the matched block.

        ``old_text`` must occur exactly once unless ``replace_all`` is set.
        Copy whitespace and indentation exactly: CRLF is normalized to LF,
        line numbers copied from the read and edit views (``42<TAB>``) and
        markdown fences are stripped, and an
        exact miss falls back to a whitespace-loose match that re-indents
        ``new_text`` to the file. The edit is rejected, and nothing is written,
        on no match, a multi-match, an empty ``old_text``, a result that no
        longer parses or that the target's parser rejects, a duplicate
        definition, or a test that inspects the target's source or bytecode;
        the file is unverified until the next ``run_tests``.

        Several edits: pass ``edits``, a list of ``{old_text, new_text}``
        items, each with an optional ``replace_all``, instead of ``old_text``
        and ``new_text``. They apply in list order, each matched against the
        file as the previous ones left it, and the checks run once on the
        final result: all or nothing, and an error names the failing edit by
        its position. Use the list for changes that depend on each other,
        such as a new version of a test and the deletion of the old one, a
        block header and its body, or swapping two names: one at a time, the
        first of them would be rejected. Several separate edit_tests calls in
        one reply also apply in call order, each against the file after the
        previous calls.

        Args:
            old_text: exact text to find; add surrounding context to make it
                unique. To insert, this is the anchor line you keep in new_text.
            new_text: replacement text, required for a single edit; ""
                deletes the matched block.
            replace_all: replace every occurrence instead of requiring one;
                in a list, set it on the item.
            edits: several edits applied in order as one change, with
                old_text and new_text left out.
        """
        self.log_thinking(self.edit_tool_name)

        def reject(kind: str, message: str, old: Optional[str] = None,
                   new: Optional[str] = None) -> str:
            # A list item's texts when given, else the call's own.
            old_prefix = (old_text if old is None else old or "")[:120].replace("\n", " \\n ")
            new_prefix = ((new_text if new is None else new) or "")[:120].replace("\n", " \\n ")
            self.log(
                f"[{self.edit_tool_name}] REJECTED ({kind}): "
                f"old[:120]={old_prefix!r} new[:120]={new_prefix!r}",
                level=1,
            )
            return self.track_output(message)

        path = self.target_path()
        if path is None or not path.exists():
            return reject("no file yet", self.tool_msg("edit_no_file"))

        blocked = self.sandbox_guard(path)
        if blocked is not None:
            return reject("sandbox escape", blocked)

        try:
            existing = path.read_text(encoding="utf-8")
        except Exception as exc:
            return reject("read failed", self.tool_msg(
                "edit_read_failed", name=path.name, exc=str(exc)))

        if edits is not None:
            if old_text or new_text or replace_all:
                return reject("both forms", self.tool_msg("edit_list_mixed"))
            return self._edit_list(path, existing, list(edits), reject)
        if new_text is None:
            # Deletion is an explicit "", never an omitted new_text.
            return reject("no new_text", self.tool_msg(
                "edit_missing_new_text" if old_text else "edit_no_arguments"))

        matched = self._match_edit(
            existing.replace("\r\n", "\n"), old_text, new_text, replace_all,
            path.name, after_edit=self._edited_in_this_reply())
        if matched[0] == "reject":
            return reject(matched[1], matched[2])
        if matched[0] == "unchanged":
            return matched[1]
        match = matched[1]

        gate = self._edit_gates(existing, match.content)
        if gate is not None:
            return reject(*gate)

        self._write_edit(path, match.content)

        return self._edit_echo(
            match.content, match.new_norm, match.occurrences,
            replace_all, match.used_loose_match, match.loose_indent_diff,
            path.name, match.landed_at,
        )

    def _current_request(self) -> Optional[int]:
        """The number of the model request whose tool calls run now, or
        None without a usage tracker."""
        tracker = self.session.usage_tracker_ref
        return None if tracker is None else getattr(tracker, "requests_used", None)

    def _edited_in_this_reply(self) -> bool:
        """Whether an earlier call of the reply being executed wrote this
        agent's file."""
        current = self._current_request()
        return current is not None and getattr(self, "_last_edit_request", None) == current

    def _match_edit(
        self, existing_norm: str, old_text: str, new_text: str,
        replace_all: bool, name: str, *, after_edit: bool = False,
        list_index: int = 0,
    ) -> tuple:
        """Match one edit against ``existing_norm`` (LF line endings) and
        apply it in memory. Returns ``("reject", kind, message)``,
        ``("unchanged", message)`` or ``("ok", _EditMatch)``. Writes nothing
        and runs no gate. ``after_edit`` says that an earlier call of the
        same reply wrote the file, which a miss then mentions.
        ``list_index`` is the edit's position in an ``edits`` list, 0 for a
        single edit: from the second on, ``existing_norm`` holds the earlier
        edits of the list, which are not on disk."""
        if not old_text:
            return ("reject", "empty old_text", self.tool_msg("edit_empty_old"))
        if (agent_helpers.contains_history_marker(old_text)
                or agent_helpers.contains_history_marker(new_text)):
            return ("reject", "history marker in args",
                    self.tool_msg("edit_history_marker"))
        try:
            (new_text or "").encode("utf-8")
        except UnicodeEncodeError as exc:
            return ("reject", "unencodable new_text", self.tool_msg(
                "edit_bad_characters", position=exc.start))

        old_sent = strip_markdown_code_fences(old_text).replace("\r\n", "\n")
        new_sent = strip_markdown_code_fences(new_text or "").replace("\r\n", "\n")
        # Pasted view line numbers are removed from an old_text the file lacks
        # as sent, and from new_text unless old_text was found as sent with
        # numbers: then they are the file's content.
        found_as_sent = old_sent in existing_norm
        old_norm = old_sent if found_as_sent else strip_line_number_prefixes(old_sent)
        numbered_content = found_as_sent and strip_line_number_prefixes(old_sent) != old_sent
        new_norm = new_sent if numbered_content else strip_line_number_prefixes(new_sent)
        if not old_norm:
            # Only prefixes or fences: an empty needle matches everywhere.
            return ("reject", "old_text empty after cleaning",
                    self.tool_msg("edit_empty_after_cleaning"))

        occurrences = existing_norm.count(old_norm)
        used_loose_match = False
        loose_indent_diff = 0
        loose_at = None
        if occurrences == 0:
            loose = whitespace_loose_match(old_norm, existing_norm)
            if loose is not None:
                actual_block, loose_indent_diff, loose_at = loose
                old_norm = actual_block
                new_norm = apply_indent_diff(new_norm, loose_indent_diff)
                occurrences = 1
                used_loose_match = True
                self.log(
                    f"[{self.edit_tool_name}] whitespace-loose match found "
                    f"(indent diff {loose_indent_diff:+d}). Proceeding.", level=1)
        if occurrences == 0:
            hint_lines = []
            compared = old_norm.split("\n")
            line_limit = _limits.value("edit_evidence_lines")
            char_limit = _limits.value("edit_evidence_chars")
            for ln in compared if line_limit is None else compared[:line_limit]:
                stripped = ln.strip()
                if stripped and len(stripped) >= 8 and stripped in existing_norm:
                    shown = stripped if char_limit is None else stripped[:char_limit]
                    # The marker stays outside the quotes: only the quoted
                    # text appeared in the file.
                    hint_lines.append(repr(shown) + (
                        " " + _limits.marker(len(stripped) - len(shown), "chars", "the line")
                        if len(shown) < len(stripped) else ""))
            hint = ""
            if hint_lines:
                hint = (
                    "\n\nPartial-match evidence: these substrings from your "
                    "old_text DID appear in the file:\n"
                    + "\n".join(f"  - {h}" for h in hint_lines)
                    + "\n-> confirms the BLOCK is roughly there but the EXACT "
                    "bytes drifted (whitespace, indentation, a blank line, "
                    "a trailing comma, or a quote character).")
            if list_index > 1:
                earlier = "edit 1" if list_index == 2 else f"edits 1 to {list_index - 1}"
                return ("reject", "not found", self.tool_msg(
                    "edit_list_not_found", name=name, earlier=earlier,
                    line_count=agent_helpers.line_count(existing_norm), hint=hint))
            if after_edit:
                hint += self.tool_msg("edit_not_found_after_edit")
            return ("reject", "not found", self.tool_msg(
                "edit_not_found", name=name,
                line_count=agent_helpers.line_count(existing_norm), hint=hint))

        if occurrences > 1 and not replace_all:
            if list_index > 1:
                # Counted in the list's working copy, not on disk.
                earlier = "edit 1" if list_index == 2 else f"edits 1 to {list_index - 1}"
                return ("reject", "multi-match", self.tool_msg(
                    "edit_list_multi_match", occurrences=occurrences, name=name,
                    earlier=earlier))
            message = self.tool_msg("edit_multi_match", occurrences=occurrences, name=name)
            if list_index:
                message += self.tool_msg("edit_list_replace_all_note")
            return ("reject", "multi-match", message)
        if old_norm == new_norm:
            return ("unchanged", self.tool_msg("edit_unchanged", name=name))

        if loose_at is not None:
            # The line-aligned block the loose match found: the same text
            # can also occur earlier inside another line.
            starts = [loose_at]
            content = existing_norm[:loose_at] + new_norm + existing_norm[loose_at + len(old_norm):]
        else:
            # Where each replacement starts in the new content: the text
            # before the first match is unchanged, and each later match
            # moves by the length change of the ones before it.
            starts, at = [], existing_norm.find(old_norm)
            while at >= 0 and (replace_all or not starts):
                starts.append(at + len(starts) * (len(new_norm) - len(old_norm)))
                at = existing_norm.find(old_norm, at + len(old_norm))
            content = (
                existing_norm.replace(old_norm, new_norm) if replace_all
                else existing_norm.replace(old_norm, new_norm, 1))
        return ("ok", _EditMatch(
            content=content, old_norm=old_norm, new_norm=new_norm,
            occurrences=occurrences, used_loose_match=used_loose_match,
            loose_indent_diff=loose_indent_diff, landed_at=starts[0],
            starts=starts))

    def _edit_gates(self, existing: str, new_content: str):
        """The checks a write must pass, in order: ``(kind, message)`` of the
        first that fails, or None."""
        parse_err = agent_helpers.source_structure_error(
            new_content, self.source_language()
        )
        if parse_err is not None:
            # The compiler names the error and its place (javac's missing
            # semicolon at its token). The structural parser's can span a class.
            validation = self.test_backend().validate_test_source(
                new_content, self.deps.source_dir
            )
            if (not validation.ok and validation.status == "syntax_error"
                    and validation.line):
                parse_err = agent_helpers.source_validation_error(
                    new_content, validation)
            return ("post-edit syntax error", self.tool_msg(
                "edit_post_edit_syntax_error", parse_err=parse_err))

        validation = self.test_backend().validate_test_source(
            new_content, self.deps.source_dir
        )
        validation_err = agent_helpers.source_validation_error(
            new_content, validation
        )
        if validation_err is not None:
            return ("target parser rejected edit", self.tool_msg(
                "edit_post_edit_syntax_error", parse_err=validation_err))

        dup_reject = self._shadowed_definition_reject(new_content)
        if dup_reject is not None:
            return ("duplicate test name", dup_reject)

        validity_reject = self._generated_test_validity_reject(
            existing, new_content,
        )
        if validity_reject is not None:
            return ("generated-test validity", validity_reject)
        return None

    def _write_edit(self, path: Path, new_content: str) -> None:
        """Write an accepted edit and record that the file changed."""
        path.write_text(new_content, encoding="utf-8")
        self._last_edit_request = self._current_request()
        self._persist_generated_test_validity_write()
        self.progress_made()
        self.session.last_tests_passed = False
        self.after_write()

    def _edit_list(self, path: Path, existing: str, edits: List[Any], reject) -> str:
        """Apply ``edits`` in order to a copy, run the checks once on the
        result and write it, or write nothing."""
        count = len(edits)
        if not count:
            return reject("empty edits", self.tool_msg("edit_list_empty"))
        limit = _limits.value("edit_list_items")
        if limit is not None and count > limit:
            return reject("too many edits", self.tool_msg(
                "edit_list_too_many", count=count, limit=limit))
        after_reply_edit = self._edited_in_this_reply()
        original = current = existing.replace("\r\n", "\n")
        none_applied = f"none of the {count} edits was applied" if count > 1 else "it was not applied"
        # Changed regions as [start, end, item numbers] in ``current``.
        regions: List[list] = []
        loose_notes = []
        for index, item in enumerate(edits, 1):
            item = item if isinstance(item, dict) else {}
            old_item, new_item = item.get("old_text") or "", item.get("new_text")
            if new_item is None:
                matched = ("reject", "no new_text", self.tool_msg("edit_missing_new_text"))
            else:
                matched = self._match_edit(
                    current, old_item, new_item, bool(item.get("replace_all", False)),
                    path.name, after_edit=index == 1 and after_reply_edit, list_index=index)
            if matched[0] != "ok":
                return reject(f"edits item {index}: {matched[1] if matched[0] == 'reject' else 'unchanged'}",
                              self.tool_msg("edit_list_item_failed", index=index, count=count,
                                            none_applied=none_applied, reason=matched[-1]),
                              old=old_item, new=new_item)
            match = matched[1]
            old_len, new_len = len(match.old_norm), len(match.new_norm)
            # Map the earlier regions through this edit's replacements, at
            # their places in ``current``: each start in the new content less
            # the length change of the replacements before it.
            befores = [start - n * (new_len - old_len) for n, start in enumerate(match.starts)]
            regions = self._remap_regions(regions, befores, old_len, new_len, index)
            current = match.content
            if match.used_loose_match:
                loose_notes.append(
                    f"edit {index}: whitespace-loose match, re-indented its new_text "
                    f"by {match.loose_indent_diff:+d} chars to match the file")

        if current == original:
            return self.tool_msg("edit_list_unchanged", count=count, name=path.name)

        gate = self._edit_gates(existing, current)
        if gate is not None:
            kind, message = gate
            return reject(f"{kind} (edits)", self.tool_msg(
                "edit_list_gate_failed", none_applied=none_applied, reason=message,
                result_of=f"all {count} edits" if count > 1 else "the 1 edit"))

        self._write_edit(path, current)
        return self._edit_list_echo(current, regions, count, loose_notes, path.name)

    @staticmethod
    def _remap_regions(regions: List[list], befores: List[int], old_len: int, new_len: int,
                       index: int) -> List[list]:
        """``regions`` (sorted, disjoint ``[start, end, items]``) after each
        ``[at, at + old_len)`` at ``befores`` (ascending, in the same text)
        became ``new_len`` long. Each replacement is added as a region of
        edit ``index``. A region that overlaps or touches it is merged into
        it, one before it stays and one after it moves by the length change.
        One pass from the last replacement to the first, the same result as
        applying them one at a time in that order."""
        delta = new_len - old_len
        # Regions passed, right to left. A replacement further left moves
        # each of them by delta, so they are stored less delta times the
        # replacements applied when stored.
        passed: List[list] = []
        applied = 0
        left = len(regions)
        for at in reversed(befores):
            landed = [at, at + new_len, {index}]
            while passed:
                start, end, items = passed[-1]
                start, end = start + delta * applied, end + delta * applied
                if start > at + old_len or (start == at + old_len and end > start):
                    break
                passed.pop()
                landed = [min(landed[0], start), max(landed[1], end + delta), landed[2] | items]
            while left:
                start, end, items = regions[left - 1]
                if end < at or (end == at and start < at):
                    break
                left -= 1
                if start > at + old_len or (start == at + old_len and end > start):
                    passed.append([start - delta * applied, end - delta * applied, items])
                else:
                    landed = [min(landed[0], start), max(landed[1], end + delta), landed[2] | items]
            applied += 1
            passed.append([landed[0] - delta * applied, landed[1] - delta * applied, landed[2]])
        moved = [[start + delta * applied, end + delta * applied, items]
                 for start, end, items in reversed(passed)]
        return sorted(regions[:left] + moved, key=lambda region: (region[0], region[1]))

    def _edit_list_echo(self, content: str, regions: List[list], count: int,
                        loose_notes: List[str], name: str) -> str:
        """The success result of a list: one line-numbered view per changed
        region, in final line numbers, merged where their views would meet."""
        lines = agent_helpers.text_lines(content)
        n_lines = agent_helpers.line_count(content)
        context_window = 5
        # [first, last, items, parts]: a view's lines and its regions, each
        # region as (first, last, items).
        spans = []
        for start, end, items in regions:
            # A deletion at the end of the file shows its last line.
            first = min(content.count("\n", 0, start) + 1, max(1, n_lines))
            last = (min(content.count("\n", 0, max(start, end - 1)) + 1, max(1, n_lines))
                    if end > start else first)
            if spans and first <= spans[-1][1] + 2 * context_window + 1:
                spans[-1][1] = max(spans[-1][1], last)
                spans[-1][2] |= items
                spans[-1][3].append((first, last, items))
            else:
                spans.append([first, last, set(items), [(first, last, items)]])
        max_region_lines = _limits.value("edit_echo_lines")
        views = []
        for first, last, items, parts in spans:
            lo = max(1, first - context_window)
            hi = min(len(lines), last + context_window)
            span = last - first + 1
            each = max(1, ((max_region_lines or 0) - 6) // 2)
            # Cut only when the kept ends leave lines out.
            if max_region_lines is not None and span > max_region_lines and span > 2 * each:
                omitted_first, omitted_last = first + each, last - each
                # Only lines inside one edit's new_text are text the model
                # just submitted. Merged views can omit unchanged lines.
                submitted = any(len(part_items) == 1 and part_first < omitted_first
                                and omitted_last < part_last
                                for part_first, part_last, part_items in parts)
                shown = [agent_helpers.number_line(i, lines[i - 1]) for i in range(lo, omitted_first)]
                shown.append("  " + _limits.marker(
                    span - 2 * each, "lines", "the changed region",
                    self._omission_more(span - 2 * each, submitted)))
                shown.extend(agent_helpers.number_line(i, lines[i - 1]) for i in range(omitted_last + 1, hi + 1))
            else:
                shown = [agent_helpers.number_line(i, lines[i - 1]) for i in range(lo, hi + 1)]
            label = ", ".join(str(item) for item in sorted(items))
            plural = "s" if len(items) > 1 else ""
            views.append(f"Edit{plural} {label}, lines {first}-{last}:\n"
                         + ("\n".join(shown) if shown else "(empty file)"))
        view_limit = _limits.value("edit_list_views")
        if view_limit is not None and len(views) > view_limit:
            hidden = len(views) - view_limit
            views = views[:view_limit] + [_limits.marker(
                hidden, "items", "further changed regions",
                self._omission_more(hidden, submitted=False))]
        loose_note = ("\n\n" + "\n".join(loose_notes)) if loose_notes else ""
        return self.tool_msg(
            "edit_ok_list", name=name, n_lines=n_lines, context_window=context_window,
            applied=f"all {count} edits in order" if count > 1 else "1 edit",
            views_heading=("Views of the changed regions" if len(spans) > 1
                           else "View of the changed region"),
            after="all edits" if count > 1 else "the edit",
            the_edits="the edits" if count > 1 else "the edit",
            regions="\n\n".join(views), loose_note=loose_note,
        )

    @staticmethod
    def _omission_more(count: int, submitted: bool) -> str:
        """Where the ``count`` things a view leaves out are, for its
        marker: in the new_text the model just submitted, or in read_tests."""
        if submitted:
            return ("it is from the new_text you just submitted" if count == 1
                    else "they are the new_text you just submitted")
        return "read_tests shows it" if count == 1 else "read_tests shows them"

    def _edit_echo(
        self, new_content: str, new_norm: str, occurrences: int,
        replace_all: bool, used_loose_match: bool, loose_indent_diff: int,
        name: str, landed_at: int = 0,
    ) -> str:
        """The success echo: a numbered view of the changed region with 5 lines
        of context each side, its middle cut past ``edit_echo_lines``, since the
        model holds the new_text it submitted. ``landed_at`` is the character
        offset where the first replacement starts."""
        n_lines = agent_helpers.line_count(new_content)

        # A deletion at the end of the file lands at its last line.
        edit_start_line = min(new_content.count("\n", 0, max(0, landed_at)) + 1, max(1, n_lines))
        new_line_count = (
            new_norm.count("\n") + (1 if new_norm and not new_norm.endswith("\n") else 0)
            if new_norm else 0
        )
        edit_end_line = max(edit_start_line, edit_start_line + max(0, new_line_count - 1))

        new_lines_list = agent_helpers.text_lines(new_content)
        context_window = 5
        lo = max(1, edit_start_line - context_window)
        hi = min(len(new_lines_list), edit_end_line + context_window)
        max_region_lines = _limits.value("edit_echo_lines")
        inserted_span = edit_end_line - edit_start_line + 1
        # A longer insertion keeps its first and last lines, six fewer than
        # the limit in all: 12 from each end at the default 30. It is cut
        # only when the kept ends leave lines out.
        each = max(1, ((max_region_lines or 0) - 6) // 2)
        if (max_region_lines is not None and inserted_span > max_region_lines
                and inserted_span > 2 * each):
            head_end = edit_start_line + each - 1
            tail_start = edit_end_line - each + 1
            snippet_lines = [
                agent_helpers.number_line(i, new_lines_list[i - 1]) for i in range(lo, head_end + 1)]
            snippet_lines.append("  " + _limits.marker(
                inserted_span - 2 * each, "lines", "the inserted content",
                self._omission_more(inserted_span - 2 * each, submitted=True)))
            snippet_lines.extend(
                agent_helpers.number_line(i, new_lines_list[i - 1]) for i in range(tail_start, hi + 1))
        else:
            snippet_lines = [
                agent_helpers.number_line(i, new_lines_list[i - 1]) for i in range(lo, hi + 1)]
        snippet = "\n".join(snippet_lines) if snippet_lines else "(empty file)"

        replaced_msg = (
            f"all {occurrences} occurrences"
            if (replace_all and occurrences > 1) else "1 occurrence")
        loose_note = ""
        if used_loose_match:
            loose_note = (
                f" (whitespace-loose match: re-indented your new_text by "
                f"{loose_indent_diff:+d} chars to match the file)")
        return self.tool_msg(
            "edit_ok",
            replaced_msg=replaced_msg, name=name, loose_note=loose_note,
            n_lines=n_lines, edit_start_line=edit_start_line,
            edit_end_line=edit_end_line, lo=lo, hi=hi,
            context_window=context_window, snippet=snippet,
        )
