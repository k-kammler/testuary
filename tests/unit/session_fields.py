"""The session fields a loop's tool call writes, for tests that pin a path's
reply together with the state it leaves."""
import copy
import dataclasses

# What every tool call writes (the call count, the tool's name and time,
# the read and loop-breaker counters), whatever its path.
_EVERY_CALL = frozenset({
    "session_tool_calls", "current_tool_name",
    "last_tool_event_at", "consecutive_reads_by_tool", "read_ranges_seen_by_tool",
    "loop_breaker",
})


def written_fields(session, act):
    """``(act(), changed)``: what ``act()`` returns, and each field of the
    dataclass ``session`` whose value it changed, with the new value,
    leaving out the fields every tool call writes."""
    names = [field.name for field in dataclasses.fields(session)
             if field.name not in _EVERY_CALL]
    before = copy.deepcopy({name: getattr(session, name) for name in names})
    result = act()
    return result, {name: getattr(session, name) for name in names
                    if getattr(session, name) != before[name]}
