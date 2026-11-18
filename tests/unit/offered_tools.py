"""The tools an editing agent offers the model in its current agent mode."""
from __future__ import annotations


def offered_tools(agent) -> list:
    """The unwrapped tool methods of ``agent``'s current agent mode: the
    mode sets ``build_graph`` binds, picked by the agent mode it routes on."""
    return agent.tool_mode_sets(*agent.tool_methods())[agent.effective_agent_mode()]
