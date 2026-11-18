"""The coding agents of the coverage and mutation phases.

``coverage_agent`` and ``mutation_agent`` are the two phases' agents, both
built on ``_base.CodingAgent``. ``_langgraph`` sends the model requests and
runs the tool loop, ``_context`` keeps the loop's context, ``_edit_tool``
is the edit tool, ``_helpers`` holds the agents' shared scaffolding and
``_limits`` the limits on what the model sees. The starter phase
(``testuary.starters``) and Finishing (``testuary.measurement``) need no
agent. ``testuary.target_phases`` runs a target's four phases in order.
"""
