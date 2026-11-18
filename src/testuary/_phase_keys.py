"""The phase keys of the hints and prompt profile files, and the key of a
run phase."""

PHASES = ("initial", "coverage", "mutation")
"""The phases whose prompts read hints and a prompt profile: the starter
phase and the two loops."""


def phase_key(phase: str) -> str:
    """The hints and prompt profile files' key of the run phase ``phase``:
    ``initial`` for the starter phase, else the phase's own name, which no
    phase block holds for a phase that reads no hints."""
    return "initial" if phase == "starter" else phase
