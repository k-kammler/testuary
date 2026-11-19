"""Testuary's boundary to the language backend: the package imports only the
backend's public modules and names, never a module or name one of whose parts
starts with an underscore."""
from __future__ import annotations

import ast
from pathlib import Path

import testuary

#: The imported package (its source tree, or its installed files).
PACKAGE = Path(testuary.__file__).resolve().parent


def _private(path: str) -> bool:
    return any(part.startswith("_") for part in path.split("."))


def private_imports() -> set:
    """The private imports of the backend in the package, each file named
    from the directory that holds the package."""
    found = set()
    for file in sorted(PACKAGE.rglob("*.py")):
        rel = file.relative_to(PACKAGE.parent).as_posix()
        tree = ast.parse(file.read_text(encoding="utf-8"), rel)
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom) and node.level == 0
                    and (node.module or "").split(".")[0] == "langadapter"):
                if _private(node.module):
                    found.add(f"{rel}: {node.module}")
                    continue
                for alias in node.names:
                    path = f"{node.module}.{alias.name}"
                    if _private(path):
                        found.add(f"{rel}: {path}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "langadapter" and _private(alias.name):
                        found.add(f"{rel}: {alias.name}")
    return found


def test_no_private_import_of_the_backend():
    found = sorted(private_imports())
    assert not found, (
        "private imports of the language backend. Use its public API: " + ", ".join(found))


def test_a_private_import_is_recognized():
    """The scan's rule on the forms an import takes."""
    assert _private("langadapter.python._implementation")
    assert _private("langadapter.execution._as_text")
    assert not _private("langadapter.runs")
    assert not _private("langadapter.python.adapter")
