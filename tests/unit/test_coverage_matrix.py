"""``GenericBackend.capture_coverage_matrix``, the per-line coverage sidecar."""
from __future__ import annotations

import json
from pathlib import Path

from testuary.languages.generic import GenericBackend
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def test_capture_coverage_matrix_records_covered_and_missing(tmp_path: Path) -> None:
    src = tmp_path / "mod.py"
    src.write_text(
        "def f(x):\n"
        "    if x > 0:\n"
        "        return 'pos'\n"
        "    return 'nonpos'\n",
        encoding="utf-8",
    )
    test = tmp_path / "test_mod.py"
    test.write_text(
        "import mod\n"
        "\n"
        "def test_pos():\n"
        "    assert mod.f(1) == 'pos'\n",
        encoding="utf-8",
    )
    out = tmp_path / "coverage_matrix.json"
    GenericBackend("python", (".py",)).capture_coverage_matrix(
        target=src, source_dir=tmp_path, test_paths=[str(test)],
        out_path=out, timeout=120,
    )

    assert out.exists(), "coverage matrix sidecar was not written"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert Path(data["target_file"]).name == "mod.py"
    assert "captured_at" not in data

    statuses = {r["line"]: r["status"] for r in data["lines"]}
    assert statuses.get(3) == "covered"
    assert statuses.get(4) == "missing"

    s = data["summary"]
    assert s["covered_lines"] >= 1
    assert s["missing_lines"] >= 1
    assert s["total_lines"] == s["covered_lines"] + s["missing_lines"]

    branches = {(b["from"], b["to"]): b["status"] for b in data["branches"]}
    assert branches.get((2, 3)) == "taken"
    assert branches.get((2, 4)) == "missing"
    assert s["covered_branches"] >= 1
    assert s["missing_branches"] >= 1


def test_capture_coverage_matrix_keeps_container_posix_target(
        monkeypatch, tmp_path, in_run) -> None:
    """In container mode the in-container POSIX target path is passed
    verbatim, with forward slashes, even when ``Path()`` on a Windows host
    spells it with backslashes."""
    from types import SimpleNamespace

    from testuary import project

    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "benchmark")
    in_run()
    # The work directory is a real host directory, the target an in-container path.
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    in_run()
    monkeypatch.setattr(project, "project_repo_dir", lambda: "/home/user/xarray")
    (tmp_path / "test_seed.py").write_text("def test_seed(): pass\n", encoding="utf-8")
    seen = {}

    def project_backend(self, root=None):
        seen["root"] = root
        return SimpleNamespace(
            coverage_matrix_in_place=True, setup_files=lambda root, paths: (),
            request_options=lambda purpose, target=None: {
                "isolation_config": True, "module_under_test": target},
            tests=SimpleNamespace(
                run_suite=lambda request: seen.setdefault("request", request) and SimpleNamespace(
                    coverage=None)))

    monkeypatch.setattr(GenericBackend, "project_backend", project_backend)

    target = Path("/home/user/xarray/xarray/core/groupby.py")
    GenericBackend("python", (".py",)).capture_coverage_matrix(
        target=target, source_dir=Path("/home/user/xarray"),
        test_paths=[str(tmp_path / "test_seed.py")],
        out_path=tmp_path / "out.json", timeout=120,
    )

    assert seen["root"] == "/home/user/xarray"
    request = seen["request"]
    assert request.coverage_target == "/home/user/xarray/xarray/core/groupby.py"
    assert request.module_under_test == request.coverage_target


def test_capture_coverage_matrix_pins_the_run_config(tmp_path: Path) -> None:
    """The sidecar is written even when the project's ``[coverage:run]`` turns
    on parallel data files, since the run uses its own rcfile."""
    src = tmp_path / "mod.py"
    src.write_text(
        "def f(x):\n"
        "    if x > 0:\n"
        "        return 'pos'\n"
        "    return 'nonpos'\n",
        encoding="utf-8",
    )
    (tmp_path / "setup.cfg").write_text(
        "[coverage:run]\nparallel = True\n", encoding="utf-8",
    )
    test = tmp_path / "test_mod.py"
    test.write_text(
        "import mod\n"
        "\n"
        "def test_pos():\n"
        "    assert mod.f(1) == 'pos'\n",
        encoding="utf-8",
    )
    out = tmp_path / "coverage_matrix.json"
    GenericBackend("python", (".py",)).capture_coverage_matrix(
        target=src, source_dir=tmp_path, test_paths=[str(test)],
        out_path=out, timeout=120,
    )
    assert out.exists(), "coverage matrix sidecar was not written"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["covered_lines"] >= 1

