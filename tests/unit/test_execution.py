"""The project execution backends: LocalExecutor (default) and
ContainerExecutor, plus the resolver in project."""
import subprocess
from pathlib import Path

import pytest

from testuary import project as execution
from testuary import project
from python_runtime import project_python

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def test_container_executor_builds_docker_exec(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return subprocess.CompletedProcess(argv, 0, "out", "")

    # Unsupervised guest commands launch through the backend's run_capture.
    from langadapter import execution as backend_execution
    monkeypatch.setattr(backend_execution, "run_capture", fake_run)
    env = {"TESTUARY_X": "1", "TESTUARY_TIMING_SERIALIZE": "false",
           "PYTHONPATH": "/p", "PYTHONHASHSEED": "0",
           "HOME": "/home/host", "PATH": "/usr/bin"}
    execution.ContainerExecutor("c1", runner="docker").run(
        ["py", "-m", "pytest"], cwd="/testbed", env=env, timeout=99)
    argv = seen["argv"]
    assert argv[:2] == ["docker", "exec"]
    assert "-w" in argv and "/testbed" in argv
    assert "c1" in argv
    assert argv[-3:] == ["py", "-m", "pytest"]
    joined = " ".join(argv)
    # Run variables cross, the host's base environment (HOME, PATH) does not.
    assert "TESTUARY_X=1" in joined
    assert "TESTUARY_TIMING_SERIALIZE=false" in joined
    assert "PYTHONPATH=/p" in joined
    # PYTHONHASHSEED must cross into the container: in-container hash
    # randomization would make set/dict order (and coverage) nondeterministic.
    assert "PYTHONHASHSEED=0" in joined
    assert "HOME=/home/host" not in joined
    assert "PATH=/usr/bin" not in joined
    assert seen["kw"].get("timeout") == 99


def test_get_executor_container(monkeypatch, in_run):
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "tg_x")
    in_run()
    ex = project.get_executor()
    assert isinstance(ex, execution.ContainerExecutor)
    assert ex.container == "tg_x"


def test_get_executor_container_requires_name(monkeypatch, in_run):
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    monkeypatch.delenv("TESTUARY_PROJECT_CONTAINER", raising=False)
    in_run()
    with pytest.raises(RuntimeError):
        project.get_executor()


def test_the_project_python_in_a_container_skips_the_host_check(monkeypatch, in_run):
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    in_run()
    ctr_py = "/home/user/miniconda3/envs/x/bin/python"
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", ctr_py)
    in_run()
    # The interpreter lives in the container, not on this host. No error.
    assert project_python() == ctr_py


def test_a_missing_local_project_python_is_refused(monkeypatch, in_run):
    monkeypatch.delenv("TESTUARY_MAIN_PROJECT_EXEC", raising=False)
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "/no/such/python/xyz")
    in_run()
    with pytest.raises(RuntimeError):
        project_python()


def test_project_work_dir(monkeypatch, in_run):
    monkeypatch.delenv("TESTUARY_PROJECT_WORK_DIR", raising=False)
    in_run()
    assert project.project_work_dir() is None
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", "/wz")
    in_run()
    # A host path, resolved as the run resolves every host path.
    assert project.project_work_dir() == str(Path("/wz").resolve())


def test_project_repo_dir(monkeypatch, in_run):
    monkeypatch.delenv("TESTUARY_PROJECT_SOURCE_DIR", raising=False)
    in_run()
    assert project.project_repo_dir() is None
    monkeypatch.setenv("TESTUARY_PROJECT_SOURCE_DIR", "/home/swe/repo")
    in_run()
    assert project.project_repo_dir() == "/home/swe/repo"


# ---------------------------------------------------------------------------
# Project file I/O: through the executor in container mode, direct otherwise.
# ---------------------------------------------------------------------------

def test_read_write_project_text_local(monkeypatch, tmp_path, in_run):
    monkeypatch.delenv("TESTUARY_MAIN_PROJECT_EXEC", raising=False)
    in_run()
    f = tmp_path / "f.py"
    f.write_text("abc\n", encoding="utf-8")
    assert f.read_text(encoding="utf-8") == "abc\n"
    assert project.read_project_text(str(f)) == "abc\n"


def test_read_project_text_container_decodes_executor_output(monkeypatch, in_run):
    """A container's file is read with coreutils' base64, as the container's
    other commands use coreutils: no language's runtime, so a Java or C
    project in a container needs no Python selected."""
    import base64
    import hashlib
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    monkeypatch.delenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", raising=False)
    in_run()
    monkeypatch.setattr(project, "project_runtime",
                        lambda *args, **kwargs: pytest.fail("a container read needs no Python"))
    content = b"hello\x00bytes\r\n" * 20
    encoded = base64.encodebytes(content).decode()  # wrapped, as base64 prints it

    class _Exec:
        def run(self, argv, **kw):
            assert list(argv) == ["base64", "--", "/repo/m.py"]
            return subprocess.CompletedProcess(argv, 0, encoded, "")

    monkeypatch.setattr(project, "get_executor", lambda: _Exec())
    assert project.read_project_text("/repo/m.py") == content.decode().replace("\r\n", "\n")
    assert project.sha256_project_file("/repo/m.py") == hashlib.sha256(content).hexdigest()


def test_read_project_text_container_raises_on_failure(monkeypatch, in_run):
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    in_run()

    class _Exec:
        def run(self, argv, **kw):
            return subprocess.CompletedProcess(argv, 1, "", "no such file")

    monkeypatch.setattr(project, "get_executor", lambda: _Exec())
    with pytest.raises(OSError):
        project.read_project_text("/repo/missing.py")
    assert project.sha256_project_file("/repo/missing.py") is None


def test_read_file_content_routes_through_container(monkeypatch, in_run):
    """Every phase loads the source through read_file_content. In container
    mode it goes through the executor, not a host read."""
    import base64
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    in_run()

    class _Exec:
        def run(self, argv, **kw):
            return subprocess.CompletedProcess(
                argv, 0, base64.b64encode(b"def f(): pass\n").decode(), "")

    monkeypatch.setattr(project, "get_executor", lambda: _Exec())
    assert project.read_file_content("/repo/pkg/mod.py") == "def f(): pass\n"


def test_the_target_source_reads_the_same_in_both_execution_modes(monkeypatch, tmp_path, in_run):
    """UTF-8 with undecodable bytes replaced and CRLF or CR read as LF: the
    prompt's source and the measurement's read give one text locally and
    from a container."""
    import base64
    data = b"def f():\r\n    return '\xff'\rx = 1\n"
    target = tmp_path / "calc.py"
    target.write_bytes(data)
    monkeypatch.delenv("TESTUARY_MAIN_PROJECT_EXEC", raising=False)
    in_run()
    local = (project.read_file_content(str(target)), project.read_project_text(str(target)))
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    in_run()

    class _Exec:
        def run(self, argv, **kw):
            return subprocess.CompletedProcess(argv, 0, base64.b64encode(data).decode(), "")

    monkeypatch.setattr(project, "get_executor", lambda: _Exec())
    container = (project.read_file_content(str(target)), project.read_project_text(str(target)))
    expected = "def f():\n    return '\ufffd'\nx = 1\n"
    assert local == container == (expected, expected)


def test_scratch_dir_local_creates_and_removes(monkeypatch, tmp_path, in_run):
    """Local mode: a tempdir under the work directory, removed host-side on exit."""
    monkeypatch.delenv("TESTUARY_MAIN_PROJECT_EXEC", raising=False)
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    in_run()
    seen = {}
    with project.scratch_dir(prefix="testuary_t_") as td:
        seen["td"] = td
        assert Path(td).is_dir()
        assert Path(td).parent == tmp_path
        (Path(td) / "f.txt").write_text("x", encoding="utf-8")
    assert not Path(seen["td"]).exists()


def test_scratch_dir_container_removes_via_executor(monkeypatch, tmp_path, in_run):
    """Container mode: the executor removes the temp as the writer's user,
    so root-owned in-container files go too, which a host delete could not."""
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    in_run()
    calls = []

    class _Exec:
        def run(self, argv, *, cwd=None, env=None, timeout=None, input=None):
            calls.append([str(a) for a in argv])
            return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(project, "get_executor", lambda: _Exec())
    with project.scratch_dir(prefix="testuary_c_") as td:
        held = td
    assert any(c[:2] == ["rm", "-rf"] and c[-1] == held for c in calls)


def test_scratch_dir_container_survives_executor_failure(monkeypatch, tmp_path, in_run):
    """An executor-side removal failure does not propagate: the disposable
    temp may leak in the work directory but the run continues."""
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    in_run()

    class _Exec:
        def run(self, argv, **kw):
            raise RuntimeError("docker exec failed")

    monkeypatch.setattr(project, "get_executor", lambda: _Exec())
    with project.scratch_dir(prefix="testuary_c_") as td:
        assert Path(td).is_dir()
    # No exception escaped the context manager.
