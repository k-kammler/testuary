"""Settings and catalogue files stay out of project copies.

By default the source directory is the settings directory, so models.ini (which can
hold a key) lies beside the subject. Copies leave out every file read as
settings, copy subject files and resources, and leave the originals unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from testuary import orchestrator
from testuary import config
from testuary.orchestrator import RunRequest
from testuary import target_phases as phases
from testuary.settings import resolve_settings

KEY = "sk-synthetic-fixture-key"
CONFIGURATION = ("models.ini", "config.ini", "hints.json", "profile.json")
SUBJECT = ("module.py", "pyproject.toml", "resources/data.txt", "package/helper.py")

# A stand-in for a generated test: it searches the tree it runs in.
CHILD = r"""
import json, os, sys
found, keyed = [], []
for top, dirs, files in os.walk(sys.argv[1]):
    for name in files:
        path = os.path.join(top, name)
        rel = os.path.relpath(path, sys.argv[1]).replace(os.sep, "/")
        found.append(rel)
        with open(path, "rb") as handle:
            if sys.argv[2].encode() in handle.read():
                keyed.append(rel)
print(json.dumps({"found": found, "keyed": keyed}))
"""


def _project(tmp_path):
    root = tmp_path / "project"
    files = {
        "models.ini": f"[explicit]\nprovider = openai\nmodel = fixture\napi_url = https://example.test/v1\n"
                      f"api_key = {KEY}\n",
        "config.ini": "[main]\nprompt_profile = profile.json\n",
        "hints.json": "{}\n",
        "profile.json": '{"format": 1, "phases": {}}\n',
        "module.py": "def add(a, b):\n    return a + b\n",
        "pyproject.toml": "",
        "resources/data.txt": "resource\n",
        "package/helper.py": "VALUE = 1\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return root


def _snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_settings_name_the_configuration_files_they_read(tmp_path):
    root = _project(tmp_path)
    settings = resolve_settings(base_dir=root, home=root, environment={})
    assert {Path(path).name for path in settings.configuration_files} == {
        "models.ini", "config.ini"}
    assert config.configuration_exclusions(root, settings) == tuple(sorted(CONFIGURATION))
    # Prepared copies read the same absolute paths.
    assert sorted(Path(path).name for path in settings.consumer_files) == sorted(CONFIGURATION)
    assert all(Path(path).is_absolute() for path in settings.consumer_files)
    # Files outside the root are not exclusions of that root.
    assert config.configuration_exclusions(root / "package", settings) == ()
    # Explicit catalogues read nothing from disk.
    assert resolve_settings(base_dir=root, home=root, environment={}, models={},
                            config_files=[]).configuration_files == ()


def test_the_verification_copy_leaves_the_configuration_files_out(tmp_path):
    from langadapter import Environment, Interpreter, ProjectContext
    from testuary.readiness import _configuration_inputs

    root = _project(tmp_path)
    settings = resolve_settings(base_dir=root, home=root, environment={})
    context = ProjectContext(str(root), Interpreter(sys.executable), environment=Environment({}, inherit=False))
    selection = _configuration_inputs(context, settings)
    assert selection.exclude == tuple(sorted(CONFIGURATION))
    assert not any(selection.excludes(name) for name in SUBJECT)


def test_the_backend_measurement_copies_leave_the_same_files_out(tmp_path):
    from langadapter import Environment, Interpreter, ProjectContext
    from testuary.languages.preparation import project_inputs

    root = _project(tmp_path)
    settings = resolve_settings(base_dir=root, home=root, environment={})
    context = ProjectContext(str(root), Interpreter(sys.executable), environment=Environment({}, inherit=False))
    with config.use_run_settings(settings):
        selection = project_inputs(context)
    assert selection.exclude == tuple(sorted(CONFIGURATION))


def test_a_generated_test_child_in_the_default_journey_finds_no_catalogue_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])
    root = _project(tmp_path)
    before = _snapshot(root)
    output = tmp_path / "output"
    # No explicit source directory, so the settings directory is the project. The
    # harness's own TESTUARY_ variables are dropped.
    launch = {name: value for name, value in os.environ.items()
              if not name.startswith(("TESTUARY_", "LANGADAPTER_"))}
    settings = resolve_settings(base_dir=root, home=root, environment=launch,
                                overrides={"api": {"model": "explicit"},
                                           "project": {"output_dir": str(output)},
                                           "starter": {"concurrent_requests": 1}})
    seen = []

    def body(ctx):
        work = Path(ctx.top_file["file_path"]).parent
        child = subprocess.run([sys.executable, "-I", "-c", CHILD, str(output / "_work"), KEY],
                               cwd=work, capture_output=True, text=True, timeout=60, check=True)
        seen.append(json.loads(child.stdout))
        return phases.TestPipelineResult(stopped_after_phase="starter")

    monkeypatch.setattr(phases, "run_target_phases", body)
    request = RunRequest(("module.py",), stop_after_phase="starter")
    assert orchestrator.run_pipeline(request, settings=settings) == 0
    [observed] = seen
    names = {Path(rel).name for rel in observed["found"]}
    assert observed["keyed"] == []
    assert not names & set(CONFIGURATION)
    assert all(any(rel.endswith(subject) for rel in observed["found"]) for subject in SUBJECT)
    assert _snapshot(root) == before


def test_a_catalogue_outside_an_explicit_root_changes_nothing(tmp_path):
    root = _project(tmp_path)
    subject = root / "package"
    settings = resolve_settings(base_dir=root, home=root, environment={},
                                overrides={"project": {"source_dir": str(subject)}})
    with config.use_run_settings(settings):
        orchestrator._copy_project_to_work(projects_root=subject, work_src=tmp_path / "copy")
    assert sorted(path.name for path in (tmp_path / "copy").rglob("*")) == ["helper.py"]


def test_a_virtual_environment_inside_the_project_is_left_out(tmp_path):
    """The work copy leaves out every directory holding a pyvenv.cfg, since the
    run uses the environment in place. Other directories, even one named like an
    environment, are copied, and the original stays."""
    root = _project(tmp_path)
    package = root / ".venv" / "lib" / "site-packages" / "dependency"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (root / ".venv" / "pyvenv.cfg").write_bytes(b"home = /usr/bin\n")
    (root / "tools" / ".tox" / "py312").mkdir(parents=True)
    (root / "tools" / ".tox" / "py312" / "pyvenv.cfg").write_bytes(b"home = /usr/bin\n")
    (root / "env").mkdir()
    (root / "env" / "data.txt").write_bytes(b"a directory named like an environment\n")
    settings = resolve_settings(base_dir=root, home=root, environment={})
    with config.use_run_settings(settings):
        orchestrator._copy_project_to_work(projects_root=root, work_src=tmp_path / "copy")
    copied = {path.relative_to(tmp_path / "copy").as_posix() for path in (tmp_path / "copy").rglob("*")
              if path.is_file()}
    assert not any(name.startswith((".venv/", "tools/.tox/")) for name in copied)
    assert {"module.py", "resources/data.txt", "env/data.txt"} <= copied
    assert (root / ".venv" / "pyvenv.cfg").is_file()


@pytest.mark.parametrize("selected", ["hints.json", "nested/hints.json"])
def test_a_selected_hints_file_inside_the_project_is_left_out(tmp_path, selected):
    root = _project(tmp_path)
    (root / selected).parent.mkdir(parents=True, exist_ok=True)
    (root / selected).write_bytes(b"{}\n")
    settings = resolve_settings(base_dir=root, home=root, environment={}, overrides={"main": {"hints_file": selected}})
    with config.use_run_settings(settings):
        orchestrator._copy_project_to_work(projects_root=root, work_src=tmp_path / "copy")
    copied = {path.relative_to(tmp_path / "copy").as_posix() for path in (tmp_path / "copy").rglob("*")
              if path.is_file()}
    assert selected not in copied and "module.py" in copied and "resources/data.txt" in copied


def _directory_link(link, target):
    if os.name == "nt":
        command = Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"
        result = subprocess.run([str(command), "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, check=False)
        assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    else:
        link.symlink_to(target, target_is_directory=True)


def _child_sees(tree):
    child = subprocess.run([sys.executable, "-I", "-c", CHILD, str(tree), KEY],
                           capture_output=True, text=True, timeout=60, check=True)
    return json.loads(child.stdout)


def test_configuration_reached_through_a_directory_link_stays_out(tmp_path):
    # project/conf links outside the project, and settings and hints are
    # read through it.
    root = _project(tmp_path)
    shared = tmp_path / "shared-conf"
    shared.mkdir()
    (shared / "config.ini").write_bytes(b"[main]\nseed = 3\n")
    (shared / "hints.json").write_bytes(f'{{"_note": "{KEY}"}}\n'.encode())
    (shared / "data.txt").write_bytes(b"a resource reached through the link\n")
    _directory_link(root / "conf", shared)
    before = _snapshot(shared)
    try:
        settings = resolve_settings(base_dir=root, home=root, environment={}, config_files=["conf/config.ini"],
                                    overrides={"main": {"hints_file": "conf/hints.json"}})
        assert {"conf/config.ini", "conf/hints.json"} <= set(config.configuration_exclusions(root, settings))
        with config.use_run_settings(settings):
            orchestrator._copy_project_to_work(projects_root=root, work_src=tmp_path / "copy")
        seen = _child_sees(tmp_path / "copy")
        assert seen["keyed"] == []
        assert "conf/config.ini" not in seen["found"] and "conf/hints.json" not in seen["found"]
        assert "conf/data.txt" in seen["found"] and "module.py" in seen["found"]
        assert _snapshot(shared) == before
    finally:
        if os.name == "nt":
            os.rmdir(root / "conf")
        else:
            (root / "conf").unlink()


def test_a_linked_model_catalogue_stays_out(tmp_path):
    root = _project(tmp_path)
    shared = tmp_path / "shared-models.ini"
    (root / "models.ini").replace(shared)
    try:
        (root / "models.ini").symlink_to(shared)
    except OSError:
        pytest.skip("creating a file symbolic link needs a privilege here")
    settings = resolve_settings(base_dir=root, home=root, environment={})
    with config.use_run_settings(settings):
        orchestrator._copy_project_to_work(projects_root=root, work_src=tmp_path / "copy")
    seen = _child_sees(tmp_path / "copy")
    assert seen["keyed"] == [] and "models.ini" not in seen["found"]
    assert KEY in shared.read_text(encoding="utf-8")


def test_a_hard_link_to_the_catalogue_stays_out_of_the_verification_copy(tmp_path):
    # Only the file's identity shows that the hard link extra/creds.ini is
    # the catalogue.
    from langadapter import Environment, Interpreter, ProjectContext
    from testuary.readiness import _configuration_inputs

    root = _project(tmp_path)
    (root / "extra").mkdir()
    try:
        os.link(root / "models.ini", root / "extra" / "creds.ini")
    except OSError:
        pytest.skip("hard links are not available here")
    settings = resolve_settings(base_dir=root, home=root, environment={})
    assert "extra/creds.ini" in config.configuration_exclusions(root, settings)
    context = ProjectContext(str(root), Interpreter(sys.executable), environment=Environment({}, inherit=False))
    selection = _configuration_inputs(context, settings)
    assert selection.excludes("extra/creds.ini")
    assert not any(selection.excludes(name) for name in SUBJECT)
    # A file with one link costs no search of the tree.
    assert config.configuration_aliases(root, [root / "hints.json"]) == ()


def test_configuration_read_in_another_letter_case_stays_out(tmp_path, monkeypatch):
    # On a case-insensitive POSIX filesystem (macOS by default) normcase is a
    # no-op, so Models.ini or Conf/Config.ini must still match. A no-op
    # normcase reproduces that here.
    root = _project(tmp_path)
    (root / "models.ini").rename(root / "Models.ini")
    (root / "conf").mkdir()
    (root / "conf" / "config.ini").write_bytes(b"[main]\nseed = 3\n")
    (root / "conf" / "hints.json").write_bytes(f'{{"_note": "{KEY}"}}\n'.encode())
    (root / "sub").mkdir()
    (root / "sub" / "models.ini").write_bytes(b"[unread]\nprovider = openai\n")
    spelled = root / "Conf" / "Config.ini"
    if not spelled.exists() or not (root / "conf" / "config.ini").samefile(spelled):
        pytest.skip("the filesystem here is case-sensitive")
    monkeypatch.setattr(os.path, "normcase", lambda path: path)
    settings = resolve_settings(base_dir=root, home=root, environment={}, config_files=["Conf/Config.ini"],
                                overrides={"main": {"hints_file": "Conf/Hints.json"}})
    exclusions = set(config.configuration_exclusions(root, settings))
    # Prepared copies compare names as the directory lists them.
    assert {"Models.ini", "conf/config.ini", "conf/hints.json"} <= exclusions
    with config.use_run_settings(settings):
        orchestrator._copy_project_to_work(projects_root=root, work_src=tmp_path / "copy")
    seen = _child_sees(tmp_path / "copy")
    assert seen["keyed"] == []
    assert not {"Models.ini", "conf/config.ini", "conf/hints.json"} & set(seen["found"])
    assert "sub/models.ini" in seen["found"] and "module.py" in seen["found"]
