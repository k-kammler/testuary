"""The check of target files against the project checkout's HEAD."""
from types import SimpleNamespace

from testuary import orchestrator
from testuary import config


def _git(monkeypatch, stat_stdout):
    def run(args, **kwargs):
        if "--quiet" in args:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=stat_stdout, stderr="")
    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr(config, "project_exec_mode", lambda: "local")


def _target(tmp_path):
    (tmp_path / ".git").mkdir()
    target = tmp_path / "pkg" / "mod.py"
    target.parent.mkdir()
    target.write_text("x = 1\n", encoding="utf-8")
    return target


def test_a_drifted_file_reports_the_stat_summary(tmp_path, monkeypatch):
    target = _target(tmp_path)
    _git(monkeypatch, " pkg/mod.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)\n")
    assert orchestrator._check_canonical_against_git(tmp_path, [target]) == [
        (target, " 1 file changed, 1 insertion(+), 1 deletion(-)")]


def test_a_drifted_file_without_a_stat_line_is_still_reported(tmp_path, monkeypatch):
    target = _target(tmp_path)
    _git(monkeypatch, "")
    assert orchestrator._check_canonical_against_git(tmp_path, [target]) == [(target, "")]
