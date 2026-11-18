"""The computer's name reaches no record unless the user names it: the
records' clock label ``clock_domain_id`` and the work directory of
``work_dir_job_namespace = always`` carry ``[main] device_name``, which
is empty by default."""
from __future__ import annotations

import os
import re
import socket

import pytest

from testuary import phase_state, project

HOST = "host-sentinel-7"


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: HOST)
    for name in ("TESTUARY_CLOCK_DOMAIN_ID", "TESTUARY_EXECUTION_ID",
                 "TESTUARY_MAIN_WORK_DIR_NAMESPACE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """A system whose boot has the ID ``boot-0f1e``."""
    path = tmp_path / "boot_id"
    path.write_text("boot-0f1e\n", encoding="ascii")
    monkeypatch.setattr(phase_state, "_BOOT_ID", path)
    return path


def test_the_clock_label_names_no_computer_by_default(host, boot, in_run, tmp_path,
                                                      monkeypatch):
    assert phase_state.clock_domain_id() == "boot-boot-0f1e"
    in_run()
    assert phase_state.clock_domain_id() == "boot-boot-0f1e"
    # Without a boot ID the label is this process's.
    monkeypatch.setattr(phase_state, "_BOOT_ID", tmp_path / "missing")
    label = phase_state.clock_domain_id()
    assert re.fullmatch(rf"process-{os.getpid()}-[0-9a-f]{{12}}", label)
    assert phase_state.clock_domain_id() == label


def test_the_device_name_labels_the_clock(host, boot, in_run, tmp_path, monkeypatch):
    in_run(overrides={"main": {"device_name": " lab-pc-3 "}})
    assert phase_state.clock_domain_id() == "boot-lab-pc-3-boot-0f1e"
    monkeypatch.setattr(phase_state, "_BOOT_ID", tmp_path / "missing")
    assert re.fullmatch(rf"process-lab-pc-3-{os.getpid()}-[0-9a-f]{{12}}",
                        phase_state.clock_domain_id())


def test_the_clock_variable_replaces_the_whole_label(host, boot, in_run, monkeypatch):
    monkeypatch.setenv("TESTUARY_CLOCK_DOMAIN_ID", "cluster-clock")
    in_run(overrides={"main": {"device_name": "lab-pc-3"}})
    assert phase_state.clock_domain_id() == "cluster-clock"


def test_an_own_work_dir_names_the_process_and_the_device_name_only(host, in_run):
    in_run(overrides={"main": {"work_dir_job_namespace": "always"}})
    namespace = project.resolve_work_dir_namespace()
    assert re.fullmatch(rf"job_local_{os.getpid()}_[0-9a-f]{{8}}", namespace)
    assert project.resolve_work_dir_namespace() == namespace
    in_run(overrides={"main": {"work_dir_job_namespace": "always",
                               "device_name": "lab pc/3"}})
    assert project.resolve_work_dir_namespace() == namespace.replace(
        "job_local_", "job_local_lab_pc_3_")
