from pathlib import Path

import pytest

from openollama_app import bootstrap

ROOT = Path("/x")


def test_plan_runs_when_textual_importable():
    assert bootstrap.plan(ROOT, True, False, False) == "run"


def test_plan_reexecs_into_existing_venv():
    assert bootstrap.plan(ROOT, False, True, True) == "reexec"


def test_plan_installs_when_uv_present():
    assert bootstrap.plan(ROOT, False, False, True) == "install"


def test_plan_manual_when_no_uv():
    assert bootstrap.plan(ROOT, False, False, False) == "manual"


def test_plan_inside_a_venv_without_textual_installs_instead_of_looping():
    assert bootstrap.plan(ROOT, False, True, True, in_venv=True) == "install"


def test_plan_inside_a_venv_without_textual_or_uv_is_manual():
    assert bootstrap.plan(ROOT, False, True, False, in_venv=True) == "manual"


def test_failed_install_removes_the_half_built_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.subprocess, "check_call", lambda cmd: (tmp_path / ".venv").mkdir(exist_ok=True))
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd: 1)
    with pytest.raises(SystemExit):
        bootstrap.install(tmp_path)
    assert not (tmp_path / ".venv").exists()
