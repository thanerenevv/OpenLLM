from pathlib import Path

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
