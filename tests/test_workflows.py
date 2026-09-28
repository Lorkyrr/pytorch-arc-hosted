"""Propriedades de segurança/robustez do gpu.yaml, o workflow que roda com o socket do Docker do host.

Um erro aqui não quebra teste nenhum de Python — só aparece na primeira run real (ou num ataque).
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

GPU = yaml.safe_load(Path(".github/workflows/gpu.yaml").read_text(encoding="utf-8"))
STEPS = GPU["jobs"]["run"]["steps"]


def _step(name):
    return next(s for s in STEPS if s.get("name") == name)


def test_never_triggered_by_push_or_pull_request():
    triggers = GPU[True]  # PyYAML (YAML 1.1) lê a chave `on` como True
    assert set(triggers) == {"workflow_run", "workflow_dispatch"}


def test_no_concurrency_group_that_would_drop_pending_runs():
    # O GitHub mantém só 1 run pendente por grupo: um treino na fila seria cancelado
    # pelo próximo benchmark. maxRunners: 1 + nvidia.com/gpu: 1 já serializam sem descartar.
    assert "concurrency" not in GPU
    assert "concurrency" not in GPU["jobs"]["run"]


def test_workflow_run_guard_rejects_forks_and_pull_requests():
    guard = GPU["jobs"]["run"]["if"]
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in guard
    assert "github.event.workflow_run.path == '.github/workflows/image.yaml'" in guard
    assert '["push","schedule","workflow_dispatch"]' in guard.replace(" ", "")


def test_every_action_is_pinned_by_commit_sha():
    uses = [s["uses"] for s in STEPS if "uses" in s]
    assert uses, "esperava ao menos o upload-artifact"
    for ref in uses:
        assert re.fullmatch(r"actions/[\w-]+@[0-9a-f]{40}", ref), f"{ref}: só actions/* e fixado por SHA"


def test_checkpoint_upload_survives_failure_timeout_and_cancel():
    upload = next(s for s in STEPS if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"].replace(" ", "").startswith("always()&&")


def test_run_step_removes_orphaned_containers_before_starting():
    script = _step("Rodar")["run"]
    cleanup = 'docker ps -aq --filter "name=^pah-" | xargs -r docker rm -f'
    assert cleanup in script
    assert script.index(cleanup) < script.index("docker run")
