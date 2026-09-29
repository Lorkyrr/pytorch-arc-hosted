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


JOB_LABEL = "pah-arc-hosted.job"


def test_run_step_removes_orphaned_containers_before_starting():
    script = _step("Rodar")["run"]
    cleanup = f'docker ps -aq --filter "label={JOB_LABEL}" | xargs -r docker rm -f'
    assert cleanup in script
    assert script.index(cleanup) < script.index("docker run")


def test_orphan_cleanup_can_only_hit_job_containers():
    # Regressão de 2026-09-29: `--filter name=^pah-` casou com `pah-control-plane`, o node do
    # cluster kind "pah", e o job apagou o próprio cluster. Filtro por label exclusivo, nunca por nome.
    script = _step("Rodar")["run"]
    assert "--filter \"name=" not in script and "--filter name=" not in script
    assert f"--label {JOB_LABEL}" in script  # o docker run marca o container que cria


CHECK_STEP = "Descobrir a GPU que o Kubernetes reservou pra este pod"
UUID = "GPU-18e10a1a-6e1e-a359-305e-37f93ae6d626"


def _run_check(tmp_path, env_value, smi_output):
    """Executa o script real do passo, com um nvidia-smi falso no PATH."""
    import shutil
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if smi_output is not None:
        smi = bin_dir / "nvidia-smi"
        smi.write_text(f"#!/bin/sh\nprintf '{smi_output}'\n")
        smi.chmod(0o755)
    github_env = tmp_path / "github_env"
    github_env.touch()
    # PATH isolado: o nvidia-smi REAL da máquina de teste não pode vazar pro caso "sem GPU"
    (bin_dir / "tr").symlink_to(shutil.which("tr"))
    env = {"PATH": str(bin_dir), "GITHUB_ENV": str(github_env)}
    if env_value is not None:
        env["NVIDIA_VISIBLE_DEVICES"] = env_value
    proc = subprocess.run(
        [shutil.which("bash"), "-e", "-c", _step(CHECK_STEP)["run"]], env=env, capture_output=True, text=True
    )
    return proc.returncode, github_env.read_text()


def test_gpu_uuid_from_device_plugin_envvar(tmp_path):
    assert _run_check(tmp_path, UUID, smi_output=None) == (0, f"GPU_UUID={UUID}\n")


def test_gpu_uuid_from_nvidia_smi_when_toolkit_sets_void(tmp_path):
    # Toolkit >= 1.20 injeta a GPU via CDI e troca a variável por "void" (visto em 2026-09-29)
    assert _run_check(tmp_path, "void", smi_output=f"{UUID}\\n") == (0, f"GPU_UUID={UUID}\n")


@pytest.mark.parametrize(
    ("env_value", "smi_output"),
    [
        ("void", None),  # sem nvidia-smi: o pod não recebeu GPU
        ("", ""),  # nada em lugar nenhum
        ("all", f"{UUID}\\nGPU-00000000-0000-0000-0000-000000000000\\n"),  # 2 GPUs visíveis: não é uma reserva
    ],
)
def test_gpu_check_refuses_anything_but_exactly_one_uuid(tmp_path, env_value, smi_output):
    code, written = _run_check(tmp_path, env_value, smi_output)
    assert code != 0 and written == ""


def test_docker_run_uses_the_resolved_uuid():
    script = _step("Rodar")["run"]
    assert '--gpus "device=$GPU_UUID"' in script
    assert "NVIDIA_VISIBLE_DEVICES" not in script
