# pytorch-arc-hosted — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repositório novo que roda benchmark e treino de ResNet/CIFAR-10 numa RTX 3050 real. A imagem é buildada no GitHub, publicada no GHCR e executada via Docker num runner ARC self-hosted, dentro de um cluster kind.

**Architecture:** O pacote Python `pytorch_arc_hosted` roda dentro de uma única imagem Docker (base PyTorch mais nova, resolvida a cada build). `image.yaml` builda e publica; `gpu.yaml` é disparado por `workflow_run` e executa a imagem no daemon Docker do host via socket montado, preso à GPU que o k8s reservou para o pod (`--gpus device=$NVIDIA_VISIBLE_DEVICES`). `scripts/cluster.sh` sobe kind + device plugin + ARC.

**Tech Stack:** Python ≥3.10, PyTorch/torchvision (da imagem base), pytest, ruff, Docker/Buildx, GitHub Actions, GHCR, kind, Helm, ARC (`gha-runner-scale-set`), NVIDIA k8s-device-plugin, Bash.

**Spec:** [docs/specs/2026-09-24-pytorch-arc-hosted-design.md](../specs/2026-09-24-pytorch-arc-hosted-design.md). A **seção 8 (revisões)** sobrepõe as anteriores.

## Global Constraints

- **Nenhum download pesado na máquina do autor durante a implementação:** nada de `pip install torch`, `docker pull/build`, `kind create` ou `helm install`. Consultas de metadados (APIs do GitHub e do Docker Hub) são permitidas.
- Localmente não existe torch. Testes que precisam dele usam `pytest.importorskip("torch")` e rodam de verdade no `ci.yaml`.
- Sempre a versão mais nova: base resolvida por `scripts/latest_pytorch_base.py` com `MAX_CUDA: "13.4"`, chart do ARC sem `--version`, device plugin na release mais recente, runner `ghcr.io/actions/actions-runner:latest`, Actions nos majors `checkout@v7`, `setup-python@v7`, `upload-artifact@v7`, `docker/login-action@v4`, `docker/setup-buildx-action@v4`, `docker/build-push-action@v7`, e Dependabot semanal.
- Nunca usar `pytorch/pytorch:latest` (parada em 2024-02-23).
- Identificadores em inglês; comentários e saída de console em português (pt-BR com acentos).
- Imagem: `ghcr.io/lorkyrr/pytorch-arc-hosted` (minúsculo). Volume do dataset: `pah-cifar10`. Cluster kind: `pah`. Runner set: `arc-runner-set-gpu`. Secret: `pah-github-token`.
- `gpu.yaml` nunca tem gatilho `push`/`pull_request`.
- Commits terminam com `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Checagens locais disponíveis: `ruff`, `pytest` (sistema, Python 3.14, sem torch), `shellcheck`, `bash -n`, `python3 -c 'import yaml'` (se houver).

## Review Focus

1. **`--epochs 0` / `--batch-size -1` / `--lr 0`:** devem ser rejeitados no parsing, sem crash no meio do treino (`max()` de histórico vazio). O teste fica na Task 1 (`test_rejects_invalid`).
2. **Treino curto (`--epochs 1` ou `2`):** os marcos do LR nunca caem na época 0, e o treino completa e salva checkpoint. O teste fica na Task 4 (`test_lr_milestones_*`, `test_fit_*`).
3. **Container sem CUDA no CI de GPU:** o job tem que falhar (código 2), não ficar verde rodando em CPU. O teste fica na Task 1 (`test_require_gpu_fails_without_cuda`, que roda no ci.yaml onde não há GPU).
4. **Tag base nova com CUDA acima do driver** (ex.: `cuda13.6` com driver 13.4): o resolvedor tem que descartá-la. O teste fica na Task 5 (`test_respects_max_cuda`).
5. **Versões com dois dígitos** (`2.10.0` vs `2.9.1`): a comparação tem que ser numérica, não lexicográfica. O teste fica na Task 5 (`test_compares_versions_numerically`).

---

### Task 1: Esqueleto do repo + CLI

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `pytorch_arc_hosted/__init__.py`, `pytorch_arc_hosted/__main__.py`, `pytorch_arc_hosted/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `cli.ARCHS: tuple[str, ...] = ("resnet20", "resnet56", "resnet110")`; `cli.build_parser() -> argparse.ArgumentParser` (atributos `mode`, `require_gpu`, `arch`, `epochs`, `batch_size`, `lr`, `amp`, `seed`, `data_dir`, `out_dir`); `cli.main(argv: Sequence[str] | None = None) -> int` (0 = ok, 2 = `--require-gpu` sem CUDA).
- Consumes (importes tardios, dentro de `main`): `env.report_environment() -> torch.device` (Task 3), `bench.run_benchmark(device) -> None` (Task 3), `train.train_cifar10(device, *, arch, epochs, batch_size, lr, amp, seed, data_dir, out_dir) -> list[EpochStats]` (Task 4).

- [ ] **Step 1: Arquivos de projeto**

`pyproject.toml`:
```toml
[project]
name = "pytorch-arc-hosted"
version = "0.1.0"
description = "Benchmark de GPU e treino de ResNet/CIFAR-10 numa GPU real, via GitHub Actions + ARC + Docker"
requires-python = ">=3.10"
dependencies = ["torch", "torchvision"]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = [".", "scripts"]
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
data/
out/
*.pth
```

`LICENSE`: texto MIT padrão, `Copyright (c) 2026 Lorkyrr`.

`pytorch_arc_hosted/__init__.py`:
```python
"""Benchmark de GPU e treino de ResNet/CIFAR-10, feito pra rodar em Docker num runner ARC."""
```

`pytorch_arc_hosted/__main__.py`:
```python
import sys

from .cli import main

sys.exit(main())
```

- [ ] **Step 2: Teste que falha**

`tests/test_cli.py`:
```python
import pytest

from pytorch_arc_hosted.cli import build_parser, main


def parse(*argv):
    return build_parser().parse_args(argv)


def test_train_defaults():
    args = parse("train")
    assert (args.mode, args.arch, args.epochs, args.batch_size, args.lr) == ("train", "resnet20", 30, 128, 0.1)
    assert args.amp is True and args.seed is None and args.require_gpu is False
    assert (args.data_dir, args.out_dir) == ("/data", "/out")


def test_train_flags_come_after_the_subcommand():
    args = parse("train", "--arch", "resnet56", "--no-amp", "--seed", "7", "--require-gpu")
    assert (args.arch, args.amp, args.seed, args.require_gpu) == ("resnet56", False, 7, True)


def test_benchmark_accepts_require_gpu():
    assert parse("benchmark", "--require-gpu").require_gpu is True


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("train", "--arch", "resnet18"),
        ("train", "--epochs", "0"),
        ("train", "--batch-size", "-1"),
        ("train", "--lr", "0"),
        ("train", "--epochs", "abc"),
    ],
)
def test_rejects_invalid(argv):
    with pytest.raises(SystemExit):
        parse(*argv)


def test_require_gpu_fails_without_cuda():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("este teste é para máquina sem GPU")
    assert main(["benchmark", "--require-gpu"]) == 2
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `pytest tests/test_cli.py -v`
Expected: erro de import (`No module named 'pytorch_arc_hosted.cli'`).

- [ ] **Step 4: Implementar `cli.py`**

```python
"""Linha de comando: `python -m pytorch_arc_hosted {benchmark,train} [opções]`.

torch só é importado dentro de `main()`, depois do parsing: `--help` e os
testes de parsing rodam sem torch instalado.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

ARCHS = ("resnet20", "resnet56", "resnet110")


def _positive(kind: Callable[[str], float]) -> Callable[[str], float]:
    def parse(value: str) -> float:
        number = kind(value)
        if number <= 0:
            raise argparse.ArgumentTypeError(f"precisa ser > 0 (recebi {value})")
        return number

    return parse


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--require-gpu",
        action="store_true",
        help="sai com código 2 se CUDA não estiver disponível (no CI, job sem GPU não pode ficar verde)",
    )

    parser = argparse.ArgumentParser(
        prog="python -m pytorch_arc_hosted",
        description="Benchmark de GPU e treino de ResNet no CIFAR-10.",
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("benchmark", parents=[common], help="cuBLAS, cuDNN, AMP e treino sintético")

    train = modes.add_parser("train", parents=[common], help="treina uma ResNet (He et al., 2015) no CIFAR-10")
    train.add_argument("--arch", choices=ARCHS, default="resnet20")
    train.add_argument("--epochs", type=_positive(int), default=30)
    train.add_argument("--batch-size", type=_positive(int), default=128)
    train.add_argument("--lr", type=_positive(float), default=0.1)
    train.add_argument("--no-amp", dest="amp", action="store_false", help="desliga a precisão mista (FP32 puro)")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--data-dir", default="/data")
    train.add_argument("--out-dir", default="/out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .env import report_environment

    device = report_environment()
    if args.require_gpu and device.type != "cuda":
        print("[ERRO] --require-gpu: CUDA indisponível. Confira o --gpus do docker run e o CUDA máximo do driver.")
        return 2

    if args.mode == "benchmark":
        from .bench import run_benchmark

        run_benchmark(device)
    else:
        from .train import train_cifar10

        train_cifar10(
            device,
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            amp=args.amp,
            seed=args.seed,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
        )
    return 0
```

- [ ] **Step 5: Rodar e ver passar**

Run: `pytest tests/test_cli.py -v && ruff check .`
Expected: 9 passed, 1 skipped (`test_require_gpu_fails_without_cuda`, sem torch local), e ruff limpo.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore LICENSE pytorch_arc_hosted tests/test_cli.py
git commit -m "feat: esqueleto do pacote e CLI benchmark/train"
```

---

### Task 2: ResNet CIFAR (6n+2)

**Files:**
- Create: `pytorch_arc_hosted/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `cli.ARCHS`.
- Produces: `model.RESNET_BLOCKS: dict[str, int]`; `model.build_resnet(arch: str, num_classes: int = 10) -> ResNetCIFAR` (`ValueError` se `arch` desconhecida); `ResNetCIFAR.forward(x: Tensor[N,3,32,32]) -> Tensor[N,num_classes]`.

- [ ] **Step 1: Teste que falha**

`tests/test_model.py`:
```python
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from pytorch_arc_hosted.cli import ARCHS  # noqa: E402
from pytorch_arc_hosted.model import RESNET_BLOCKS, build_resnet  # noqa: E402


def test_cli_choices_match_model_variants():
    assert tuple(RESNET_BLOCKS) == ARCHS


def test_resnet20_parameter_count():
    # Opção B (projeção 1x1 nas 2 transições): 272.474. O paper (opção A) cita ~0,27M.
    assert sum(p.numel() for p in build_resnet("resnet20").parameters()) == 272_474


@pytest.mark.parametrize("arch", ARCHS)
def test_depth_is_6n_plus_2(arch):
    model = build_resnet(arch)
    weighted = [
        m for m in model.modules() if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3))
    ]
    assert len(weighted) == 6 * RESNET_BLOCKS[arch] + 2


@pytest.mark.parametrize("arch", ARCHS)
def test_output_shape(arch):
    model = build_resnet(arch).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="resnet18"):
        build_resnet("resnet18")
```

- [ ] **Step 2: Rodar**

Run: `pytest tests/test_model.py -v`
Expected (local, sem torch): `1 skipped`. A falha real (`ModuleNotFoundError: pytorch_arc_hosted.model`) só aparece no CI. Aceitável, pela Global Constraint.

- [ ] **Step 3: Implementar `model.py`**

```python
"""ResNet-(6n+2) para CIFAR-10, como no paper original (He et al., 2015, seção 4.2).

Construída do zero, não é `torchvision.models`: aquelas são as ResNets de
ImageNet (stem 7x7 + maxpool, pensadas pra 224x224), grandes demais pra 32x32.
"""

from __future__ import annotations

import torch
from torch import nn

# n blocos por estágio -> profundidade 6n+2
RESNET_BLOCKS = {"resnet20": 3, "resnet56": 9, "resnet110": 18}
STAGE_CHANNELS = (16, 32, 64)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Projeção 1x1 só quando a forma muda ("opção B" do paper); senão, identidade
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, blocks_per_stage: int, num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3, STAGE_CHANNELS[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(STAGE_CHANNELS[0]),
            nn.ReLU(inplace=True),
        ]
        in_channels = STAGE_CHANNELS[0]
        for stage, out_channels in enumerate(STAGE_CHANNELS):
            for block in range(blocks_per_stage):
                # o 1º bloco dos estágios 2 e 3 reduz a resolução pela metade (32 -> 16 -> 8)
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_resnet(arch: str, num_classes: int = 10) -> ResNetCIFAR:
    if arch not in RESNET_BLOCKS:
        raise ValueError(f"arquitetura desconhecida: {arch!r} (opções: {', '.join(RESNET_BLOCKS)})")
    return ResNetCIFAR(RESNET_BLOCKS[arch], num_classes)
```

- [ ] **Step 4: Verificar**

Run: `ruff check . && pytest -q`
Expected: ruff limpo e testes torch pulados localmente. A passagem real é conferida na Task 10 (CI).

- [ ] **Step 5: Commit**

```bash
git add pytorch_arc_hosted/model.py tests/test_model.py
git commit -m "feat: ResNet-20/56/110 para CIFAR-10"
```

---

### Task 3: Ambiente + benchmarks

**Files:**
- Create: `pytorch_arc_hosted/env.py`, `pytorch_arc_hosted/bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Produces: `env.banner(text: str) -> None`, `env.section(text: str) -> None`, `env.report_environment() -> torch.device`; `bench.gflops(n: int, seconds: float) -> float` (`ValueError` se `seconds <= 0`), `bench.run_benchmark(device: torch.device) -> None`. Constantes de módulo lidas na hora da chamada (os testes fazem monkeypatch): `MATMUL_SIZES`, `CONV_BATCH`, `CONV_SIDE`, `CONV_REPEATS`, `AMP_SIZE`, `SYNTH_BATCH`, `SYNTH_SIDE`, `SYNTH_CLASSES`, `SYNTH_STEPS`, `SYNTH_LR`.

- [ ] **Step 1: Teste que falha**

`tests/test_bench.py`:
```python
import pytest

torch = pytest.importorskip("torch")

from pytorch_arc_hosted import bench  # noqa: E402


def test_gflops_known_value():
    # 2 * 1000^3 = 2e9 operações em 2 s = 1 GFLOPS
    assert bench.gflops(1000, 2.0) == pytest.approx(1.0)


def test_gflops_rejects_non_positive_time():
    with pytest.raises(ValueError):
        bench.gflops(10, 0.0)


def test_benchmark_smoke_on_cpu(monkeypatch, capsys):
    monkeypatch.setattr(bench, "MATMUL_SIZES", (64,))
    monkeypatch.setattr(bench, "CONV_SIDE", 32)
    monkeypatch.setattr(bench, "CONV_REPEATS", 2)
    monkeypatch.setattr(bench, "SYNTH_SIDE", 16)
    monkeypatch.setattr(bench, "SYNTH_STEPS", 2)
    bench.run_benchmark(torch.device("cpu"))
    out = capsys.readouterr().out
    assert "GFLOPS" in out
    assert "pulado" in out  # AMP só roda em GPU
    assert "BENCHMARK CONCLUÍDO" in out
```

- [ ] **Step 2: Implementar `env.py`**

```python
"""Detecção do device, relatório do ambiente e helpers de saída no console."""

from __future__ import annotations

import torch

TENSOR_CORES_MIN_MAJOR = 7  # compute capability 7.0 (Volta) em diante


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def section(text: str) -> None:
    print(f"\n--- {text} ---")


def report_environment() -> torch.device:
    banner("AMBIENTE PYTORCH + GPU")
    print(f"[INFO] PyTorch {torch.__version__} | compilado com CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[AVISO] CUDA indisponível: rodando na CPU.")
        return torch.device("cpu")

    props = torch.cuda.get_device_properties(0)
    tensor_cores = "sim" if props.major >= TENSOR_CORES_MIN_MAJOR else "não"
    print(f"[INFO] GPU: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB | SMs: {props.multi_processor_count}")
    print(f"[INFO] Compute capability {props.major}.{props.minor} | Tensor Cores: {tensor_cores}")
    print(f"[INFO] cuDNN {torch.backends.cudnn.version()}")
    return torch.device("cuda")
```

- [ ] **Step 3: Implementar `bench.py`**

```python
"""Micro-benchmarks de GPU: cuBLAS, cuDNN, precisão mista e um treino sintético.

Os tamanhos cabem nos 4 GB de VRAM de uma RTX 3050 Laptop. Um OOM num tamanho
de matmul é registrado e o benchmark segue.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial

import torch
from torch import nn

from .env import banner, section

MATMUL_SIZES = (1000, 2000, 4000, 6000)
CONV_BATCH, CONV_SIDE, CONV_REPEATS = 16, 224, 20
AMP_SIZE = 4000
SYNTH_BATCH, SYNTH_SIDE, SYNTH_CLASSES, SYNTH_STEPS, SYNTH_LR = 32, 64, 10, 30, 1e-3


def gflops(n: int, seconds: float) -> float:
    """GFLOPS de um matmul n x n: 2*n^3 operações (uma multiplicação e uma soma por termo)."""
    if seconds <= 0:
        raise ValueError(f"tempo precisa ser > 0 (recebi {seconds})")
    return 2 * n**3 / seconds / 1e9


def _timed(device: torch.device, fn: Callable[[], object]) -> float:
    # Kernels CUDA são assíncronos: sem synchronize mediríamos só o enfileiramento
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def bench_matmul(device: torch.device) -> None:
    section("[1/4] Multiplicação de matrizes (cuBLAS)")
    warm = torch.randn(256, 256, device=device)
    _timed(device, partial(torch.mm, warm, warm))  # inicializa o cuBLAS fora da medição
    for n in MATMUL_SIZES:
        a = b = None
        try:
            a = torch.randn(n, n, device=device)
            b = torch.randn(n, n, device=device)
            seconds = _timed(device, partial(torch.mm, a, b))
            print(f" -> {n}x{n}: {seconds:.4f}s | {gflops(n, seconds):.1f} GFLOPS")
        except torch.OutOfMemoryError:
            print(f" -> {n}x{n}: [FALHOU] sem VRAM suficiente")
        finally:
            del a, b
            if device.type == "cuda":
                torch.cuda.empty_cache()


def bench_conv(device: torch.device) -> None:
    section("[2/4] Convolução 2D em lote (cuDNN)")
    images = torch.randn(CONV_BATCH, 3, CONV_SIDE, CONV_SIDE, device=device)
    net = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ).to(device)

    def repeat() -> None:
        for _ in range(CONV_REPEATS):
            net(images)

    with torch.no_grad():
        net(images)  # warm-up: a 1ª chamada inclui a escolha de algoritmo do cuDNN
        seconds = _timed(device, repeat) / CONV_REPEATS
    print(f" -> {CONV_BATCH} imagens {CONV_SIDE}x{CONV_SIDE}: {seconds * 1000:.2f} ms/lote")
    print(f" -> throughput: {CONV_BATCH / seconds:.1f} imagens/s")


def bench_amp(device: torch.device) -> None:
    # Micro-benchmark isolado de um matmul. A AMP de verdade, no laço de treino, está em train.py.
    section("[3/4] Precisão mista: FP32 vs FP16 (autocast)")
    if device.type != "cuda":
        print(" -> pulado (precisa de GPU)")
        return
    a = torch.randn(AMP_SIZE, AMP_SIZE, device=device)
    b = torch.randn(AMP_SIZE, AMP_SIZE, device=device)

    def fp16() -> None:
        with torch.autocast("cuda", dtype=torch.float16):
            torch.mm(a, b)

    fp32 = partial(torch.mm, a, b)
    fp32()
    fp16()  # warm-up dos dois caminhos
    t32, t16 = _timed(device, fp32), _timed(device, fp16)
    print(f" -> FP32: {t32:.4f}s | FP16: {t16:.4f}s | ganho: {t32 / t16:.2f}x")


def bench_synthetic_training(device: torch.device) -> None:
    section("[4/4] Treino sintético (forward + backward + otimizador)")
    flat = 64 * (SYNTH_SIDE // 2) ** 2
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(flat, 256),
        nn.ReLU(),
        nn.Linear(256, SYNTH_CLASSES),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SYNTH_LR)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(SYNTH_BATCH, 3, SYNTH_SIDE, SYNTH_SIDE, device=device)
    y = torch.randint(0, SYNTH_CLASSES, (SYNTH_BATCH,), device=device)
    losses: list[torch.Tensor] = []

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())  # sem .item() aqui: forçaria sync a cada passo

    def steps() -> None:
        for _ in range(SYNTH_STEPS):
            step()

    step()  # warm-up
    seconds = _timed(device, steps) / SYNTH_STEPS
    print(f" -> {SYNTH_STEPS} passos, lote {SYNTH_BATCH} ({SYNTH_SIDE}x{SYNTH_SIDE}): {seconds * 1000:.2f} ms/passo")
    print(f" -> throughput: {SYNTH_BATCH / seconds:.1f} amostras/s | loss final: {losses[-1].item():.4f}")


def run_benchmark(device: torch.device) -> None:
    banner("BENCHMARK")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for bench in (bench_matmul, bench_conv, bench_amp, bench_synthetic_training):
        bench(device)
    if device.type == "cuda":
        section("Memória da GPU")
        print(f" -> pico alocado: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    banner("BENCHMARK CONCLUÍDO")
```

- [ ] **Step 4: Verificar**

Run: `ruff check . && pytest -q`
Expected: ruff limpo e testes torch pulados localmente.

- [ ] **Step 5: Commit**

```bash
git add pytorch_arc_hosted/env.py pytorch_arc_hosted/bench.py tests/test_bench.py
git commit -m "feat: relatório do ambiente e micro-benchmarks de GPU"
```

---

### Task 4: Dados + treino com AMP

**Files:**
- Create: `pytorch_arc_hosted/data.py`, `pytorch_arc_hosted/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `model.build_resnet`, `env.banner`, `env.section`.
- Produces: `data.cifar10_loaders(data_dir: str, batch_size: int, pin_memory: bool, num_workers: int = 2) -> tuple[DataLoader, DataLoader]`; `train.EpochStats(epoch, train_loss, train_acc, test_loss, test_acc, seconds)`; `train.lr_milestones(epochs: int) -> list[int]`; `train.fit(model, train_loader, test_loader, *, device, epochs, lr, amp, checkpoint_path: Path, arch: str) -> list[EpochStats]`. O checkpoint é um dict `{"arch", "epoch", "test_acc", "state_dict"}`, salvo na 1ª época e a cada melhora estrita. Também produz `train.write_summary(history, arch, path: Path) -> None` e `train.train_cifar10(device, *, arch, epochs, batch_size, lr, amp, seed, data_dir, out_dir) -> list[EpochStats]` (grava `<out_dir>/<arch>_cifar10.pth` e `<out_dir>/summary.md`).

- [ ] **Step 1: Teste que falha**

`tests/test_train.py`:
```python
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from pytorch_arc_hosted.model import build_resnet  # noqa: E402
from pytorch_arc_hosted.train import EpochStats, fit, lr_milestones, write_summary  # noqa: E402


def _loader(n=32):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


def test_lr_milestones_default_run():
    assert lr_milestones(30) == [15, 22]


@pytest.mark.parametrize("epochs", [1, 2, 3])
def test_lr_milestones_short_runs_never_hit_epoch_zero(epochs):
    assert min(lr_milestones(epochs)) >= 1


def test_fit_learns_and_saves_reloadable_best_checkpoint(tmp_path):
    torch.manual_seed(0)
    loader = _loader()
    ckpt = tmp_path / "m.pth"
    history = fit(
        build_resnet("resnet20"),
        loader,
        loader,
        device=torch.device("cpu"),
        epochs=4,
        lr=0.05,
        amp=False,
        checkpoint_path=ckpt,
        arch="resnet20",
    )
    assert [s.epoch for s in history] == [1, 2, 3, 4]
    assert history[-1].train_loss < history[0].train_loss

    saved = torch.load(ckpt, weights_only=True)
    best = max(history, key=lambda s: s.test_acc)
    assert (saved["arch"], saved["epoch"], saved["test_acc"]) == ("resnet20", best.epoch, best.test_acc)
    build_resnet("resnet20").load_state_dict(saved["state_dict"])


def test_fit_single_epoch_still_writes_checkpoint(tmp_path):
    ckpt = tmp_path / "m.pth"
    fit(build_resnet("resnet20"), _loader(16), _loader(16), device=torch.device("cpu"),
        epochs=1, lr=0.1, amp=False, checkpoint_path=ckpt, arch="resnet20")
    assert ckpt.exists()


def test_write_summary_reports_first_best_epoch(tmp_path):
    history = [
        EpochStats(1, 2.0, 10.0, 2.1, 12.5, 1.0),
        EpochStats(2, 1.5, 40.0, 1.6, 55.0, 1.0),
        EpochStats(3, 1.2, 50.0, 1.7, 55.0, 1.0),
    ]
    path = tmp_path / "summary.md"
    write_summary(history, "resnet20", path)
    text = path.read_text(encoding="utf-8")
    assert "**Melhor acurácia no teste:** 55.00% (época 2 de 3)" in text
    assert text.count("\n| ") == 1 + 3  # cabeçalho + 3 épocas (a linha separadora começa com "|-")
```

Nota para quem implementar: o ruff formata a chamada longa do `test_fit_single_epoch_still_writes_checkpoint` em várias linhas se `ruff format` for usado. O CI só roda `ruff check`, então o formato acima é aceito.

- [ ] **Step 2: Implementar `data.py`**

```python
"""CIFAR-10 via torchvision: download automático, augmentation e DataLoaders."""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def cifar10_loaders(
    data_dir: str, batch_size: int, pin_memory: bool, num_workers: int = 2
) -> tuple[DataLoader, DataLoader]:
    normalize = [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), *normalize])
    test_tf = transforms.Compose(normalize)

    # download=True é no-op quando o volume pah-cifar10 já tem os arquivos
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return DataLoader(train_set, shuffle=True, **common), DataLoader(test_set, shuffle=False, **common)
```

- [ ] **Step 3: Implementar `train.py`**

```python
"""Treino da ResNet no CIFAR-10 com precisão mista (AMP) de verdade.

A AMP aqui é a do laço de treino (autocast + GradScaler). Não confundir com
`bench.bench_amp`, que só mede um matmul isolado.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import cifar10_loaders
from .env import banner, section
from .model import build_resnet

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_DECAY_AT = (0.5, 0.75)  # frações do treino em que o LR cai 10x
LR_GAMMA = 0.1


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    seconds: float


def lr_milestones(epochs: int) -> list[int]:
    # max(1, ...): em treinos curtos int(epochs * 0.5) daria 0, marco que o MultiStepLR nunca alcança
    return sorted({max(1, int(epochs * fraction)) for fraction in LR_DECAY_AT})


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                # O GradScaler evita que gradientes pequenos virem zero em FP16 (underflow)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += targets.size(0)
    return total_loss / seen, 100.0 * correct / seen


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    amp: bool,
    checkpoint_path: Path,
    arch: str,
) -> list[EpochStats]:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones(epochs), gamma=LR_GAMMA)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = nn.CrossEntropyLoss()

    history: list[EpochStats] = []
    best_acc = -1.0  # garante checkpoint já na 1ª época
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = _run_epoch(model, train_loader, loss_fn, device, amp, optimizer, scaler)
        test_loss, test_acc = _run_epoch(model, test_loader, loss_fn, device, amp)
        scheduler.step()
        stats = EpochStats(epoch, train_loss, train_acc, test_loss, test_acc, time.perf_counter() - start)
        history.append(stats)
        print(
            f"[época {epoch:03d}/{epochs}] treino loss={train_loss:.4f} acc={train_acc:.2f}% | "
            f"teste loss={test_loss:.4f} acc={test_acc:.2f}% | {stats.seconds:.1f}s"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {"arch": arch, "epoch": epoch, "test_acc": test_acc, "state_dict": model.state_dict()},
                checkpoint_path,
            )
    return history


def write_summary(history: list[EpochStats], arch: str, path: Path) -> None:
    best = max(history, key=lambda s: s.test_acc)  # empate: fica a 1ª, igual ao checkpoint
    lines = [
        f"## {arch} no CIFAR-10",
        "",
        f"**Melhor acurácia no teste:** {best.test_acc:.2f}% (época {best.epoch} de {len(history)})",
        "",
        "| Época | loss treino | acc treino | loss teste | acc teste | tempo |",
        "|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {s.epoch} | {s.train_loss:.4f} | {s.train_acc:.2f}% | {s.test_loss:.4f} | {s.test_acc:.2f}% "
            f"| {s.seconds:.1f}s |"
            for s in history
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_cifar10(
    device: torch.device,
    *,
    arch: str,
    epochs: int,
    batch_size: int,
    lr: float,
    amp: bool,
    seed: int | None,
    data_dir: str,
    out_dir: str,
) -> list[EpochStats]:
    banner(f"TREINO: {arch} NO CIFAR-10")
    if seed is not None:
        torch.manual_seed(seed)
    amp = amp and device.type == "cuda"  # CPU não tem Tensor Cores: FP16 lá só atrapalha
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] épocas={epochs} lote={batch_size} lr={lr} amp={'sim' if amp else 'não'} seed={seed}")

    train_loader, test_loader = cifar10_loaders(data_dir, batch_size, pin_memory=device.type == "cuda")
    model = build_resnet(arch).to(device)
    print(f"[INFO] parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    section("Progresso")
    history = fit(
        model,
        train_loader,
        test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        amp=amp,
        checkpoint_path=out / f"{arch}_cifar10.pth",
        arch=arch,
    )
    write_summary(history, arch, out / "summary.md")

    best = max(history, key=lambda s: s.test_acc)
    section("Resultado")
    print(f" -> melhor acurácia no teste: {best.test_acc:.2f}% (época {best.epoch})")
    print(f" -> checkpoint e resumo em {out}")
    return history
```

- [ ] **Step 4: Verificar**

Run: `ruff check . && pytest -q`
Expected: ruff limpo e testes torch pulados localmente.

- [ ] **Step 5: Commit**

```bash
git add pytorch_arc_hosted/data.py pytorch_arc_hosted/train.py tests/test_train.py
git commit -m "feat: treino da ResNet no CIFAR-10 com AMP, checkpoint e resumo"
```

---

### Task 5: Resolvedor da imagem base mais nova

**Files:**
- Create: `scripts/latest_pytorch_base.py`
- Test: `tests/test_latest_pytorch_base.py`

**Interfaces:**
- Produces: `parse_cuda(value: str) -> tuple[int, int]`; `pick_latest(tags: Iterable[str], max_cuda: tuple[int, int] | None = None) -> str` (`ValueError` se nenhuma tag servir); `fetch_tags(max_pages: int = 5) -> list[str]`; a CLI `python3 scripts/latest_pytorch_base.py [--max-cuda X.Y]` imprime `pytorch/pytorch:<tag>`.
- Consumido por: `image.yaml` (Task 7).

- [ ] **Step 1: Teste que falha** (roda localmente: só stdlib)

`tests/test_latest_pytorch_base.py`:
```python
import pytest

from latest_pytorch_base import parse_cuda, pick_latest

TAGS = [
    "latest",
    "2.14.0-cuda12.6-cudnn9-runtime",
    "2.14.0-cuda13.2-cudnn9-runtime",
    "2.14.0-cuda13.2-cudnn9-devel",
    "2.13.0-cuda13.2-cudnn9-runtime",
]


def test_picks_newest_torch_then_newest_cuda():
    assert pick_latest(TAGS) == "2.14.0-cuda13.2-cudnn9-runtime"


def test_respects_max_cuda():
    assert pick_latest(TAGS + ["2.15.0-cuda13.6-cudnn9-runtime"], max_cuda=(13, 4)) == "2.14.0-cuda13.2-cudnn9-runtime"
    assert pick_latest(TAGS, max_cuda=(12, 9)) == "2.14.0-cuda12.6-cudnn9-runtime"


def test_compares_versions_numerically():
    assert pick_latest(["2.9.1-cuda12.8-cudnn9-runtime", "2.10.0-cuda12.8-cudnn9-runtime"]) == (
        "2.10.0-cuda12.8-cudnn9-runtime"
    )


def test_fails_when_nothing_matches():
    with pytest.raises(ValueError):
        pick_latest(["latest", "2.14.0-cuda13.2-cudnn9-devel"])


def test_parse_cuda():
    assert parse_cuda("13.4") == (13, 4)
    with pytest.raises(ValueError):
        parse_cuda("13")
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_latest_pytorch_base.py -v`
Expected: `ModuleNotFoundError: No module named 'latest_pytorch_base'`.

- [ ] **Step 3: Implementar**

`scripts/latest_pytorch_base.py`:
```python
#!/usr/bin/env python3
"""Descobre a tag mais nova de pytorch/pytorch no formato <torch>-cuda<X.Y>-cudnn<N>-runtime.

`pytorch/pytorch:latest` não serve: está parada em 2024-02-23 (PyTorch 2.2.1).
Critério: maior versão do torch e, empatando, maior CUDA, sem passar de --max-cuda
(o CUDA máximo que o driver do host suporta; ver `nvidia-smi`).

Uso: python3 scripts/latest_pytorch_base.py --max-cuda 13.4
Só stdlib: roda no runner GitHub-hosted sem instalar nada.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Iterable, Sequence

REPO = "pytorch/pytorch"
HUB_URL = f"https://hub.docker.com/v2/repositories/{REPO}/tags?page_size=100&ordering=last_updated"
TAG_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)-cuda(\d+)\.(\d+)-cudnn\d+-runtime$")


def parse_cuda(value: str) -> tuple[int, int]:
    major, dot, minor = value.partition(".")
    if not dot:
        raise ValueError(f"CUDA no formato X.Y (recebi {value!r})")
    return int(major), int(minor)


def pick_latest(tags: Iterable[str], max_cuda: tuple[int, int] | None = None) -> str:
    candidates = []
    for tag in tags:
        match = TAG_RE.match(tag)
        if not match:
            continue
        numbers = tuple(int(g) for g in match.groups())
        torch_version, cuda_version = numbers[:3], numbers[3:]
        if max_cuda is not None and cuda_version > max_cuda:
            continue
        candidates.append((torch_version, cuda_version, tag))
    if not candidates:
        raise ValueError("nenhuma tag <torch>-cuda<X.Y>-cudnn<N>-runtime compatível encontrada")
    return max(candidates)[2]


def fetch_tags(max_pages: int = 5) -> list[str]:
    url: str | None = HUB_URL
    tags: list[str] = []
    for _ in range(max_pages):
        if url is None:
            break
        with urllib.request.urlopen(url, timeout=30) as response:
            page = json.load(response)
        tags += [result["name"] for result in page["results"]]
        url = page.get("next")
    return tags


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Imprime a imagem pytorch/pytorch runtime mais nova.")
    parser.add_argument("--max-cuda", type=parse_cuda, default=None, help="ex.: 13.4 (CUDA máximo do driver)")
    args = parser.parse_args(argv)
    print(f"{REPO}:{pick_latest(fetch_tags(), args.max_cuda)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Rodar e ver passar**

Run: `chmod +x scripts/latest_pytorch_base.py && pytest tests/test_latest_pytorch_base.py -v && ruff check .`
Expected: 5 passed e ruff limpo. Opcional (só metadados, ~100 KB): `python3 scripts/latest_pytorch_base.py --max-cuda 13.4` imprime `pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime` ou mais nova.

- [ ] **Step 5: Commit**

```bash
git add scripts/latest_pytorch_base.py tests/test_latest_pytorch_base.py
git commit -m "feat: resolvedor da imagem base PyTorch mais nova (limitada pelo CUDA do driver)"
```

---

### Task 6: Dockerfile, .dockerignore, compose

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `compose.yaml`

**Interfaces:**
- Produces: a imagem com `ENTRYPOINT ["python", "-m", "pytorch_arc_hosted"]` e `CMD ["benchmark"]`, `ARG BASE`, volume `/data` e saída em `/out`.

- [ ] **Step 1: `Dockerfile`**

```dockerfile
# Imagem única do projeto: roda no CI (gpu.yaml) e localmente (compose.yaml).
#
# BASE é resolvida a cada build no image.yaml (scripts/latest_pytorch_base.py):
# a tag mais nova <torch>-cuda<X.Y>-cudnn<N>-runtime que o driver do host aguenta.
# O default abaixo só vale pra build local sem --build-arg. Não troque por
# pytorch/pytorch:latest: essa tag está parada em 2024 (PyTorch 2.2.1).
ARG BASE=pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime
FROM ${BASE}

LABEL org.opencontainers.image.source="https://github.com/Lorkyrr/pytorch-arc-hosted"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# torch e torchvision vêm da base. Se um dia deixarem de vir, o build falha aqui, e não no meio do treino.
RUN python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"

WORKDIR /app
COPY pytorch_arc_hosted/ pytorch_arc_hosted/

ENTRYPOINT ["python", "-m", "pytorch_arc_hosted"]
CMD ["benchmark"]
```

- [ ] **Step 2: `.dockerignore`**

```
*
!pytorch_arc_hosted/
**/__pycache__
```

- [ ] **Step 3: `compose.yaml`**

```yaml
# Uso local, fora do CI. Mesma imagem e mesmo volume de dataset do gpu.yaml.
#   docker compose run --rm app                         # benchmark
#   docker compose run --rm app train --epochs 30       # treino (checkpoint em ./out)
services:
  app:
    image: ghcr.io/lorkyrr/pytorch-arc-hosted:latest
    build: .
    shm_size: 1gb # os workers do DataLoader trocam lotes por memória compartilhada
    volumes:
      - cifar10:/data
      - ./out:/out
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  cifar10:
    name: pah-cifar10 # sem o prefixo do compose: é o mesmo volume que o CI usa
```

- [ ] **Step 4: Verificar sem build**

Run: `docker compose config --quiet && echo compose-ok`. Isso só valida a sintaxe, sem pull/build. Também confira a lista do contexto: `grep -c . .dockerignore` deve dar 3.
Expected: `compose-ok`.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore compose.yaml
git commit -m "feat: imagem Docker única (base PyTorch resolvida no build) e compose local"
```

---

### Task 7: Workflows + Dependabot

**Files:**
- Create: `.github/workflows/ci.yaml`, `.github/workflows/image.yaml`, `.github/workflows/gpu.yaml`, `.github/dependabot.yml`

**Interfaces:**
- Consumes: `scripts/latest_pytorch_base.py --max-cuda`, a CLI (`benchmark|train --require-gpu ...`), runner label `arc-runner-set-gpu`, volume `pah-cifar10`, `/out/summary.md`.
- O nome do workflow `Imagem (GHCR)` é referenciado literalmente pelo `gpu.yaml`.

- [ ] **Step 1: `ci.yaml`**

```yaml
name: CI

on:
  push:
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.x" # sempre o Python estável mais novo
      - name: Instalar torch (CPU) e ferramentas
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
          pip install pytest ruff
      - name: Lint
        run: ruff check .
      - name: Testes (inclui ResNet, treino e benchmark em CPU)
        run: pytest -v
```

- [ ] **Step 2: `image.yaml`**

```yaml
name: Imagem (GHCR)

on:
  push:
    branches: [main]
    paths:
      - Dockerfile
      - .dockerignore
      - pytorch_arc_hosted/**
      - scripts/latest_pytorch_base.py
      - .github/workflows/image.yaml
  schedule:
    - cron: "0 6 * * 1" # segunda 06:00 UTC: pega base nova mesmo sem commit
  workflow_dispatch:

permissions:
  contents: read
  packages: write

env:
  IMAGE: ghcr.io/lorkyrr/pytorch-arc-hosted
  # Maior CUDA que o driver do host suporta (canto superior direito do nvidia-smi).
  # Atualize junto com o driver. Hoje: driver 615 -> CUDA 13.4.
  MAX_CUDA: "13.4"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7

      - name: Resolver a imagem base mais nova
        id: base
        run: echo "image=$(python3 scripts/latest_pytorch_base.py --max-cuda "$MAX_CUDA")" >> "$GITHUB_OUTPUT"

      - uses: docker/setup-buildx-action@v4

      - uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v7
        with:
          context: .
          pull: true
          push: true
          build-args: BASE=${{ steps.base.outputs.image }}
          labels: org.opencontainers.image.base.name=${{ steps.base.outputs.image }}
          tags: |
            ${{ env.IMAGE }}:${{ github.sha }}
            ${{ env.IMAGE }}:latest

      - name: Resumo
        env:
          BASE_IMAGE: ${{ steps.base.outputs.image }}
        run: |
          {
            echo "### Imagem publicada"
            echo "- \`$IMAGE:$GITHUB_SHA\` (e \`:latest\`)"
            echo "- base: \`$BASE_IMAGE\` (MAX_CUDA=$MAX_CUDA)"
          } >> "$GITHUB_STEP_SUMMARY"
```

- [ ] **Step 3: `gpu.yaml`**

```yaml
name: GPU (RTX 3050)

on:
  workflow_run:
    workflows: ["Imagem (GHCR)"]
    types: [completed]
  workflow_dispatch:
    inputs:
      mode:
        description: Modo
        type: choice
        options: [benchmark, train]
        default: benchmark
      arch:
        description: Arquitetura (treino)
        type: choice
        options: [resnet20, resnet56, resnet110]
        default: resnet20
      epochs:
        description: Épocas (treino)
        default: "30"
      batch_size:
        description: Tamanho do lote (treino)
        default: "128"
      lr:
        description: Taxa de aprendizado inicial (treino)
        default: "0.1"
      amp:
        description: Precisão mista (AMP)
        type: boolean
        default: true
      seed:
        description: Semente (vazio = aleatória)
        default: ""
      image_tag:
        description: Tag da imagem no GHCR (sha de um commit ou latest)
        default: latest

permissions:
  contents: read
  packages: read

# Só existe uma GPU física: jobs esperam na fila, nunca se cancelam
concurrency:
  group: gpu
  cancel-in-progress: false

jobs:
  run:
    # Segurança: este job roda na SUA máquina, com acesso ao Docker do host.
    # Nunca tem gatilho de push/PR, e só aceita imagem buildada a partir da main.
    if: >-
      github.event_name == 'workflow_dispatch' ||
      (github.event.workflow_run.conclusion == 'success' && github.event.workflow_run.head_branch == 'main')
    runs-on: arc-runner-set-gpu
    timeout-minutes: 120
    env:
      IMAGE: ghcr.io/lorkyrr/pytorch-arc-hosted:${{ github.event.workflow_run.head_sha || inputs.image_tag }}
      CONTAINER: pah-${{ github.run_id }}-${{ github.run_attempt }}
      MODE: ${{ inputs.mode || 'benchmark' }}
    steps:
      - name: Conferir a GPU que o Kubernetes reservou pra este pod
        run: |
          echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<vazio>}"
          case "${NVIDIA_VISIBLE_DEVICES:-}" in
            "" | all | none | void)
              echo "::error::o pod não recebeu uma GPU específica do device plugin. Rode 'scripts/cluster.sh status'."
              exit 1 ;;
          esac

      - uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Baixar a imagem (só as camadas que o host ainda não tem)
        run: docker pull "$IMAGE"

      - name: Rodar
        env:
          ARCH: ${{ inputs.arch }}
          EPOCHS: ${{ inputs.epochs }}
          BATCH_SIZE: ${{ inputs.batch_size }}
          LR: ${{ inputs.lr }}
          AMP: ${{ inputs.amp }}
          SEED: ${{ inputs.seed }}
        run: |
          args=("$MODE" --require-gpu)
          if [ "$MODE" = train ]; then
            args+=(--arch "$ARCH" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR")
            if [ "$AMP" = false ]; then args+=(--no-amp); fi
            if [ -n "$SEED" ]; then args+=(--seed "$SEED"); fi
          fi
          # O docker daqui fala com o daemon do HOST (socket montado): o container
          # é "irmão" do pod, preso à mesma GPU que o k8s reservou para este pod.
          docker run --name "$CONTAINER" --gpus "device=$NVIDIA_VISIBLE_DEVICES" --shm-size=1g \
            -v pah-cifar10:/data "$IMAGE" "${args[@]}"

      - name: Coletar resultados e limpar
        if: always()
        run: |
          mkdir -p out
          docker cp "$CONTAINER:/out/." out/ 2>/dev/null || true
          if [ -f out/summary.md ]; then cat out/summary.md >> "$GITHUB_STEP_SUMMARY"; fi
          docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

      - uses: actions/upload-artifact@v7
        if: env.MODE == 'train'
        with:
          name: ${{ inputs.arch }}-cifar10-${{ github.run_id }}
          path: out/
          if-no-files-found: warn
```

- [ ] **Step 4: `dependabot.yml`**

```yaml
version: 2
updates:
  - package-ecosystem: github-actions
    directory: /
    schedule:
      interval: weekly
```

- [ ] **Step 5: Verificar sintaxe localmente**

Run:
```bash
python3 -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/**/*.y*ml', recursive=True)]; print('yaml-ok')"
grep -n "pull_request\|push:" .github/workflows/gpu.yaml || echo "gpu.yaml sem push/PR: ok"
```
Expected: `yaml-ok` e `gpu.yaml sem push/PR: ok`. (Se não houver PyYAML, trocar por `ruby -ryaml -e` ou pular; o GitHub valida no push, na Task 10.)

- [ ] **Step 6: Commit**

```bash
git add .github
git commit -m "ci: testes em CPU, build da imagem no GHCR e execução na GPU via ARC"
```

---

### Task 8: Kubernetes + `cluster.sh`

**Files:**
- Create: `k8s/kind-config.yaml`, `k8s/runner-values.yaml`, `scripts/cluster.sh`

**Interfaces:**
- Produces: `scripts/cluster.sh up [owner/repo] | status | down`; cluster `pah` (contexto `kind-pah`); namespace `arc-systems`; releases `arc` (controller) e `arc-runner-set-gpu`; secret `pah-github-token` (chave `github_token`).
- `runner-values.yaml` recebe via `--set`: `githubConfigUrl` e `template.spec.securityContext.supplementalGroups[0]`. **Não** usar `--set` em `template.spec.containers[...]`: o Helm substitui listas inteiras e o container perderia `command`/`resources`.

- [ ] **Step 1: `k8s/kind-config.yaml`**

```yaml
# Cluster kind "pah": um node, com a GPU do host e o socket do Docker do host.
# Pré-requisito no host: Docker com runtime padrão "nvidia" e
# accept-nvidia-visible-devices-as-volume-mounts = true (ver README).
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraMounts:
      # Caminho "mágico" do NVIDIA Container Runtime: montar algo aqui faz ele
      # injetar driver e /dev/nvidia* no container do node.
      - hostPath: /dev/null
        containerPath: /var/run/nvidia-container-devices/all
      # Binários do Container Toolkit (a injeção automática só traz as libs do driver)
      - hostPath: /usr/bin/nvidia-container-runtime
        containerPath: /usr/bin/nvidia-container-runtime
        readOnly: true
      - hostPath: /usr/bin/nvidia-container-cli
        containerPath: /usr/bin/nvidia-container-cli
        readOnly: true
      - hostPath: /usr/bin/nvidia-ctk
        containerPath: /usr/bin/nvidia-ctk
        readOnly: true
      # Socket do Docker do HOST: os pods runner o recebem via hostPath e rodam
      # `docker pull/run` no daemon do host (cache de camadas persistente).
      - hostPath: /var/run/docker.sock
        containerPath: /var/run/docker.sock
containerdConfigPatches:
  # O containerd INTERNO do node (não o Docker do host) também precisa do runtime
  # nvidia como padrão; sem isso, o device plugin não enxerga a GPU.
  - |-
    [plugins."io.containerd.grpc.v1.cri".containerd]
      default_runtime_name = "nvidia"
    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
      runtime_type = "io.containerd.runc.v2"
      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
        BinaryName = "/usr/bin/nvidia-container-runtime"
```

- [ ] **Step 2: `k8s/runner-values.yaml`**

```yaml
# Values do chart gha-runner-scale-set (ARC): o runner set com GPU.
# Aplicado pelo scripts/cluster.sh, que ainda injeta via --set:
#   githubConfigUrl                                     -> https://github.com/<owner>/<repo>
#   template.spec.securityContext.supplementalGroups[0] -> GID do /var/run/docker.sock do host
# (o GID muda de máquina pra máquina; a imagem do runner cria o grupo docker com GID 123)
githubConfigSecret: pah-github-token
minRunners: 0
maxRunners: 1 # uma GPU física = um runner por vez

template:
  spec:
    containers:
      - name: runner
        image: ghcr.io/actions/actions-runner:latest
        command: ["/home/runner/run.sh"]
        resources:
          limits:
            # Reserva a GPU no scheduler. O device plugin injeta
            # NVIDIA_VISIBLE_DEVICES=<UUID>, que o gpu.yaml repassa ao docker run.
            nvidia.com/gpu: 1
        volumeMounts:
          - name: docker-sock
            mountPath: /var/run/docker.sock
    volumes:
      - name: docker-sock
        hostPath:
          path: /var/run/docker.sock
          type: Socket
```

- [ ] **Step 3: `scripts/cluster.sh`**

```bash
#!/usr/bin/env bash
# Ambiente local do pytorch-arc-hosted: cluster kind com a GPU do host + ARC.
#
#   scripts/cluster.sh up [owner/repo]   cria/atualiza tudo (idempotente, não apaga nada)
#   scripts/cluster.sh status            GPU alocável, versões e pods do ARC
#   scripts/cluster.sh down              apaga o cluster (cache de imagens e dataset ficam no host)
#
# Sempre instala as versões mais novas (chart do ARC, device plugin, runner).
# Token do GitHub: $GITHUB_TOKEN ou `gh auth token`, nunca gravado em disco.
set -euo pipefail

CLUSTER=pah
CONTEXT=kind-$CLUSTER
NAMESPACE=arc-systems
RUNNER_SET=arc-runner-set-gpu
SECRET=pah-github-token
CHARTS=oci://ghcr.io/actions/actions-runner-controller-charts
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\n[ERRO] %s\n' "$*" >&2; exit 1; }
k() { kubectl --context "$CONTEXT" "$@"; }
h() { helm --kube-context "$CONTEXT" "$@"; }

check_prereqs() {
  local missing=() bin
  for bin in docker kind kubectl helm nvidia-smi curl; do
    command -v "$bin" >/dev/null || missing+=("comando '$bin' não encontrado")
  done
  if command -v docker >/dev/null; then
    if ! docker info >/dev/null 2>&1; then
      missing+=("sem acesso ao Docker (seu usuário está no grupo 'docker'?)")
    elif [ "$(docker info --format '{{.DefaultRuntime}}')" != nvidia ]; then
      missing+=("runtime padrão do Docker não é 'nvidia' (README: 'Preparar o host', passo 3)")
    fi
  fi
  grep -Eq '^[[:space:]]*accept-nvidia-visible-devices-as-volume-mounts[[:space:]]*=[[:space:]]*true' \
    /etc/nvidia-container-runtime/config.toml 2>/dev/null ||
    missing+=("falta 'accept-nvidia-visible-devices-as-volume-mounts = true' em /etc/nvidia-container-runtime/config.toml")
  [ -S /var/run/docker.sock ] || missing+=("socket /var/run/docker.sock não existe")
  if [ ${#missing[@]} -gt 0 ]; then
    printf ' - %s\n' "${missing[@]}" >&2
    die "pré-requisitos faltando (ver README, seção 'Preparar o host')"
  fi
}

check_other_clusters() {
  [ "${ALLOW_OTHER_CLUSTERS:-0}" = 1 ] && return
  local others
  others=$(kind get clusters 2>/dev/null | grep -vx "$CLUSTER" || true)
  [ -z "$others" ] && return
  die "outro(s) cluster(s) kind rodando: $(echo "$others" | tr '\n' ' ')
Cada um com device plugin próprio disputaria a mesma GPU. Derrube com:
  kind delete cluster --name <nome>
(ou rode com ALLOW_OTHER_CLUSTERS=1 se tiver certeza de que nenhum usa a GPU)"
}

repo_slug() {
  local url
  if [ -n "${1:-}" ]; then
    echo "$1"
    return
  fi
  url=$(git -C "$ROOT" remote get-url origin 2>/dev/null) || die "sem 'origin' no git: passe owner/repo como argumento"
  url=${url%.git}
  echo "${url#*github.com[:/]}"
}

github_token() {
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "$GITHUB_TOKEN"
  elif command -v gh >/dev/null && gh auth token 2>/dev/null; then
    :
  else
    die "defina GITHUB_TOKEN ou faça 'gh auth login'"
  fi
}

latest_release() { # $1 = owner/repo no GitHub
  local json
  # Guardar antes de filtrar: com pipefail, `curl | grep -m1` pode falhar por SIGPIPE
  json=$(curl -fsSL "https://api.github.com/repos/$1/releases/latest")
  grep -m1 '"tag_name"' <<<"$json" | cut -d'"' -f4
}

wait_for_gpu() {
  local gpus
  for _ in $(seq 60); do
    gpus=$(k get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}')
    if [ "${gpus:-0}" -ge 1 ]; then
      echo "GPU alocável no node: $gpus"
      return
    fi
    sleep 2
  done
  die "o node não anunciou nvidia.com/gpu em 2 min (kubectl --context $CONTEXT -n kube-system logs ds/nvidia-device-plugin-daemonset)"
}

cmd_up() {
  local slug token plugin gid
  check_prereqs
  check_other_clusters
  slug=$(repo_slug "${1:-}")
  [[ $slug =~ ^[^/]+/[^/]+$ ]] || die "repositório inválido: '$slug' (esperado owner/repo)"
  token=$(github_token)

  if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
    log "Cluster '$CLUSTER' já existe: reaproveitando"
  else
    log "Criando o cluster '$CLUSTER'"
    kind create cluster --name "$CLUSTER" --config "$ROOT/k8s/kind-config.yaml"
  fi
  docker exec "$CLUSTER-control-plane" nvidia-smi -L ||
    die "a GPU não aparece dentro do node do kind (README: 'Solução de problemas')"

  plugin=$(latest_release NVIDIA/k8s-device-plugin)
  [ -n "$plugin" ] || die "não consegui descobrir a versão mais nova do NVIDIA device plugin"
  log "NVIDIA device plugin $plugin"
  k apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/$plugin/deployments/static/nvidia-device-plugin.yml"
  k -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=300s
  wait_for_gpu

  log "ARC: controller + runner set '$RUNNER_SET' para $slug"
  k create namespace "$NAMESPACE" --dry-run=client -o yaml | k apply -f -
  # Token via stdin: não aparece na linha de comando (ps) nem em arquivo
  printf '%s' "$token" |
    k -n "$NAMESPACE" create secret generic "$SECRET" --from-file=github_token=/dev/stdin \
      --dry-run=client -o yaml | k apply -f -
  h upgrade --install arc "$CHARTS/gha-runner-scale-set-controller" -n "$NAMESPACE" --wait
  gid=$(stat -c %g /var/run/docker.sock)
  h upgrade --install "$RUNNER_SET" "$CHARTS/gha-runner-scale-set" -n "$NAMESPACE" --wait \
    -f "$ROOT/k8s/runner-values.yaml" \
    --set githubConfigUrl="https://github.com/$slug" \
    --set "template.spec.securityContext.supplementalGroups[0]=$gid"

  cmd_status
}

cmd_status() {
  kind get clusters 2>/dev/null | grep -qx "$CLUSTER" || die "o cluster '$CLUSTER' não existe (rode: scripts/cluster.sh up)"
  log "GPU alocável"
  k get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
  log "Device plugin"
  k -n kube-system get daemonset nvidia-device-plugin-daemonset \
    -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
  log "ARC (versões dos charts e pods)"
  h list -n "$NAMESPACE"
  k -n "$NAMESPACE" get pods
}

cmd_down() {
  kind delete cluster --name "$CLUSTER"
  echo "O cache de imagens e o volume pah-cifar10 continuam no Docker do host."
}

case "${1:-}" in
  up) cmd_up "${2:-}" ;;
  status) cmd_status ;;
  down) cmd_down ;;
  *)
    echo "uso: $0 up [owner/repo] | status | down" >&2
    exit 64
    ;;
esac
```

- [ ] **Step 4: Verificar sem executar**

Run:
```bash
chmod +x scripts/cluster.sh
bash -n scripts/cluster.sh && shellcheck scripts/cluster.sh && echo shell-ok
python3 -c "import yaml; [yaml.safe_load(open(f)) for f in ('k8s/kind-config.yaml','k8s/runner-values.yaml')]; print('yaml-ok')"
scripts/cluster.sh 2>&1 | head -1   # só imprime o uso, não toca em nada
```
Expected: `shell-ok`, `yaml-ok` e `uso: scripts/cluster.sh up [owner/repo] | status | down`. **Não** rodar `up` (Global Constraint).

- [ ] **Step 5: Commit**

```bash
git add k8s scripts/cluster.sh
git commit -m "feat: cluster kind com GPU + ARC num único script idempotente"
```

---

### Task 9: README, CLAUDE.md (handoff) e SAGA

**Files:**
- Create: `README.md`, `CLAUDE.md`, `SAGA-DA-RTX3050.md`

**Interfaces:**
- Consumes: todos os nomes acima (cluster `pah`, runner set, volume, imagem, `MAX_CUDA`, workflows).

- [ ] **Step 1: `README.md` (pt-BR)**, com estas seções, nesta ordem:
  1. O que é (um parágrafo) + diagrama ASCII do fluxo push → image.yaml → gpu.yaml → docker run na GPU.
  2. **Estado atual**: código e workflows prontos; o cluster ainda não foi criado nesta máquina; a primeira execução real na GPU ainda não aconteceu.
  3. **Preparar o host (uma vez)**, em passos numerados com os comandos exatos: driver NVIDIA (`nvidia-smi`, anotar o "CUDA Version"); Docker + usuário no grupo `docker`; NVIDIA Container Toolkit + `sudo nvidia-ctk runtime configure --runtime=docker --set-as-default` + `sudo systemctl restart docker`; a linha `accept-nvidia-visible-devices-as-volume-mounts = true` em `/etc/nvidia-container-runtime/config.toml` (via `sudo nvidia-ctk config --in-place --set accept-nvidia-visible-devices-as-volume-mounts=true`); kind, kubectl e helm nas versões mais novas; `gh auth login`. Também o aviso de que o runtime padrão `nvidia` vale para **todos** os containers da máquina.
  4. **Primeira vez (quando a internet estiver boa)**, com o tamanho dos downloads: derrubar o cluster antigo (`kind delete cluster --name kind`); `scripts/cluster.sh up`; opcionalmente pré-baixar a imagem (`docker pull ghcr.io/lorkyrr/pytorch-arc-hosted:latest`, ~3,3 GB na 1ª vez); disparar `gh workflow run gpu.yaml`; acompanhar com `gh run watch`.
  5. **Rodar treino**: pela UI (Actions → GPU (RTX 3050) → Run workflow) e por `gh workflow run gpu.yaml -f mode=train -f arch=resnet56 -f epochs=50 -f seed=42`. Onde ficam checkpoint (artifact) e resumo (Summary da run).
  6. **Uso local com Docker Compose** (`docker compose run --rm app`, `... train --epochs 30`).
  7. **Como "sempre a versão mais nova" funciona** (resolvedor, `MAX_CUDA`, schedule semanal, Dependabot, chart/plugin/runner no `up`) e **como atualizar o ARC** (`scripts/cluster.sh down && scripts/cluster.sh up`, porque o `helm upgrade` não atualiza CRDs).
  8. **Segurança** (socket = root no host; `gpu.yaml` sem push/PR; em Settings → Actions → "Fork pull request workflows", exigir aprovação para todos os colaboradores externos).
  9. **Solução de problemas**: tabela sintoma → causa → correção, com as linhas de `kind-config` herdadas do projeto anterior (nvidia-smi ausente no node, `ERROR_LIBRARY_NOT_FOUND`, `Insufficient nvidia.com/gpu`, cluster antigo disputando a GPU, `permission denied` no socket → GID, job com `NVIDIA_VISIBLE_DEVICES` vazio, `--require-gpu` falhando → `MAX_CUDA` acima do driver, "Re-run jobs" rodando SHA antigo).
  10. **Desenvolvimento**: `ruff check .`, `pytest` (o que pula sem torch), mapa do repo.
  11. Licença.

- [ ] **Step 2: `CLAUDE.md` (inglês, denso: é o handoff)**, com as seções: What this is; **Handoff: current state** (feito e verificado, feito e não verificado, não feito, com datas absolutas); Decisions and why (as tabelas da spec condensadas, incluindo a proibição de `pytorch/pytorch:latest`, só Docker, host socket e não dind, `--gpus device=`, `supplementalGroups`, sempre a versão mais nova); Repository map; Commands; Conventions (identificadores em inglês, saída/comentários em pt-BR, sem download local enquanto a internet do autor estiver ruim); Gotchas (a tabela do README + Helm substitui listas no `--set` + `helm upgrade` não atualiza CRDs do ARC + Python `3.x` no CI pode quebrar quando sair um Python novo sem wheel do torch); Next steps (ordem exata do que o autor faz a seguir); Open verification points (os 3 da spec seção 7, marcando o que já foi verificado: CLI docker na imagem do runner = verificado em 2026-09-24 pelo Dockerfile upstream, que instala Docker 29.8.1).

- [ ] **Step 3: `SAGA-DA-RTX3050.md`**: copiar `../refactored-pytorch-CTTK/SAGA-DA-RTX3050.md` na íntegra como **Parte I**, com uma nota no topo dizendo que ela descreve os repos anteriores e que os links de arquivo apontam para aqueles repos. Depois, acrescentar a **Parte II: pytorch-arc-hosted**, no mesmo tom de estudo (o quê / por quê / como verificar), com os capítulos:
  - 19. Recomeço: o que cinco projetos ensinaram (a dor-raiz era onde moram 3–5 GB de dependências)
  - 20. Só Docker, e o socket do host como cache (por que não dind, não python nativo)
  - 21. Fechando a GPU "parcialmente virtual": `NVIDIA_VISIBLE_DEVICES` → `--gpus device=`
  - 22. Imagem no GHCR ativada por `workflow_run` (e por que usar `head_sha` e não `latest`)
  - 23. A tag `latest` que parou em 2024, e o resolvedor com `MAX_CUDA`
  - 24. Repo público + runner self-hosted: o risco do socket
  - 25. Sem root: `supplementalGroups` com o GID do socket
  - 26. Dois clusters, uma GPU
  - 27. O que ainda falta (TODOs reais da Parte II)
  - 28. Comandos de referência rápida (Parte II)

  Atualizar o índice.

- [ ] **Step 4: Verificar links internos**

Run: `grep -o '](\([^)#]*\)' README.md CLAUDE.md | sed 's/.*](//' | sort -u | while read -r p; do [ -e "$p" ] || echo "QUEBRADO: $p"; done`
Expected: nenhuma linha `QUEBRADO` (links externos `http` são ignorados pelo filtro; ajustar o filtro se preciso).

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md SAGA-DA-RTX3050.md
git commit -m "docs: README, CLAUDE.md de handoff e SAGA parte II"
```

---

### Task 10: Publicar no GitHub e verificar o CI remoto

**Files:** nenhum novo (o `CLAUDE.md` é atualizado com o resultado).

- [ ] **Step 1: Push**

```bash
git push -u origin main
```

- [ ] **Step 2: Acompanhar CI e imagem** (rodam no GitHub; nada é baixado localmente)

```bash
gh run list --limit 5
gh run watch <id do CI> --exit-status
gh run watch <id do Imagem (GHCR)> --exit-status
```
Expected: `CI` verde (os testes torch rodam de verdade aqui) e `Imagem (GHCR)` verde, com o resumo mostrando a base resolvida. Se falhar, voltar à task dona do erro (ex.: contagem de parâmetros → Task 2) com `superpowers:systematic-debugging`.

- [ ] **Step 3: Cancelar o `gpu.yaml` enfileirado**

O `workflow_run` vai enfileirar um job para `arc-runner-set-gpu`, runner que ainda não existe. Cancelar, para não disparar o pull de ~3,3 GB quando o cluster subir:
```bash
gh run list --workflow gpu.yaml --limit 1 --json databaseId --jq '.[0].databaseId' | xargs -r gh run cancel
```

- [ ] **Step 4: Registrar o estado no `CLAUDE.md`** (seção Handoff: SHAs, status das runs, base resolvida) e fazer commit + push.

```bash
git add CLAUDE.md
git commit -m "docs: estado do handoff após o primeiro CI"
git push
```
