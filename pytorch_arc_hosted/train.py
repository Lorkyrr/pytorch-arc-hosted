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
