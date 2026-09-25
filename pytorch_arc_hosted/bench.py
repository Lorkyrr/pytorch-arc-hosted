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
