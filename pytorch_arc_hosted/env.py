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
