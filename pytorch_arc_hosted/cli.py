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
