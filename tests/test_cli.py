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
