import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from pytorch_arc_hosted.model import build_resnet  # noqa: E402
from pytorch_arc_hosted.train import EpochStats, fit, lr_milestones, write_summary  # noqa: E402

CPU = torch.device("cpu")


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
        device=CPU,
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
    fit(
        build_resnet("resnet20"),
        _loader(16),
        _loader(16),
        device=CPU,
        epochs=1,
        lr=0.1,
        amp=False,
        checkpoint_path=ckpt,
        arch="resnet20",
    )
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
