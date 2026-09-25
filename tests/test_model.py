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
