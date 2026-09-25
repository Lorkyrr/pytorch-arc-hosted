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
