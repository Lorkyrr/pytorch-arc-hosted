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
import time
import urllib.request
from collections.abc import Iterable, Sequence

REPO = "pytorch/pytorch"
HUB_URL = f"https://hub.docker.com/v2/repositories/{REPO}/tags?page_size=100&ordering=last_updated"
ATTEMPTS = 4  # o Docker Hub (e a rede de quem roda) às vezes engasga
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


def _get_json(url: str) -> dict:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except (OSError, json.JSONDecodeError) as error:  # TimeoutError e URLError são OSError
            if attempt == ATTEMPTS:
                raise
            print(f"[aviso] tentativa {attempt}/{ATTEMPTS} falhou ({error}); tentando de novo", file=sys.stderr)
            time.sleep(2**attempt)
    raise AssertionError("inalcançável")


def fetch_tags(max_pages: int = 5) -> list[str]:
    url: str | None = HUB_URL
    tags: list[str] = []
    for _ in range(max_pages):
        if url is None:
            break
        page = _get_json(url)
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
