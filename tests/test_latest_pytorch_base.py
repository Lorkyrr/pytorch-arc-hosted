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
