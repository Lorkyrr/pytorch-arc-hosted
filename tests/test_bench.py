import pytest

torch = pytest.importorskip("torch")

from pytorch_arc_hosted import bench  # noqa: E402


def test_gflops_known_value():
    # 2 * 1000^3 = 2e9 operações em 2 s = 1 GFLOPS
    assert bench.gflops(1000, 2.0) == pytest.approx(1.0)


def test_gflops_rejects_non_positive_time():
    with pytest.raises(ValueError):
        bench.gflops(10, 0.0)


def test_benchmark_smoke_on_cpu(monkeypatch, capsys):
    monkeypatch.setattr(bench, "MATMUL_SIZES", (64,))
    monkeypatch.setattr(bench, "CONV_SIDE", 32)
    monkeypatch.setattr(bench, "CONV_REPEATS", 2)
    monkeypatch.setattr(bench, "SYNTH_SIDE", 16)
    monkeypatch.setattr(bench, "SYNTH_STEPS", 2)
    bench.run_benchmark(torch.device("cpu"))
    out = capsys.readouterr().out
    assert "GFLOPS" in out
    assert "pulado" in out  # AMP só roda em GPU
    assert "BENCHMARK CONCLUÍDO" in out
