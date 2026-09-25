# Imagem única do projeto: roda no CI (gpu.yaml) e localmente (compose.yaml).
#
# BASE é resolvida a cada build no image.yaml (scripts/latest_pytorch_base.py):
# a tag mais nova <torch>-cuda<X.Y>-cudnn<N>-runtime que o driver do host aguenta.
# O default abaixo só vale pra build local sem --build-arg. Não troque por
# pytorch/pytorch:latest: essa tag está parada em 2024 (PyTorch 2.2.1).
ARG BASE=pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime
FROM ${BASE}

LABEL org.opencontainers.image.source="https://github.com/Lorkyrr/pytorch-arc-hosted"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# torch e torchvision vêm da base. Se um dia deixarem de vir, o build falha aqui, e não no meio do treino.
RUN python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"

WORKDIR /app
COPY pytorch_arc_hosted/ pytorch_arc_hosted/

ENTRYPOINT ["python", "-m", "pytorch_arc_hosted"]
CMD ["benchmark"]
