# pytorch-arc-hosted

Benchmark de GPU e treino de ResNet no CIFAR-10 **na sua RTX 3050 de verdade**,
disparados pelo GitHub Actions. A imagem Docker é buildada no GitHub, publicada
no GHCR e executada por um runner self-hosted do
[Actions Runner Controller (ARC)](https://github.com/actions/actions-runner-controller)
que roda num cluster [kind](https://kind.sigs.k8s.io/) na sua máquina.

> Projeto de estudo. É a reescrita do zero de `pytorch-gpu-sandbox`,
> `pytorch-gpu-refactor` e `refactored-pytorch-CTTK`. A história completa, com
> os erros no caminho, está em [SAGA-DA-RTX3050.md](SAGA-DA-RTX3050.md).

```
git push (main)
   ├─► CI             (GitHub)       ruff + pytest com torch CPU
   └─► Imagem (GHCR)  (GitHub)       resolve a base PyTorch mais nova → build → ghcr.io/lorkyrr/pytorch-arc-hosted:<sha>
            │ terminou com sucesso (workflow_run)
            ▼
        GPU (RTX 3050)  (SUA máquina, runner ARC no kind)
            pod do runner: reserva nvidia.com/gpu: 1 → recebe NVIDIA_VISIBLE_DEVICES=<UUID>
            docker pull  (daemon do host: só baixa as camadas novas)
            docker run --gpus device=<UUID> -v pah-cifar10:/data  <imagem>  benchmark|train
            checkpoint → artifact da run · resumo → Summary da run
```

## Estado atual (2026-09-28)

- ✅ Código, testes, Dockerfile, workflows, `cluster.sh` e docs prontos.
- ✅ O CI (35 testes, com torch) e o build da imagem já rodaram com sucesso no GitHub. A imagem está publicada e pública em `ghcr.io/lorkyrr/pytorch-arc-hosted` (torch 2.14.0 + CUDA 13.2).
- ⏳ **O cluster ainda não foi criado nesta máquina**, e a primeira execução real
  na GPU ainda não aconteceu. Siga "Primeira vez" abaixo quando a internet
  estiver boa.

## Preparar o host (uma vez)

A maior parte disso já está feita nesta máquina (ver [CLAUDE.md](CLAUDE.md)).
Os passos servem para reinstalar ou montar em outra máquina.

1. **Driver NVIDIA.** Rode `nvidia-smi` e anote o **"CUDA Version"** do canto
   superior direito. Esse é o CUDA **máximo** que o driver aguenta. Ele vai no
   `MAX_CUDA` de [.github/workflows/image.yaml](.github/workflows/image.yaml)
   (hoje `13.4`, driver 615).
2. **Docker Engine**, com seu usuário no grupo `docker`:
   ```bash
   sudo usermod -aG docker "$USER"   # depois, saia e entre de novo na sessão
   ```
3. **NVIDIA Container Toolkit**, com o runtime `nvidia` como **padrão** do Docker:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
   sudo systemctl restart docker
   ```
   ⚠️ Isso vale para **todos** os containers da máquina, não só para este projeto.
   É o que faz o kind conseguir repassar a GPU para o node.
4. **Deixar a GPU ser pedida por volume** (o truque que o kind usa):
   ```bash
   sudo nvidia-ctk config --in-place --set accept-nvidia-visible-devices-as-volume-mounts=true
   ```
5. **kind, kubectl e helm** nas versões mais novas (binários oficiais em
   `/usr/local/bin`; o helm pelo script oficial `get-helm-3`, não por snap/apt).
6. **GitHub CLI** logado: `gh auth login`. O `cluster.sh` usa `gh auth token`
   para registrar o runner; nenhum token vai para arquivo.

O `scripts/cluster.sh up` confere os itens 2 a 4 e lista **tudo** o que faltar de
uma vez. Ele não instala nada sozinho.

## Primeira vez (quando a internet estiver boa)

Downloads da primeira vez, aproximados: imagem do node kind (~0,5 GB), runner
do ARC (~0,5 GB), controller e device plugin (~0,2 GB), **base PyTorch
(~3,3 GB)** e CIFAR-10 (~170 MB). Depois disso, cada job só baixa a camada do
código (KBs). A base só é baixada de novo quando sair uma versão nova do PyTorch.

```bash
# 1. Derrube os clusters dos projetos antigos: dois clusters disputariam a mesma GPU
kind get clusters                 # nesta máquina: kind e nvidia (em 2026-09-28)
kind delete cluster --name kind
kind delete cluster --name nvidia
```

```bash
# 2. Suba o cluster + device plugin + ARC (idempotente: pode rodar de novo)
scripts/cluster.sh up
```

```bash
# 3. (Opcional) Pré-baixe a imagem com calma, antes do primeiro job
docker pull ghcr.io/lorkyrr/pytorch-arc-hosted:latest
```

```bash
# 4. Dispare o benchmark e acompanhe
gh workflow run gpu.yaml
```

```bash
gh run watch
```

A partir daí, todo `push` na `main` que mexa no código ou no Dockerfile
builda a imagem e roda o benchmark na GPU sozinho. Toda segunda-feira, o
`image.yaml` rebuilda com a base mais nova e também dispara o benchmark.

## Rodar um treino

Pela interface: **Actions → GPU (RTX 3050) → Run workflow**, escolha `train` e
os parâmetros. Ou pelo terminal:

```bash
gh workflow run gpu.yaml -f mode=train -f arch=resnet56 -f epochs=50 -f seed=42
```

- **Checkpoint** (`<arch>_cifar10.pth`, um dict com `arch`, `epoch`,
  `test_acc` e `state_dict`): fica nos **Artifacts** da run.
- **Resumo** (melhor acurácia + tabela por época): fica no **Summary** da run.
- O dataset fica no volume Docker `pah-cifar10` do host, baixado uma única vez.

Referência dos projetos anteriores (ResNet-20, FP32): 30 épocas / lote 128 /
LR 0,1 → **89,40%**; 50 épocas / lote 64 / LR 0,05 → **90,81%**.

## Uso local (sem CI)

A mesma imagem e o mesmo volume de dataset do CI:

```bash
docker compose run --rm app
```

```bash
docker compose run --rm app train --epochs 30
```

O checkpoint e o `summary.md` aparecem em `./out`. Opções do treino:
`--arch {resnet20,resnet56,resnet110}`, `--epochs`, `--batch-size`, `--lr`,
`--no-amp`, `--seed`, `--require-gpu`.

## "Sempre a versão mais nova": como funciona

| O quê | Como fica atualizado |
|---|---|
| Base PyTorch | [scripts/latest_pytorch_base.py](scripts/latest_pytorch_base.py) resolve, a cada build, a tag `<torch>-cuda<X.Y>-cudnn<N>-runtime` mais nova com CUDA ≤ `MAX_CUDA`. **Não** usamos `pytorch/pytorch:latest`: ela está parada em 2024-02-23 (PyTorch 2.2.1). |
| Rebuild sem commit | `schedule` semanal no `image.yaml` (segunda, 06:00 UTC) |
| Actions (`checkout`, `docker/*`…) | [Dependabot](.github/dependabot.yml) abre PR toda semana |
| ARC (controller + runner set) | `cluster.sh up` instala o chart mais novo (sem `--version`) |
| NVIDIA device plugin | `cluster.sh up` busca a release mais recente na API do GitHub |
| Runner | `ghcr.io/actions/actions-runner:latest` |
| Python do CI | `3.x` (o estável mais novo) |

**Atualizar o ARC num cluster que já existe:** o `helm upgrade` não atualiza os
CRDs do ARC. Como o cluster é descartável (dataset e cache ficam no host), o
caminho seguro é recriar:

```bash
scripts/cluster.sh down && scripts/cluster.sh up
```

**Atualizou o driver NVIDIA?** Atualize o `MAX_CUDA` no `image.yaml` com o novo
"CUDA Version" do `nvidia-smi`.

**Limpar imagens antigas do projeto** (cada push deixa uma tag `:<sha>`, mas as
camadas pesadas são compartilhadas):

```bash
docker image prune -a --filter "label=org.opencontainers.image.source=https://github.com/Lorkyrr/pytorch-arc-hosted" --filter "until=168h"
```

## Segurança

O runner recebe o **socket do Docker do host**, e isso equivale a root na sua
máquina para qualquer código que rode nele. Por isso:

- O [gpu.yaml](.github/workflows/gpu.yaml) **nunca** tem gatilho de `push` ou
  `pull_request`. Ele só roda por `workflow_dispatch` (quem tem escrita no
  repo) ou depois de um build da imagem **a partir da `main`**.
- O runner set é registrado **só neste repositório**.
- Em **Settings → Actions → General**, na parte de workflows de pull requests
  de forks, exija **aprovação para todos os colaboradores externos**.

## Solução de problemas

| Sintoma | Causa | Correção |
|---|---|---|
| `cluster.sh up` para com "outro(s) cluster(s) kind rodando" | Cluster de projeto antigo com device plugin próprio | `kind delete cluster --name <nome>` |
| "a GPU não aparece dentro do node do kind" | Runtime padrão do Docker não é `nvidia`, ou falta `accept-nvidia-visible-devices-as-volume-mounts` | Passos 3 e 4 de "Preparar o host", depois `cluster.sh down && cluster.sh up` |
| Device plugin em `CrashLoopBackOff` / `ERROR_LIBRARY_NOT_FOUND` | O containerd **interno** do node não usa o runtime nvidia | Confira o `containerdConfigPatches` em [k8s/kind-config.yaml](k8s/kind-config.yaml) |
| `0/1 nodes are available: 1 Insufficient nvidia.com/gpu` | Outro pod segura a única GPU, ou o device plugin morreu | `scripts/cluster.sh status`; `cluster.sh up` reinstala o plugin |
| `permission denied ... docker.sock` no job | GID do socket mudou (reinstalação do Docker) | `scripts/cluster.sh up` (relê o GID e reaplica) |
| Job falha em "Conferir a GPU que o Kubernetes reservou" | O pod não recebeu `NVIDIA_VISIBLE_DEVICES` com um UUID | Veja [CLAUDE.md](CLAUDE.md), "Open verification points", item 1 |
| `[ERRO] --require-gpu: CUDA indisponível` | A base resolvida exige CUDA acima do driver, ou o `--gpus` não chegou | Confira o `MAX_CUDA` contra o `nvidia-smi` e a base no Summary do `Imagem (GHCR)` |
| A run nova usa código velho | "Re-run jobs" repete a run no **SHA original** | Use "Run workflow" (cria uma run nova) |
| Job de GPU fica em "Queued" (toda segunda, depois do rebuild semanal) | Cluster desligado, sem runner. Ele expira em 24h, mas se o cluster subir antes o job roda e baixa a imagem. | `scripts/cluster.sh up`, ou `gh run cancel <id>` se ainda não quiser o download |

## Desenvolvimento

Sem GPU e sem torch instalado, dá para rodar o lint e parte dos testes:

```bash
ruff check .
```

```bash
pytest
```

Os testes que precisam de torch se pulam sozinhos (`pytest.importorskip`) e
rodam de verdade no `ci.yaml`, com torch CPU.

```
pytorch_arc_hosted/        pacote (python -m pytorch_arc_hosted)
  cli.py                     argparse: benchmark | train (torch importado só depois do parsing)
  env.py                     relatório do ambiente + helpers de console
  bench.py                   cuBLAS, cuDNN, AMP isolada, treino sintético
  model.py                   ResNet-20/56/110 (He et al., 2015), do zero
  data.py                    CIFAR-10 (torchvision) + augmentation
  train.py                   laço com AMP real, checkpoint, summary.md
scripts/
  cluster.sh                 up | status | down do kind + GPU + ARC
  latest_pytorch_base.py     resolve a imagem base PyTorch mais nova
k8s/
  kind-config.yaml           GPU e socket do Docker do host para dentro do node
  runner-values.yaml         runner set do ARC com nvidia.com/gpu: 1
.github/workflows/
  ci.yaml · image.yaml · gpu.yaml
docs/specs · docs/plans     design e plano de implementação
```

## Licença

MIT. Veja [LICENSE](LICENSE).
