# pytorch-arc-hosted — Design

Data: 2026-09-24 · Status: aguardando revisão

## 1. Objetivo

Treinar e benchmarkar PyTorch na RTX 3050 (4 GB) do autor, disparado pelo GitHub
Actions, usando runners self-hosted do Actions Runner Controller (ARC) num cluster
`kind` local. A execução acontece **sempre dentro de uma imagem Docker publicada
no GHCR**. Não existe variante "python nativo".

Reescrita do zero dos projetos `pytorch-gpu-sandbox`, `pytorch-gpu-refactor` e
`refactored-pytorch-CTTK`: mesma intenção, estrutura nova.

### Critérios de sucesso

1. Um `git push` na `main` builda a imagem no GitHub-hosted, publica no
   `ghcr.io/lorkyrr/pytorch-arc-hosted:<sha>` e, ao terminar, dispara
   automaticamente o benchmark na GPU real com **essa** imagem.
2. `workflow_dispatch` permite treinar a ResNet (arquitetura, épocas, batch, LR,
   AMP, seed, tag da imagem), e o checkpoint aparece como artifact do run.
3. Depois do primeiro pull, os jobs de GPU não re-baixam a base do PyTorch: só
   as camadas do código. O dataset CIFAR-10 é baixado uma única vez.
4. `scripts/cluster.sh up` recria o ambiente inteiro (kind + GPU + ARC) a partir
   de um host com os pré-requisitos, e é idempotente.
5. O container de treino usa exatamente a GPU que o Kubernetes reservou para o
   pod do runner.

### Fora de escopo

- Variante sem Docker, Docker-in-Docker (dind) e `containerMode: kubernetes`.
- Instalação automática de pacotes do sistema (driver, Docker, Container
  Toolkit). O script **diagnostica e informa**, não roda `apt`/`sudo`.
- Multi-GPU, time-slicing, distribuição de treino.
- Runner set sem GPU.

## 2. Arquitetura

```
git push main
   │
   ├─► ci.yaml     (ubuntu-latest)  ruff + pytest com torch CPU
   │
   └─► image.yaml  (ubuntu-latest)  docker build → push ghcr.io/lorkyrr/pytorch-arc-hosted:{sha,latest}
            │ workflow_run: completed + success
            ▼
        gpu.yaml   (runs-on: arc-runner-set-gpu)      ◄── também workflow_dispatch
            │
            │  pod do runner (imagem stock ghcr.io/actions/actions-runner)
            │    limits: nvidia.com/gpu: 1      → NVIDIA_VISIBLE_DEVICES=<UUID>
            │    volume: /var/run/docker.sock   (do host, via extraMount do node kind)
            │    supplementalGroups: [GID do docker.sock do host]
            ▼
        docker pull ghcr.io/...:<sha>                   (daemon do HOST → cache persistente)
        docker run --gpus "device=$NVIDIA_VISIBLE_DEVICES" \
                   -v pah-cifar10:/data --name pah-<run_id> <imagem> <modo> ...
        docker cp pah-<run_id>:/out  →  upload-artifact + $GITHUB_STEP_SUMMARY
        docker rm -f pah-<run_id>                       (sempre)
```

### Decisões e motivos

| Decisão | Motivo |
|---|---|
| Socket do Docker do **host** montado no runner | O cache de camadas vive no host e sobrevive a `kind delete cluster`. Com dind, o cache morreria junto com o pod (~5 GB por job). |
| `--gpus "device=$NVIDIA_VISIBLE_DEVICES"` | O device plugin injeta o UUID da GPU alocada no pod. Isso amarra o container irmão à reserva do k8s e fecha a lacuna de "GPU fora da contabilidade" dos projetos anteriores. Se a variável estiver vazia, o job **falha** com mensagem clara. Nunca cai silenciosamente para `all`. |
| `supplementalGroups` com o GID do socket | Acesso ao socket sem `runAsUser: 0` nem `RUNNER_ALLOW_RUNASROOT`. O `cluster.sh` lê o GID com `stat -c %g /var/run/docker.sock` e passa via `helm --set`. |
| Dataset em volume nomeado `pah-cifar10` | Persiste no host entre jobs e recriações do cluster, sem subir/baixar 170 MB via `actions/cache`. |
| Checkpoint via `docker cp` | O daemon do host não enxerga o filesystem do pod, então bind mount do workspace não funciona. |
| Base `pytorch/pytorch` com tag **fixa** (`*-cuda12.*-cudnn9-runtime`) | Camadas estáveis = pull do host reaproveita tudo. `:latest` mudaria as camadas por baixo. Na implementação, escolher a tag mais recente desse formato no Docker Hub; o driver 615 (CUDA ≤ 13.4) cobre CUDA 12. |
| Build no GitHub-hosted, não no runner local | A rede do GitHub baixa a base rápido; o host só puxa o delta. |

### Segurança (repo público + runner self-hosted)

Montar o `docker.sock` equivale a root no host para qualquer job que rode ali.
Mitigações obrigatórias:

- `gpu.yaml` só tem os gatilhos `workflow_dispatch` e `workflow_run`. **Nunca**
  `pull_request`/`push`. `workflow_run` só executa a versão do workflow na
  branch padrão, e `image.yaml` só roda em `push` na `main`, não em PR.
- `gpu.yaml` filtra `workflow_run.conclusion == 'success'` e
  `workflow_run.head_branch == 'main'` (`image.yaml` não tem gatilho de PR,
  então o evento de origem só pode ser push ou dispatch de quem tem escrita).
- O runner set é registrado **só neste repositório** (`githubConfigUrl` no nível
  de repo).
- README documenta: manter "Require approval for all outside collaborators"
  nas configurações de Actions do repo.

## 3. Componentes

### 3.1 Pacote `pytorch_arc_hosted/`

Identificadores em inglês; comentários e saída de console em português.

| Módulo | Responsabilidade |
|---|---|
| `__main__.py` | `python -m pytorch_arc_hosted` → `cli.main()` |
| `cli.py` | argparse com subcomandos `benchmark` e `train`, e flags de treino: `--arch {resnet20,resnet56,resnet110}`, `--epochs 30`, `--batch-size 128`, `--lr 0.1`, `--no-amp`, `--seed`, `--data-dir /data`, `--out-dir /out`, `--require-gpu` (falha se não houver CUDA; usada no CI) |
| `env.py` | Detecta o device e imprime o relatório (torch, CUDA, cuDNN, GPU, VRAM, compute capability, Tensor Cores). Sem CUDA: aviso + CPU. |
| `bench.py` | matmul (tamanhos que cabem em 4 GB, OOM por tamanho não derruba o run), conv2d com warm-up, FP32 vs autocast FP16 (pulado em CPU), loop de treino sintético, memória de pico. `gflops()` é função pura. |
| `model.py` | ResNet CIFAR 6n+2 (He et al., 2015), blocos básicos, atalho por projeção 1×1 quando muda stride/canais. `build_resnet(arch)`; nome desconhecido → `ValueError`. |
| `data.py` | CIFAR-10 via torchvision: RandomCrop(32, pad 4) + flip no treino, normalização com as estatísticas fixas do CIFAR-10, `pin_memory` só em CUDA. |
| `train.py` | SGD (momentum 0.9, nesterov, wd 5e-4) + MultiStepLR em 50% e 75% (γ=0.1). AMP real (`autocast` + `GradScaler`), ligado só em CUDA. Seed opcional. Salva o melhor checkpoint em `out_dir/<arch>_cifar10.pth` como dict `{arch, epoch, test_acc, state_dict}`. Escreve `out_dir/summary.md` (tabela por época + melhor acurácia). |

Hiperparâmetros iguais aos projetos anteriores, de propósito: permite comparar
com os resultados registrados (89,40% em 30 épocas; 90,81% em 50).

### 3.2 Container

- `Dockerfile`: `FROM pytorch/pytorch:<tag fixa>`, `WORKDIR /app`, instala
  apenas o pacote (`pip install --no-deps .`, porque torch/torchvision já vêm
  na base), `ENTRYPOINT ["python", "-m", "pytorch_arc_hosted"]`,
  `CMD ["benchmark"]`, `ENV PYTHONUNBUFFERED=1`, e a label
  `org.opencontainers.image.source=https://github.com/Lorkyrr/pytorch-arc-hosted`
  (vincula o pacote do GHCR ao repo).
- `.dockerignore`: tudo que não é `pyproject.toml` + `pytorch_arc_hosted/`.
- `compose.yaml` (uso local): `image: ghcr.io/lorkyrr/pytorch-arc-hosted:latest`
  com `build: .`, GPU via `deploy.resources.reservations.devices`, volume
  `pah-cifar10:/data` (o mesmo nome do CI, então o dataset é compartilhado) e
  bind `./out:/out`.

### 3.3 Workflows (`.github/workflows/`)

| Arquivo | Runner | Gatilho | Faz |
|---|---|---|---|
| `ci.yaml` | ubuntu-latest | push, pull_request | `pip install torch torchvision --index-url …/whl/cpu`, `pip install -e .[dev]`, `ruff check`, `pytest` |
| `image.yaml` | ubuntu-latest | push em `main` (paths: `Dockerfile`, `pyproject.toml`, `pytorch_arc_hosted/**`), workflow_dispatch | `docker/login-action` (GITHUB_TOKEN, `packages: write`), `docker/build-push-action` → tags `:<sha>` e `:latest` |
| `gpu.yaml` | arc-runner-set-gpu | `workflow_run` de `image.yaml` (success, main) → benchmark; `workflow_dispatch` (mode, arch, epochs, batch_size, lr, amp, seed, image_tag=`latest`) | login GHCR (`packages: read`), pull, run com `--gpus device=`, cp, step summary, upload-artifact (só train), `docker rm -f` em `always()`. `concurrency: gpu`, `cancel-in-progress: false`, `timeout-minutes: 120`. |

A tag da imagem no `gpu.yaml` vem de `github.event.workflow_run.head_sha` quando
o gatilho é `workflow_run`, e do input `image_tag` no dispatch.

### 3.4 Kubernetes (`k8s/`)

- `kind-config.yaml`: node único. `extraMounts`: `/dev/null →
  /var/run/nvidia-container-devices/all` e `/var/run/docker.sock`. Patch do
  containerd do node com `default_runtime_name = "nvidia"`. (Os binários do
  Container Toolkit ficam como `extraMounts` só se, na implementação, o node
  não os tiver. Verificar.)
- `runner-values.yaml`: values do `gha-runner-scale-set`. `githubConfigSecret`,
  container `runner` com a imagem stock, `command: ["/home/runner/run.sh"]`,
  `limits: nvidia.com/gpu: 1`, volume hostPath do socket.
  `githubConfigUrl` e `supplementalGroups` entram via `helm --set` (nada
  específico da máquina fica commitado).
- NVIDIA device plugin: aplicado por URL com **tag de release fixa** (sem
  snapshot commitado). Chart do ARC também com versão fixa. As versões ficam
  em variáveis no topo do `cluster.sh`.

### 3.5 `scripts/cluster.sh`

Um arquivo, `set -euo pipefail`, três subcomandos:

- `up [owner/repo]`: repo padrão = derivado do `git remote origin`.
  1. Pré-requisitos (falha listando **tudo** o que falta, de uma vez): `docker`,
     `kind`, `kubectl`, `helm`, `nvidia-smi`, runtime `nvidia` registrado no
     Docker, `accept-nvidia-visible-devices-as-volume-mounts = true` em
     `/etc/nvidia-container-runtime/config.toml`.
  2. Token: `$GITHUB_TOKEN` ou `gh auth token`. Nunca lido de arquivo nem escrito
     em disco.
  3. Cria o cluster `pah` **só se não existir** (não destrói nada).
  4. Aplica o device plugin, espera `nvidia.com/gpu` alocável no node.
  5. Namespace + secret (`kubectl create secret … --dry-run=client -o yaml |
     kubectl apply -f -`), controller e runner set via `helm upgrade --install`.
- `down`: `kind delete cluster --name pah`. Os volumes Docker e o cache de
  imagens do host continuam intactos.
- `status`: nodes, GPU alocável, pods do `arc-systems`.

## 4. Tratamento de erros

- OOM num tamanho de matmul: registra `[FALHOU]`, `empty_cache()`, segue.
- Sem CUDA: benchmark roda em CPU (sem o teste FP16); `train` roda em CPU com
  aviso. No `gpu.yaml`, o container verifica `torch.cuda.is_available()` via
  `--require-gpu` (flag do CLI, ligada no CI) e sai com código ≠ 0 se não
  houver GPU. Assim, um job "verde" sem GPU não passa despercebido.
- `NVIDIA_VISIBLE_DEVICES` vazio no pod: o step falha antes do `docker run`.
- `docker rm -f` em `if: always()`, para não deixar container órfão ocupando
  a GPU.
- `cluster.sh`: pré-requisitos agregados numa lista única; cada etapa
  idempotente.

## 5. Testes

Rodam no `ci.yaml` com torch CPU; `pytest` também roda dentro da imagem.

- `test_model.py`: contagem de parâmetros da resnet20 (~0,27M) e forma da
  saída `(N, 10)` para as 3 arquiteturas; nome inválido → `ValueError`.
- `test_bench.py`: `gflops()` com valores conhecidos.
- `test_train.py`: um passo de treino num batch sintético em CPU reduz a loss
  (overfit de um batch em poucos passos); o checkpoint salvo recarrega no
  modelo.
- `test_cli.py`: parsing dos subcomandos e defaults.

Sem testes para shell/YAML além do `ruff` no Python. O workflow de GPU é o
teste de integração.

## 6. Documentação e repo

- `README.md` (pt-BR): o que é, pré-requisitos do host, `cluster.sh up`, como
  disparar treino, uso local com compose, nota de segurança.
- `CLAUDE.md`: referência densa para agentes (mapa do repo, comandos, gotchas
  herdados dos projetos anteriores e ainda relevantes).
- `pyproject.toml`: metadados, deps `torch`, `torchvision`, extra `dev`
  (`pytest`, `ruff`), config do ruff/pytest.
- `LICENSE` MIT, `.gitignore`.
- Remoto: `https://github.com/Lorkyrr/pytorch-arc-hosted.git` (público, vazio),
  branch `main`.

## 7. Pontos a verificar na implementação

1. O device plugin injeta `NVIDIA_VISIBLE_DEVICES=<UUID>` no pod com a
   configuração do kind acima (estratégia `envvar`). Se injetar outra coisa,
   ajustar o mapeamento, mantendo a regra de falhar em vez de usar `all`.
2. A imagem `ghcr.io/actions/actions-runner` traz o CLI `docker`.
3. A primeira publicação no GHCR cria o pacote privado. Com o login via
   `GITHUB_TOKEN` no `gpu.yaml` o pull funciona igual. Tornar público é opcional
   e manual.

## 8. Revisões pós-aprovação (2026-09-24)

Pedidos do autor depois da primeira versão da spec. **Sobrepõem** as seções acima
quando houver conflito.

1. **Nada de download local durante a implementação.** O código é escrito e
   verificado sem baixar torch/imagens na máquina do autor (internet ruim no
   momento). Testes que precisam de torch rodam no `ci.yaml` (GitHub-hosted).
   O README descreve os passos para o autor executar depois.
2. **Sempre a versão mais nova de tudo**, aceitando re-download. Isso substitui
   as "tags fixas" das seções 2 e 3.4:
   - Base da imagem: **resolvida a cada build** por
     `scripts/latest_pytorch_base.py`: maior versão do torch, depois maior CUDA,
     no formato `<torch>-cuda<X.Y>-cudnn<N>-runtime`, limitada por
     `MAX_CUDA` (CUDA máximo do driver do host; hoje `13.4`, driver 615).
     `pytorch/pytorch:latest` **não** é usado: está parado em 2024-02-23
     (PyTorch 2.2.1).
   - `image.yaml` também roda semanalmente (`schedule`) e com `pull: true`.
   - ARC: chart sem `--version`. Device plugin: tag da release mais recente,
     resolvida via API do GitHub no `cluster.sh`. Runner:
     `ghcr.io/actions/actions-runner:latest`.
   - Actions nos majors mais novos + Dependabot (`github-actions`) semanal.
   - Cada execução imprime as versões resolvidas (step summary do
     `image.yaml`, relatório do ambiente no container, `cluster.sh status`).
3. **Conflito de GPU com o cluster antigo:** já existe um cluster kind `kind`
   (projetos anteriores) com device plugin próprio. O `cluster.sh up` avisa e
   para se houver outro cluster kind rodando com GPU. O README manda derrubá-lo.
4. `Dockerfile` não roda `pip install`: copia só o pacote e roda
   `python -m pytorch_arc_hosted`. Torch/torchvision vêm da base, e um
   `RUN python -c "import torchvision"` garante isso no build. `pytest` não
   roda dentro da imagem (a seção 5 fica restrita ao `ci.yaml`).
5. `CLAUDE.md` é também o **documento de handoff**: estado atual, decisões,
   pendências e próximos passos, para continuar o projeto numa sessão nova sem
   o histórico desta conversa.
6. `SAGA-DA-RTX3050.md` vem para este repo (Parte I = projetos anteriores,
   intacta) e ganha a Parte II com a jornada do `pytorch-arc-hosted`.
