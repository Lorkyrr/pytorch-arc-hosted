# CLAUDE.md

Guidance for Claude Code in this repo. **It is also the handoff document:** a fresh
session should be able to continue the work from this file alone, without the
conversation that built it. Keep the "Handoff" section current whenever state changes.

## What this is

Learning project (author: Lorkyrr, beginner, Brazilian; answer in pt-BR). It runs a GPU
benchmark and ResNet/CIFAR-10 training **on the author's real RTX 3050 Laptop (4 GB)**,
triggered by GitHub Actions. The flow:

1. `image.yaml` (GitHub-hosted) resolves the newest PyTorch base image, builds, and pushes to
   `ghcr.io/lorkyrr/pytorch-arc-hosted:{<sha>,latest}`.
2. `gpu.yaml` fires on `workflow_run` (or `workflow_dispatch`). It runs on `arc-runner-set-gpu`, a
   self-hosted ARC runner inside a local `kind` cluster, and executes the image with `docker run`
   on the **host's** Docker daemon (socket mounted into the pod), pinned to the GPU the k8s device
   plugin reserved for that pod.

Ground-up rewrite (2026-09-24, Claude Opus 5.5, with superpowers skills) of the author's earlier
repos, which are siblings of this directory: `../pytorch-gpu-sandbox` (v1.0, the original
kind+ARC setup), `../pytorch-gpu-refactor` (modular package, tried Docker Build Cloud → GHCR),
`../refactored-pytorch-CTTK` (same, plus a 678-line auto-cluster script and ~15 overlapping docs),
`../pytorch-gpu-sandbox-debug`, `../Pytorch-Full`, and `../Pytorch 2`. Those were built with Sonnet
and no skills. **Do not copy their structure back in.** Their root pain was where 3–5 GB of CUDA
wheels live, which caused two competing variants (python/docker), `get-pip.py`, PEP 668 hacks, a
never-built runner image, and a disabled push trigger.

Design and plan (read before any structural change):
- Spec: [docs/specs/2026-09-24-pytorch-arc-hosted-design.md](docs/specs/2026-09-24-pytorch-arc-hosted-design.md). **Section 8 overrides the earlier sections.**
- Plan: [docs/plans/2026-09-24-pytorch-arc-hosted.md](docs/plans/2026-09-24-pytorch-arc-hosted.md) (10 tasks, executed inline).
- Narrative: [SAGA-DA-RTX3050.md](SAGA-DA-RTX3050.md). Part I covers the old repos; Part II (chapters 19+) covers this one.

## Handoff: current state

**As of 2026-09-24** (the author's internet was bad, so **nothing heavy was downloaded locally**):

| Item | State |
|---|---|
| Package, tests, Dockerfile, compose, 3 workflows, Dependabot, kind config, runner values, `cluster.sh`, docs | Done and committed on `main` |
| Local checks | `ruff check .` clean. `pytest`: the torch-free tests pass (CLI parsing, base-image resolver). The torch tests skip locally because torch is not installed on the host, on purpose. `shellcheck` is clean on `cluster.sh` and on every workflow `run:` block. `docker compose config` is OK. |
| Base image resolver, real run | Verified 2026-09-24: `pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime` |
| Push to GitHub, CI, and image build | See "Remote verification log" at the end of this file |
| kind cluster `pah` | **Not created.** The old cluster `kind` from the earlier repos **still exists** and must be deleted first (`kind delete cluster --name kind`). `cluster.sh up` refuses to run while it exists. |
| First real GPU run | **Not done.** Blocked on the cluster (and on good internet, since the first pull is ~3.3 GB). |

### Next steps (in order)

1. When the internet is good: `kind delete cluster --name kind`, then `scripts/cluster.sh up`.
2. Optional: `docker pull ghcr.io/lorkyrr/pytorch-arc-hosted:latest` to pre-warm the host cache.
3. `gh workflow run gpu.yaml`, then `gh run watch`. Check the first step's output for
   `NVIDIA_VISIBLE_DEVICES=GPU-…` (open verification point 1).
4. Train: `gh workflow run gpu.yaml -f mode=train -f epochs=30`. Compare with the old baseline
   (89.40% for resnet20, 30 epochs, bs128, lr 0.1, FP32). AMP is on by default now.
5. Record the results in this file and in SAGA chapter 27.

### Open verification points

1. **Does the pod get `NVIDIA_VISIBLE_DEVICES=<GPU UUID>`, and do job steps inherit it?** The
   device plugin's default `DEVICE_LIST_STRATEGY=envvar` should set it. `gpu.yaml` fails loudly if
   it is empty, `all`, `none`, or `void`, and **never** falls back to `--gpus all` (that would bring
   back the "GPU outside k8s accounting" gap from the old repos). If it doesn't arrive, the likely
   fix is reading it inside the pod (`nvidia-smi --query-gpu=uuid --format=csv,noheader`, since the
   nvidia runtime injects nvidia-smi into the pod) and still passing a specific UUID.
2. **Runner image has the `docker` CLI:** verified 2026-09-24. The upstream `actions/runner`
   `images/Dockerfile` installs Docker 29.8.1 static binaries and buildx, and creates the group
   `docker` with GID 123. Our pod adds the **host socket's GID** (1002 on this host) via
   `supplementalGroups`.
3. **GHCR package visibility:** the first push creates the package as private. `gpu.yaml` logs in
   with `GITHUB_TOKEN` (`packages: read`), so pulls work either way. Making it public is optional
   (package settings on GitHub).

## Cache strategy (where bytes live, and what not to re-download)

This is the core design concern. The author's link is slow and unstable, and the first ~3.3 GB
pull is the expensive part. Each layer below is independent:

| Cache | Where | Survives | Status |
|---|---|---|---|
| **Docker layer cache (host)** | Host Docker daemon; the runner pod talks to it through the mounted `/var/run/docker.sock` | Jobs, pods, **`kind delete cluster`**, reboots | Designed. Warms on the first `gpu.yaml` run or on a manual `docker pull`. After that, a pull of a new `:<sha>` only fetches the code layer (KBs), because the base layers are shared by digest. |
| **Dataset** | Named Docker volume `pah-cifar10` on the host (the CI uses `-v pah-cifar10:/data`, and compose names the volume explicitly, so there's no project prefix) | Same as above; shared between CI and local compose | Designed; fills on the first `train` |
| **Image build cache (GitHub → GHCR)** | Inline cache metadata inside `ghcr.io/lorkyrr/pytorch-arc-hosted:latest` (`cache-from: type=registry,ref=…:latest`, `cache-to: type=inline` in `image.yaml`) | Across builds, as long as `:latest` exists | Implemented. When the resolved base did not change, BuildKit reuses the `RUN` layer and doesn't re-extract the base. `pull: true` still checks for a newer base digest. |
| **Base image choice** | `scripts/latest_pytorch_base.py` (Docker Hub API, 4 retries with backoff) | n/a | Implemented. It picks the newest `<torch>-cuda<X.Y>-cudnn<N>-runtime` with CUDA ≤ `MAX_CUDA`. A new PyTorch release means **one** new ~3.3 GB pull on the host, which the author has accepted ("always newest, even if it re-downloads"). |
| **kind node / ARC / device-plugin images** | The node's **internal** containerd | Pods and restarts, but **not `kind delete cluster`** | Not cached across cluster recreation (~1.2 GB per recreate). Considered and deferred: `kind load docker-image` from the host cache (needs pinned tags, which conflicts with "always newest"; it's also flaky with multi-arch images on Docker's containerd image store). |
| **CI pip (torch CPU, ~200 MB)** | GitHub-hosted runner | n/a | Not cached. It runs on GitHub's network, costs the author nothing, and `3.x` + unpinned torch make cache keys churn anyway. Add `actions/setup-python` `cache: pip` with `cache-dependency-path: pyproject.toml` if CI time ever matters. |

Rejected alternatives (don't re-propose without new facts):
- **ARC dind mode:** the daemon and its cache die with each pod, which means ~3.3 GB per job.
- **Native python runner with wheels in the pod:** the old repos' failure mode (PEP 668, `get-pip.py`, 3–5 GB per job).
- **A custom fat runner image:** it's re-pulled into kind's containerd on every cluster recreate, and every dependency bump means GBs.
- **`type=gha` cache with `mode=max`:** it would store the ~3 GB base inside the 10 GB GHA cache for no gain.
- **`type=registry` `:buildcache` with `mode=max`:** it duplicates the base layers in GHCR, and inline covers our single-stage Dockerfile.

## Decisions and why

| Decision | Why |
|---|---|
| **Docker only** (one Dockerfile for CI and local) | The author wants a single path; a python-native variant was explicitly rejected |
| Host Docker socket in the runner pod (DooD) | The layer cache persists on the host (see the cache table) |
| `docker run --gpus "device=$NVIDIA_VISIBLE_DEVICES"` | It pins the sibling container to the GPU k8s reserved for the pod, which closes the old "partially virtual GPU" gap |
| `supplementalGroups: [<host socket GID>]`, injected by `cluster.sh` | Socket access without `runAsUser: 0` or `RUNNER_ALLOW_RUNASROOT` |
| **Never `pytorch/pytorch:latest`** | It's frozen at 2024-02-23 (PyTorch 2.2.1). The old repos silently ran that. |
| `MAX_CUDA: "13.4"` in `image.yaml` | Driver 615 supports up to CUDA 13.4. A newer-CUDA base would make `cuda.is_available()` false, which `--require-gpu` catches. |
| **Always newest** (floating chart, device plugin, runner, Actions majors, Python `3.x`) plus Dependabot | Author's explicit preference; reproducibility traded for freshness. Resolved versions are printed (the image build summary, the container's env report, and `cluster.sh status`). |
| `gpu.yaml` only has `workflow_run` and `workflow_dispatch`, plus a `head_branch == 'main'` guard | Public repo + self-hosted runner + host socket = root on the host. Fork PRs must never reach it. |
| `--require-gpu` in CI | Otherwise a CUDA failure silently falls back to CPU and the job goes green |
| `concurrency: gpu`, `maxRunners: 1` | One physical GPU |
| `cluster.sh up` refuses if another kind cluster exists | Two device plugins on one GPU = contention (the old cluster `kind` exists on this host) |
| Token via `$GITHUB_TOKEN` or `gh auth token`, piped to `kubectl create secret --from-file=/dev/stdin` | Never on disk and never in argv. The old repos used a `token.md` file. |
| ResNet built from scratch (6n+2, option-B projection shortcuts); same hyperparameters as the old repos | Comparable to the old baselines. resnet20 has 272,474 params (asserted in the tests). |

## Repository map

```
pytorch_arc_hosted/            python -m pytorch_arc_hosted {benchmark,train}
  cli.py                         argparse; torch imported lazily inside main() (parsing tests need no torch)
  env.py                         report_environment() -> device; banner/section console helpers
  bench.py                       gflops(), matmul/conv/amp/synthetic benches; sizes fit 4 GB VRAM
  model.py                       RESNET_BLOCKS, BasicBlock, ResNetCIFAR, build_resnet(arch)
  data.py                        cifar10_loaders(data_dir, batch_size, pin_memory)
  train.py                       EpochStats, lr_milestones, fit(), write_summary(), train_cifar10()
scripts/
  cluster.sh                     up [owner/repo] | status | down  (kind "pah" + device plugin + ARC)
  latest_pytorch_base.py         newest pytorch/pytorch runtime tag with CUDA <= --max-cuda (stdlib only)
k8s/
  kind-config.yaml               GPU passthrough (/dev/null -> /var/run/nvidia-container-devices/all) + docker.sock
  runner-values.yaml             gha-runner-scale-set values: nvidia.com/gpu: 1, docker.sock hostPath
.github/workflows/
  ci.yaml                        GitHub-hosted: torch CPU, ruff, pytest (push + PR)
  image.yaml                     GitHub-hosted: resolve base -> build+push GHCR (push main, weekly, dispatch)
  gpu.yaml                       self-hosted arc-runner-set-gpu: pull + docker run on the GPU
.github/dependabot.yml           weekly github-actions bumps
tests/                           pytest; torch tests use importorskip
Dockerfile · .dockerignore · compose.yaml
docs/specs · docs/plans          design + implementation plan
SAGA-DA-RTX3050.md               study narrative (Part I old repos, Part II this repo)
```

Interfaces (keep stable, since tests and workflows depend on them): the CLI flags `--arch --epochs
--batch-size --lr --no-amp --seed --data-dir(/data) --out-dir(/out) --require-gpu`; the outputs
`/out/<arch>_cifar10.pth` (a dict with `arch`, `epoch`, `test_acc`, `state_dict`) and
`/out/summary.md` (appended to `$GITHUB_STEP_SUMMARY`); the names `pah` (cluster),
`kind-pah` (context), `arc-systems`, `arc`, `arc-runner-set-gpu`, `pah-github-token` (key
`github_token`), `pah-cifar10`, and `ghcr.io/lorkyrr/pytorch-arc-hosted` (lowercase is required by
GHCR). The workflow name `Imagem (GHCR)` is referenced literally by `gpu.yaml`.

## Commands

```bash
ruff check .                     # lint (CI runs only `check`, not `format`)
pytest                           # torch tests skip without torch
scripts/cluster.sh up            # create/update kind + GPU + ARC (idempotent; never deletes)
scripts/cluster.sh status
scripts/cluster.sh down          # deletes cluster only; host image cache + pah-cifar10 stay
gh workflow run gpu.yaml                                   # benchmark
gh workflow run gpu.yaml -f mode=train -f arch=resnet56 -f epochs=50 -f seed=42
gh workflow run image.yaml                                 # rebuild with newest base now
python3 scripts/latest_pytorch_base.py --max-cuda 13.4     # which base would be used
docker compose run --rm app [train --epochs 30]            # local, same image + volume
```

## Conventions

- Identifiers in English; comments and console output in **pt-BR with correct accents**.
- **No heavy local downloads without asking**: the author's internet is unreliable. Metadata API
  calls are fine. torch-dependent verification happens in `ci.yaml`.
- TDD. A test that needs torch uses `pytest.importorskip("torch")` at module top, followed by
  `# noqa: E402` imports.
- `ruff` rules: E, F, I, B, UP; line length 120. `latest_pytorch_base` is declared first-party
  in the isort config (it lives in `scripts/`, which is on the pytest `pythonpath`).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Superpowers workflow: brainstorming → spec → writing-plans → executing-plans. The inline-execution
  ledger lives in `.superpowers/sdd/<plan>/progress.md` (git-ignored, local only).

## Host facts (this machine, 2026-09-24)

Ubuntu, NVIDIA driver 615 (CUDA ≤ 13.4), RTX 3050 Laptop 4 GB (cc 8.6). Docker's default
runtime is already `nvidia`, and `accept-nvidia-visible-devices-as-volume-mounts = true` is
already set. The `/var/run/docker.sock` GID is **1002**. kind v0.33.0; `gh` is logged in as
`Lorkyrr`; `shellcheck` is available. The host Python is 3.14.4 with **no torch**. The old kind
cluster `kind` still exists.

Newest versions seen on 2026-09-24 (floating; for reference only): PyTorch base
`2.14.0-cuda13.2-cudnn9-runtime`, ARC charts `0.14.2`, device plugin `v0.20.1`,
actions/runner `v2.337.0`, `actions/checkout@v7`, `setup-python@v7`, `upload-artifact@v7`,
`docker/login-action@v4`, `setup-buildx-action@v4`, `build-push-action@v7`.

## Gotchas

| Symptom / trap | Cause | Fix |
|---|---|---|
| `pytorch/pytorch:latest` looks convenient | Frozen since 2024 | Use the resolver; never hardcode `:latest` |
| `helm --set template.spec.containers[0].image=…` | Helm **replaces whole lists** from `--set`, so the container loses `command`/`resources` (GPU!) | Change the list in `runner-values.yaml`; only `githubConfigUrl` and `supplementalGroups[0]` go through `--set` |
| New ARC version, but CRDs stay old | `helm upgrade` doesn't update CRDs | `cluster.sh down && cluster.sh up` (the cluster is disposable) |
| CI breaks right after a new Python release | `python-version: "3.x"` moves before torch publishes wheels | Temporarily pin the previous minor in `ci.yaml` |
| `curl … \| grep -m1` aborts under `set -o pipefail` | SIGPIPE on curl | Capture into a variable first (see `latest_release`) |
| Docker Hub API timeouts | Unstable link | The resolver retries 4x with backoff; `curl --retry 4` in `cluster.sh` |
| nvidia-smi missing inside the kind node | Default runtime isn't nvidia, or the accept-volume-mounts line is missing | README "Preparar o host" steps 3–4 |
| Device plugin `ERROR_LIBRARY_NOT_FOUND` | The node's **internal** containerd lacks the nvidia runtime | `containerdConfigPatches` in `k8s/kind-config.yaml` |
| `Insufficient nvidia.com/gpu` | The GPU is held by another pod, or the plugin is gone after a cluster recreate | `cluster.sh up` reinstalls it |
| Docker socket `permission denied` in the job | The socket GID changed | `cluster.sh up` rereads the GID |
| A "fixed" workflow still runs old code | "Re-run jobs" reuses the original SHA | "Run workflow" creates a new run |
| Bind mounts from the job into `docker run` don't work | The daemon is the host's, and the pod FS isn't visible to it | Named volume for data, `docker cp` for outputs |
| Job cancelled but training keeps running | Killing the CLI doesn't kill the host container | The cleanup step runs on `always()` and does `docker rm -f` |

## Remote verification log

(Filled in by plan Task 10 after the first push.)
