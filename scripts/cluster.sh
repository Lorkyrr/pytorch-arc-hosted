#!/usr/bin/env bash
# Ambiente local do pytorch-arc-hosted: cluster kind com a GPU do host + ARC.
#
#   scripts/cluster.sh up [owner/repo]   cria/atualiza tudo (idempotente, não apaga nada)
#   scripts/cluster.sh status            GPU alocável, versões e pods do ARC
#   scripts/cluster.sh down              apaga o cluster (cache de imagens e dataset ficam no host)
#
# Sempre instala as versões mais novas (chart do ARC, device plugin, runner).
# Token do GitHub: $GITHUB_TOKEN ou `gh auth token`, nunca gravado em disco.
set -euo pipefail

CLUSTER=pah
CONTEXT=kind-$CLUSTER
NAMESPACE=arc-systems
RUNNER_SET=arc-runner-set-gpu
SECRET=pah-github-token
CHARTS=oci://ghcr.io/actions/actions-runner-controller-charts
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\n[ERRO] %s\n' "$*" >&2; exit 1; }
k() { kubectl --context "$CONTEXT" "$@"; }
h() { helm --kube-context "$CONTEXT" "$@"; }

check_prereqs() {
  local missing=() bin
  for bin in docker kind kubectl helm nvidia-smi curl; do
    command -v "$bin" >/dev/null || missing+=("comando '$bin' não encontrado")
  done
  if command -v docker >/dev/null; then
    if ! docker info >/dev/null 2>&1; then
      missing+=("sem acesso ao Docker (seu usuário está no grupo 'docker'?)")
    elif [ "$(docker info --format '{{.DefaultRuntime}}')" != nvidia ]; then
      missing+=("runtime padrão do Docker não é 'nvidia' (README: 'Preparar o host', passo 3)")
    fi
  fi
  grep -Eq '^[[:space:]]*accept-nvidia-visible-devices-as-volume-mounts[[:space:]]*=[[:space:]]*true' \
    /etc/nvidia-container-runtime/config.toml 2>/dev/null ||
    missing+=("falta 'accept-nvidia-visible-devices-as-volume-mounts = true' em /etc/nvidia-container-runtime/config.toml")
  [ -S /var/run/docker.sock ] || missing+=("socket /var/run/docker.sock não existe")
  if [ ${#missing[@]} -gt 0 ]; then
    printf ' - %s\n' "${missing[@]}" >&2
    die "pré-requisitos faltando (ver README, seção 'Preparar o host')"
  fi
}

check_other_clusters() {
  [ "${ALLOW_OTHER_CLUSTERS:-0}" = 1 ] && return
  local others
  others=$(kind get clusters 2>/dev/null | grep -vx "$CLUSTER" || true)
  [ -z "$others" ] && return
  die "outro(s) cluster(s) kind rodando: $(echo "$others" | tr '\n' ' ')
Cada um com device plugin próprio disputaria a mesma GPU. Derrube com:
  kind delete cluster --name <nome>
(ou rode com ALLOW_OTHER_CLUSTERS=1 se tiver certeza de que nenhum usa a GPU)"
}

repo_slug() {
  local url
  if [ -n "${1:-}" ]; then
    echo "$1"
    return
  fi
  url=$(git -C "$ROOT" remote get-url origin 2>/dev/null) || die "sem 'origin' no git: passe owner/repo como argumento"
  url=${url%.git}
  echo "${url#*github.com[:/]}"
}

github_token() {
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "$GITHUB_TOKEN"
  elif command -v gh >/dev/null && gh auth token 2>/dev/null; then
    :
  else
    die "defina GITHUB_TOKEN ou faça 'gh auth login'"
  fi
}

latest_release() { # $1 = owner/repo no GitHub
  local json
  # Guardar antes de filtrar: com pipefail, `curl | grep -m1` pode falhar por SIGPIPE
  json=$(curl -fsSL --retry 4 "https://api.github.com/repos/$1/releases/latest")
  grep -m1 '"tag_name"' <<<"$json" | cut -d'"' -f4
}

wait_for_gpu() {
  local gpus
  # 5 min: na 1ª subida o plugin leva ~2 min entre o pod ficar Ready e registrar a GPU
  # no kubelet (medido em 2026-09-29); 2 min não bastavam.
  for _ in $(seq 150); do
    gpus=$(k get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}')
    if [ "${gpus:-0}" -ge 1 ]; then
      echo "GPU alocável no node: $gpus"
      return
    fi
    sleep 2
  done
  die "o node não anunciou nvidia.com/gpu em 5 min (kubectl --context $CONTEXT -n kube-system logs ds/nvidia-device-plugin-daemonset)"
}

cmd_up() {
  local slug token plugin gid
  check_prereqs
  check_other_clusters
  slug=$(repo_slug "${1:-}")
  [[ $slug =~ ^[^/]+/[^/]+$ ]] || die "repositório inválido: '$slug' (esperado owner/repo)"
  token=$(github_token)

  if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
    log "Cluster '$CLUSTER' já existe: reaproveitando"
  else
    log "Criando o cluster '$CLUSTER'"
    kind create cluster --name "$CLUSTER" --config "$ROOT/k8s/kind-config.yaml"
  fi
  docker exec "$CLUSTER-control-plane" nvidia-smi -L ||
    die "a GPU não aparece dentro do node do kind (README: 'Solução de problemas')"

  plugin=$(latest_release NVIDIA/k8s-device-plugin)
  [ -n "$plugin" ] || die "não consegui descobrir a versão mais nova do NVIDIA device plugin"
  log "NVIDIA device plugin $plugin"
  k apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/$plugin/deployments/static/nvidia-device-plugin.yml"
  k -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=300s
  wait_for_gpu

  log "ARC: controller + runner set '$RUNNER_SET' para $slug"
  k create namespace "$NAMESPACE" --dry-run=client -o yaml | k apply -f -
  # Token via stdin: não aparece na linha de comando (ps) nem em arquivo
  printf '%s' "$token" |
    k -n "$NAMESPACE" create secret generic "$SECRET" --from-file=github_token=/dev/stdin \
      --dry-run=client -o yaml | k apply -f -
  h upgrade --install arc "$CHARTS/gha-runner-scale-set-controller" -n "$NAMESPACE" --wait
  gid=$(stat -c %g /var/run/docker.sock)
  h upgrade --install "$RUNNER_SET" "$CHARTS/gha-runner-scale-set" -n "$NAMESPACE" --wait \
    -f "$ROOT/k8s/runner-values.yaml" \
    --set githubConfigUrl="https://github.com/$slug" \
    --set "template.spec.securityContext.supplementalGroups[0]=$gid"

  cmd_status
}

cmd_status() {
  kind get clusters 2>/dev/null | grep -qx "$CLUSTER" || die "o cluster '$CLUSTER' não existe (rode: scripts/cluster.sh up)"
  log "GPU alocável"
  k get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
  log "Device plugin"
  k -n kube-system get daemonset nvidia-device-plugin-daemonset \
    -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
  log "ARC (versões dos charts e pods)"
  h list -n "$NAMESPACE"
  k -n "$NAMESPACE" get pods
}

cmd_down() {
  kind delete cluster --name "$CLUSTER"
  echo "O cache de imagens e o volume pah-cifar10 continuam no Docker do host."
}

case "${1:-}" in
  up) cmd_up "${2:-}" ;;
  status) cmd_status ;;
  down) cmd_down ;;
  *)
    echo "uso: $0 up [owner/repo] | status | down" >&2
    exit 64
    ;;
esac
