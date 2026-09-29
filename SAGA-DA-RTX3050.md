# A Saga da RTX 3050 — do `docker run` ao cluster de verdade

> Documento de estudo, não de referência de API. Feito pra ser lido de cima a
> baixo, sem internet, num dia em que a conexão não colabora. Cada capítulo
> explica **o quê**, **por quê** e **como verificar com os próprios olhos**
> que o que está escrito é verdade no seu ambiente — porque aprender de
> verdade é conseguir refazer sem depender de mim pedindo pra eu escrever o
> YAML de novo.

> **Nota (2026-09-24, repositório `pytorch-arc-hosted`):** este documento tem
> duas partes. A **Parte I** (capítulos 1 a 18) foi escrita nos repositórios
> anteriores (`pytorch-gpu-sandbox` e derivados) e está aqui **sem alterações**.
> Os links de arquivo dela (`main.py`, `k8s/gpu-runner-values.yaml`,
> `scripts/2-setup-arc.sh`…) apontam para aqueles repositórios e **não existem
> neste**. A **Parte II** (capítulos 19 em diante) conta a reescrita do zero que
> virou este repositório, e os links dela apontam para arquivos daqui.

## Como usar este documento

Cada capítulo é independente o bastante pra ser lido fora de ordem, mas eles
contam uma história em sequência: começamos com um script Python rodando
dentro de um container Docker no seu notebook, e terminamos com um cluster
Kubernetes de verdade, com sua RTX 3050 sendo reservada como recurso
agendável, treinando um ResNet-20 sozinho sempre que você dispara o
workflow (hoje é sob demanda, via `workflow_dispatch` — ver capítulo 17
sobre por que o gatilho automático em `push` foi removido). No meio do
caminho, quebramos as coisas umas seis vezes de formas diferentes — e cada
quebra ensina algo que o caminho feliz não ensina.

Sempre que um capítulo referenciar um arquivo do repositório, o caminho está
entre colchetes tipo `[main.py](main.py)` — abra o arquivo de verdade ao
lado deste texto, não confie só na minha descrição.

---

## Índice

1. [O ponto de partida: o que já existia](#1-o-ponto-de-partida-o-que-já-existia)
2. [Por que CI numa GPU real, e não um runner "de graça" da GitHub](#2-por-que-ci-numa-gpu-real-e-não-um-runner-de-graça-da-github)
3. [Primeira versão do workflow: CI via Docker](#3-primeira-versão-do-workflow-ci-via-docker)
4. [O bug do bind mount: Docker-fora-do-Docker](#4-o-bug-do-bind-mount-docker-fora-do-docker)
5. [Como uma GPU física vira um recurso do Kubernetes](#5-como-uma-gpu-física-vira-um-recurso-do-kubernetes)
6. [ARC: quem cria os pods que rodam seus workflows](#6-arc-quem-cria-os-pods-que-rodam-seus-workflows)
7. [GPU "virtual" vs. GPU alocada de verdade](#7-gpu-virtual-vs-gpu-alocada-de-verdade)
8. [Dois workflows, um só runner: evitando colisão de nomes](#8-dois-workflows-um-só-runner-evitando-colisão-de-nomes)
9. [Tirando o Docker do CI: rodando Python nativo no pod](#9-tirando-o-docker-do-ci-rodando-python-nativo-no-pod)
10. [A saga de três erros seguidos no `pip install`](#10-a-saga-de-três-erros-seguidos-no-pip-install)
11. [A armadilha do "Re-run jobs"](#11-a-armadilha-do-re-run-jobs)
12. [Banda larga é infraestrutura: o custo real de baixar PyTorch](#12-banda-larga-é-infraestrutura-o-custo-real-de-baixar-pytorch)
13. [Git merge de branches: o rename que deu errado](#13-git-merge-de-branches-o-rename-que-deu-errado)
14. [VS Code integrado: debug remoto e tasks automatizadas](#14-vs-code-integrado-debug-remoto-e-tasks-automatizadas)
15. [A pasta-mãe: cópia total pra rodar em outra máquina](#15-a-pasta-mãe-cópia-total-pra-rodar-em-outra-máquina)
16. [Glossário rápido](#16-glossário-rápido)
17. [O que ainda falta (TODOs reais)](#17-o-que-ainda-falta-todos-reais)
18. [Comandos de referência rápida](#18-comandos-de-referência-rápida)

**Parte II: pytorch-arc-hosted**

19. [Recomeço: o que cinco projetos ensinaram](#19-recomeço-o-que-cinco-projetos-ensinaram)
20. [Só Docker, e o socket do host como cache](#20-só-docker-e-o-socket-do-host-como-cache)
21. [Fechando a GPU "parcialmente virtual"](#21-fechando-a-gpu-parcialmente-virtual)
22. [A imagem no GHCR, ativada por `workflow_run`](#22-a-imagem-no-ghcr-ativada-por-workflow_run)
23. [A tag `latest` que parou em 2024](#23-a-tag-latest-que-parou-em-2024)
24. [Repo público + runner self-hosted: o risco do socket](#24-repo-público--runner-self-hosted-o-risco-do-socket)
25. [Sem root: `supplementalGroups`](#25-sem-root-supplementalgroups)
26. [Dois clusters, uma GPU](#26-dois-clusters-uma-gpu)
27. [O que ainda falta (Parte II)](#27-o-que-ainda-falta-parte-ii)
28. [Comandos de referência rápida (Parte II)](#28-comandos-de-referência-rápida-parte-ii)
29. [A revisão final: o `concurrency` que apagava treinos](#29-a-revisão-final-o-concurrency-que-apagava-treinos)
30. [A primeira execução real: quatro quedas até o primeiro "success"](#30-a-primeira-execução-real-quatro-quedas-até-o-primeiro-success)

---

# Parte I: os repositórios anteriores

## 1. O ponto de partida: o que já existia

Antes de tudo isso, o repositório já tinha:

- [main.py](main.py) — um script com dois modos (`argparse` com um argumento
  posicional `modo`):
  - `benchmark` (padrão): roda testes de `torch.mm` (multiplicação de
    matrizes, usa a biblioteca cuBLAS por baixo), convolução 2D (cuDNN),
    precisão mista FP16 (`torch.autocast`, usa os Tensor Cores da GPU) e um
    laço de treino simulado — tudo cronometrado, pra você ver números reais
    de desempenho da sua placa.
  - `train`: treina de verdade uma **ResNet-20** (a variante de 6n+2 camadas
    do paper original do ResNet, pensada pra imagens pequenas 32x32 como as
    do CIFAR-10) do zero — sem usar `torchvision.models`, a arquitetura
    inteira (`BlocoResidual`, `ResNetCIFAR`) está escrita à mão no arquivo.
- [Dockerfile](Dockerfile) — parte da imagem oficial `pytorch/pytorch:latest`
  (que já vem com PyTorch + CUDA compilados) e só adiciona `torchvision` e
  `torchaudio`.
- [compose.yaml](compose.yaml) — sobe esse container localmente, com
  `deploy.resources.reservations.devices` pedindo a GPU NVIDIA do host e um
  volume `.:/app` que espelha a pasta do projeto pra dentro do container (é
  por isso que o `./data` baixado e o checkpoint `.pth` gerado aparecem de
  volta no seu notebook depois do treino).
- [compose.debug.yaml](compose.debug.yaml) — variante que troca o comando
  padrão por uma instalação do `debugpy` na hora e sobe `main.py` esperando
  um debugger remoto conectar na porta 5678, antes de rodar qualquer linha.

Esse era o "modo local": tudo dentro de um container Docker, na sua máquina,
disparado manualmente. O que fizemos a partir daqui foi levar isso pra rodar
**automaticamente em CI**, usando **a GPU física do seu notebook**, dentro de
um cluster Kubernetes que você mesmo hospeda.

---

## 2. Por que CI numa GPU real, e não um runner "de graça" da GitHub

GitHub Actions te dá runners hospedados de graça (`ubuntu-latest`,
`windows-latest` etc.), mas **nenhum deles tem GPU** no plano gratuito — só
existe runner com GPU nos planos pagos de "larger runners", que cobram por
minuto e por hora de GPU alocada. Pra um projeto pessoal de aprendizado, isso
não faz sentido financeiro.

A alternativa é um **runner self-hosted**: em vez da GitHub prover a máquina
que executa o job, você mesmo aponta uma máquina sua (física ou virtual) pra
rodar os jobs. A GitHub só manda "tem um job esperando" e sua máquina pega o
trabalho. Isso significa: **zero custo de nuvem**, mas **você** é responsável
por manter essa máquina de pé, seguro e com os pré-requisitos certos (nesse
caso: driver NVIDIA, Docker, Kubernetes).

O caminho que você montou (documentado nos scripts irmãos
[1-create-gpu-cluster.sh](scripts/1-create-gpu-cluster.sh),
[2-setup-arc.sh](scripts/2-setup-arc.sh),
[3-teardown-cluster.sh](scripts/3-teardown-cluster.sh) e no arquivo de referência
[kubernetes-gpu-arc-referencia.sh](scripts/kubernetes-gpu-arc-referencia.sh)) foi:

1. Um cluster Kubernetes local via **kind** (`Kubernetes IN Docker` — cria
   "nodes" de Kubernetes que são, na real, containers Docker comuns).
2. Dentro desse cluster, o **Actions Runner Controller (ARC)** — um
   componente que fica de olho no GitHub e cria/destrói pods runner sob
   demanda, cada um executando um job de workflow e depois sumindo.
3. Um jeito de esses pods enxergarem a RTX 3050 física do notebook.

Os capítulos 5 e 6 destrincham como cada peça disso funciona. Antes, vamos
ver o workflow em si — porque foi por ele que começamos.

---

## 3. Primeira versão do workflow: CI via Docker

O primeiro workflow funcional (hoje preservado como referência em
[.github/workflows/pytorch-gpu-docker.yaml](.github/workflows/pytorch-gpu-docker.yaml))
seguia a lógica mais óbvia possível: fazer no CI exatamente o que você já
fazia local com `docker compose`.

```yaml
on:
  workflow_dispatch:
    inputs:
      mode:
        type: choice
        options: [benchmark, train]
```

`workflow_dispatch` é o gatilho **manual** — o botão "Run workflow" na aba
Actions do GitHub. `inputs.mode` com `type: choice` vira um menu suspenso na
hora de disparar, deixando escolher entre rodar o benchmark ou o treino sem
precisar editar o YAML toda vez.

```yaml
jobs:
  run:
    runs-on: arc-runner-set-gpu
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t pytorch-cifar10:${{ github.sha }} .
      - run: docker run --gpus all pytorch-cifar10:${{ github.sha }} python3 main.py benchmark
```

- `runs-on: arc-runner-set-gpu` é o nome do **runner scale set** que o ARC
  gerencia (configurado no capítulo 6) — é assim que o job "escolhe" rodar
  na sua GPU, e não num runner qualquer da GitHub.
- `actions/checkout@v4` é uma **action** (um pacote de automação reutilizável,
  mantido pela própria GitHub nesse caso) que clona o repositório dentro do
  workspace do job — sem isso, o job não teria acesso a `main.py`,
  `Dockerfile` etc.
- `docker build` + `docker run --gpus all` são os mesmos comandos que você já
  rodava local via `docker compose`, só que direto, sem o compose por cima.

Esse workflow funcionou. Ele expôs dois problemas reais, que os capítulos 4 e
7 explicam em detalhe: um bug técnico (bind mount quebrado) e um problema
conceitual mais sutil (a GPU "não pertencer" de verdade ao pod do
Kubernetes).

---

## 4. O bug do bind mount: Docker-fora-do-Docker

A primeira tentativa de extrair o checkpoint treinado do container usava um
bind mount, do jeito que `compose.yaml` já fazia local:

```yaml
docker run --gpus all -v ${{ github.workspace }}:/app pytorch-cifar10:sha ...
```

Isso **quebrou** com um erro enganoso: `python3: can't open file
'/app/main.py': No such file or directory` — como se o `COPY . /app` do
Dockerfile nunca tivesse rodado.

**O que realmente aconteceu**: o runner (o pod do ARC) e o **daemon Docker**
que executa `docker run` não são a mesma coisa, nem compartilham o mesmo
sistema de arquivos. Isso é chamado de **Docker-fora-do-Docker (DooD, "Docker
outside of Docker")** — em vez do pod ter seu próprio Docker completo rodando
dentro dele (o que seria DinD, "Docker in Docker"), ele só tem o **socket**
`/var/run/docker.sock` montado, que é uma forma de "telefonar" pro Docker que
já está rodando em outro lugar (no caso, no host físico, via o mecanismo que
o [kind-gpu-config.yaml](k8s/kind-gpu-config.yaml) monta — ver capítulo 5).

Quando você escreve `-v ${{ github.workspace }}:/app`, o caminho
`${{ github.workspace }}` (algo como
`/home/runner/_work/pytorch-gpu-sandbox/pytorch-gpu-sandbox`) existe no
**pod**, mas o comando é interpretado pelo **daemon remoto**, que procura
esse caminho no *seu próprio* sistema de arquivos — onde ele não existe (ou
existe vazio). O resultado é um diretório vazio sendo montado por cima do
`/app` que o build tinha acabado de preencher.

**A correção** (em
[.github/workflows/pytorch-gpu-docker.yaml](.github/workflows/pytorch-gpu-docker.yaml))
foi trocar o bind mount por `docker cp`, que copia arquivos de dentro de um
container através da **API do Docker** — não depende de sistema de arquivos
compartilhado, só da conexão com o daemon (que já funciona, é o mesmo socket
usado pra `docker build`/`docker run`):

```bash
docker run --gpus all --name treino-$RUN_ID pytorch-cifar10:sha python3 main.py train
docker cp treino-$RUN_ID:/app/resnet20_cifar10.pth ./resnet20_cifar10.pth
docker cp treino-$RUN_ID:/app/data ./data
docker rm -f treino-$RUN_ID
```

**A lição que fica**: sempre que você tiver um container remoto (seja via
DooD, seja um servidor Docker de verdade em outra máquina), bind mount
**não funciona** — ele monta caminhos do lado do daemon, não do lado de quem
digitou o comando. `docker cp` funciona nesses casos porque só depende da
conexão de rede/socket com o daemon, nunca do sistema de arquivos local de
quem está pedindo.

---

## 5. Como uma GPU física vira um recurso do Kubernetes

Esse é o capítulo mais denso, mas é a base de tudo o que vem depois. A
pergunta que ele responde: **como um pod dentro de um cluster kind consegue
"ver" a RTX 3050 do notebook, que fisicamente está fora do cluster?**

### 5.1 kind cria nodes que são containers Docker

O `kind` (Kubernetes IN Docker) não instala um Kubernetes "de verdade" numa
VM — ele sobe um container Docker comum e roda **dentro** dele todo o
software de um node de Kubernetes (`kubelet`, `containerd`, etc.). Rode
`docker ps` com o cluster de pé e você vai ver um container chamado
`kind-control-plane` — esse container *é* o seu cluster de um node só.

Isso importa porque significa que **dar acesso à GPU pro node do kind é
literalmente o mesmo problema de dar acesso à GPU pra qualquer container
Docker comum** — não tem mágica especial de Kubernetes aqui.

### 5.2 O NVIDIA Container Toolkit faz o Docker "emprestar" a GPU

No host (fora de qualquer container), o **NVIDIA Container Toolkit** ensina o
Docker a, quando pedido, montar dentro de um container os arquivos de
dispositivo (`/dev/nvidia0` etc.) e as bibliotecas do driver
(`libcuda.so`, `nvidia-smi`) que já estão instaladas no host. Sem isso,
`docker run --gpus all` simplesmente não teria o que fazer.

O comando chave, rodado uma vez no host:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

Isso registra um **runtime alternativo** chamado `nvidia` no Docker e o torna
o runtime **padrão** — ou seja, todo `docker run` (não só os que pedem GPU)
passa por esse runtime. Isso não vaza acesso à GPU pra containers que não
pediram: o runtime só intervém de verdade quando a variável de ambiente
`NVIDIA_VISIBLE_DEVICES` está definida no container (o que `--gpus all`
define automaticamente); sem ela, ele se comporta como o runtime padrão
`runc`, sem tocar em nada de GPU.

### 5.3 Trazendo isso pra dentro do node do kind

Como o node do kind é "só" um container Docker, ele **herda** esse runtime
padrão do host automaticamente — quando o Docker sobe o container
`kind-control-plane`, ele já nasce com acesso à GPU, do mesmo jeito que
qualquer outro container nasceria.

Mas tem uma pegadinha: dentro desse container-node roda um **outro**
gerenciador de containers, o `containerd`, que é quem o `kubelet` usa pra
criar os pods do cluster (não confundir com o Docker do host — são dois
softwares diferentes, mesmo fazendo coisas parecidas). Esse `containerd`
*interno* não sabe nada sobre `nvidia` como runtime, a não ser que a gente
configure explicitamente.

É pra isso que serve a parte `containerdConfigPatches` do
[kind-gpu-config.yaml](k8s/kind-gpu-config.yaml):

```yaml
containerdConfigPatches:
  - |-
    [plugins."io.containerd.grpc.v1.cri".containerd]
      default_runtime_name = "nvidia"
    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
      runtime_type = "io.containerd.runc.v2"
      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
        BinaryName = "/usr/bin/nvidia-container-runtime"
```

Isso é literalmente o mesmo tipo de configuração que fizemos no Docker do
host (seção 5.2), só que escrita no formato de config do `containerd` — e
aplicada especificamente ao `containerd` **de dentro do node**, não ao do
host. Sem isso, pods criados pelo Kubernetes (como o próprio device plugin da
seção 5.4) simplesmente não enxergam a GPU, mesmo que
`docker exec kind-control-plane nvidia-smi` já funcione perfeitamente — são
duas camadas diferentes.

Os `extraMounts` do mesmo arquivo resolvem um problema relacionado: o
runtime `nvidia-container-runtime` referenciado acima precisa **existir**
como binário dentro do node. `hostPath: /usr/bin/nvidia-container-runtime` →
`containerPath: /usr/bin/nvidia-container-runtime` copia esse binário (e os
irmãos `nvidia-container-cli`, `nvidia-ctk`) do host pra dentro do container
do node na hora da criação do cluster.

### 5.4 O Device Plugin: quem avisa o Kubernetes que existe 1 GPU

Mesmo com tudo isso, o **scheduler** do Kubernetes (a peça que decide em qual
node cada pod vai rodar) não sabe, por padrão, que existe uma GPU disponível
— pra ele, "GPU" não é um recurso nativo como CPU ou memória.

É aí que entra o **NVIDIA k8s-device-plugin**
([k8s/nvidia-device-plugin.yaml](k8s/nvidia-device-plugin.yaml), um
`DaemonSet` oficial da NVIDIA, aplicado direto com `kubectl create -f <url>`,
sem Helm nem o "GPU Operator" completo — uma escolha deliberadamente mais
simples, que só faz uma coisa). Um `DaemonSet` é um tipo de objeto do
Kubernetes que garante "uma cópia deste pod rodando em cada node" — nesse
caso, um pod que:

1. Detecta a GPU física disponível no node (via as bibliotecas do driver
   injetadas pelo runtime `nvidia`, seção 5.2/5.3).
2. Anuncia pro `kubelet` do node: "eu tenho 1 unidade do recurso
   `nvidia.com/gpu` disponível aqui".

Depois disso, `kubectl describe node kind-control-plane` mostra
`nvidia.com/gpu: 1` na seção `Allocatable` — esse número é exatamente o que
permite um pod **pedir** essa GPU (capítulo 6) e o scheduler **recusar**
agendar um segundo pod que peça a mesma GPU quando ela já estiver em uso
(diferente de CPU/memória, que podem ser fatiadas entre vários pods, GPU
inteira normalmente vai pra um pod só de cada vez, a não ser que você
configure "time-slicing" — que propositalmente não fizemos aqui, porque você
só tem uma placa e quer ela inteira pro treino).

**Detalhe que já mordeu vocês**: o `kind delete cluster` apaga o namespace
`kube-system` inteiro, e o device plugin mora lá — então toda vez que o
cluster é recriado, o device plugin precisa ser reinstalado, senão qualquer
pod que peça `nvidia.com/gpu` fica preso com o erro `0/1 nodes are
available: 1 Insufficient nvidia.com/gpu`. É por isso que
[2-setup-arc.sh](scripts/2-setup-arc.sh) reinstala o device plugin toda vez que
roda, em vez de assumir que ele já está lá.

---

## 6. ARC: quem cria os pods que rodam seus workflows

O **Actions Runner Controller (ARC)** é instalado via Helm (o "gerenciador de
pacotes" do Kubernetes — empacota vários YAMLs relacionados num "chart"
instalável com um comando só) em duas partes:

### 6.1 O controller

```bash
helm upgrade --install arc \
  --namespace arc-systems \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller
```

Esse é o "cérebro": um pod que fica de olho no GitHub (via o token que você
fornece) e observa quando há jobs esperando pra rodar num runner scale set
que ele gerencia. Só existe **um** controller, mesmo que você tenha vários
runner scale sets diferentes.

### 6.2 Os runner scale sets

Cada `helm upgrade --install <nome> ... gha-runner-scale-set` cria um
**runner scale set** — um "molde" de pod que o controller usa pra criar
runners sob demanda (cria quando tem job esperando, destrói quando o job
termina — diferente de um runner "sempre ligado"). Você tem dois:

- `arc-runner-set` — sem GPU, pra jobs que não precisam dela.
- `arc-runner-set-gpu` — o que interessa aqui, configurado via um arquivo de
  values customizado.

O `<nome>` que você dá no `helm upgrade --install <nome>` é exatamente o
valor que aparece em `runs-on:` nos workflows — é essa string que conecta o
YAML do workflow ao runner scale set certo.

### 6.3 O values.yaml: o que faz o pod ser "com GPU"

[k8s/gpu-runner-values.yaml](k8s/gpu-runner-values.yaml) (e sua variante
[k8s/gpu-runner-values.docker.yaml](k8s/gpu-runner-values.docker.yaml), ver
capítulo 8) customizam o "molde" de pod que esse runner scale set cria. A
parte que importa mais:

```yaml
template:
  spec:
    containers:
      - name: runner
        resources:
          limits:
            nvidia.com/gpu: 1
```

Isso é um **resource request/limit** — a mesma mecânica usada pra CPU
(`cpu: "2"`) ou memória (`memory: "4Gi"`), só que aplicada ao recurso
customizado que o device plugin (seção 5.4) anunciou. Ao ver isso, o
scheduler do Kubernetes só agenda esse pod num node que tenha
`nvidia.com/gpu` disponível na capacidade **e** reserva essa unidade
exclusivamente pra esse pod enquanto ele existir — nenhum outro pod
concorrente consegue pedir a mesma GPU ao mesmo tempo.

A variante Docker (`gpu-runner-values.docker.yaml`) também monta o socket do
Docker do host:

```yaml
volumeMounts:
  - name: docker-sock
    mountPath: /var/run/docker.sock
volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
      type: Socket
```

(Esse é o mecanismo de Docker-fora-do-Docker do capítulo 4 — o
`hostPath` aqui se refere ao "host" do pod, que é o **node do kind**, não seu
notebook diretamente; é o `kind-gpu-config.yaml` que, por sua vez, traz o
socket do notebook real pra dentro do node via seu próprio `extraMounts`.
Duas camadas de "host", uma dentro da outra.)

Também precisa de `securityContext.runAsUser: 0` e
`env: RUNNER_ALLOW_RUNASROOT=1`, porque o usuário padrão da imagem do runner
não tem permissão de acesso ao socket montado — sem isso, o erro é
`permission denied while trying to connect to the Docker API`. E sem a
variável `RUNNER_ALLOW_RUNASROOT`, a imagem se recusa a nem iniciar como
root (entra em crash loop rápido, poucos segundos de `Running` antes de
`Error`).

A variante Python (capítulo 9) **não precisa de nada disso**: sem Docker
envolvido, não tem socket pra montar nem motivo pra rodar como root.

---

## 7. GPU "virtual" vs. GPU alocada de verdade

Esse foi o ponto de virada conceitual da conversa inteira, então vale um
capítulo só pra ele.

Com a variante Docker (capítulo 3) já funcionando, ainda restava uma
pergunta: o `resources.limits.nvidia.com/gpu: 1` do capítulo 6 realmente
garante que **o treino** está usando a GPU reservada, ou só garante que **o
pod do runner** tem uma reserva, sem relação com o que o treino de fato
consome?

A resposta, seguindo a cadeia de comandos: o workflow Docker faz `docker
build` + `docker run --gpus all` **através do socket montado**. Esse comando
não fala com o `containerd` do node (que é quem o Kubernetes usa pra aplicar
o `nvidia.com/gpu: 1`) — ele fala **diretamente com o daemon Docker do
host**, que é um programa totalmente diferente, sem noção nenhuma de "quotas
do Kubernetes". Ou seja: o container de treino, que é quem de fato ocupa a
GPU com CUDA por vários minutos, nasce e morre **completamente por fora** do
que o scheduler do Kubernetes está controlando.

Na prática, isso funciona hoje porque só existe um pod pedindo GPU por vez —
mas o `nvidia.com/gpu: 1` não é o que garante isso; é a política de só
disparar um workflow de cada vez que garante. Se dois pods diferentes
tivessem esse mesmo socket montado, os dois conseguiriam rodar
`docker run --gpus all` simultaneamente e brigar pela mesma placa física, e
o Kubernetes não teria como impedir — porque ele nunca fica sabendo que isso
está acontecendo.

Foi essa lacuna que motivou o capítulo 9: tirar o Docker do meio e rodar o
treino **dentro do mesmo processo/pod** que efetivamente reservou a GPU, pra
não ter dois sistemas de contabilidade (o Kubernetes de um lado, o daemon
Docker do outro) que não se falam.

---

## 8. Dois workflows, um só runner: evitando colisão de nomes

Quando a versão "Python puro" (capítulo 9) ficou pronta numa branch separada
(`gpu-sandbox-python3`), surgiu uma pergunta prática: dava pra ter as duas
versões (Docker e Python) convivendo no repositório, sem uma apagar a outra
no merge?

A resposta técnica é simples — **arquivos com nomes diferentes não
conflitam** — mas exigiu renomear tudo que as duas branches tinham criado
com o mesmo nome:

| Antes (colisão) | Depois |
|---|---|
| `.github/workflows/pytorch-gpu.yaml` (2 versões diferentes) | [pytorch-gpu-docker.yaml](.github/workflows/pytorch-gpu-docker.yaml) + [pytorch-gpu-python.yaml](.github/workflows/pytorch-gpu-python.yaml) |
| `k8s/gpu-runner-values.yaml` (2 versões diferentes) | [gpu-runner-values.docker.yaml](k8s/gpu-runner-values.docker.yaml) + [gpu-runner-values.yaml](k8s/gpu-runner-values.yaml) (Python ficou com o nome "padrão") |

Como só existe **uma** RTX 3050, só **um** dos dois `values.yaml` pode estar
de fato aplicado no cluster a qualquer momento — os dois workflows existem
no repositório, mas só o que corresponde ao `values.yaml` atualmente
aplicado (via `helm upgrade -f ...`) vai funcionar de verdade se disparado.
O outro fica como referência/documentação, pronto pra reativar trocando o
values de volta.

O capítulo 13 conta a história de como esse merge quase deu errado por causa
de detecção automática de rename do Git.

---

## 9. Tirando o Docker do CI: rodando Python nativo no pod

Seguindo a lógica do capítulo 7, a mudança foi: em vez de o workflow chamar
`docker build`/`docker run`, ele chama `python3 main.py ...` **diretamente**,
como um passo normal do job — sem container nenhum por baixo além do próprio
pod do runner.

Pra isso funcionar, o pod do runner precisa ter Python **e** PyTorch/CUDA já
instalados nele mesmo (não numa imagem Docker que ele constrói depois). A
solução de longo prazo é [k8s/runner-image/Dockerfile](k8s/runner-image/Dockerfile)
— uma imagem customizada que estende `ghcr.io/actions/actions-runner:latest`
(a imagem oficial do runner) instalando Python e as dependências de
[requirements.txt](requirements.txt) **no momento do build da imagem**, não
a cada job:

```dockerfile
FROM ghcr.io/actions/actions-runner:latest
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
USER runner
```

Essa imagem **ainda não foi construída e publicada** (ver capítulo 17) — o
campo `image:` em
[k8s/gpu-runner-values.yaml](k8s/gpu-runner-values.yaml) continua um
placeholder. Até isso acontecer, o workflow tem um "paraquedas": instala tudo
na unha a cada execução (capítulo 10) — funciona, mas é lento, porque baixa
os pacotes de novo toda vez que o cache do pip não bate.

Uma vez que a imagem custom existir de verdade e o `nvidia.com/gpu: 1` do
pod for usado pelo **próprio** processo Python (sem Docker no meio, sem
socket, sem segundo daemon) — a lacuna do capítulo 7 se fecha: o mesmo pod
que reservou a GPU no Kubernetes é literalmente o processo que a está
usando.

---

## 10. A saga de três erros seguidos no `pip install`

Essa sequência é um estudo de caso perfeito de "cada erro corrigido revela o
próximo problema" — vale entender os três, porque cada um ensina algo sobre
como distribuições Linux e o Python moderno lidam com pacotes.

### Erro 1 — `ModuleNotFoundError: No module named 'torch'`

O mais óbvio: a imagem do runner (ainda a padrão, sem o
`k8s/runner-image/Dockerfile` publicado) nunca teve PyTorch instalado.
Correção inicial: adicionar um passo `pip install -r requirements.txt` antes
de rodar `main.py`.

### Erro 2 — `/usr/bin/python3: No module named pip`

A correção do erro 1 assumia que `pip` já existia pra atualizar
(`pip install --upgrade pip`). Só que **Debian e Ubuntu removem
deliberadamente** o `pip`/`ensurepip` do pacote `python3` — é uma escolha de
empacotamento deles, não um bug: eles preferem que você instale pacotes
Python via `apt` (`apt install python3-nome-do-pacote`) sempre que possível,
e cortam o instalador padrão do Python pra empurrar essa convenção. Sem
`apt` disponível com privilégio de root (removemos o `runAsUser: 0` do
values.yaml Python de propósito, já que não precisa mais de socket nenhum),
a saída foi usar o **`get-pip.py`** — um script oficial do projeto pip
(mantido pela PyPA, "Python Packaging Authority") que baixa e instala o pip
sozinho, sem depender do `apt` nem do `ensurepip`:

```bash
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --user
```

`--user` instala num diretório dentro do `$HOME` do usuário atual
(`~/.local/lib/...`), em vez do diretório de pacotes do sistema — não precisa
de root pra escrever ali.

### Erro 3 — `error: externally-managed-environment` (PEP 668)

Mesmo com um pip novinho instalado, o **primeiro** `pip install` de verdade
(seja o do próprio `get-pip.py`, seja o de `requirements.txt` depois) falhou
com essa mensagem. Isso é a **PEP 668** — uma proposta formal do ecossistema
Python (não específica do Debian) que distribuições como o Ubuntu passaram a
adotar a partir do Python 3.11/3.12: o sistema marca seu Python "gerenciado
externamente" (por um arquivo `EXTERNALLY-MANAGED` que o pip procura antes de
instalar qualquer coisa), pra evitar que um `pip install` acidental
sobrescreva um pacote que o `apt` também gerencia e quebre alguma ferramenta
do sistema operacional que dependa daquele pacote Python.

A saída recomendada pelo próprio pip pra quando você **sabe** que está num
ambiente descartável (como um container de CI que morre no fim do job, não
um sistema de verdade que alguém vai usar amanhã) é a flag
`--break-system-packages`:

```bash
python3 /tmp/get-pip.py --user --break-system-packages
python3 -m pip install --user --break-system-packages -r requirements.txt
```

**Detalhe que também mordeu**: na primeira tentativa, só passamos a flag no
segundo comando (o `pip install -r requirements.txt`), esquecendo que o
**próprio** `get-pip.py` já dispara esse mesmo bloqueio internamente. Como o
script roda com `bash -e` (que aborta a execução no primeiro comando que
falhar), o erro acontecia ali, e o terceiro comando nunca chegava a rodar —
por isso só aparecia **um** bloco de erro no log, não dois. A lição: quando
um script `set -e`/`bash -e` falha, o log geralmente só mostra o **primeiro**
comando que quebrou, então vale sempre suspeitar da linha mais **cedo**
possível na sequência, não só da última.

---

## 11. A armadilha do "Re-run jobs"

Depois de corrigir o workflow, uma run continuava falhando com o **mesmo**
código antigo, mesmo já tendo sido commitado (e confirmadamente enviado pro
GitHub — `git fetch` + `git log origin/main` provavam isso).

A causa: o botão **"Re-run jobs"** no GitHub Actions não dispara um workflow
novo — ele **repete exatamente a mesma run**, que já nasceu amarrada a um
commit SHA específico (o que estava em `origin/main` no momento em que ela
foi originalmente criada). Reexecutar essa run sempre vai fazer checkout
daquele mesmo commit antigo, **pra sempre**, não importa quantos commits
novos você empilhe depois.

A forma de rodar contra o código atual é sempre um **novo** disparo — o botão
**"Run workflow"** na página do workflow (não dentro de uma run específica),
que cria uma run nova, amarrada ao `HEAD` atual da branch no momento do
clique.

**Como verificar sem se enganar de novo**: no log de qualquer run, o passo
"Checkout do código" imprime a linha `commit = '<sha>'` — compare esse SHA
com o `git log --oneline -1` local antes de confiar no resultado da run.

---

## 12. Banda larga é infraestrutura: o custo real de baixar PyTorch

Uma vez que o `pip install -r requirements.txt` (capítulo 10) começou a
rodar de verdade, ele ficou "preso" por minutos — não travado, só **lento**:
o pacote `torch` com suporte a CUDA vem acompanhado de várias dependências
`nvidia-*` (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cufft-cu12`,
`nvidia-nccl-cu12` e outras), cada uma um wheel de centenas de MB — o total
facilmente passa de 3–5 GB.

Numa conexão de banda limitada, isso pode levar horas, não minutos — e o
problema se agrava porque adicionamos um gatilho `push:` no workflow (pra
testar cada commit sem precisar clicar manualmente), o que dispara esse
download gigante **a cada commit**, a não ser que o `actions/cache` (que
guarda `~/.cache/pip` entre runs, usando como chave o hash do
`requirements.txt`) já tenha sido populado por uma run anterior bem-sucedida.

Esse é o argumento mais forte a favor de terminar o
[k8s/runner-image/Dockerfile](k8s/runner-image/Dockerfile) do capítulo 9:
pagar esse custo de download **uma vez**, no build da imagem (que você
controla, faz quando quiser, sem pressa) — e nunca mais pagar de novo em CI,
porque a imagem já nasce com tudo instalado.

---

## 13. Git merge de branches: o rename que deu errado

Ao fazer o merge da branch `gpu-sandbox-python3` de volta pra `main`
(capítulo 8), duas mudanças aconteceram em paralelo nos **mesmos arquivos
originais**: a `main` renomeou `k8s/gpu-runner-values.yaml` para
`k8s/gpu-runner-values.docker.yaml`, enquanto a `gpu-sandbox-python3`
continuou **editando o conteúdo** desse mesmo arquivo original (sem
renomear).

O Git tenta ser esperto nessas horas: ele **detecta renames automaticamente**
comparando similaridade de conteúdo entre arquivos deletados de um lado e
criados do outro. O problema é que, com edições grandes nos dois lados ao
mesmo tempo, essa detecção pode escolher a combinação errada — foi
exatamente o que aconteceu: o merge produziu um `gpu-runner-values.docker.yaml`
com o comentário de cabeçalho da versão Docker, mas o **corpo** (a parte que
realmente importa: imagem, volumes, resources) da versão Python — e o
arquivo `gpu-runner-values.yaml` sequer sobrou na árvore de arquivos.

**Como percebemos**: depois de qualquer merge com renomes envolvidos, vale
sempre **ler o conteúdo final**, não só confiar que "o merge terminou sem
conflitos" significa "o resultado está certo" — merge automático sem
conflitos textuais **não é o mesmo** que merge semanticamente correto.

**Como corrigimos**: usamos `git show <commit>:<caminho>` pra recuperar o
conteúdo exato de cada versão em cada branch, direto do histórico (que nunca
mentiu, só a reconciliação automática errou), e reescrevemos os dois
arquivos manualmente com o conteúdo certo em cada um.

```bash
git show 8357f20:k8s/gpu-runner-values.docker.yaml > /tmp/correto-docker.yaml
git show gpu-sandbox-python3:k8s/gpu-runner-values.yaml > /tmp/correto-python.yaml
```

---

## 14. VS Code integrado: debug remoto e tasks automatizadas

O `launch.json` original do projeto tinha dois problemas: estava na raiz do
repositório (o VS Code só reconhece automaticamente
`.vscode/launch.json`, dentro dessa pasta específica) e não tinha nenhuma
configuração de **attach** — só uma config de C++ irrelevante e uma de
"debugar arquivo atual" que não usa Docker.

### 14.1 `launch.json`: `launch` vs. `attach`

Uma configuração de debug do tipo `request: "launch"` **inicia** um processo
novo diretamente pelo VS Code. Uma do tipo `request: "attach"` **conecta**
num processo que já está rodando em outro lugar, esperando conexão — que é
exatamente o que `compose.debug.yaml` faz ao rodar `debugpy --wait-for-client
--listen 0.0.0.0:5678`: o processo Python já sobe e trava esperando alguém
se conectar na porta 5678.

```json
{
  "name": "Python: Anexar ao container (Docker debug)",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "localhost", "port": 5678 },
  "pathMappings": [
    { "localRoot": "${workspaceFolder}", "remoteRoot": "/app" }
  ]
}
```

`pathMappings` é o que permite colocar um breakpoint no `main.py` **local**
(no seu editor) e o VS Code saber que isso corresponde ao `/app/main.py`
**dentro do container** — sem isso, o debugger conecta, mas não sabe
correlacionar os arquivos, e breakpoints simplesmente não param em lugar
nenhum.

### 14.2 `preLaunchTask` e o problema de "esperar a hora certa"

`"preLaunchTask": "Docker Compose: subir debug"` faz o VS Code rodar uma
`task` automaticamente antes de tentar o attach. O problema: `docker compose
up --build` demora (build da imagem, depois `pip install debugpy`, só então
o processo começa a escutar) — se o VS Code tentar conectar cedo demais, a
porta ainda nem está aberta, e o attach falha.

A solução foi um **marcador de texto**: adicionamos um `echo` no comando do
[compose.debug.yaml](compose.debug.yaml), bem entre a instalação do debugpy
e o início dele:

```yaml
command: ["sh", "-c", "pip install debugpy -t /tmp && echo '[debugpy] pronto para anexar' && python /tmp/debugpy --wait-for-client --listen 0.0.0.0:5678 main.py"]
```

E em [.vscode/tasks.json](.vscode/tasks.json), um **background problem
matcher** — um recurso do VS Code que permite marcar uma task de longa
duração como "iniciada" (`beginsPattern`) e depois "pronta" (`endsPattern`),
usando expressões regulares que procuram por linhas específicas no output:

```json
"background": {
  "activeOnStart": true,
  "beginsPattern": ".",
  "endsPattern": "\\[debugpy\\] pronto para anexar"
}
```

Só quando essa linha aparece no log é que o VS Code considera a task
"terminada" e libera o `preLaunchTask`, deixando o attach prosseguir — F5
sozinho, sem precisar ficar olhando o terminal esperando a hora certa.

### 14.3 `tasks.json`: segredos sem aparecer no terminal

As tasks que chamam [2-setup-arc.sh](scripts/2-setup-arc.sh) precisam de um
`GITHUB_TOKEN`. Em vez de pedir pra você colar ele direto no comando (o que
apareceria ecoado no painel do terminal, ficando visível no histórico),
usamos um **input** do VS Code:

```json
"inputs": [
  {
    "id": "githubToken",
    "type": "promptString",
    "description": "GITHUB_TOKEN fine-grained...",
    "password": true
  }
]
```

E injetamos ele via `options.env` da task (não dentro da string de
`command`):

```json
{
  "command": "bash \"${workspaceFolder}/scripts/2-setup-arc.sh\" Lorkyrr/pytorch-gpu-sandbox",
  "options": { "env": { "GITHUB_TOKEN": "${input:githubToken}" } }
}
```

`options.env` define variáveis de ambiente pro processo da task, sem elas
aparecerem na linha de comando impressa no terminal — só o `command` em si é
ecoado, e ele não menciona o token em lugar nenhum.

Por fim, `dependsOn` + `dependsOrder: "sequence"` encadeiam duas tasks (criar
cluster → configurar ARC) rodando uma depois da outra, com o VS Code
perguntando o token **uma vez só**, reaproveitado nas duas.

---

## 15. A pasta-mãe: cópia total pra rodar em outra máquina

Pra testar diferença de hardware sem depender de internet, criamos uma cópia
completa do projeto em `/home/mateus/Coding/pytorch-gpu-sandbox-debug` — não
um `git clone` (que baixaria o histórico de novo da rede e ignoraria tudo
que está no `.gitignore`), mas um `cp -a` **bruto**, copiando literalmente
tudo, byte a byte, inclusive:

- `.git/` — o histórico inteiro já está no disco local, não precisa buscar
  de novo de lugar nenhum. O resultado é um **segundo working copy do mesmo
  repositório** (mesmo remoto, mesma branch) — não um clone novo. `git
  commit`/`push` funcionam lá exatamente como aqui, com o cuidado de não
  divergir histórico entre as duas pastas sem reconciliar depois.
- `data/` — o CIFAR-10 já baixado (~341 MB), evitando repetir aquele
  download que a documentação original já registrou como levando ~19
  minutos na primeira vez.
- `.venv/`, `__pycache__/` — tudo, mesmo sabendo que parte disso (o `.venv`,
  nesse caso) podia estar quebrado — e estava: o `python3` de dentro dele
  tinha um shebang (a linha `#!/caminho/...` no topo do arquivo executável,
  que diz ao sistema operacional qual interpretador usar) apontando pra um
  ambiente virtual de **outro** projeto seu, que não existe mais.

**Descoberta importante nesse processo**: as imagens Docker já construídas
(`pytorch-cifar10:*`, `pytorch-app-debug`, ~11 GB cada) **não pertencem à
pasta do projeto** — elas ficam guardadas no armazenamento do **daemon
Docker**, que é global pra máquina inteira. Isso significa: rodar `docker
compose up --build` dentro da pasta-cópia reaproveita esse cache
automaticamente (mesmo daemon, mesma máquina), sem baixar nada — mas se você
um dia copiar essa pasta pra um computador **fisicamente diferente**, esse
cache **não viaja junto**; só o que está dentro da pasta em si (código,
`data/`, `.git`) acompanha a cópia.

---

## 16. Glossário rápido

- **Runner (self-hosted)**: máquina sua que executa jobs do GitHub Actions,
  em vez de uma máquina fornecida pela GitHub.
- **kind**: ferramenta que cria clusters Kubernetes usando containers Docker
  como "nodes", útil pra desenvolvimento/teste local sem precisar de VMs ou
  nuvem.
- **containerd**: o software que efetivamente cria e gerencia containers por
  baixo do Kubernetes — geralmente invisível, mas é ele (não o Docker) quem
  o `kubelet` chama pra subir cada pod.
- **NVIDIA Container Toolkit**: conjunto de ferramentas que ensina
  runtimes de container (Docker, containerd) a expor a GPU física do host
  dentro de containers.
- **Device plugin (Kubernetes)**: um pod especial que anuncia ao Kubernetes
  a existência de um recurso de hardware (GPU, nesse caso) num node
  específico, permitindo que outros pods "peçam" esse recurso via
  `resources.limits`.
- **ARC (Actions Runner Controller)**: sistema que roda dentro de um cluster
  Kubernetes e cria/destrói pods runner do GitHub Actions sob demanda.
- **DooD (Docker-fora-do-Docker)**: um container acessa o Docker de outra
  máquina/camada através de um socket montado, em vez de ter seu próprio
  daemon Docker interno (que seria DinD, Docker-dentro-do-Docker).
- **Bind mount**: mecanismo do Docker que expõe um diretório do **host**
  (do ponto de vista de quem executa o comando `docker run`) dentro de um
  container. Só funciona se quem roda o comando e o daemon Docker
  compartilharem o mesmo sistema de arquivos.
- **PEP 668**: proposta do ecossistema Python que faz distribuições Linux
  bloquearem `pip install` fora de um ambiente virtual, evitando conflito
  com pacotes gerenciados pelo `apt`.
- **Background problem matcher (VS Code)**: mecanismo que permite marcar uma
  task de longa duração como "pronta" observando padrões de texto no output,
  em vez de assumir que ela terminou assim que o processo começou.

---

## 17. O que ainda falta (TODOs reais)

Isto não é retórica — são pendências reais deste repositório, no estado em
que ele está agora:

1. **Construir e publicar `k8s/runner-image/Dockerfile`** — enquanto isso
   não acontece, `pytorch-gpu-python.yaml` continua pagando o custo lento do
   capítulo 10/12 a cada run. Depois de publicada, trocar o `image:`
   placeholder em [k8s/gpu-runner-values.yaml](k8s/gpu-runner-values.yaml)
   e reaplicar com `helm upgrade`.
2. ~~**Decidir se o gatilho `push:` continua** em
   [.github/workflows/pytorch-gpu-python.yaml](.github/workflows/pytorch-gpu-python.yaml)~~
   — **resolvido**: o gatilho foi removido (commit `0abdb15`), o workflow
   só dispara por `workflow_dispatch` agora. Motivo registrado no próprio
   arquivo e no capítulo 12: numa conexão lenta, baixar o `torch` com CUDA a
   cada `push` era caro demais sem o cache do pip garantido.
3. **`.venv` local quebrado** — só importa se algum dia você quiser rodar
   sem Docker nesta pasta principal; hoje ninguém usa esse `.venv` (tudo
   passa por Docker ou pelo pod de CI). Continua quebrado (o `pyvenv.cfg`
   ainda aponta pra outro projeto) — mas como `.venv/` está no
   `.gitignore`, isso não viaja para quem clona o repositório, só afeta
   esta máquina.
4. ~~**Nenhum push foi feito** de vários commits recentes~~ — **resolvido**:
   `git log origin/main..HEAD` está vazio, a branch local está em dia com
   `origin/main`. Ainda vale reconferir isso (`git fetch && git status`)
   antes de confiar em qualquer run de CI, já que a combinação "só eu
   commito, você dá o push" pode voltar a divergir a qualquer momento.

> **Atualização:** o item que faltava — os 4 scripts (`1-create-gpu-cluster.sh`,
> `2-setup-arc.sh`, `3-teardown-cluster.sh`,
> `kubernetes-gpu-arc-referencia.sh`) viviam numa pasta **fora** deste
> repositório git, então um `git clone` sozinho não trazia o que as tasks do
> VS Code chamavam — foi resolvido: os 4 agora estão versionados em
> [scripts/](scripts/), e `2-setup-arc.sh` passou a instalar o device plugin
> e o ARC (controller + os dois runner sets) em duas trilhas paralelas em vez
> de uma cadeia serial, pra rodar mais rápido. As versões exatas de
> Docker/kind/kubectl/Helm/driver NVIDIA/CTK usadas nesta máquina estão no
> [README](README.md#host-environment-this-was-built-and-tested-on).

---

## 18. Comandos de referência rápida

```bash
# Rodar local (Docker) — benchmark
docker compose up --build

# Rodar local (Docker) — treino
docker compose run --rm app python3 main.py train --epochs 30

# Debug remoto local
docker compose -f compose.debug.yaml up --build   # ou F5 no VS Code

# Ver se o node do kind anuncia a GPU
kubectl describe node kind-control-plane | grep -A3 "Allocatable:"

# Ver pods do ARC
kubectl get pods -n arc-systems

# Trocar qual workflow está "ativo" (Docker vs Python)
helm upgrade --install arc-runner-set-gpu \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  -n arc-systems -f k8s/gpu-runner-values.yaml          # python
  # -f k8s/gpu-runner-values.docker.yaml                # ou docker

# Recriar o cluster do zero
./scripts/1-create-gpu-cluster.sh
export GITHUB_TOKEN='...'
./scripts/2-setup-arc.sh Lorkyrr/pytorch-gpu-sandbox        # variante docker (padrão)
# ./scripts/2-setup-arc.sh Lorkyrr/pytorch-gpu-sandbox python   # ou variante python, se já publicou a imagem custom

# Derrubar o cluster (libera GPU/RAM/CPU)
./scripts/3-teardown-cluster.sh

# Conferir se uma run de CI rodou no commit certo
git log --oneline -1
# comparar com a linha "commit = '<sha>'" no passo "Checkout do código" do log da run
```

---

# Parte II: pytorch-arc-hosted

## 19. Recomeço: o que cinco projetos ensinaram

Em 24/09/2026 o projeto foi reescrito do zero, num repositório novo
(`pytorch-arc-hosted`), usando o Claude Opus 5.5 com um fluxo de trabalho em
etapas: **brainstorming → spec → plano → execução com TDD**. Os repositórios da
Parte I tinham sido feitos com o Sonnet, sem esse processo. O primeiro passo não
foi escrever código: foi ler os cinco projetos anteriores e perguntar **qual era
a dor de verdade**.

A resposta não estava no código Python (ResNet, AMP e benchmarks já estavam
bons). Estava em **onde moram os 3–5 GB de dependências CUDA**. Quase tudo que
deu errado na Parte I é sintoma disso:

- Duas variantes de workflow (python e docker) disputando o mesmo runner set
  (capítulo 8).
- `get-pip.py`, `--break-system-packages` e PEP 668 (capítulo 10).
- O gatilho `push` desligado porque baixar o torch a cada commit era inviável
  (capítulo 12).
- A imagem customizada do runner que nunca chegou a ser publicada (capítulo
  17, item 1).

**Lição:** antes de refatorar, pergunte o que custa caro de verdade. Aqui era
banda, não organização de código.

**Como verificar:** compare o capítulo 17 (TODOs da Parte I) com o
[CLAUDE.md](CLAUDE.md), seção "Cache strategy". Cada TODO antigo virou uma
linha daquela tabela.

---

## 20. Só Docker, e o socket do host como cache

Decisão do autor: **existe só a versão Docker.** Manter "python nativo" e
"docker" lado a lado não fazia sentido. Sobrava escolher **qual** Docker:

| Opção | Onde fica o cache das camadas | Custo por job |
|---|---|---|
| Docker-in-Docker (dind, o modo "oficial" do ARC) | Dentro do pod, e morre com ele | ~3,3 GB por job |
| Imagem de runner "gorda" com torch | No containerd **interno** do node kind, que morre no `kind delete cluster` | GBs a cada recriação do cluster |
| **Socket do Docker do host (DooD)** | No **daemon do host** | Só a camada do código (KBs) |

Com o socket do host montado no pod, `docker pull` e `docker run` falam com o
mesmo daemon que você usa no terminal. O cache sobrevive a tudo: jobs, pods,
recriação do cluster e reboot. É o mesmo fato que o capítulo 15 descobriu por
acidente ("as imagens Docker não pertencem à pasta do projeto, ficam no
daemon"), agora usado **de propósito**.

O dataset segue a mesma lógica: um volume Docker nomeado, `pah-cifar10`, no
host. O `actions/cache` num runner self-hosted **baixa do GitHub** a cada job,
e subir e baixar 170 MB pela sua rede é pior do que não cachear.

**Como verificar** (depois do primeiro job):
```bash
docker volume inspect pah-cifar10
docker images ghcr.io/lorkyrr/pytorch-arc-hosted
```
No segundo job, o log do passo "Baixar a imagem" deve mostrar `Already exists`
em quase todas as camadas.

---

## 21. Fechando a GPU "parcialmente virtual"

O capítulo 7 deixou uma lacuna honesta. Na variante docker, o pod pedia
`nvidia.com/gpu: 1`, mas o `docker run --gpus all` falava direto com o daemon
do host, **fora** da contabilidade do Kubernetes. A reserva do pod virava só um
"cadeado", sem ligação real com quem usava a placa.

O fechamento é simples, depois de descoberto. Quando o device plugin da NVIDIA
entrega uma GPU a um pod, ele injeta no container a variável
`NVIDIA_VISIBLE_DEVICES=GPU-<UUID>`: o UUID **daquela** GPU. O `gpu.yaml` repassa
esse valor:

```bash
docker run --gpus "device=$NVIDIA_VISIBLE_DEVICES" ...
```

Assim, o container irmão usa **exatamente** a GPU que o Kubernetes reservou para
o pod. Numa máquina com várias GPUs, isso impediria dois jobs de treinarem na
mesma placa. Com uma GPU só, deixa a contabilidade verdadeira.

A regra de segurança: se a variável vier vazia, `all`, `none` ou `void`, o job
**falha** logo no primeiro passo. Ele nunca volta silenciosamente para
`--gpus all`, porque isso reabriria a lacuna sem ninguém perceber.

**Como verificar:** o primeiro passo do `gpu.yaml` imprime
`NVIDIA_VISIBLE_DEVICES=...`. Compare com `nvidia-smi -L` no host: o UUID tem
que ser o mesmo. Este é o **ponto de verificação 1** do CLAUDE.md, e ainda não
foi confirmado numa execução real.

---

## 22. A imagem no GHCR, ativada por `workflow_run`

O autor pediu para ver a imagem **hospedada online** e **ativada por workflow**.
Ficou assim:

1. [image.yaml](.github/workflows/image.yaml) roda num runner **do GitHub**
   (rede rápida, de graça): builda e publica
   `ghcr.io/lorkyrr/pytorch-arc-hosted:<sha>` e `:latest`.
2. [gpu.yaml](.github/workflows/gpu.yaml) tem o gatilho
   `workflow_run: workflows: ["Imagem (GHCR)"]`. Quando o build termina com
   sucesso, ele acorda sozinho no seu runner com GPU.

Por que o `gpu.yaml` usa `github.event.workflow_run.head_sha`, e não `:latest`?
Porque entre o fim do build e o início do job outro push pode ter publicado
outra `:latest`. O SHA garante que a GPU roda **exatamente** a imagem que
acabou de ser construída. É o mesmo cuidado do capítulo 11 ("Re-run jobs"
preso ao SHA antigo), só que a favor.

Duas curiosidades:
- O nome da imagem no GHCR **precisa ser minúsculo** (`lorkyrr`, não
  `Lorkyrr`), e isso deu vários commits de correção no `pytorch-gpu-refactor`.
- O build usa **cache inline**: os metadados de cache viajam dentro da própria
  `:latest`. Se a base não mudou, o próximo build reaproveita as camadas sem
  baixar os ~3 GB de novo, e sem criar nenhum artefato extra no registry.

**Como verificar:** na aba Actions, uma run de "Imagem (GHCR)" deve ser seguida
de uma run de "GPU (RTX 3050)". O Summary do build mostra a base escolhida.

---

## 23. A tag `latest` que parou em 2024

O autor pediu **sempre a versão mais nova de tudo**, mesmo que isso custe
baixar de novo. A reação óbvia seria "então use `pytorch/pytorch:latest`". Uma
consulta à API do Docker Hub mostrou o contrário:

```
latest  2024-02-23  3.66 GB
```

A tag `latest` do PyTorch **parou de ser atualizada em fevereiro de 2024**. É o
PyTorch 2.2.1 que aparece nos logs de `testes-de-ambiente/` da Parte I. Os
projetos anteriores rodaram com um PyTorch de dois anos atrás **sem saber**.

Hoje o PyTorch só publica tags explícitas, como
`2.14.0-cuda13.2-cudnn9-runtime`. Por isso "sempre a mais nova" virou um
programa, [scripts/latest_pytorch_base.py](scripts/latest_pytorch_base.py):

1. Lê as tags do Docker Hub.
2. Fica só com as `<torch>-cuda<X.Y>-cudnn<N>-runtime` (descarta `devel` e
   `latest`).
3. Descarta as que pedem CUDA acima do `MAX_CUDA`, o máximo que o **seu driver**
   suporta (615 → 13.4). Sem esse limite, "mais nova" viraria "quebrada": o
   `torch.cuda.is_available()` daria `False`.
4. Escolhe a maior versão do torch e, no empate, o maior CUDA, comparando
   **números** (`2.10.0 > 2.9.1`), não texto.

Na primeira execução real, a rede instável daquele dia (chuva) derrubou a
consulta com `TimeoutError`. Isso virou um teste (primeiro falhando, depois
passando) e uma retentativa com espera crescente. O que quebra na sua rede ruim
também pode quebrar no GitHub num dia ruim do Docker Hub.

**Como verificar:**
```bash
python3 scripts/latest_pytorch_base.py --max-cuda 13.4
nvidia-smi   # "CUDA Version" no canto superior direito
```

---

## 24. Repo público + runner self-hosted: o risco do socket

Montar `/var/run/docker.sock` num container equivale a dar **root no host**
para quem controla esse container: basta um
`docker run -v /:/host ...` e ele alcança o seu disco inteiro. Na Parte I isso
também acontecia (variante docker), mas o assunto não foi discutido.

Num repositório **público**, qualquer pessoa pode abrir um pull request. Se um
workflow que roda no seu runner self-hosted disparasse em `pull_request`, o
código de um desconhecido rodaria na sua máquina, com root. As defesas:

- O `gpu.yaml` **não tem** gatilho `push` nem `pull_request`, só
  `workflow_dispatch` (exige permissão de escrita) e `workflow_run`.
- O `workflow_run` sempre executa a versão do workflow que está na branch
  padrão, e o job ainda confere `head_branch == 'main'`.
- **Descoberto na revisão final (28/09):** o `workflow_run` casa os workflows
  pelo **nome**. Um PR de fork que criasse um workflow chamado
  `Imagem (GHCR)` também dispararia o `gpu.yaml`. Por isso a guarda agora
  confere também o repositório de origem, o **arquivo** do workflow e o evento
  (`push`, `schedule` ou `workflow_dispatch`, nunca PR).
- No runner com root só rodam actions `actions/*` **fixadas por SHA de commit**
  (uma tag como `@v7` pode ser movida por quem controla o repositório da
  action). O login no GHCR virou `docker login` puro.
- [tests/test_workflows.py](tests/test_workflows.py) transforma tudo isso em
  teste: se alguém recolocar um `pull_request` ou uma tag mutável, o CI quebra.
- O runner set é registrado **só neste repositório**.
- Recomendação no README: exigir aprovação para workflows de forks.

**Como verificar:**
```bash
grep -n "pull_request\|push:" .github/workflows/gpu.yaml
```
Não deve aparecer nada.

---

## 25. Sem root: `supplementalGroups`

Na Parte I, o runner só conseguia usar o socket rodando como root
(`runAsUser: 0` + `RUNNER_ALLOW_RUNASROOT=1`, capítulo 6 e a tabela de gotchas).
Funciona, mas é a solução mais larga possível.

O motivo real do `permission denied` era um **número**. O socket do host
pertence a um grupo `docker` com um GID (nesta máquina, **1002**). A imagem do
runner cria o *seu próprio* grupo `docker`, com GID **123**. Os nomes são iguais,
mas o kernel só compara números.

A correção é dar ao processo do runner o GID do host como grupo extra:

```yaml
securityContext:
  supplementalGroups: [1002]
```

Como esse número muda de máquina para máquina, ele não fica escrito no repo: o
[scripts/cluster.sh](scripts/cluster.sh) lê o GID com
`stat -c %g /var/run/docker.sock` e passa por `helm --set`.

Um detalhe do Helm que custaria caro: o `--set` **substitui listas inteiras**.
Um `--set template.spec.containers[0].image=...` apagaria o `command` e o
`resources` do container, e com eles o pedido de GPU. Por isso só
`supplementalGroups[0]` e `githubConfigUrl` passam por `--set`.

**Como verificar:**
```bash
stat -c '%g %G' /var/run/docker.sock
```

---

## 26. Dois clusters, uma GPU

Ao preparar o ambiente novo, apareceu um detalhe: o cluster `kind` da Parte I
**ainda existia**, com o seu próprio device plugin anunciando a mesma RTX 3050.
Dois clusters são dois schedulers que não se conhecem, e cada um acha que tem
uma GPU livre.

O `cluster.sh up` agora se recusa a continuar se houver outro cluster kind
rodando e diz como derrubá-lo. A variável `ALLOW_OTHER_CLUSTERS=1` existe para
quem tiver certeza de que o outro cluster não usa a GPU.

**Como verificar:**
```bash
kind get clusters
```

---

## 27. O que ainda falta (Parte II)

Estado em 24/09/2026. A implementação foi feita **sem baixar nada pesado** na
máquina do autor, por causa da internet ruim naquele dia:

1. **Subir o cluster novo** (`kind delete cluster --name kind` e depois
   `scripts/cluster.sh up`) quando a internet estiver boa.
2. **Primeira execução real na GPU**, confirmando o capítulo 21
   (`NVIDIA_VISIBLE_DEVICES` com o UUID).
3. **Primeiro treino com AMP** e comparação com os 89,40% / 90,81% da Parte I,
   que eram FP32 com PyTorch 2.2.1. A partir daqui, o PyTorch é o 2.14.
4. Registrar os resultados aqui e no [CLAUDE.md](CLAUDE.md).

---

## 28. Comandos de referência rápida (Parte II)

```bash
# Ambiente
scripts/cluster.sh up           # cria/atualiza kind + GPU + ARC (idempotente)
scripts/cluster.sh status       # GPU alocável, versões, pods do ARC
scripts/cluster.sh down         # apaga só o cluster; cache e dataset ficam

# Rodar na GPU via GitHub Actions
gh workflow run gpu.yaml                                          # benchmark
gh workflow run gpu.yaml -f mode=train -f arch=resnet56 -f epochs=50 -f seed=42
gh run watch

# Rebuildar a imagem agora (base mais nova)
gh workflow run image.yaml

# Local, sem CI
docker compose run --rm app
docker compose run --rm app train --epochs 30

# Qual base seria usada hoje
python3 scripts/latest_pytorch_base.py --max-cuda 13.4
```

---

## 29. A revisão final: o `concurrency` que apagava treinos

Depois das 10 tarefas, um revisor independente (um agente com contexto limpo,
que não escreveu o código) leu o branch inteiro. Ele não achou nada crítico, mas
achou um problema que nenhum teste pegaria.

O `gpu.yaml` tinha `concurrency: {group: gpu, cancel-in-progress: false}`,
herdado da Parte I, com o comentário "jobs esperam na fila, nunca se cancelam".
**Isso é só metade da verdade.** O `cancel-in-progress: false` protege o job que
**está rodando**. Para os que estão **esperando**, o GitHub guarda **no máximo 1
pendente** por grupo: quando chega um terceiro, o pendente anterior é cancelado.

Cenário real: um benchmark está rodando, você dispara um treino de 50 épocas
(fica pendente), e aí chega o rebuild de segunda-feira com o seu benchmark
automático. **Seu treino some**, com status "cancelled".

A correção foi **remover** o bloco. A serialização já existia num lugar melhor:
o runner set tem `maxRunners: 1` e cada pod pede `nvidia.com/gpu: 1`. A fila do
ARC segura todos os jobs, sem descartar nenhum.

Na mesma passada, mais duas correções:
- **Checkpoint no timeout.** O upload do artifact agora usa `always()`. Um
  ResNet-110 que estoure os 120 minutos ainda entrega o melhor checkpoint até
  ali.
- **Container órfão.** Se o pod morrer no meio (`cluster.sh down`, reboot), o
  container de treino continua no host segurando a GPU. O próximo job agora
  remove os containers de job órfãos antes de começar. (Ver capítulo 30: a
  primeira versão desta limpeza filtrava por **nome** e apagou o próprio cluster.)

Cada correção entrou com um teste que falhou antes, em
[tests/test_workflows.py](tests/test_workflows.py). Os achados menores ficaram
listados no [CLAUDE.md](CLAUDE.md), seção "Known minor issues".

**Lição:** comentário não é garantia. A frase "nunca se cancelam" estava errada
desde a Parte I, e ninguém tinha lido a documentação do `concurrency` até o fim.

---

## 30. A primeira execução real: quatro quedas até o primeiro "success"

Em 29/09, com internet boa, o projeto rodou de verdade pela primeira vez. Deu
certo na quinta tentativa, e cada falha ensinou algo que nenhum teste local
pegaria.

**1. A GPU que demorou a aparecer.** O `cluster.sh up` esperava 2 minutos pelo
`nvidia.com/gpu` no node e desistiu. O log do device plugin mostrou que ele se
registrou no kubelet **segundos depois** do prazo. Nada estava errado, só
lento: na primeira subida, a NVML leva uns 2 minutos para inicializar dentro do
kind. A espera virou 5 minutos.

**2. O prazo do kubelet.** O pod do runner ficou em `ErrImagePull`
(`DeadlineExceeded`). O kubelet dá uns 2 minutos para cada tentativa de baixar
uma imagem, e a do runner (1,56 GB) não chegava, disputando a banda com o pull
de 3,3 GB da imagem do projeto. Pior: o job que aquele runner tinha
"reservado" **não volta para a fila** quando o runner morre. O run fica
"queued" para sempre e precisa ser cancelado e disparado de novo. A solução
permanente: o `cluster.sh` baixa a imagem pelo Docker do host (sem prazo) e a
entrega ao node com `kind load docker-image`.

**3. O `void`.** O primeiro passo do `gpu.yaml` barrou o job:
`NVIDIA_VISIBLE_DEVICES=void`. O capítulo 21 apostava que a variável traria o
UUID da GPU. Um pod de teste mostrou a verdade: a variável vinha `void`, **mas
o `nvidia-smi` lá dentro enxergava a GPU**. O NVIDIA Container Toolkit 1.20
entrega a placa por **CDI** e apaga a variável de propósito, para ninguém
"reinjetar" GPUs a partir dela. O UUID passou a vir do `nvidia-smi` do pod, que
só enxerga a GPU reservada. A regra de ouro ficou igual: exatamente um UUID,
nunca `all`.

**4. O job que apagou o próprio cluster.** Esta foi grave, e o erro foi meu (do
Claude). No capítulo 29, a limpeza de containers órfãos filtrava por nome:
`docker ps --filter "name=^pah-"`. O node do kind se chama
**`pah-control-plane`**. O job rodou `docker rm -f` no próprio node em que
estava rodando. O runner morreu no meio do passo, sem avisar o GitHub, que
continuou mostrando "baixando a imagem" como se nada tivesse acontecido. A
pista veio do `docker events`: `kill → die → destroy` no `pah-control-plane`
às 07:34:10. O volume e o contexto do kubectl continuavam lá, então tinha sido
`docker rm -f`, não `kind delete`.

A correção: o `docker run` marca o container com
`--label pah-arc-hosted.job`, e a limpeza filtra **só esse label**. O teste
novo não verifica mais "a limpeza existe". Ele verifica **o que ela pode
atingir**, que é a propriedade que importa.

**Resultado** (run 36557180170):
- PyTorch 2.14.0 com CUDA 13.2 na RTX 3050.
- Até 3.383 GFLOPS no cuBLAS.
- 340 imagens/s no cuDNN.
- FP16 2,05× mais rápido que FP32.
- Pico de 841 MB de VRAM.

**Lições:**
- Prefixo de nome não é identidade: use labels.
- "O job está travado" pode significar "o job morreu e ninguém avisou". Olhe
  a infraestrutura (`docker events`), não só a tela do GitHub.
- Uma premissa de design (o UUID na variável) só vira fato depois de testada
  no ambiente real. Deixar o plano B anotado no CLAUDE.md economizou horas.
