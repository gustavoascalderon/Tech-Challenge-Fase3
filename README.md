# Triagem Automática de Laudos Médicos — Tech Challenge Fase 3 (FIAP PosTech MLET)

Classificador de texto (NLP) que recebe o texto de um laudo e devolve a urgência
(`normal` / `atencao` / `urgente`), servido por **API REST (FastAPI) em Docker**, com
**CI/CD (GitHub Actions)**, **retreino orquestrado (Airflow)**, **monitoramento
(Prometheus + Grafana)** e **otimização de latência (ONNX Runtime)**.

> ⚠️ Projeto acadêmico. Não deve ser usado para decisões clínicas reais (veja
> [Limitações](#limitações-e-honestidade-dos-resultados)).

## Sumário
1. [Arquitetura](#arquitetura)
2. [Decisão arquitetural de nuvem](#decisão-arquitetural-de-nuvem-etapa-1)
3. [Dataset e mapeamento de urgência](#dataset-e-mapeamento-de-urgência)
4. [Como executar (passo a passo)](#como-executar-passo-a-passo)
5. [Resultados](#resultados)
6. [CI/CD](#cicd-etapa-2)
7. [Airflow](#airflow-etapa-2)
8. [Monitoramento](#monitoramento-etapa-3)
9. [Otimização de latência](#otimização-de-latência-etapa-4)
10. [Estrutura do repositório](#estrutura-do-repositório)
11. [Limitações](#limitações-e-honestidade-dos-resultados)
12. [Vídeo STAR](#vídeo-star)

## Arquitetura

```
 laudo (texto) ──► FastAPI (/classificar) ──► ONNX Runtime (TF-IDF + Reg. Logística) ──► urgência
                      │  /metrics (prometheus_client)
                      ▼
                 Prometheus ──► Grafana (dashboard com 6 painéis)

 GitHub push ──► GitHub Actions: lint (ruff) → testes (pytest) → build Docker

 Airflow (semanal): carregar_dados ──► treinar_modelo ──► exportar_onnx ──► models/
```

| Camada | Tecnologia |
|---|---|
| Modelo | scikit-learn: `TfidfVectorizer` + `LogisticRegression` |
| Otimização | `skl2onnx` + `onnxruntime` |
| API | FastAPI + Uvicorn |
| Métricas | `prometheus_client` |
| Orquestração | Apache Airflow (DAG) |
| Observabilidade | Prometheus + Grafana (Docker Compose) |
| CI/CD | GitHub Actions |

## Decisão arquitetural de nuvem (Etapa 1)

**Pergunta:** batch ou tempo real? **Resposta: tempo real para a triagem + batch para o retreino.**

| Critério | Batch | Tempo real (escolhido p/ inferência) |
|---|---|---|
| Objetivo clínico | Classificar laudos acumulados | Priorizar o laudo **assim que chega** |
| Latência aceitável | Minutos/horas | Dezenas de ms |
| Risco de atraso | Laudo urgente espera a próxima janela | Mínimo |
| Custo | Menor (computa só quando roda) | Maior, mas o modelo é leve (~0,1 ms de inferência) |

**Proposta de deploy (agnóstica de provedor; exemplo AWS):**
- **Inferência (tempo real):** a imagem Docker desta API roda em um serviço de contêineres
  gerenciado (**AWS ECS Fargate** atrás de um Application Load Balancer; equivalentes: Cloud Run no
  GCP, Azure Container Apps), com *autoscaling* por CPU/requisições. Como o modelo é pequeno
  (~0,5 MB) e só usa CPU, não há necessidade de GPU.
- **Retreino (batch):** DAG do Airflow (ex.: **MWAA**) executada semanalmente; o novo `modelo.onnx`
  é publicado em um bucket (S3) e a imagem é reconstruída/publicada (ECR) pelo CI/CD.
- **Observabilidade:** Prometheus/Grafana gerenciados (Amazon Managed Prometheus/Grafana) ou a
  mesma stack deste repositório.
- **Dados sensíveis (LGPD):** laudos são dados de saúde → região `sa-east-1`, criptografia em
  trânsito e em repouso, acesso por IAM mínimo e **sem registrar o texto dos laudos em logs**
  (a API só registra métricas agregadas).

## Dataset e mapeamento de urgência

Foi usado o **Medical Abstracts TC Corpus** (resumos médicos, ~11 mil amostras de treino;
[repositório oficial](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)). Ele rotula a
**condição** do paciente (5 classes), **não** a urgência. Como o enunciado aceita qualquer dataset
texto + alvo com ≥ 2.000 amostras, adotamos um **mapeamento proxy** (decisão do projeto, **não**
critério clínico validado):

| Condição original | Urgência (proxy) |
|---|---|
| Neoplasias (1) e Doenças cardiovasculares (4) | `urgente` |
| Doenças do sistema digestivo (2) e do sistema nervoso (3) | `atencao` |
| Condições patológicas gerais (5) | `normal` |

**Auditoria dos dados** (`python scripts/auditar_dados.py` → `results/auditoria_dados.md`). Ela encontrou
problemas no dataset original que foram corrigidos antes do treino:

| Problema encontrado | Quantidade | Tratamento |
|---|---|---|
| Textos **idênticos em treino e teste** (vazamento) | 988 | removidos do teste |
| Textos duplicados no treino | 2.105 | mantido apenas um |
| Textos com **urgência conflitante** no treino (mesmo resumo com rótulos diferentes) | 1.880 | removidos (rótulo ambíguo) |

Após a limpeza: **7.565** laudos de treino e **1.788** de teste (≥ 2.000 exigidos no treino).
Os CSVs originais ficam em `data/raw/` (veja `data/README.md`).

## Como executar (passo a passo)

Pré-requisitos: **Python 3.12**, **Docker Desktop** e Git. Comandos para **Windows (PowerShell)**;
em Linux/Mac troque a ativação do ambiente por `source .venv/bin/activate`.

### 1) Ambiente Python e testes
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

### 2) Auditar os dados, treinar, validar e medir a latência
```powershell
python scripts/auditar_dados.py
python scripts/treinar.py
python scripts/validar_modelo.py
python scripts/auditar_artefatos.py
python scripts/avaliar_modelo_original.py
python scripts/benchmark_latencia.py
python scripts/validar_revisao.py
```
Gera `models/modelo.joblib`, `models/modelo.onnx` e os relatórios em `results/`:

| Script | Relatório em `results/` | O que comprova |
|---|---|---|
| `auditar_dados.py` | `auditoria_dados.json` / `.md` | vazamento, duplicatas e rótulos conflitantes nos dados |
| `validar_modelo.py` | `validacao_modelo.json` / `.md` | métricas, matriz de confusão, ONNX x sklearn, contrato da API |
| `auditar_artefatos.py` | `auditoria_artefatos_recebidos.json` | inventário (tamanho e SHA-256) dos modelos e dados, versões fixadas e coerência das métricas |
| `avaliar_modelo_original.py` | `metricas_modelo_original.json` | referência "antes": modelo treinado sem a limpeza dos dados |
| `benchmark_latencia.py` | `latencia_modelo.json` / `.md` | latência sklearn x ONNX |
| `validar_revisao.py` | `validacao_revisao.json` | consolidado: testes, lint, auditorias, YAMLs, dashboard, DAG, referências do README e comparação antes x depois |

> O SHA-256 dos modelos só vale para a máquina e o momento em que foram gerados (treinar de novo
> gera arquivos binários diferentes). O que importa para o avaliador é a coerência registrada no relatório.

### 3) Subir a API + Prometheus + Grafana (Docker)
```powershell
docker compose up --build
```
| Serviço | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Métricas brutas | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana (admin / admin) | http://localhost:3000 → dashboard **Triagem de Laudos** |

Teste rápido da API:
```powershell
curl -X POST http://localhost:8000/classificar -H "Content-Type: application/json" `
  -d '{\"texto\": \"patient presents with malignant tumor and requires urgent chemotherapy evaluation\"}'
```

### 4) Gerar tráfego para o dashboard
Em outro terminal (com o `.venv` ativo):
```powershell
python scripts/gerar_trafego.py --segundos 90
```
Abra o Grafana: os painéis passam a mostrar requisições, latência e taxa de erro
(o script envia ~8% de requisições inválidas de propósito para o painel de erros).

### 5) Baseline de latência local (Etapa 1)
Para comparar modelo original x otimizado **via HTTP**, rode a API com cada backend:
```powershell
# baseline (sklearn): edite MODEL_BACKEND no docker-compose.yml para "sklearn", suba e meça
python scripts/latencia_api.py --n 300
# otimizado (onnx): volte para "onnx", suba novamente e meça
python scripts/latencia_api.py --n 300
```

### 6) Airflow (opcional, para a DAG)
```powershell
docker compose -f docker-compose.airflow.yml up
```
Abra http://localhost:8080 (usuário/senha aparecem no log: `standalone | Login with ...`),
ative a DAG **treino_triagem_laudos** e clique em *Trigger*.

## Resultados

### Modelo (conjunto de teste limpo, 1.788 laudos)
| Métrica | Valor |
|---|---|
| Acurácia | **74,4 %** |
| F1 macro | **71,9 %** |
| Baseline (sempre prever a classe majoritária) | 49,8 % |

Por classe (F1): `urgente` 0,83 · `atencao` 0,70 · `normal` 0,62.
Matriz de confusão e demais verificações: `python scripts/validar_modelo.py` → `results/validacao_modelo.md`.

> **Como interpretar:** esta acurácia **não é comparável** com a de uma avaliação sem limpeza. O teste
> original continha textos repetidos do treino e resumos de rótulo ambíguo; removê-los torna a avaliação
> mais honesta, porém mais fácil (só laudos de rótulo inequívoco). O *benchmark* ficou mais confiável.

**Comparação justa (mesmo teste limpo, 1.788 laudos):**

| Modelo | Acurácia | F1 macro |
|---|---|---|
| Treinado com os dados originais (sem limpeza) | 73,3 % | 70,8 % |
| Treinado com os dados limpos (final) | **74,4 %** | **71,9 %** |

Ou seja, limpar o treino rendeu cerca de +1 ponto percentual. Detalhes em
`results/metricas_modelo_original.json` e `results/validacao_revisao.json`.

### Latência: modelo original (sklearn) x otimizado (ONNX Runtime)

Inferência de 1 laudo por vez (500 laudos, após aquecimento), medida em CPU:

| Modelo | média (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| sklearn (original) | 0.477 | 0.47 | 0.602 | 0.675 |
| ONNX Runtime (otimizado) | 0.089 | 0.087 | 0.13 | 0.162 |

**Speedup médio: 5,4x** no tempo de inferência. A concordância das previsões ONNX x sklearn no
teste foi de **99,3 %** (a conversão do `TfidfVectorizer` para ONNX pode divergir levemente na
tokenização — comportamento documentado do `sklearn-onnx`).

### Latência ponta a ponta (HTTP, API local, 300 requisições)
| Backend | média (ms) | p95 (ms) |
|---|---|---|
| sklearn | 2.34 | 3.00 |
| ONNX | 1.77 | 2.42 |

O ganho total é menor que 5,4x porque, na API, a maior parte dos ~2 ms é pilha HTTP
(parsing, validação, serialização), não o modelo. Os números acima foram medidos fora do Docker;
**meça de novo no seu contêiner** (passo 5) e use os seus valores no vídeo.

## CI/CD (Etapa 2)

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml), disparado em todo `push` na
`main` e em *pull requests*:

1. **lint** — `ruff check` e `ruff format --check`
2. **test** — `pytest` (18 testes: dados, auditoria, modelo, ONNX, pipeline de treino, API, métricas e DAG) e
   `python scripts/auditar_dados.py` (falha o CI se houver vazamento ou dados fora do padrão)
3. **build** — `docker build` da imagem (só roda se lint e testes passarem)

## Airflow (Etapa 2)

DAG: [`dags/treino_triagem_dag.py`](dags/treino_triagem_dag.py) (`treino_triagem_laudos`, semanal):

`carregar_dados` → `treinar_modelo` → `exportar_onnx`

As três tarefas chamam funções de `src/triagem/pipeline_treino.py`, as mesmas usadas por
`scripts/treinar.py` (uma única fonte de verdade).

## Monitoramento (Etapa 3)

A API expõe `/metrics` (via `prometheus_client`):

| Métrica | Tipo | Uso |
|---|---|---|
| `triagem_requisicoes_total{metodo,endpoint,status}` | Counter | total de requisições |
| `triagem_latencia_requisicao_segundos` | Histogram | latência p50/p95/p99 |
| `triagem_latencia_inferencia_segundos` | Histogram | tempo só do modelo |
| `triagem_erros_total{endpoint,status}` | Counter | respostas ≥ 400 |
| `triagem_predicoes_total{urgencia}` | Counter | classificações por classe |
| `triagem_requisicoes_em_andamento` | Gauge | requisições simultâneas |

Dashboard provisionado automaticamente
([`monitoring/grafana/dashboards/triagem.json`](monitoring/grafana/dashboards/triagem.json)),
com **6 painéis**: total de requisições, requisições/s, latência (p50/p95/p99), taxa de erro (%),
tempo de inferência e classificações por urgência.

## Otimização de latência (Etapa 4)

Técnica aplicada: **conversão para ONNX Runtime** do pipeline completo (TF-IDF + classificador
em um único grafo), em `src/triagem/onnx_export.py`.
Decisões que importam:
- O texto é normalizado (minúsculas) **antes** do modelo, no treino e na inferência, para evitar o
  operador de normalização com *locale* do ONNX (que falha em imagens Docker sem `en_US.UTF-8`).
- Foram comparados Regressão Logística e Random Forest com TF-IDF: a regressão logística teve
  praticamente a mesma acurácia no teste limpo (74,4% x 73,9%), é menor e mais simples; por isso é o modelo final.
  O Random Forest mostrou ganho de latência relativo maior com ONNX (≈ 30x na exploração), mas gera um
  modelo maior e teve acurácia ligeiramente menor.

## Estrutura do repositório

```
├── .github/workflows/ci.yml       # lint → test → build
├── dags/treino_triagem_dag.py     # DAG do Airflow
├── data/raw/                      # CSVs do Medical Abstracts TC Corpus
├── monitoring/                    # prometheus.yml + Grafana (datasource e dashboard)
├── docs/                          # roteiro do vídeo STAR, checklist de entrega, guia de commits
├── results/                       # relatórios: auditoria dos dados, validação do modelo, latência
├── scripts/                       # treinar, auditar_dados, validar_modelo, benchmark, tráfego
├── src/triagem/                   # código da aplicação (config, dados, auditoria, modelo, onnx, api...)
├── tests/                         # pytest
├── Dockerfile
├── docker-compose.yml             # API + Prometheus + Grafana
├── docker-compose.airflow.yml     # Airflow (opcional)
└── requirements*.txt
```

## Limitações e honestidade dos resultados

- **Rótulos proxy:** o dataset não tem urgência clínica real; o mapeamento condição → urgência é
  uma escolha do projeto. Um sistema real exigiria laudos rotulados por médicos.
- **Qualidade moderada:** 74,4% de acurácia (baseline 49,8%) em um teste que **exclui** os laudos de rótulo
  ambíguo; em dados reais, com casos ambíguos, o desempenho tende a ser pior. O corpus tem rótulos ruidosos
  (um resumo pode tratar de várias condições). O foco desta fase é o **ciclo de vida MLOps**
  (deploy, CI/CD, monitoramento e latência), não a acurácia.
- **Idioma:** o corpus está em inglês; para laudos em português seria preciso retreinar com dados em PT-BR.
- **Verificações:** testes, treino, ONNX, API e métricas foram executados localmente.
  `docker compose`, o workflow do GitHub Actions e a execução da DAG dentro do Airflow **precisam
  ser validados no seu ambiente** (o roteiro de execução acima cobre isso).

## Vídeo STAR

Roteiro de 5 minutos em [`docs/ROTEIRO_VIDEO_STAR.md`](docs/ROTEIRO_VIDEO_STAR.md).

**Link do vídeo:** _(cole aqui o link após gravar)_
