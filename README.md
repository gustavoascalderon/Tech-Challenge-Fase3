# Tech Challenge - Fase 3 - Machine Learning Engineering

**FIAP PosTech | Programa de Pós-Graduação em Machine Learning Engineering**

---

## Sobre o Projeto

Protótipo de **triagem automática de textos médicos** que classifica urgência em
`normal`, `atencao` ou `urgente`. A Fase 3 demonstra o ciclo de implantação de um modelo
NLP leve, com serviço de inferência em Docker, integração contínua, retreino orquestrado,
monitoramento e otimização de latência.

### Objetivo

Treinar um classificador TF-IDF + Regressão Logística, exportá-lo para ONNX Runtime e
comparar os backends em qualidade e tempo de resposta. Disponibilizar a inferência via
FastAPI, automatizar lint, testes e build, executar retreino no Airflow e visualizar métricas
no Prometheus e Grafana.

> Projeto acadêmico. Os dados são resumos médicos em inglês e os rótulos de urgência são
> aproximações definidas pelo projeto, sem validação clínica. Não utilizar para decisões clínicas.

### Evolução da Publicação

Esta versão acompanha a conclusão dos commits 5 a 8 pelo integrante B.
Os estados abaixo indicam arquivos integrados, não a conclusão automática de todas as validações.

---

## Arquitetura e Tecnologias

### Stack Técnico

| Componente | Tecnologia |
|---|---|
| **ML Framework** | Scikit-Learn: TF-IDF + Regressão Logística |
| **Otimização** | skl2onnx + ONNX Runtime, CPU |
| **Dados** | Pandas + CSVs públicos |
| **Dependências** | pip, requirements.txt e requirements-dev.txt; Python 3.12 |
| **API** | FastAPI + Uvicorn + Pydantic |
| **Container** | Docker + Dockerfile |
| **Logging** | logging da biblioteca padrão |
| **Instrumentação** | prometheus_client |
| **Retreino** | Apache Airflow |
| **Monitoramento** | Prometheus + Grafana via Docker Compose |
| **Code Quality** | Ruff: lint e formatação |
| **Testing** | pytest + FastAPI TestClient |
| **CI e build** | GitHub Actions |

### Estrutura do Projeto

```text
src/triagem/__init__.py
src/triagem/config.py
src/triagem/dados.py
src/triagem/modelo.py
src/triagem/onnx_export.py
src/triagem/pipeline_treino.py
src/triagem/predictor.py
src/triagem/metricas.py
src/triagem/api/__init__.py
src/triagem/api/main.py
scripts/treinar.py
scripts/gerar_trafego.py
scripts/benchmark_latencia.py
scripts/latencia_api.py
scripts/auditar_dados.py
data/README.md
data/raw/
models/
results/
dags/treino_triagem_dag.py
monitoring/
tests/
.github/workflows/ci.yml
docs/
pyproject.toml
requirements.txt
requirements-dev.txt
.dockerignore
.gitignore
Dockerfile
docker-compose.airflow.yml
docker-compose.yml
README.md
```

### Decisão de Deploy em Nuvem

**Inferência em tempo real e retreino em batch.** A API atende cada texto na chegada;
o treinamento pode ocorrer em uma janela periódica.

| Critério | Batch | Tempo real |
|---|---|---|
| Aplicação no projeto | Retreino agendado | Classificação por requisição |
| Espera | Depende da janela de execução | Depende da duração da requisição |
| Disponibilidade | Recursos podem ser alocados por execução | Serviço de inferência disponível continuamente |

Proposta teórica na AWS: API em ECS Fargate atrás de ALB, imagens no ECR, dados e artefatos
no S3 e Airflow no MWAA. O modelo utiliza CPU. Prometheus e Grafana podem ter operação gerenciada.
O repositório não provisiona essa infraestrutura; a implementação a demonstrar é local em Docker.
A promoção de um modelo do Airflow para a API precisaria de integração adicional.

---

## Quickstart

### Pré-requisitos

- Python 3.12 e Git.
- Docker com Docker Compose para validar a API em contêiner e as stacks locais.
- CSVs públicos em `data/raw/`, incluídos no projeto.

### Opção 1: Execução com Docker (Recomendado)

A API e o Dockerfile estão integrados nesta publicação. Na raiz do projeto:

```bash
docker build -t triagem-laudos:local .
docker run --rm -p 8000:8000 triagem-laudos:local
```

O build treina e exporta o modelo; a API utiliza ONNX por padrão. Para usar sklearn,
encerre a instância anterior e execute:

```bash
docker run --rm -p 8000:8000 -e MODEL_BACKEND=sklearn triagem-laudos:local
```

| Serviço | Endereço |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Saúde da API | http://localhost:8000/health |
| Métricas | http://localhost:8000/metrics |

Exemplo de classificação no macOS/Linux:

```bash
curl -X POST http://localhost:8000/classificar \
  -H 'Content-Type: application/json' \
  -d '{"texto":"patient presents with malignant tumor and requires further examination"}'
```

No Windows PowerShell:

```powershell
$corpo = @{texto = 'patient presents with malignant tumor and requires further examination'} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/classificar -Method Post -ContentType 'application/json' -Body $corpo
```

A resposta contém urgência, confiança, probabilidades, backend e tempo de inferência.
O Compose completo de API, Prometheus e Grafana entra na publicação C.

### Opção 2: Execução Local

Após clonar o repositório, execute na raiz.


macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Treine e exporte o modelo:

```bash
python scripts/treinar.py
```

Para iniciar a API localmente no macOS/Linux:

```bash
PYTHONPATH=src python -m uvicorn triagem.api.main:app --host 127.0.0.1 --port 8000
```

No PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m uvicorn triagem.api.main:app --host 127.0.0.1 --port 8000
```

Encerre a instância anterior antes de utilizar novamente a porta 8000.

---

## Pipeline de ML

### Etapa 1: Preparação de Dados

Executada por `etapa_carregar_dados()` no fluxo de `python scripts/treinar.py`.
Não há script separado de preparação nem um novo split 80/20 nesta implementação.

**Responsabilidades:**

- Ler os CSVs de treino e teste fornecidos pela fonte.
- Validar as colunas `condition_label` e `medical_abstract`.
- Remover registros ausentes e duplicatas internas por texto.
- Normalizar o texto em minúsculas e espaços.
- Mapear cinco condições para três classes proxy de urgência.
- Conferir pelo menos 2.000 registros no conjunto de treino.

**Saída:** DataFrames em memória com `texto` e `urgencia`; os CSVs originais são preservados.

#### Dataset e Mapeamento de Urgência

Foi utilizado o **Medical Abstracts TC Corpus**, com resumos médicos em inglês e cinco
classes de condições médicas: [repositório oficial](https://github.com/sebischair/Medical-Abstracts-TC-Corpus).
Autores: Tim Schopf, Daniel Braun e Florian Matthes. A referência e a licença dos dados
estão em [data/README.md](data/README.md).

O corpus rotula **condições médicas**, não urgência. Para demonstrar o pipeline técnico,
o projeto utiliza o seguinte **mapeamento proxy**, sem validação clínica:

| Condição original | Urgência proxy |
|---|---|
| Neoplasias (1) e doenças cardiovasculares (4) | `urgente` |
| Doenças do sistema digestivo (2) e do sistema nervoso (3) | `atencao` |
| Condições patológicas gerais (5) | `normal` |

Os CSVs originais ficam em `data/raw/`. A preparação atual remove duplicatas dentro de
cada arquivo, resultando em **9.445 textos de treino e 2.770 de teste**. O conjunto de treino
supera o mínimo de 2.000 amostras previsto na atividade.

**Limitação identificada na revisão dos dados:** existem **988 textos compartilhados entre
treino e teste**, sendo **949 com rótulos proxy diferentes**. A deduplicação interna não remove
essa sobreposição. Assim, o teste atual não é composto integralmente por textos inéditos.
É necessário definir uma política para conflitos de rótulo e para separação por texto antes
de consolidar uma avaliação independente. O pipeline original foi preservado nesta versão.

O script e o relatório da auditoria dos dados entram na publicação C.

### Etapa 2: Treinamento

```bash
python scripts/treinar.py
```

O comando executa preparação, treino, avaliação e exportação no mesmo fluxo.

**Responsabilidades:**

- Construir TF-IDF + Regressão Logística.
- Treinar com os textos normalizados e os rótulos proxy.
- Calcular métricas no arquivo de teste preparado.
- Persistir o pipeline, as classes e as métricas.

**Saída:**

- `models/modelo.joblib`
- `models/classes.json`
- `models/metricas_treino.json`

Os artefatos em `models/` são ignorados pelo Git e regenerados pelo script ou pelo build Docker.

### Etapa 3: Avaliação

A função `treinar()` calcula acurácia, F1 macro e relatório por classe. O indicador majoritário
registrado é a frequência da classe mais comum no teste; não há um `DummyClassifier` treinado
separadamente. A sobreposição e os conflitos de texto documentados acima limitam a avaliação.

As métricas da sua execução ficam em `models/metricas_treino.json`. Os relatórios
versionados e a comparação dos dois backends serão consolidados na publicação C.

### Etapa 4: Otimização e Comparação de Latência

A função `exportar_onnx()` converte TF-IDF e classificador em um grafo ONNX executado em CPU.
A normalização é feita antes do modelo e compartilhada com a inferência.
O comando de treino também exporta `models/modelo.onnx` e calcula a concordância das classes.

Os scripts `scripts/benchmark_latencia.py` e `scripts/latencia_api.py`, com relatórios
comparativos, serão integrados pelo integrante C. O baseline HTTP deve usar a API em Docker.

---

## Clean Code Implementado

### Nomenclatura Descritiva

- Funções de dados: `normalizar_texto()` e `carregar_dataset()`.
- Funções de treinamento: `construir_pipeline()`, `treinar()` e `salvar()`.
- Etapas compartilhadas: `etapa_carregar_dados()`, `etapa_treinar()` e `etapa_exportar_onnx()`.

### Organização por Responsabilidades

O pacote `src/triagem/` separa configuração, dados, modelo e exportação ONNX. A publicação
B acrescenta predictors, métricas e API, mantendo essas responsabilidades em módulos próprios.
`Config` e `Resultado` usam dataclasses, enquanto o protocolo `Predictor` permite alternar
entre sklearn e ONNX sem alterar o contrato da API.

### Docstrings e Logging

Os módulos e as funções principais documentam seu propósito; o pipeline utiliza o módulo
padrão `logging` para registrar carga, treinamento e exportação. As mensagens do pipeline
registram resumos, sem imprimir o conteúdo integral dos textos.

Os testes da API usam injeção de um predictor falso; os testes de modelo treinam dados
sintéticos temporários. A suíte e o workflow de qualidade são integrados pelo integrante C.

---

## Configuração

### Variáveis de Ambiente

A configuração utiliza `os.environ` em `src/triagem/config.py`. No ambiente Python,
exporte as variáveis antes de iniciar o processo. O código não carrega um arquivo `.env`
automaticamente. O Compose da API define o backend em `docker-compose.yml`.

| Variável | Padrão | Descrição |
|---|---|---|
| `DADOS_TREINO` | `data/raw/medical_tc_train.csv`, relativo à raiz do projeto | CSV de treinamento |
| `DADOS_TESTE` | `data/raw/medical_tc_test.csv`, relativo à raiz do projeto | CSV de teste |
| `DIR_MODELOS` | `models/`, relativo à raiz do projeto | Diretório dos artefatos |
| `DIR_RESULTADOS` | `results/`, relativo à raiz do projeto | Diretório dos relatórios |
| `MODEL_BACKEND` | `onnx` | Backend de inferência: `onnx` ou `sklearn` |

No macOS/Linux, por exemplo:

```bash
export MODEL_BACKEND=onnx
```

No PowerShell:

```powershell
$env:MODEL_BACKEND = 'onnx'
```

### Hiperparâmetros

| Parâmetro | Valor atual | Local |
|---|---|---|
| `max_features` | 5.000 | `Config`, em `src/triagem/config.py` |
| `random_state` | 42 | `Config`, em `src/triagem/config.py` |
| `sublinear_tf` | `True` | TF-IDF, em `src/triagem/modelo.py` |
| `lowercase` | `False` | TF-IDF; o texto já chega normalizado |
| `C` | 5 | Regressão Logística, em `src/triagem/modelo.py` |
| `max_iter` | 2.000 | Regressão Logística, em `src/triagem/modelo.py` |

Os hiperparâmetros da tabela não são sobrescritos por variáveis de ambiente na implementação
atual. Mudanças devem ser feitas na configuração ou na construção do pipeline, seguidas de retreino.

---

## Monitoramento (Prometheus + Grafana)

### Visualizar Métricas

A API expõe as métricas abaixo em `/metrics` por meio de `prometheus_client`.
A coleta Prometheus e o dashboard Grafana serão integrados por C.

| Métrica | Tipo | Uso |
|---|---|---|
| `triagem_requisicoes_total{metodo,endpoint,status}` | Counter | Total de requisições |
| `triagem_latencia_requisicao_segundos{endpoint}` | Histogram | Latência por endpoint |
| `triagem_latencia_inferencia_segundos` | Histogram | Tempo de inferência |
| `triagem_erros_total{endpoint,status}` | Counter | Respostas com status maior ou igual a 400 |
| `triagem_predicoes_total{urgencia}` | Counter | Predições por classe |
| `triagem_requisicoes_em_andamento` | Gauge | Requisições em processamento |

### Dashboard e Geração de Tráfego

O provisionamento Grafana, o Compose completo e `scripts/gerar_trafego.py` serão
publicados por C. O dashboard previsto possui seis painéis: total de requisições,
requisições/s, latência, taxa de erro, inferência e classificações por urgência.

---

## Automação (GitHub Actions + Airflow)

### Integração Contínua e Build

O workflow `.github/workflows/ci.yml` entra na publicação C: lint e testes em paralelo,
seguidos de build Docker se ambos passarem. A implementação prevista não publica imagem
nem realiza deploy automático.

### Orquestração de Retreino

A DAG semanal `treino_triagem_laudos` está em
[`dags/treino_triagem_dag.py`](dags/treino_triagem_dag.py). Suas tarefas são
`carregar_dados` → `treinar_modelo` → `exportar_onnx`, usando as funções compartilhadas
em `src/triagem/pipeline_treino.py`.

Para subir o Airflow local:

```bash
docker compose -f docker-compose.airflow.yml up -d
docker compose -f docker-compose.airflow.yml logs airflow
```

Abra http://localhost:8080. Consulte o usuário e a senha gerados pelo modo `standalone`
no log, ative a DAG e dispare uma execução manual. Confira as três tarefas concluídas
e os artefatos em `models/`.

Para conferir erros de importação:

```bash
docker compose -f docker-compose.airflow.yml exec airflow airflow dags list-import-errors
```

O primeiro início instala dependências adicionais. Se houver falha de escrita, confira as
permissões de `models/` montado no contêiner. A execução da DAG faz parte da validação da entrega.
Para encerrar, use `docker compose -f docker-compose.airflow.yml down`.

### Disponibilização de Novos Modelos

A DAG grava artefatos em `models/`, mas não atualiza automaticamente a API. A API carrega
seu modelo na inicialização, a partir da imagem, cujo próprio build executa o treinamento.
Promover o artefato do Airflow para a API requer uma integração adicional; ela não está
implementada neste protótipo.

---

## Testes

### Executar Testes

A suíte `tests/` será publicada por C. A configuração de Ruff e pytest já está em
`pyproject.toml`, mas os testes completos ainda não estão integrados nesta etapa.
Não declarar execução da suíte ou aprovação do workflow remoto antes das respectivas evidências.

---

## Vídeo de Apresentação (Método STAR)

**Duração:** até cinco minutos.  
**Link:** PREENCHER_LINK_DO_VIDEO.

| Parte | Conteúdo |
|---|---|
| **Situation** | Cenário de priorização da leitura de textos médicos |
| **Task** | API em Docker, CI/build, retreino, monitoramento e otimização |
| **Action** | Arquitetura e demonstração dos componentes em funcionamento |
| **Result** | Qualidade, latências medidas, limitações dos dados e lições aprendidas |

O roteiro e o link final serão consolidados por C após as demonstrações e validações da equipe.

---

## Referências e Documentação

### Oficiais

- [Scikit-Learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [Apache Airflow 2.10.5](https://airflow.apache.org/docs/apache-airflow/2.10.5/)
- [Prometheus](https://prometheus.io/docs/introduction/overview/)
- [Grafana](https://grafana.com/docs/grafana/latest/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Docker](https://docs.docker.com/)

### Datasets

- [Medical Abstracts TC Corpus — fonte dos CSVs](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
- [Publicação de referência do corpus](https://doi.org/10.1145/3582768.3582795)
- [Documentação dos dados utilizados](data/README.md)

---

## 👥 Time

| Nome | RA / RM |
|---|---|
| Gustavo Andres Silva Calderon | rm374865 |
| Lilian Dorini Almagro | rm374368 |
| Matheus Angelo Filetti | rm373015 |

---

## Licença

**Código do projeto:** a licença de distribuição deve ser definida pela equipe. O pacote
atual não contém um arquivo `LICENSE`; por isso, esta documentação não declara uma licença MIT
sem o respectivo arquivo.

**Dados:** o Medical Abstracts TC Corpus informa a licença
[CC BY-SA 3.0 Unported](https://github.com/sebischair/Medical-Abstracts-TC-Corpus/blob/main/LICENSE).
A atribuição da fonte e a documentação do conjunto estão em [data/README.md](data/README.md).
A licença dos dados deve ser distinguida da licença escolhida para o código do projeto.

---

**Desenvolvido para FIAP PosTech - Machine Learning Engineering | Fase 3**
