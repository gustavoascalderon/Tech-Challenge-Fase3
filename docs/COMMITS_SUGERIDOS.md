# Histórico de commits semântico (sugestão)

O enunciado pede histórico "semântico e organizado". Prefixos: `feat`, `fix`, `docs`, `test`, `ci`,
`build`, `chore`, `refactor`, `perf`. Sugestão de sequência, um commit por passo:

```
chore: estrutura inicial do projeto e dependencias
feat(dados): carga do dataset e mapeamento condicao -> urgencia
feat(dados): auditoria de qualidade e remocao de vazamento treino/teste
feat(scripts): auditoria de artefatos, referencia do modelo original e validacao da revisao
feat(modelo): pipeline TF-IDF + regressao logistica
feat(api): endpoint /classificar com FastAPI
build: Dockerfile da API de inferencia
test: testes de dados, modelo e API
ci: workflow GitHub Actions (lint, pytest, build)
feat(airflow): DAG de treino e retreino
feat(metricas): instrumentacao com prometheus_client
build: docker compose com API, Prometheus e Grafana
feat(grafana): dashboard com 6 paineis
perf(modelo): exportacao para ONNX Runtime
docs: benchmark de latencia sklearn x ONNX
docs: README, roteiro do video e checklist de entrega
```

Comandos (PowerShell), por exemplo:
```powershell
git add src/triagem/dados.py
git commit -m "feat(dados): carga do dataset e mapeamento condicao -> urgencia"
```
