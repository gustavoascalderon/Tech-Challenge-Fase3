# Checklist de entrega — Tech Challenge Fase 3

Marque cada item **depois de verificar no seu computador**. Itens com 🖥️ só você consegue validar
(dependem de Docker, GitHub ou gravação).

## Repositório GitHub
- [ ] Repositório criado e código enviado (`git push`)
- [ ] 🖥️ Histórico de commits semântico (veja `docs/COMMITS_SUGERIDOS.md`)
- [ ] 🖥️ Aba **Actions** com o workflow `CI` verde (lint, test, build)
- [ ] Repositório com acesso liberado ao professor (público ou convidado)

## Etapa 1 — API e decisão de nuvem
- [x] API FastAPI (`src/triagem/api/main.py`) — `/classificar`, `/health`, `/metrics`
- [x] `Dockerfile` funcional
- [ ] 🖥️ `docker compose up --build` sobe sem erro
- [ ] 🖥️ Baseline de latência local medido (`scripts/latencia_api.py`) e anotado no README
- [x] Decisão arquitetural (batch x tempo real) no README

## Etapa 2 — CI/CD e Airflow
- [x] `.github/workflows/ci.yml` (lint + pytest + auditoria de dados + build)
- [x] `dags/treino_triagem_dag.py`
- [ ] 🖥️ DAG executada no Airflow (`docker-compose.airflow.yml`) — tire um print do grafo verde

## Etapa 3 — Monitoramento
- [x] Métricas com `prometheus_client`
- [x] `docker-compose.yml` com API + Prometheus + Grafana
- [x] Dashboard com 6 painéis (mínimo exigido: 3) em `monitoring/grafana/dashboards/triagem.json`
- [ ] 🖥️ **Print do dashboard com dados** (rode `scripts/gerar_trafego.py`) salvo em `docs/dashboard_grafana.png`
- [ ] 🖥️ Painel "Taxa de erro" mostrando valores (o gerador envia ~8% de requisições inválidas)

## Etapa 4 — Otimização de latência
- [x] Modelo treinado (TF-IDF + Regressão Logística)
- [x] Conversão para ONNX Runtime
- [x] Comparação de latência original x otimizado (`results/latencia_modelo.md`)
- [ ] 🖥️ Refazer o benchmark na sua máquina e atualizar a tabela do README, se os números forem diferentes

## Relatórios de auditoria e validação (extras)
- [x] `results/auditoria_dados.md` (vazamento, duplicatas, rótulos conflitantes)
- [x] `results/validacao_modelo.md` (métricas, matriz de confusão, ONNX, contrato da API)
- [x] `results/auditoria_artefatos_recebidos.json` (inventário e hashes dos artefatos, versões fixadas)
- [x] `results/metricas_modelo_original.json` (referência antes da limpeza dos dados)
- [x] `results/validacao_revisao.json` (consolidado da revisão: tudo aprovado)

## Documentação e vídeo
- [x] README com arquitetura, instruções de execução e limitações
- [ ] 🖥️ Vídeo STAR gravado (≤ 5 min) — roteiro em `docs/ROTEIRO_VIDEO_STAR.md`
- [ ] 🖥️ Link do vídeo colado no README (seção "Vídeo STAR")
- [ ] Entrega na plataforma da FIAP antes do prazo (vale 90% da nota das disciplinas da fase)
