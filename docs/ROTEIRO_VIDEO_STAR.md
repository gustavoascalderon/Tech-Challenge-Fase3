# Roteiro do vídeo STAR (máx. 5 min)

Dica: grave a tela com Docker, Grafana e o terminal já abertos. Ensaie uma vez com cronômetro.

## S — Situation (0:00–0:45)
- Hospital de referência recebe muitos laudos; um laudo urgente não pode esperar na fila.
- Triagem automática por texto ajuda a **priorizar** a leitura médica.
- Deixe claro: é um protótipo acadêmico; os rótulos de urgência são **proxy** (dataset público de
  condições médicas), não validados clinicamente.

## T — Task (0:45–1:30)
- Servir um classificador de texto leve por API REST em Docker.
- Requisitos da fase: **CI/CD** (GitHub Actions), **retreino** (Airflow), **monitoramento**
  (Prometheus + Grafana, ≥ 3 painéis) e **otimização de latência** (ONNX).
- Decisão de nuvem: inferência em **tempo real** (contêiner gerenciado) + retreino em **batch**.

## A — Action (1:30–3:15)
1. Mostre a arquitetura (diagrama do README).
2. Dados: mostre `results/auditoria_dados.md` (vazamento encontrado e removido).
2. Modelo: TF-IDF + Regressão Logística; comparou com Random Forest e escolheu pelo melhor resultado.
3. Otimização: pipeline exportado para ONNX Runtime; texto normalizado fora do modelo.
4. CI/CD: abra a aba **Actions** com lint → test → build verdes.
5. Airflow: mostre a DAG `treino_triagem_laudos` (3 tarefas) e a execução.
6. Monitoramento: mostre `/metrics` e o dashboard do Grafana.

## R — Result (3:15–5:00)
1. Rode `python scripts/gerar_trafego.py --segundos 60` e mostre os painéis reagindo
   (requisições, latência, taxa de erro com os 422 propositais).
2. Mostre a tabela de latência: sklearn x ONNX (≈ 5,4x no modelo; use **os seus números** medidos
   no Docker).
3. Resultado do modelo: 74,4% de acurácia vs 49,8% do baseline, no teste limpo; seja transparente: os rótulos de urgência são proxy e o teste exclui laudos ambíguos.
4. Lições aprendidas (exemplos reais deste projeto):
   - O ganho do ONNX no modelo (5,4x) diluiu no total HTTP (~2 ms de pilha web): medir ponta a ponta importa.
   - O operador de normalização do ONNX depende de *locale* e quebra em contêiner enxuto;
     normalizar o texto antes resolveu.
   - A conversão do TF-IDF para ONNX não é 100% idêntica (99,3% de concordância): valide sempre.
   - A auditoria achou 988 textos iguais em treino e teste e 1.880 com rótulo conflitante: sem ela, a métrica seria enganosa.
   - Próximos passos: laudos em português com rótulos reais, monitorar *drift* do texto.
