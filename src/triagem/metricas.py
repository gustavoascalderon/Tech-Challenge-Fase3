"""Metricas Prometheus (prometheus_client). Definidas uma unica vez no import."""

from prometheus_client import Counter, Gauge, Histogram

REQUISICOES = Counter(
    "triagem_requisicoes_total",
    "Total de requisicoes HTTP recebidas",
    ["metodo", "endpoint", "status"],
)
LATENCIA_REQUISICAO = Histogram(
    "triagem_latencia_requisicao_segundos",
    "Tempo total de resposta da requisicao (segundos)",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
LATENCIA_INFERENCIA = Histogram(
    "triagem_latencia_inferencia_segundos",
    "Tempo somente da inferencia do modelo (segundos)",
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)
ERROS = Counter(
    "triagem_erros_total",
    "Total de respostas com erro (status >= 400)",
    ["endpoint", "status"],
)
PREDICOES = Counter(
    "triagem_predicoes_total",
    "Total de classificacoes por classe de urgencia",
    ["urgencia"],
)
EM_ANDAMENTO = Gauge(
    "triagem_requisicoes_em_andamento",
    "Requisicoes sendo processadas neste momento",
)
