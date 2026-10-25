"""Etapas do pipeline de treino/retreino.

As mesmas funcoes sao usadas pelos scripts de linha de comando E pela DAG do Airflow
(uma unica fonte de verdade).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from triagem import modelo, onnx_export
from triagem.config import Config, obter_config
from triagem.dados import carregar_treino_teste

log = logging.getLogger(__name__)


def etapa_carregar_dados(cfg: Config | None = None) -> dict[str, int]:
    """Etapa 1: le os CSVs, aplica o mapeamento de urgencia e valida o volume minimo."""
    cfg = cfg or obter_config()
    treino, teste = carregar_treino_teste(cfg.dados_treino, cfg.dados_teste)
    if set(treino["texto"]) & set(teste["texto"]):
        raise ValueError("Vazamento: ha textos iguais no treino e no teste")
    if len(treino) < 2000:
        raise ValueError(f"Dataset de treino com {len(treino)} linhas (minimo exigido: 2000)")
    resumo = {"n_treino": len(treino), "n_teste": len(teste)}
    log.info("Dados carregados: %s", resumo)
    return resumo


def etapa_treinar(cfg: Config | None = None) -> dict[str, float]:
    """Etapa 2: treina o classificador, avalia e salva o modelo (.joblib)."""
    cfg = cfg or obter_config()
    treino, teste = carregar_treino_teste(cfg.dados_treino, cfg.dados_teste)
    pipeline, metricas = modelo.treinar(treino, teste, cfg)
    modelo.salvar(pipeline, metricas, cfg)
    resumo = {"acuracia": metricas["acuracia"], "f1_macro": metricas["f1_macro"]}
    log.info("Modelo treinado e salvo em %s: %s", cfg.modelo_joblib, resumo)
    return resumo


def etapa_exportar_onnx(cfg: Config | None = None) -> dict[str, float]:
    """Etapa 3: converte o modelo salvo para ONNX e valida a equivalencia."""
    import joblib

    cfg = cfg or obter_config()
    pipeline = joblib.load(cfg.modelo_joblib)
    onnx_export.exportar_onnx(pipeline, cfg.modelo_onnx)
    _, teste = carregar_treino_teste(cfg.dados_treino, cfg.dados_teste)
    concordancia = onnx_export.taxa_concordancia(pipeline, cfg.modelo_onnx, teste["texto"].tolist())
    log.info("ONNX salvo em %s (concordancia com sklearn: %.4f)", cfg.modelo_onnx, concordancia)
    return {"concordancia_onnx_sklearn": round(concordancia, 4)}


def executar_tudo(cfg: Config | None = None) -> dict[str, object]:
    cfg = cfg or obter_config()
    return {
        "dados": etapa_carregar_dados(cfg),
        "treino": etapa_treinar(cfg),
        "onnx": etapa_exportar_onnx(cfg),
    }


def ler_metricas(caminho: Path) -> dict[str, object]:
    return json.loads(caminho.read_text(encoding="utf-8"))
