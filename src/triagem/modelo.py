"""Treino do classificador de texto (TF-IDF + Regressao Logistica)."""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from triagem.config import Config
from triagem.dados import COLUNA_ALVO, COLUNA_TEXTO


def construir_pipeline(max_features: int = 5000, random_state: int = 42) -> Pipeline:
    """lowercase=False porque o texto ja chega normalizado (simplifica a conversao ONNX)."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(lowercase=False, max_features=max_features, sublinear_tf=True),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=5, random_state=random_state)),
        ]
    )


def treinar(
    treino: pd.DataFrame, teste: pd.DataFrame, cfg: Config
) -> tuple[Pipeline, dict[str, Any]]:
    """Treina e avalia. Retorna o pipeline e um dicionario de metricas."""
    pipeline = construir_pipeline(cfg.max_features, cfg.random_state)
    pipeline.fit(treino[COLUNA_TEXTO], treino[COLUNA_ALVO])
    previsto = pipeline.predict(teste[COLUNA_TEXTO])
    baseline = teste[COLUNA_ALVO].value_counts(normalize=True).max()
    metricas = {
        "acuracia": round(float(accuracy_score(teste[COLUNA_ALVO], previsto)), 4),
        "f1_macro": round(float(f1_score(teste[COLUNA_ALVO], previsto, average="macro")), 4),
        "baseline_classe_majoritaria": round(float(baseline), 4),
        "n_treino": int(len(treino)),
        "n_teste": int(len(teste)),
        "relatorio": classification_report(teste[COLUNA_ALVO], previsto, output_dict=True),
    }
    return pipeline, metricas


def salvar(pipeline: Pipeline, metricas: dict[str, Any], cfg: Config) -> None:
    cfg.dir_modelos.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, cfg.modelo_joblib)
    classes = [str(c) for c in pipeline.named_steps["clf"].classes_]
    cfg.classes_json.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    cfg.metricas_json.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
