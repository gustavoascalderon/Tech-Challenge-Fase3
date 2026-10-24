"""Predictors intercambiaveis: sklearn (baseline) e ONNX Runtime (otimizado)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import joblib
import numpy as np

from triagem.config import Config
from triagem.dados import normalizar_texto
from triagem.onnx_export import criar_sessao


@dataclass(frozen=True)
class Resultado:
    urgencia: str
    confianca: float
    probabilidades: dict[str, float]


class Predictor(Protocol):
    nome: str

    def prever(self, texto: str) -> Resultado: ...


def _montar(classes: list[str], probs: np.ndarray) -> Resultado:
    indice = int(np.argmax(probs))
    return Resultado(
        urgencia=classes[indice],
        confianca=round(float(probs[indice]), 4),
        probabilidades={c: round(float(p), 4) for c, p in zip(classes, probs, strict=True)},
    )


class SklearnPredictor:
    nome = "sklearn"

    def __init__(self, cfg: Config) -> None:
        self._pipeline = joblib.load(cfg.modelo_joblib)
        self._classes = [str(c) for c in self._pipeline.named_steps["clf"].classes_]

    def prever(self, texto: str) -> Resultado:
        probs = self._pipeline.predict_proba([normalizar_texto(texto)])[0]
        return _montar(self._classes, probs)


class OnnxPredictor:
    nome = "onnx"

    def __init__(self, cfg: Config) -> None:
        self._sessao = criar_sessao(cfg.modelo_onnx)
        self._classes = json.loads(cfg.classes_json.read_text(encoding="utf-8"))

    def prever(self, texto: str) -> Resultado:
        entrada = np.array([[normalizar_texto(texto)]], dtype=object)
        probs = self._sessao.run(None, {"texto": entrada})[1][0]
        return _montar(self._classes, probs)


def carregar_predictor(cfg: Config) -> Predictor:
    if cfg.backend == "sklearn":
        return SklearnPredictor(cfg)
    return OnnxPredictor(cfg)
