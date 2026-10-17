"""Configuracoes centralizadas (podem ser sobrescritas por variaveis de ambiente)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def _caminho(nome_env: str, padrao: Path) -> Path:
    return Path(os.environ.get(nome_env, str(padrao)))


@dataclass(frozen=True)
class Config:
    dados_treino: Path = field(
        default_factory=lambda: _caminho("DADOS_TREINO", RAIZ / "data/raw/medical_tc_train.csv")
    )
    dados_teste: Path = field(
        default_factory=lambda: _caminho("DADOS_TESTE", RAIZ / "data/raw/medical_tc_test.csv")
    )
    dir_modelos: Path = field(default_factory=lambda: _caminho("DIR_MODELOS", RAIZ / "models"))
    dir_resultados: Path = field(
        default_factory=lambda: _caminho("DIR_RESULTADOS", RAIZ / "results")
    )
    backend: str = field(default_factory=lambda: os.environ.get("MODEL_BACKEND", "onnx"))
    max_features: int = 5000
    random_state: int = 42

    @property
    def modelo_joblib(self) -> Path:
        return self.dir_modelos / "modelo.joblib"

    @property
    def modelo_onnx(self) -> Path:
        return self.dir_modelos / "modelo.onnx"

    @property
    def classes_json(self) -> Path:
        return self.dir_modelos / "classes.json"

    @property
    def metricas_json(self) -> Path:
        return self.dir_modelos / "metricas_treino.json"


def obter_config() -> Config:
    return Config()
