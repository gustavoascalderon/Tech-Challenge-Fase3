"""Otimizacao de latencia: exporta o pipeline treinado para ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline


def exportar_onnx(pipeline: Pipeline, destino: Path) -> Path:
    """Converte o pipeline completo (TF-IDF + classificador) em um unico grafo ONNX."""
    clf = pipeline.named_steps["clf"]
    modelo = to_onnx(
        pipeline,
        initial_types=[("texto", StringTensorType([None, 1]))],
        options={id(clf): {"zipmap": False}},
        target_opset={"": 15, "ai.onnx.ml": 3},
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(modelo.SerializeToString())
    return destino


def criar_sessao(caminho: Path) -> ort.InferenceSession:
    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = 1
    return ort.InferenceSession(str(caminho), opcoes, providers=["CPUExecutionProvider"])


def taxa_concordancia(pipeline: Pipeline, caminho_onnx: Path, textos: list[str]) -> float:
    """Fracao de previsoes ONNX identicas as do sklearn (sanidade da conversao)."""
    sessao = criar_sessao(caminho_onnx)
    entrada = np.array(textos, dtype=object).reshape(-1, 1)
    rotulos_onnx = sessao.run(None, {"texto": entrada})[0]
    rotulos_sk = pipeline.predict(textos)
    return float((rotulos_onnx == rotulos_sk).mean())
