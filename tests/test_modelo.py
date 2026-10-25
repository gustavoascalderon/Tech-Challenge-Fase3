import numpy as np

from triagem.config import Config
from triagem.modelo import construir_pipeline, salvar, treinar
from triagem.onnx_export import exportar_onnx, taxa_concordancia


def _cfg(tmp_path) -> Config:
    return Config(dir_modelos=tmp_path / "models", dir_resultados=tmp_path / "res")


def test_treino_gera_metricas_validas(df_sintetico, tmp_path):
    pipeline, metricas = treinar(df_sintetico, df_sintetico, _cfg(tmp_path))
    assert 0.0 <= metricas["acuracia"] <= 1.0
    assert metricas["acuracia"] > 0.9  # dados sinteticos sao separaveis
    assert set(pipeline.named_steps["clf"].classes_) == {"atencao", "normal", "urgente"}


def test_salvar_cria_artefatos(df_sintetico, tmp_path):
    cfg = _cfg(tmp_path)
    pipeline, metricas = treinar(df_sintetico, df_sintetico, cfg)
    salvar(pipeline, metricas, cfg)
    assert cfg.modelo_joblib.exists()
    assert cfg.classes_json.exists()
    assert cfg.metricas_json.exists()


def test_onnx_concorda_com_sklearn(df_sintetico, tmp_path):
    cfg = _cfg(tmp_path)
    pipeline = construir_pipeline().fit(df_sintetico["texto"], df_sintetico["urgencia"])
    exportar_onnx(pipeline, cfg.modelo_onnx)
    assert cfg.modelo_onnx.exists()
    concordancia = taxa_concordancia(pipeline, cfg.modelo_onnx, df_sintetico["texto"].tolist())
    assert np.isclose(concordancia, 1.0, atol=0.02)
