import pandas as pd

from triagem.config import Config
from triagem.pipeline_treino import etapa_carregar_dados, etapa_exportar_onnx, etapa_treinar


def _cfg(tmp_path, df_sintetico) -> Config:
    # reaproveita o dataset sintetico no formato original do CSV
    inverso = {"urgente": 1, "atencao": 2, "normal": 5}
    bruto = pd.DataFrame(
        {
            "condition_label": df_sintetico["urgencia"].map(inverso),
            "medical_abstract": df_sintetico["texto"],
        }
    )
    extra = pd.DataFrame(
        {
            "condition_label": [1, 2, 5] * 800,
            "medical_abstract": [
                f"complemento {i} sobre caso clinico {i % 7}" for i in range(2400)
            ],
        }
    )
    treino = pd.concat([bruto, extra], ignore_index=True)
    teste = pd.DataFrame(
        {
            "condition_label": [1, 2, 5] * 20,
            "medical_abstract": [f"teste independente {i} tumor gastric wound" for i in range(60)],
        }
    )
    treino.to_csv(tmp_path / "tr.csv", index=False)
    teste.to_csv(tmp_path / "te.csv", index=False)
    return Config(
        dados_treino=tmp_path / "tr.csv",
        dados_teste=tmp_path / "te.csv",
        dir_modelos=tmp_path / "models",
        dir_resultados=tmp_path / "res",
    )


def test_pipeline_completo_gera_modelo_e_onnx(tmp_path, df_sintetico):
    cfg = _cfg(tmp_path, df_sintetico)
    assert etapa_carregar_dados(cfg)["n_treino"] >= 2000
    assert 0.0 <= etapa_treinar(cfg)["acuracia"] <= 1.0
    assert cfg.modelo_joblib.exists()
    assert etapa_exportar_onnx(cfg)["concordancia_onnx_sklearn"] >= 0.9
    assert cfg.modelo_onnx.exists()
