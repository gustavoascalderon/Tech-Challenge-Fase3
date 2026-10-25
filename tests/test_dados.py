import pandas as pd
import pytest

from triagem.dados import CONDICAO_PARA_URGENCIA, carregar_dataset, normalizar_texto


def test_normalizar_texto_minusculas_e_espacos():
    assert normalizar_texto("  Dor   TORACICA \n forte ") == "dor toracica forte"


def test_mapeamento_cobre_as_cinco_condicoes():
    assert set(CONDICAO_PARA_URGENCIA) == {1, 2, 3, 4, 5}
    assert set(CONDICAO_PARA_URGENCIA.values()) == {"normal", "atencao", "urgente"}


def test_carregar_dataset_aplica_mapeamento(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {"condition_label": [1, 5, 3], "medical_abstract": ["Tumor A", "Geral B", "Nervo C"]}
    ).to_csv(csv, index=False)
    df = carregar_dataset(csv)
    assert df["urgencia"].tolist() == ["urgente", "normal", "atencao"]
    assert df["texto"].tolist() == ["tumor a", "geral b", "nervo c"]


def test_carregar_dataset_rejeita_colunas_erradas(tmp_path):
    csv = tmp_path / "ruim.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        carregar_dataset(csv)
