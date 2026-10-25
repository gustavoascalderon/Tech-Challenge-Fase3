import pandas as pd

from triagem.auditoria import auditar, gerar_markdown
from triagem.config import Config
from triagem.dados import preparar_dataframe, remover_sobreposicao


def _bruto(linhas):
    return pd.DataFrame(linhas, columns=["condition_label", "medical_abstract"])


def test_conflito_de_urgencia_remove_o_texto_inteiro():
    # mesmo texto: condicao 1 (urgente) e 5 (normal) -> ambiguo
    df = preparar_dataframe(_bruto([(1, "Texto X"), (5, "texto x"), (5, "Texto Y")]))
    assert df["texto"].tolist() == ["texto y"]


def test_conflito_de_condicao_com_mesma_urgencia_e_mantido_uma_vez():
    # condicoes 1 e 4 mapeiam ambas para "urgente": nao e conflito de urgencia
    df = preparar_dataframe(_bruto([(1, "Texto Z"), (4, "texto z")]))
    assert df["urgencia"].tolist() == ["urgente"]


def test_remover_sobreposicao_tira_do_teste():
    treino = pd.DataFrame({"texto": ["a", "b"], "urgencia": ["normal", "normal"]})
    teste = pd.DataFrame({"texto": ["b", "c"], "urgencia": ["normal", "urgente"]})
    assert remover_sobreposicao(treino, teste)["texto"].tolist() == ["c"]


def test_auditar_detecta_vazamento_bruto_e_aprova_apos_limpeza(tmp_path):
    treino = [(1 + i % 5, f"laudo numero {i} sobre condicao {i % 5}") for i in range(2400)]
    teste = [(1, "laudo numero 0 sobre condicao 0"), (3, "texto exclusivo do teste")]
    _bruto(treino).to_csv(tmp_path / "tr.csv", index=False)
    _bruto(teste).to_csv(tmp_path / "te.csv", index=False)
    cfg = Config(dados_treino=tmp_path / "tr.csv", dados_teste=tmp_path / "te.csv")
    r = auditar(cfg)
    assert r["sobreposicao_treino_teste_bruta"] == 1
    assert r["limpo"]["n_teste"] == 1
    assert r["aprovado"] is True
    assert "APROVADO" in gerar_markdown(r)
