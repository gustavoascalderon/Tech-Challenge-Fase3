"""Carga e preparo dos dados.

O dataset publico Medical Abstracts TC Corpus rotula resumos medicos por *condicao*
(5 classes), nao por urgencia. Como o enunciado aceita qualquer dataset com texto + alvo,
usamos um MAPEAMENTO PROXY condicao -> urgencia (documentado no README).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 1 neoplasias | 2 digestivo | 3 nervoso | 4 cardiovascular | 5 patologicas gerais
CONDICAO_PARA_URGENCIA: dict[int, str] = {
    1: "urgente",
    4: "urgente",
    2: "atencao",
    3: "atencao",
    5: "normal",
}

COLUNA_TEXTO = "texto"
COLUNA_ALVO = "urgencia"


def normalizar_texto(texto: str) -> str:
    """Normalizacao unica usada no treino E na inferencia (evita divergencia)."""
    return " ".join(str(texto).lower().split())


def preparar_dataframe(bruto: pd.DataFrame) -> pd.DataFrame:
    """Converte o CSV original em `texto` + `urgencia` e resolve duplicatas.

    Regras (auditadas em `scripts/auditar_dados.py`):
    - o mesmo texto com urgencias CONFLITANTES e ambiguo e e removido por inteiro;
    - o mesmo texto repetido com a mesma urgencia e mantido uma unica vez.
    """
    df = pd.DataFrame(
        {
            COLUNA_TEXTO: bruto["medical_abstract"].map(normalizar_texto),
            COLUNA_ALVO: bruto["condition_label"].astype(int).map(CONDICAO_PARA_URGENCIA),
        }
    ).dropna(subset=[COLUNA_ALVO])
    n_rotulos = df.groupby(COLUNA_TEXTO)[COLUNA_ALVO].transform("nunique")
    df = df[n_rotulos == 1].drop_duplicates(subset=COLUNA_TEXTO)
    return df.reset_index(drop=True)


def ler_csv_bruto(caminho: Path) -> pd.DataFrame:
    bruto = pd.read_csv(caminho)
    colunas = {"condition_label", "medical_abstract"}
    if not colunas.issubset(bruto.columns):
        raise ValueError(f"CSV {caminho} precisa das colunas {sorted(colunas)}")
    return bruto.dropna(subset=list(colunas))


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    """Le um CSV original e devolve DataFrame limpo com colunas `texto` e `urgencia`."""
    return preparar_dataframe(ler_csv_bruto(caminho))


def remover_sobreposicao(treino: pd.DataFrame, teste: pd.DataFrame) -> pd.DataFrame:
    """Remove do TESTE os textos que tambem estao no TREINO (evita vazamento de dados)."""
    repetidos = teste[COLUNA_TEXTO].isin(set(treino[COLUNA_TEXTO]))
    return teste[~repetidos].reset_index(drop=True)


def carregar_treino_teste(
    caminho_treino: Path, caminho_teste: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega treino e teste ja limpos e SEM sobreposicao entre eles."""
    treino = carregar_dataset(caminho_treino)
    teste = remover_sobreposicao(treino, carregar_dataset(caminho_teste))
    return treino, teste
