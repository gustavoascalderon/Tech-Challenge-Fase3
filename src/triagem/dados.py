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


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    """Le o CSV original e devolve DataFrame com colunas `texto` e `urgencia`."""
    bruto = pd.read_csv(caminho)
    colunas = {"condition_label", "medical_abstract"}
    if not colunas.issubset(bruto.columns):
        raise ValueError(f"CSV {caminho} precisa das colunas {sorted(colunas)}")
    bruto = bruto.dropna(subset=list(colunas)).drop_duplicates(subset="medical_abstract")
    df = pd.DataFrame(
        {
            COLUNA_TEXTO: bruto["medical_abstract"].map(normalizar_texto),
            COLUNA_ALVO: bruto["condition_label"].astype(int).map(CONDICAO_PARA_URGENCIA),
        }
    )
    return df.dropna(subset=[COLUNA_ALVO]).reset_index(drop=True)
