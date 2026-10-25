"""Referencia "ANTES": treina o mesmo modelo com a carga ORIGINAL (sem limpeza).

A carga original apenas removia textos repetidos dentro de cada arquivo; nao removia textos
com urgencia conflitante nem a sobreposicao treino/teste. O resultado serve para comparar com
o modelo revisado e provar o efeito da auditoria (results/metricas_modelo_original.json).

Uso: python scripts/avaliar_modelo_original.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from triagem.config import obter_config  # noqa: E402
from triagem.dados import (  # noqa: E402
    CONDICAO_PARA_URGENCIA,
    carregar_treino_teste,
    ler_csv_bruto,
    normalizar_texto,
)
from triagem.modelo import construir_pipeline  # noqa: E402


def carga_original(caminho: Path) -> pd.DataFrame:
    """Reproduz a carga antes da auditoria: so remove textos brutos repetidos no mesmo arquivo."""
    bruto = ler_csv_bruto(caminho).drop_duplicates(subset="medical_abstract")
    df = pd.DataFrame(
        {
            "texto": bruto["medical_abstract"].map(normalizar_texto),
            "urgencia": bruto["condition_label"].astype(int).map(CONDICAO_PARA_URGENCIA),
        }
    )
    return df.dropna(subset=["urgencia"]).reset_index(drop=True)


def avaliar(pipeline, df: pd.DataFrame) -> dict[str, float]:
    previsto = pipeline.predict(df["texto"])
    return {
        "acuracia": round(float(accuracy_score(df["urgencia"], previsto)), 4),
        "f1_macro": round(float(f1_score(df["urgencia"], previsto, average="macro")), 4),
        "baseline_classe_majoritaria": round(
            float(df["urgencia"].value_counts(normalize=True).max()), 4
        ),
        "n_teste": int(len(df)),
    }


def main() -> int:
    cfg = obter_config()
    treino = carga_original(cfg.dados_treino)
    teste = carga_original(cfg.dados_teste)
    sobreposicao = len(set(treino["texto"]) & set(teste["texto"]))
    pipeline = construir_pipeline(cfg.max_features, cfg.random_state)
    pipeline.fit(treino["texto"], treino["urgencia"])

    _, teste_limpo = carregar_treino_teste(cfg.dados_treino, cfg.dados_teste)
    resultado = {
        "descricao": "Modelo treinado com a carga ORIGINAL (antes da auditoria); referencia.",
        "n_treino": int(len(treino)),
        "textos_iguais_em_treino_e_teste": int(sobreposicao),
        "avaliacao_no_teste_original": avaliar(pipeline, teste),
        "avaliacao_no_teste_limpo": avaliar(pipeline, teste_limpo),
        "observacao": (
            "A acuracia no teste original e afetada por vazamento e rotulos ambiguos; "
            "a avaliacao no teste limpo e a comparavel com o modelo revisado."
        ),
    }
    cfg.dir_resultados.mkdir(parents=True, exist_ok=True)
    destino = cfg.dir_resultados / "metricas_modelo_original.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
