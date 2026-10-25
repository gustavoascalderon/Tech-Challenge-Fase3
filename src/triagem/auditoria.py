"""Auditoria de qualidade dos dados: duplicatas, rotulos conflitantes, vazamento e balanceamento.

Gera um dicionario (para JSON) e um relatorio em Markdown. Usada por
`scripts/auditar_dados.py` e coberta por testes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from triagem.config import Config
from triagem.dados import (
    COLUNA_ALVO,
    COLUNA_TEXTO,
    CONDICAO_PARA_URGENCIA,
    carregar_treino_teste,
    ler_csv_bruto,
    normalizar_texto,
)

MINIMO_AMOSTRAS = 2000
MINIMO_FRACAO_CLASSE = 0.05


def _perfil_bruto(bruto: pd.DataFrame) -> dict[str, Any]:
    texto = bruto["medical_abstract"].map(normalizar_texto)
    urgencia = bruto["condition_label"].astype(int).map(CONDICAO_PARA_URGENCIA)
    tmp = pd.DataFrame({"t": texto, "cond": bruto["condition_label"], "urg": urgencia})
    palavras = texto.str.split().str.len()
    return {
        "linhas": int(len(bruto)),
        "textos_duplicados": int(texto.duplicated().sum()),
        "textos_com_condicao_conflitante": int((tmp.groupby("t")["cond"].nunique() > 1).sum()),
        "textos_com_urgencia_conflitante": int((tmp.groupby("t")["urg"].nunique() > 1).sum()),
        "palavras_min": int(palavras.min()),
        "palavras_mediana": int(palavras.median()),
        "palavras_max": int(palavras.max()),
        "distribuicao_condicao": {
            str(k): round(float(v), 4)
            for k, v in bruto["condition_label"].value_counts(normalize=True).sort_index().items()
        },
    }


def _distribuicao(df: pd.DataFrame) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in df[COLUNA_ALVO].value_counts(normalize=True).items()}


def auditar(cfg: Config) -> dict[str, Any]:
    bruto_treino = ler_csv_bruto(cfg.dados_treino)
    bruto_teste = ler_csv_bruto(cfg.dados_teste)
    textos_treino = set(bruto_treino["medical_abstract"].map(normalizar_texto))
    textos_teste = set(bruto_teste["medical_abstract"].map(normalizar_texto))
    sobreposicao_bruta = len(textos_treino & textos_teste)

    treino, teste = carregar_treino_teste(cfg.dados_treino, cfg.dados_teste)
    sobreposicao_final = len(set(treino[COLUNA_TEXTO]) & set(teste[COLUNA_TEXTO]))
    dist_treino = _distribuicao(treino)
    dist_teste = _distribuicao(teste)
    menor_classe = min(dist_treino.values())
    tamanho_min_texto = int(treino[COLUNA_TEXTO].str.len().min())

    verificacoes = [
        (
            "Sem texto repetido entre treino e teste (vazamento)",
            sobreposicao_final == 0,
            f"{sobreposicao_bruta} textos repetidos no original, {sobreposicao_final} apos limpeza",
        ),
        (
            f"Treino com no minimo {MINIMO_AMOSTRAS} amostras (exigencia do enunciado)",
            len(treino) >= MINIMO_AMOSTRAS,
            f"{len(treino)} amostras",
        ),
        ("As 3 classes de urgencia estao presentes", len(dist_treino) == 3, str(dist_treino)),
        (
            f"Nenhuma classe com menos de {MINIMO_FRACAO_CLASSE:.0%} do treino",
            menor_classe >= MINIMO_FRACAO_CLASSE,
            f"menor classe: {menor_classe:.1%}",
        ),
        (
            "Sem textos nulos ou vazios",
            tamanho_min_texto > 0 and not treino[COLUNA_TEXTO].isna().any(),
            f"menor texto: {tamanho_min_texto} caracteres",
        ),
        (
            "Sem texto com urgencia conflitante apos limpeza",
            not treino[COLUNA_TEXTO].duplicated().any(),
            "duplicatas e conflitos removidos",
        ),
    ]
    return {
        "bruto": {"treino": _perfil_bruto(bruto_treino), "teste": _perfil_bruto(bruto_teste)},
        "sobreposicao_treino_teste_bruta": sobreposicao_bruta,
        "limpo": {
            "n_treino": int(len(treino)),
            "n_teste": int(len(teste)),
            "distribuicao_urgencia_treino": dist_treino,
            "distribuicao_urgencia_teste": dist_teste,
            "baseline_classe_majoritaria_teste": max(dist_teste.values()),
        },
        "verificacoes": [{"nome": n, "ok": bool(ok), "detalhe": d} for n, ok, d in verificacoes],
        "aprovado": all(ok for _, ok, _ in verificacoes),
    }


def gerar_markdown(r: dict[str, Any]) -> str:
    bt, be = r["bruto"]["treino"], r["bruto"]["teste"]
    lim = r["limpo"]
    linhas = [
        "# Relatorio de auditoria dos dados",
        "",
        f"**Resultado geral: {'APROVADO' if r['aprovado'] else 'REPROVADO'}**",
        "",
        "## Verificacoes",
        "",
        "| Verificacao | Status | Detalhe |",
        "|---|---|---|",
    ]
    for v in r["verificacoes"]:
        linhas.append(f"| {v['nome']} | {'OK' if v['ok'] else 'FALHOU'} | {v['detalhe']} |")
    linhas += [
        "",
        "## Dados originais (antes da limpeza)",
        "",
        "| Item | Treino | Teste |",
        "|---|---|---|",
        f"| Linhas | {bt['linhas']} | {be['linhas']} |",
        f"| Textos duplicados | {bt['textos_duplicados']} | {be['textos_duplicados']} |",
        "| Textos com condicao conflitante | "
        f"{bt['textos_com_condicao_conflitante']} | {be['textos_com_condicao_conflitante']} |",
        "| Textos com urgencia conflitante | "
        f"{bt['textos_com_urgencia_conflitante']} | {be['textos_com_urgencia_conflitante']} |",
        f"| Palavras (min / mediana / max) | {bt['palavras_min']} / {bt['palavras_mediana']} / "
        f"{bt['palavras_max']} | {be['palavras_min']} / {be['palavras_mediana']} / "
        f"{be['palavras_max']} |",
        "",
        f"Textos identicos presentes em treino **e** teste: "
        f"**{r['sobreposicao_treino_teste_bruta']}** (removidos do teste para evitar vazamento).",
        "",
        "## Dados limpos (usados no treino e na avaliacao)",
        "",
        f"- Treino: **{lim['n_treino']}** laudos | Teste: **{lim['n_teste']}** laudos",
        f"- Distribuicao de urgencia (treino): {lim['distribuicao_urgencia_treino']}",
        f"- Distribuicao de urgencia (teste): {lim['distribuicao_urgencia_teste']}",
        f"- Baseline (classe majoritaria no teste): {lim['baseline_classe_majoritaria_teste']:.1%}",
        "",
        "## Regras de limpeza",
        "",
        "1. Texto normalizado (minusculas, espacos unificados).",
        "2. Condicao convertida para urgencia (mapeamento proxy documentado no README).",
        "3. Texto com urgencias diferentes no mesmo conjunto: removido (rotulo ambiguo).",
        "4. Texto repetido com a mesma urgencia: mantido uma vez.",
        "5. Textos do teste que tambem aparecem no treino: removidos do teste.",
        "",
    ]
    return "\n".join(linhas)
