"""Valida o modelo treinado e gera results/validacao_modelo.md (+ .json).

Confere: metricas no teste limpo, matriz de confusao, concordancia ONNX x sklearn,
probabilidades coerentes (somam 1) e o contrato da API com o modelo real.
Uso: python scripts/validar_modelo.py   (rode antes: python scripts/treinar.py)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

from triagem.api.main import criar_app  # noqa: E402
from triagem.config import obter_config  # noqa: E402
from triagem.dados import carregar_treino_teste  # noqa: E402
from triagem.predictor import OnnxPredictor, SklearnPredictor  # noqa: E402

CLASSES = ["atencao", "normal", "urgente"]


def main() -> int:
    cfg = obter_config()
    _, teste = carregar_treino_teste(cfg.dados_treino, cfg.dados_teste)
    textos, reais = teste["texto"].tolist(), teste["urgencia"].tolist()
    sk, ox = SklearnPredictor(cfg), OnnxPredictor(cfg)
    res_sk = [sk.prever(t) for t in textos]
    res_ox = [ox.prever(t) for t in textos]
    prev_sk = [r.urgencia for r in res_sk]
    prev_ox = [r.urgencia for r in res_ox]

    concordancia = float(np.mean([a == b for a, b in zip(prev_sk, prev_ox, strict=True)]))
    somas = [sum(r.probabilidades.values()) for r in res_ox]
    baseline = max(pd_count / len(reais) for pd_count in (reais.count(c) for c in CLASSES))
    matriz = confusion_matrix(reais, prev_ox, labels=CLASSES)

    with TestClient(criar_app(OnnxPredictor(cfg))) as cliente:
        r = cliente.post("/classificar", json={"texto": textos[0]})
        contrato_ok = r.status_code == 200 and set(r.json()) >= {
            "urgencia",
            "confianca",
            "probabilidades",
            "modelo",
        }
        rejeita_curto = cliente.post("/classificar", json={"texto": "oi"}).status_code == 422

    verificacoes = {
        "acuracia_onnx_maior_que_baseline": accuracy_score(reais, prev_ox) > baseline,
        "concordancia_onnx_sklearn_>=_0.97": concordancia >= 0.97,
        "probabilidades_somam_1": bool(np.allclose(somas, 1.0, atol=1e-3)),
        "contrato_da_api_ok": bool(contrato_ok),
        "api_rejeita_texto_curto_422": bool(rejeita_curto),
    }
    resultado = {
        "n_teste": len(textos),
        "acuracia_sklearn": round(accuracy_score(reais, prev_sk), 4),
        "acuracia_onnx": round(accuracy_score(reais, prev_ox), 4),
        "f1_macro_onnx": round(f1_score(reais, prev_ox, average="macro"), 4),
        "baseline": round(baseline, 4),
        "concordancia_onnx_sklearn": round(concordancia, 4),
        "matriz_confusao": {"classes": CLASSES, "valores": matriz.tolist()},
        "verificacoes": verificacoes,
        "aprovado": all(verificacoes.values()),
    }
    cfg.dir_resultados.mkdir(parents=True, exist_ok=True)
    (cfg.dir_resultados / "validacao_modelo.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    linhas = [
        "# Relatorio de validacao do modelo",
        "",
        f"**Resultado geral: {'APROVADO' if resultado['aprovado'] else 'REPROVADO'}**",
        "",
        f"Teste limpo (sem vazamento): **{len(textos)}** laudos.",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| Acuracia sklearn | {resultado['acuracia_sklearn']:.1%} |",
        f"| Acuracia ONNX | {resultado['acuracia_onnx']:.1%} |",
        f"| F1 macro ONNX | {resultado['f1_macro_onnx']:.1%} |",
        f"| Baseline (classe majoritaria) | {baseline:.1%} |",
        f"| Concordancia ONNX x sklearn | {concordancia:.1%} |",
        "",
        "## Matriz de confusao (ONNX) - linhas: real, colunas: previsto",
        "",
        "| real \\ previsto | " + " | ".join(CLASSES) + " |",
        "|---|" + "---|" * len(CLASSES),
    ]
    for classe, linha in zip(CLASSES, matriz.tolist(), strict=True):
        linhas.append(f"| {classe} | " + " | ".join(str(v) for v in linha) + " |")
    linhas += ["", "## Verificacoes", "", "| Verificacao | Status |", "|---|---|"]
    for nome, ok in verificacoes.items():
        linhas.append(f"| {nome} | {'OK' if ok else 'FALHOU'} |")
    texto_md = "\n".join(linhas) + "\n"
    (cfg.dir_resultados / "validacao_modelo.md").write_text(texto_md, encoding="utf-8")
    print(texto_md)
    return 0 if resultado["aprovado"] else 1


if __name__ == "__main__":
    sys.exit(main())
