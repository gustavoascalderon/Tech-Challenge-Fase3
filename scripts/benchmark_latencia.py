"""Compara a latencia do modelo ORIGINAL (sklearn) x OTIMIZADO (ONNX Runtime).

Mede a inferencia de 1 laudo por vez (cenario real de API), apos aquecimento.
Uso: python scripts/benchmark_latencia.py [--n 500]
Saida: results/latencia_modelo.json e results/latencia_modelo.md
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triagem.config import obter_config  # noqa: E402
from triagem.dados import carregar_treino_teste  # noqa: E402
from triagem.predictor import OnnxPredictor, SklearnPredictor  # noqa: E402


def medir(predictor, textos: list[str]) -> dict[str, float]:
    for t in textos[:30]:  # aquecimento
        predictor.prever(t)
    tempos = []
    for t in textos:
        ini = time.perf_counter()
        predictor.prever(t)
        tempos.append((time.perf_counter() - ini) * 1000)
    tempos.sort()
    p = lambda q: tempos[min(len(tempos) - 1, int(q * len(tempos)))]  # noqa: E731
    return {
        "media_ms": round(statistics.mean(tempos), 3),
        "p50_ms": round(p(0.50), 3),
        "p95_ms": round(p(0.95), 3),
        "p99_ms": round(p(0.99), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    cfg = obter_config()
    textos = (
        carregar_treino_teste(cfg.dados_treino, cfg.dados_teste)[1]["texto"].head(args.n).tolist()
    )

    original = medir(SklearnPredictor(cfg), textos)
    otimizado = medir(OnnxPredictor(cfg), textos)
    ganho = round(original["media_ms"] / otimizado["media_ms"], 1)
    resultado = {
        "n_amostras": len(textos),
        "sklearn_original": original,
        "onnx_otimizado": otimizado,
        "speedup_media": ganho,
    }
    cfg.dir_resultados.mkdir(parents=True, exist_ok=True)
    (cfg.dir_resultados / "latencia_modelo.json").write_text(
        json.dumps(resultado, indent=2), encoding="utf-8"
    )
    md = (
        "| Modelo | media (ms) | p50 (ms) | p95 (ms) | p99 (ms) |\n|---|---|---|---|---|\n"
        f"| sklearn (original) | {original['media_ms']} | {original['p50_ms']} "
        f"| {original['p95_ms']} | {original['p99_ms']} |\n"
        f"| ONNX Runtime (otimizado) | {otimizado['media_ms']} | {otimizado['p50_ms']} "
        f"| {otimizado['p95_ms']} | {otimizado['p99_ms']} |\n\n"
        f"Speedup medio: **{ganho}x** ({len(textos)} laudos, 1 por vez).\n"
    )
    (cfg.dir_resultados / "latencia_modelo.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
