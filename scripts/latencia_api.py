"""Mede a latencia HTTP ponta a ponta da API (baseline local da Etapa 1).

Uso: python scripts/latencia_api.py --url http://localhost:8000 --n 200
Rode uma vez com MODEL_BACKEND=sklearn (baseline) e outra com onnx (otimizado).
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triagem.config import obter_config  # noqa: E402
from triagem.dados import carregar_treino_teste  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    textos = (
        carregar_treino_teste(obter_config().dados_treino, obter_config().dados_teste)[1]["texto"]
        .head(args.n)
        .tolist()
    )
    tempos: list[float] = []
    with httpx.Client(base_url=args.url, timeout=10) as cliente:
        modelo = cliente.get("/health").json().get("modelo")
        for t in textos[:20]:
            cliente.post("/classificar", json={"texto": t})
        for t in textos:
            ini = time.perf_counter()
            r = cliente.post("/classificar", json={"texto": t})
            r.raise_for_status()
            tempos.append((time.perf_counter() - ini) * 1000)
    tempos.sort()
    print(f"backend={modelo} n={len(tempos)}")
    p50 = tempos[len(tempos) // 2]
    p95 = tempos[int(0.95 * len(tempos))]
    p99 = tempos[int(0.99 * len(tempos))]
    print(f"media={statistics.mean(tempos):.2f} ms  p50={p50:.2f}  p95={p95:.2f}  p99={p99:.2f}")


if __name__ == "__main__":
    main()
