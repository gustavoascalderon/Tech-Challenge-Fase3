"""Gera trafego (incluindo erros propositais) para popular o dashboard do Grafana.

Uso: python scripts/gerar_trafego.py --url http://localhost:8000 --segundos 90
"""

import argparse
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triagem.config import obter_config  # noqa: E402
from triagem.dados import carregar_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--segundos", type=int, default=90)
    ap.add_argument("--taxa-erro", type=float, default=0.08, help="fracao de requisicoes invalidas")
    args = ap.parse_args()
    textos = carregar_dataset(obter_config().dados_teste)["texto"].head(500).tolist()
    fim = time.time() + args.segundos
    total = erros = 0
    with httpx.Client(base_url=args.url, timeout=10) as cliente:
        while time.time() < fim:
            if random.random() < args.taxa_erro:
                r = cliente.post("/classificar", json={"texto": "curto"})  # 422 proposital
            else:
                r = cliente.post("/classificar", json={"texto": random.choice(textos)})
            total += 1
            erros += r.status_code >= 400
            time.sleep(random.uniform(0.005, 0.05))
    print(f"{total} requisicoes enviadas ({erros} com erro proposital)")


if __name__ == "__main__":
    main()
