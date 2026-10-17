"""Treina o modelo e exporta para ONNX.  Uso: python scripts/treinar.py"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triagem.pipeline_treino import executar_tudo  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(executar_tudo(), indent=2, ensure_ascii=False))
