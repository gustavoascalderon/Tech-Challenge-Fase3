"""Audita a qualidade dos dados e gera o relatorio em results/.

Uso: python scripts/auditar_dados.py
Saidas: results/auditoria_dados.json e results/auditoria_dados.md
Codigo de saida 1 se alguma verificacao falhar (serve tambem como gate no CI/Airflow).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triagem.auditoria import auditar, gerar_markdown  # noqa: E402
from triagem.config import obter_config  # noqa: E402


def main() -> int:
    cfg = obter_config()
    resultado = auditar(cfg)
    cfg.dir_resultados.mkdir(parents=True, exist_ok=True)
    (cfg.dir_resultados / "auditoria_dados.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    relatorio = gerar_markdown(resultado)
    (cfg.dir_resultados / "auditoria_dados.md").write_text(relatorio, encoding="utf-8")
    print(relatorio)
    return 0 if resultado["aprovado"] else 1


if __name__ == "__main__":
    sys.exit(main())
