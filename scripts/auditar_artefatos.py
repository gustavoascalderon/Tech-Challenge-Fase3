"""Audita os artefatos gerados/recebidos: modelos, dados e versoes de bibliotecas.

Confere existencia, tamanho e hash (SHA-256), se o modelo carrega, se o ONNX tem a assinatura
esperada, se as classes batem e se as metricas salvas conferem com uma reavaliacao.
Uso: python scripts/auditar_artefatos.py   (rode antes: python scripts/treinar.py)
Saida: results/auditoria_artefatos_recebidos.json
"""

import hashlib
import json
import re
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import onnxruntime as ort  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from triagem.config import RAIZ, obter_config  # noqa: E402
from triagem.dados import carregar_treino_teste  # noqa: E402


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def descrever(caminho: Path) -> dict[str, object]:
    if not caminho.exists():
        return {"arquivo": str(caminho.relative_to(RAIZ)), "existe": False}
    return {
        "arquivo": str(caminho.relative_to(RAIZ)),
        "existe": True,
        "tamanho_bytes": caminho.stat().st_size,
        "sha256": sha256(caminho),
    }


def versoes_fixadas() -> list[dict[str, object]]:
    """Compara versoes instaladas com as fixadas (==) em requirements.txt."""
    saida = []
    for linha in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([\w.]+)", linha.strip())
        if not m:
            continue
        pacote, esperada = m.groups()
        try:
            instalada = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            instalada = None
        saida.append(
            {
                "pacote": pacote,
                "fixada": esperada,
                "instalada": instalada,
                "ok": instalada == esperada,
            }
        )
    return saida


def main() -> int:
    cfg = obter_config()
    arquivos = [cfg.modelo_joblib, cfg.modelo_onnx, cfg.classes_json, cfg.metricas_json]
    arquivos += sorted(cfg.dados_treino.parent.glob("*.csv"))
    inventario = [descrever(a) for a in arquivos]
    verificacoes: dict[str, bool] = {
        "artefatos_do_modelo_existem": all(descrever(a)["existe"] for a in arquivos[:4]),
    }
    detalhes: dict[str, object] = {}

    if verificacoes["artefatos_do_modelo_existem"]:
        pipeline = joblib.load(cfg.modelo_joblib)
        classes_pipeline = [str(c) for c in pipeline.named_steps["clf"].classes_]
        classes_json = json.loads(cfg.classes_json.read_text(encoding="utf-8"))
        verificacoes["classes_json_confere_com_modelo"] = classes_pipeline == classes_json

        sessao = ort.InferenceSession(str(cfg.modelo_onnx), providers=["CPUExecutionProvider"])
        entradas = [(i.name, i.type) for i in sessao.get_inputs()]
        saidas = [o.name for o in sessao.get_outputs()]
        detalhes["onnx_entradas"] = entradas
        detalhes["onnx_saidas"] = saidas
        verificacoes["onnx_assinatura_esperada"] = (
            entradas[0][0] == "texto" and entradas[0][1] == "tensor(string)" and len(saidas) == 2
        )

        _, teste = carregar_treino_teste(cfg.dados_treino, cfg.dados_teste)
        acuracia = float(accuracy_score(teste["urgencia"], pipeline.predict(teste["texto"])))
        salvas = json.loads(cfg.metricas_json.read_text(encoding="utf-8"))
        detalhes["acuracia_reavaliada"] = round(acuracia, 4)
        detalhes["acuracia_salva"] = salvas["acuracia"]
        verificacoes["metricas_salvas_conferem"] = abs(acuracia - salvas["acuracia"]) < 0.002

    versoes = versoes_fixadas()
    verificacoes["versoes_fixadas_conferem"] = all(v["ok"] for v in versoes)

    resultado = {
        "inventario": inventario,
        "versoes": versoes,
        "detalhes": detalhes,
        "verificacoes": verificacoes,
        "aprovado": all(verificacoes.values()),
    }
    cfg.dir_resultados.mkdir(parents=True, exist_ok=True)
    destino = cfg.dir_resultados / "auditoria_artefatos_recebidos.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verificacoes": verificacoes, "aprovado": resultado["aprovado"]}, indent=2))
    return 0 if resultado["aprovado"] else 1


if __name__ == "__main__":
    sys.exit(main())
