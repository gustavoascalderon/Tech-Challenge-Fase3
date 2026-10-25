"""Validacao consolidada da revisao do projeto: roda as checagens e grava um unico JSON.

Reune testes, lint, auditorias, validacao do modelo, arquivos de configuracao, README e a
comparacao ANTES x DEPOIS. Itens que dependem de ferramentas ausentes ficam "nao_executado"
(nao reprovam). Uso: python scripts/validar_revisao.py
Pre-requisito: rodar antes treinar, auditar_dados, validar_modelo, auditar_artefatos e
avaliar_modelo_original (veja o README).
"""

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triagem.config import RAIZ, obter_config  # noqa: E402

OK, FALHOU, NAO_EXECUTADO = "ok", "falhou", "nao_executado"


def rodar(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, check=False)
    linhas = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode, (linhas[-1] if linhas else "")


def ler_json(caminho: Path) -> dict | None:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def status(cond: bool) -> str:
    return OK if cond else FALHOU


def checar_yaml() -> str:
    try:
        import yaml
    except ImportError:
        return NAO_EXECUTADO
    arquivos = [
        ".github/workflows/ci.yml",
        "docker-compose.yml",
        "docker-compose.airflow.yml",
        "monitoring/prometheus.yml",
        "monitoring/grafana/provisioning/datasources/prometheus.yml",
        "monitoring/grafana/provisioning/dashboards/dashboards.yml",
    ]
    try:
        for a in arquivos:
            yaml.safe_load((RAIZ / a).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return FALHOU
    return OK


def checar_dashboard() -> tuple[str, int]:
    caminho = RAIZ / "monitoring/grafana/dashboards/triagem.json"
    if not caminho.exists():
        return FALHOU, 0
    paineis = json.loads(caminho.read_text(encoding="utf-8")).get("panels", [])
    return status(len(paineis) >= 3), len(paineis)


def checar_dag() -> str:
    caminho = RAIZ / "dags/treino_triagem_dag.py"
    if not caminho.exists():
        return FALHOU
    codigo = caminho.read_text(encoding="utf-8")
    ast.parse(codigo)
    tarefas = all(t in codigo for t in ("carregar_dados", "treinar_modelo", "exportar_onnx"))
    return status(tarefas and "carregar >> treinar >> exportar" in codigo)


def referencias_do_readme() -> list[str]:
    """Caminhos de arquivos citados no README que NAO existem no projeto."""
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    citados = set(re.findall(r"`([\w./\-]+\.(?:py|yml|yaml|json|md|txt))`", texto))
    citados |= set(re.findall(r"\]\(((?!http)[\w./\-]+)\)", texto))
    ignorados = {"modelo.onnx", "modelo.joblib"}
    return sorted(c for c in citados if c not in ignorados and not (RAIZ / c).exists() and "/" in c)


def main() -> int:
    cfg = obter_config()
    res = cfg.dir_resultados
    verificacoes: dict[str, object] = {}

    codigo, resumo = rodar([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    verificacoes["pytest"] = {"status": status(codigo == 0), "resumo": resumo}

    if shutil.which("ruff"):
        codigo, resumo = rodar(["ruff", "check", "."])
        verificacoes["ruff"] = {"status": status(codigo == 0), "resumo": resumo}
    else:
        verificacoes["ruff"] = {"status": NAO_EXECUTADO, "resumo": "ruff nao instalado"}

    for chave, arquivo in [
        ("auditoria_dados", "auditoria_dados.json"),
        ("validacao_modelo", "validacao_modelo.json"),
        ("auditoria_artefatos", "auditoria_artefatos_recebidos.json"),
    ]:
        dado = ler_json(res / arquivo)
        verificacoes[chave] = {
            "status": status(bool(dado and dado.get("aprovado"))),
            "arquivo": f"results/{arquivo}",
        }

    verificacoes["yaml_validos"] = {"status": checar_yaml()}
    st_dash, n_paineis = checar_dashboard()
    verificacoes["dashboard_grafana"] = {"status": st_dash, "paineis": n_paineis}
    verificacoes["dag_airflow"] = {"status": checar_dag()}
    faltando = referencias_do_readme()
    verificacoes["readme_referencias"] = {
        "status": status(not faltando),
        "arquivos_ausentes": faltando,
    }

    original = ler_json(res / "metricas_modelo_original.json")
    revisado = ler_json(cfg.metricas_json)
    comparacao = None
    if original and revisado:
        no_limpo = original["avaliacao_no_teste_limpo"]
        comparacao = {
            "original_no_teste_original": original["avaliacao_no_teste_original"],
            "original_no_teste_limpo": no_limpo,
            "revisado_no_teste_limpo": {
                "acuracia": revisado["acuracia"],
                "f1_macro": revisado["f1_macro"],
                "n_teste": revisado["n_teste"],
            },
            "textos_iguais_em_treino_e_teste_no_original": original[
                "textos_iguais_em_treino_e_teste"
            ],
        }

    obrigatorios = [v["status"] for v in verificacoes.values()]
    aprovado = FALHOU not in obrigatorios
    resultado = {
        "verificacoes": verificacoes,
        "comparacao_antes_depois": comparacao,
        "aprovado": aprovado,
    }
    res.mkdir(parents=True, exist_ok=True)
    (res / "validacao_revisao.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for nome, v in verificacoes.items():
        print(f"{v['status']:>14}  {nome}")
    print("APROVADO" if aprovado else "REPROVADO")
    return 0 if aprovado else 1


if __name__ == "__main__":
    sys.exit(main())
