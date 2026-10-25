"""Valida a estrutura da DAG sem precisar do Airflow instalado (analise estatica do arquivo)."""

import ast
from pathlib import Path

DAG = Path(__file__).resolve().parents[1] / "dags" / "treino_triagem_dag.py"


def test_dag_define_as_tres_tarefas_na_ordem():
    codigo = DAG.read_text(encoding="utf-8")
    ast.parse(codigo)  # sintaxe valida
    for task_id in ("carregar_dados", "treinar_modelo", "exportar_onnx"):
        assert f'task_id="{task_id}"' in codigo
    assert "carregar >> treinar >> exportar" in codigo
