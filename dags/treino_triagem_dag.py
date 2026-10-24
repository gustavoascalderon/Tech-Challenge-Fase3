"""DAG do Airflow: carregamento de dados -> treino -> exportacao ONNX.

Executa semanalmente (retreino) e pode ser disparada manualmente na interface.
As funcoes vivem em `triagem.pipeline_treino` (mesmo codigo dos scripts).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Permite importar o pacote `triagem` dentro do container do Airflow.
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from triagem.pipeline_treino import (  # noqa: E402
    etapa_carregar_dados,
    etapa_exportar_onnx,
    etapa_treinar,
)

argumentos_padrao = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="treino_triagem_laudos",
    description="Pipeline de treino/retreino do classificador de urgencia de laudos",
    default_args=argumentos_padrao,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["triagem", "ml", "fiap"],
) as dag:
    carregar = PythonOperator(task_id="carregar_dados", python_callable=etapa_carregar_dados)
    treinar = PythonOperator(task_id="treinar_modelo", python_callable=etapa_treinar)
    exportar = PythonOperator(task_id="exportar_onnx", python_callable=etapa_exportar_onnx)

    carregar >> treinar >> exportar
