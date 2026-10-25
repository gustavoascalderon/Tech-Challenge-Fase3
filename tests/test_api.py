import pytest
from fastapi.testclient import TestClient

from triagem.api.main import criar_app

LAUDO = "paciente apresenta massa suspeita em exame de imagem, recomenda-se biopsia"


@pytest.fixture
def cliente(predictor_falso):
    with TestClient(criar_app(predictor_falso)) as c:
        yield c


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "modelo": "falso"}


def test_classificar_retorna_urgencia(cliente):
    r = cliente.post("/classificar", json={"texto": LAUDO})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["urgencia"] == "urgente"
    assert corpo["confianca"] == 0.9
    assert "latencia_inferencia_ms" in corpo


def test_texto_curto_gera_422(cliente):
    assert cliente.post("/classificar", json={"texto": "oi"}).status_code == 422


def test_falha_do_modelo_gera_500(cliente):
    r = cliente.post("/classificar", json={"texto": "vai explodir agora mesmo"})
    assert r.status_code == 500


def test_metricas_prometheus_expostas(cliente):
    cliente.post("/classificar", json={"texto": LAUDO})
    cliente.post("/classificar", json={"texto": "x"})  # erro 422
    corpo = cliente.get("/metrics").text
    assert "triagem_requisicoes_total" in corpo
    assert "triagem_latencia_requisicao_segundos_bucket" in corpo
    assert "triagem_erros_total" in corpo
    assert 'triagem_predicoes_total{urgencia="urgente"}' in corpo
