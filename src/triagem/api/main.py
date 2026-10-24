"""API FastAPI de triagem de laudos, instrumentada com prometheus_client."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from triagem import __version__, metricas
from triagem.config import obter_config
from triagem.predictor import Predictor, carregar_predictor

ENDPOINTS_CONHECIDOS = {"/classificar", "/health", "/metrics", "/docs", "/openapi.json"}


class LaudoEntrada(BaseModel):
    texto: str = Field(..., min_length=10, max_length=20000, description="Texto do laudo")


class TriagemSaida(BaseModel):
    urgencia: str
    confianca: float
    probabilidades: dict[str, float]
    modelo: str
    latencia_inferencia_ms: float


def criar_app(predictor: Predictor | None = None) -> FastAPI:
    """Fabrica da aplicacao. `predictor` pode ser injetado (usado nos testes)."""

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = predictor or carregar_predictor(obter_config())
        yield

    app = FastAPI(
        title="Triagem automatica de laudos medicos",
        description="Classifica a urgencia (normal / atencao / urgente) de um texto de laudo.",
        version=__version__,
        lifespan=ciclo_de_vida,
    )

    @app.middleware("http")
    async def instrumentar(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Rotas desconhecidas viram "outros" para nao explodir a cardinalidade das metricas.
        endpoint = request.url.path if request.url.path in ENDPOINTS_CONHECIDOS else "outros"
        metricas.EM_ANDAMENTO.inc()
        inicio = time.perf_counter()
        status = 500
        try:
            resposta = await call_next(request)
            status = resposta.status_code
            return resposta
        finally:
            duracao = time.perf_counter() - inicio
            metricas.EM_ANDAMENTO.dec()
            if endpoint != "/metrics":
                metricas.REQUISICOES.labels(request.method, endpoint, str(status)).inc()
                metricas.LATENCIA_REQUISICAO.labels(endpoint).observe(duracao)
                if status >= 400:
                    metricas.ERROS.labels(endpoint, str(status)).inc()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "modelo": app.state.predictor.nome}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/classificar", response_model=TriagemSaida)
    def classificar(laudo: LaudoEntrada) -> TriagemSaida:
        pred: Predictor = app.state.predictor
        inicio = time.perf_counter()
        try:
            resultado = pred.prever(laudo.texto)
        except Exception as exc:  # noqa: BLE001 - qualquer falha do modelo vira HTTP 500
            raise HTTPException(status_code=500, detail="Falha na inferencia") from exc
        duracao = time.perf_counter() - inicio
        metricas.LATENCIA_INFERENCIA.observe(duracao)
        metricas.PREDICOES.labels(resultado.urgencia).inc()
        return TriagemSaida(
            urgencia=resultado.urgencia,
            confianca=resultado.confianca,
            probabilidades=resultado.probabilidades,
            modelo=pred.nome,
            latencia_inferencia_ms=round(duracao * 1000, 3),
        )

    return app


app = criar_app()
