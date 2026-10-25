import pandas as pd
import pytest

from triagem.predictor import Resultado


class PredictorFalso:
    """Predictor de mentira: deixa os testes da API rapidos e independentes de modelo treinado."""

    nome = "falso"

    def prever(self, texto: str) -> Resultado:
        if "explodir" in texto:
            raise RuntimeError("falha simulada")
        return Resultado(
            urgencia="urgente",
            confianca=0.9,
            probabilidades={"atencao": 0.05, "normal": 0.05, "urgente": 0.9},
        )


@pytest.fixture
def predictor_falso() -> PredictorFalso:
    return PredictorFalso()


@pytest.fixture
def df_sintetico() -> pd.DataFrame:
    frases = {
        "urgente": "tumor maligno carcinoma metastasis chemotherapy infarction cardiac arrest",
        "atencao": "gastric ulcer colitis neurological seizure migraine epilepsy bowel",
        "normal": "general infection inflammation routine pathology healing wound",
    }
    linhas = [(f"{texto} paciente {i}", alvo) for alvo, texto in frases.items() for i in range(60)]
    return pd.DataFrame(linhas, columns=["texto", "urgencia"])
