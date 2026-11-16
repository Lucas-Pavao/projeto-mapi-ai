"""Modelo v3 no serviço: contrato antigo intacto, chuva ausente aceita e declarada, horizonte e probabilidade corrigida."""
import pytest
from fastapi.testclient import TestClient

from mapi_ai import app as app_module
from mapi_ai.v3 import config as C

PAYLOAD = {
    "station_id": "p1", "latitude": -8.05, "longitude": -34.9,
    "current_rainfall": 30.0, "rainfall_3h_accumulated": 60.0, "rainfall_6h_accumulated": 70.0,
    "rainfall_12h_accumulated": 80.0, "rainfall_24h_accumulated": 90.0,
    "tide_level": 2.2, "timestamp": "2026-05-10T15:00:00Z",
}
DRY = {**PAYLOAD, "current_rainfall": 0.0, "rainfall_3h_accumulated": 0.0, "rainfall_6h_accumulated": 0.0,
       "rainfall_12h_accumulated": 0.0, "rainfall_24h_accumulated": 0.0, "tide_level": 1.0}

pytestmark = pytest.mark.skipif(not (C.MODELS_DIR / "metadata.json").exists(), reason="treine o v3 antes")


@pytest.fixture
def client(monkeypatch):
    from mapi_ai.v3.serve import V3Model
    monkeypatch.setattr(app_module, "v3_model", V3Model())
    return TestClient(app_module.app)


def test_contrato_antigo_intacto_e_campos_novos_por_adicao(client):
    r = client.post("/v1/predict/flood", json=PAYLOAD)
    assert r.status_code == 200, r.text
    j = r.json()
    assert 0.0 <= j["flood_probability"] <= 1.0
    assert j["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert j["model_version"].endswith("_v3")
    assert j["horizon_hours"] == C.PRIMARY_HORIZON
    assert j["probability_kind"] == "PRIOR_CORRECTED"
    assert j["flood_probability"] < j["flood_probability_uncorrected"]      # prevalência real < sintética
    assert j["training_data"] == "SYNTHETIC" and "SINTÉTICO" in j["synthetic_warning"]
    assert j["imputed_features"] == []
    assert set(j["thresholds"]) == {"medium", "high", "source"}


def test_chuva_forte_tem_mais_risco_que_seco_e_explica_pela_chuva(client):
    wet, dry = client.post("/v1/predict/flood", json=PAYLOAD).json(), client.post("/v1/predict/flood", json=DRY).json()
    assert wet["flood_probability"] > dry["flood_probability"]
    top = wet["top_contributors"]
    assert len(top) == 3 and top[0]["feature_id"].startswith("rain_")
    assert top[0]["contribution"] > 0 and top[0]["unit"] == "mm"


def test_chuva_ausente_e_aceita_e_declarada_nunca_vira_zero(client):
    semi = {**PAYLOAD, "rainfall_24h_accumulated": None}
    r = client.post("/v1/predict/flood", json=semi)
    assert r.status_code == 200, r.text
    assert r.json()["imputed_features"] == ["rain_12_24h"]
    sem_hora = {**PAYLOAD, "current_rainfall": None}
    j = client.post("/v1/predict/flood", json=sem_hora).json()
    assert j["imputed_features"] == ["rain_0_1h", "rain_1_3h"]
    zero = client.post("/v1/predict/flood", json={**PAYLOAD, "current_rainfall": 0.0}).json()
    assert j["flood_probability"] != zero["flood_probability"]            # ausente não é tratado como 0 mm


def test_sem_mare_continua_422_com_missing_features(client):
    r = client.post("/v1/predict/flood", json={**PAYLOAD, "tide_level": None})
    assert r.status_code == 422
    assert r.json()["missing_features"] == ["tide_level"]


def test_limiares_e_risco_usam_a_probabilidade_servida(client):
    j = client.post("/v1/predict/flood", json=PAYLOAD).json()
    t = j["thresholds"]
    expected = "HIGH" if j["flood_probability"] >= t["high"] else "MEDIUM" if j["flood_probability"] >= t["medium"] else "LOW"
    assert j["risk_level"] == expected


def test_contribuicoes_somam_a_diferenca_e_contam_o_indicador_de_ausencia(client):
    from mapi_ai.v3.serve import features_from_payload
    m = app_module.v3_model
    X = features_from_payload({**PAYLOAD, "current_rainfall": None}, m.features)
    c = m.contributions(X)
    assert list(c.index) == m.features
    import numpy as np
    assert c.sum() == pytest.approx(float(m.served(X)[0]) - float(np.mean(m.served(m.background))), abs=1e-9)
    assert c["rain_0_1h"] != 0.0     # a ausência da chuva da hora pesa (indicador somado à janela)
