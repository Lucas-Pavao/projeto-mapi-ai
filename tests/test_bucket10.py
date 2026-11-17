"""Balde M1a a 10 min (mapi_ai/bucket10.py) e POST /v1/bucket/point10."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from avaliacao.balde import Params, Ponto, simular_ponto
from mapi_ai import bucket10 as B
from mapi_ai import app as app_module

client = TestClient(app_module.app)


def terreno(area=1000.0, rua_rel=0.10):
    z = np.linspace(0, 2.0, 41)
    return B.Terreno(area, z, 100.0 * z, rua_rel)


def test_idf_reproduz_a_tabela_de_15_min_do_documento():
    # docs/PARAMETROS-MODELO-FISICO.md: 15 min, TR 2/5/10 = 23,4 / 28,0 / 32,0 mm
    for tr, mm in ((2, 23.4), (5, 28.0), (10, 32.0)):
        assert round(B.idf_silva_araujo(tr, 15) * 15 / 60, 1) == mm


def test_idf_em_10_min():
    assert round(B.idf_silva_araujo(5, 10), 1) == 125.4
    assert round(B.idf_silva_araujo(2, 10), 1) == 105.0
    assert round(B.idf_silva_araujo(10, 10), 1) == 143.4


@pytest.mark.parametrize("P_mm_h", [0.0, 30.0, 60.1, 80.0, 150.0, 400.0])
def test_chuva_constante_por_1h_da_a_mesma_lamina_que_o_balde_horario(P_mm_h):
    """Com a mesma capacidade (mm/h) e chuva constante por 1 h, o balde de 10 min e o horário dão a mesma lâmina."""
    t = terreno(area=2000.0, rua_rel=0.05)
    ph = Params()                                     # capacidade horária 60,1 mm/h
    pt = Ponto("S", {10: t.area_m2}, t.z_curva_m, t.v_curva_m3, t.rua_rel_m)
    h_hora, _ = simular_ponto(np.array([P_mm_h]), pt, ph)
    h10, _, _ = B.simular(np.full(6, P_mm_h / 6), t, B.Params10(), i_cap_mm_h=ph.iT_mm_h)
    assert h10[-1] == pytest.approx(h_hora[-1], abs=1e-9)


def test_sem_chuva_lamina_zero_e_projecao_sem_cruzar():
    out = B.avaliar(np.zeros(144), terreno(), B.Params10(), 0.0, 0.0)
    assert out["h_now_mm"] == 0.0
    assert out["projection"]["q50"]["minutes_to_hcrit"] is None
    assert out["version"] == "balde-M1a-10min"


def test_projecao_parte_do_volume_medido_e_q80_cruza_antes_ou_igual_ao_q50():
    t = terreno(area=5000.0, rua_rel=0.05)
    medido = np.r_[np.zeros(10), np.full(3, 30.0)]    # 30 mm em 10 min por 30 min (180 mm/h)
    out = B.avaliar(medido, t, B.Params10(), 40.0, 80.0)
    assert out["h_now_mm"] > 0
    m50, m80 = out["projection"]["q50"]["minutes_to_hcrit"], out["projection"]["q80"]["minutes_to_hcrit"]
    assert m80 is not None
    assert m50 is None or m80 <= m50


def test_serie_com_nan_e_recusada():
    with pytest.raises(ValueError):
        B.simular(np.array([1.0, np.nan]), terreno(), B.Params10())


def test_endpoint_devolve_lamina_projecao_e_versao():
    t = terreno()
    body = {"rain_10min_mm": [0.0] * 140 + [25.0, 25.0, 25.0, 25.0],
            "point": {"slug": "S", "area_m2": t.area_m2, "z_curva_m": t.z_curva_m.tolist(),
                      "v_curva_m3": t.v_curva_m3.tolist(), "rua_rel_m": t.rua_rel_m},
            "tr_anos": 5, "forecast_1h_q50_mm": 10.0, "forecast_1h_q80_mm": 20.0}
    r = client.post("/v1/bucket/point10", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["version"] == "balde-M1a-10min"
    assert len(j["projection"]["q50"]["h_mm"]) == 6
    assert j["validated"] is False
    assert j["i_cap_mm_h"] == pytest.approx(125.4, abs=0.05)


def test_endpoint_recusa_chuva_negativa_ou_nan():
    t = terreno()
    pt = {"slug": "S", "area_m2": t.area_m2, "z_curva_m": t.z_curva_m.tolist(), "v_curva_m3": t.v_curva_m3.tolist(),
          "rua_rel_m": t.rua_rel_m}
    assert client.post("/v1/bucket/point10", json={"rain_10min_mm": [-1.0], "point": pt}).status_code == 422
