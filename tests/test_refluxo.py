"""Refluxo de maré no balde a 10 min (ADR 0009 emenda 3)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mapi_ai import app as app_module
from mapi_ai import bucket10 as B

client = TestClient(app_module.app)


def test_fator_refluxo():
    f = B.fator_refluxo(np.array([0.5, 1.46, 1.76, 2.06, 2.5]), 1.46, 0.6)
    assert f.tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])


def test_sem_mare_igual_ao_de_sempre():
    z = np.linspace(0, 2.0, 41)
    t = B.Terreno(1000.0, z, 100.0 * z, 0.10)
    P = np.array([15.0] * 6)
    h0 = B.simular(P, t, B.Params10())[0]
    h1 = B.simular(P, t, B.Params10(), f_passo=np.ones(6))[0]
    assert h0.tolist() == h1.tolist()


def test_preamar_bloqueia_a_galeria_e_a_lamina_sobe():
    z = list(np.linspace(0, 2.0, 41))
    base = {"rain_10min_mm": [15.0] * 6, "point": {"slug": "X", "area_m2": 1000.0, "z_curva_m": z,
            "v_curva_m3": [100.0 * v for v in z], "rua_rel_m": 0.10}}
    baixa = client.post("/v1/bucket/point10", json={**base, "tide_nr_m": [0.4] * 12, "outlet_sill_nr_m": 1.46,
                                                     "outlet_diameter_m": 0.6}).json()
    alta = client.post("/v1/bucket/point10", json={**base, "tide_nr_m": [2.3] * 12, "outlet_sill_nr_m": 1.46,
                                                    "outlet_diameter_m": 0.6}).json()
    sem = client.post("/v1/bucket/point10", json=base).json()
    assert sem["refluxo"] is None
    assert baixa["h_now_mm"] == pytest.approx(sem["h_now_mm"])
    assert alta["refluxo"]["fator_agora"] == 0.0
    assert alta["h_now_mm"] > baixa["h_now_mm"]


def test_refluxo_incompleto_e_recusado():
    z = list(np.linspace(0, 2.0, 41))
    r = client.post("/v1/bucket/point10", json={"rain_10min_mm": [1.0] * 6, "point": {"slug": "X", "area_m2": 1.0,
                    "z_curva_m": z, "v_curva_m3": z, "rua_rel_m": 0.1}, "tide_nr_m": [1.0] * 12})
    assert r.status_code == 422
