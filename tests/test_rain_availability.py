"""Simulação do dado como chega (mapi_ai/rain/availability.py)."""
import numpy as np
import pandas as pd
import pytest

from mapi_ai.rain import availability as A
from mapi_ai.rain import config as C
from mapi_ai.rain import features as F

pytestmark = pytest.mark.skipif(not C.READINGS.exists(), reason="sem data/rain/ped_10min.csv.gz")


@pytest.fixture(scope="module")
def dados():
    from mapi_ai.rain import dataset as D
    r = D.load_readings()
    r = r[(r.ts > "2024-03-01") & (r.ts <= "2024-03-15")].reset_index(drop=True)
    return r, pd.read_csv(C.STATIONS)


def test_atraso_zero_reproduz_o_laboratorio(dados):
    r, st = dados
    times = pd.date_range("2024-03-03", "2024-03-14", freq="10min", tz="UTC")
    lab = F.features_at(r, st, -8.10791, -34.927138, times)
    sim = A.features_at_visible(r.assign(delay_min=0.0), st, -8.10791, -34.927138, times)
    pd.testing.assert_frame_equal(lab[F.BASE_FEATURES], sim[F.BASE_FEATURES], check_exact=False, atol=1e-9)


def test_atraso_so_esconde_o_que_ainda_nao_chegou(dados):
    r, st = dados
    times = pd.date_range("2024-03-03", "2024-03-14", freq="10min", tz="UTC")
    lab = F.features_at(r, st, -8.10791, -34.927138, times)
    sim = A.features_at_visible(r.assign(delay_min=25.0), st, -8.10791, -34.927138, times)
    # 25 min de atraso: o p_10 em t é o que o laboratório só via ... nunca maior que a chuva de 180 min do laboratório
    assert (sim.p_180.fillna(0) <= lab.p_180.fillna(0) + 1e-9).all()
    # a soma de 10 min com 25 min de atraso é zero quando há estação válida (nada do último passo chegou)
    assert (sim.p_10.dropna() == 0).all()


def test_sorteio_tem_semente_fixa():
    rd = pd.DataFrame({"station_id": ["A"] * 5 + ["B"] * 5, "mm": [0, 1, 0, 0, 2] * 2})
    de = pd.DataFrame({"canonical_id": ["A"] * 30, "wet": [0] * 25 + [1] * 5, "delay_min": np.arange(30.0)})
    a = A.sample_delays(rd, de, seed=42)
    b = A.sample_delays(rd, de, seed=42)
    np.testing.assert_array_equal(a, b)
    assert A.sample_delays(rd, de, scenario="lab").sum() == 0
