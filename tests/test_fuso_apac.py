"""Teste de fuso da APAC (onda 16, Fase 7.1) em dado sintético: hora local lida como UTC aparece 3 h adiantada."""
import numpy as np
import pandas as pd

from mapi_ai.rain import fuso_apac as FA


def _ped(seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-04-01", "2026-04-20", freq="10min", tz="UTC")
    mm = np.zeros(len(t))
    for k in (300, 900, 1500, 2100):   # 4 eventos
        mm[k:k + 12] = rng.gamma(2.0, 2.0, 12)
    return pd.Series(mm, index=t)


def test_hora_local_e_detectada_como_utc_menos_3():
    ped = _ped()
    apac = ped.copy()
    apac.index = apac.index - pd.Timedelta(hours=3)     # APAC em hora local, lida como se fosse UTC
    out = FA.testa(apac, ped)
    assert out["n_eventos"] == 4 and out["defasagem_moda_min"] == 180
    assert out["conclusao"].startswith("FUSO CONFIRMADO") and "UTC-3" in out["conclusao"]


def test_sem_chuva_e_inconclusivo():
    t = pd.date_range("2026-11-09", "2026-11-23", freq="10min", tz="UTC")
    z = pd.Series(0.0, index=t)
    assert FA.testa(z, z)["conclusao"].startswith("INCONCLUSIVO")
