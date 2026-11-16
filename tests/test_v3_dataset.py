"""Modelo v3: alvo com antecedência sem vazamento, lacuna que se propaga, janelas disjuntas pelo contrato."""
import numpy as np
import pandas as pd
import pytest

from mapi_ai.v3 import config as C
from mapi_ai.v3 import dataset as D
from mapi_ai.v3.evaluate import episodes, prior_shift


def test_janelas_disjuntas_somam_o_acumulado_e_propagam_ausencia():
    df = pd.DataFrame({"current_rainfall": [2.0, 1.0, np.nan], "rainfall_3h_accumulated": [5.0, np.nan, 4.0],
                       "rainfall_6h_accumulated": [9.0, 9.0, 9.0], "rainfall_12h_accumulated": [9.0, 12.0, 12.0],
                       "rainfall_24h_accumulated": [20.0, 30.0, np.nan]})
    w = D.windows_from_accumulated(df)
    assert w.iloc[0].tolist() == [2.0, 3.0, 4.0, 0.0, 11.0]
    assert w.iloc[0].sum() == df.iloc[0]["rainfall_24h_accumulated"]
    # acumulado de 3 h ausente: as duas janelas que dependem dele ficam ausentes; as outras não
    assert np.isnan(w.iloc[1]["rain_1_3h"]) and np.isnan(w.iloc[1]["rain_3_6h"])
    assert w.iloc[1]["rain_0_1h"] == 1.0 and w.iloc[1]["rain_6_12h"] == 3.0
    # hora atual ausente: 0-1 h e 1-3 h ausentes, nunca zero
    assert np.isnan(w.iloc[2]["rain_0_1h"]) and np.isnan(w.iloc[2]["rain_1_3h"]) and np.isnan(w.iloc[2]["rain_12_24h"])


def test_acumulado_com_hora_ausente_fica_ausente():
    h = pd.Series([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])
    acc = D.accumulate(h)
    assert np.isnan(acc["rainfall_3h_accumulated"].iloc[3]) and np.isnan(acc["rainfall_3h_accumulated"].iloc[4])
    assert acc["rainfall_3h_accumulated"].iloc[5] == 3.0


def test_rotulo_com_antecedencia_olha_de_t_a_t_mais_h():
    y0 = pd.Series([0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    assert D.horizon_label(y0, 0).tolist() == y0.tolist()
    h3 = D.horizon_label(y0, 3)
    assert h3.iloc[:5].tolist() == [1, 1, 1, 1, 0]          # t=0 vê até t=3
    assert np.isnan(h3.iloc[-3:]).all()                     # fim da série: janela incompleta, sem rótulo


@pytest.fixture(scope="module")
def cont():
    if not C.CONTINUOUS.exists():
        pytest.skip("data/v3/continuous.csv.gz não existe: rode python -m mapi_ai.v3.dataset")
    return D.load("continuous")


def test_sem_vazamento_nenhuma_entrada_usa_chuva_depois_de_t(cont):
    """As entradas de t saem só de horas <= t: refazê-las com a série cortada em t dá o mesmo valor."""
    g = cont[cont.slug == cont.slug.iloc[0]].reset_index(drop=True)
    obs = g["current_rainfall"]
    for t in (500, 5000, 20000):
        acc = D.accumulate(obs.iloc[:t + 1])                 # só o passado e a hora t
        w = D.windows_from_accumulated(acc).iloc[-1]
        for name in C.WINDOWS:
            a, b = w[name], g.loc[t, name]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-9), (t, name)
    # e mexer no FUTURO não muda as entradas de t
    fut = obs.copy()
    fut.iloc[5001:] = 999.0
    w = D.windows_from_accumulated(D.accumulate(fut)).iloc[5000]
    for name in C.WINDOWS:
        a, b = w[name], g.loc[5000, name]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-9)


def test_lacuna_e_aplicada_depois_do_rotulo(cont):
    gap = cont[cont.gap_injected]
    assert len(gap) > 0
    assert gap["rain_0_1h"].isna().all()                    # o modelo não vê a chuva da hora com lacuna
    assert gap.loc[gap.labeled, "flood_h0"].notna().all()   # mas o rótulo existe: veio da chuva verdadeira
    assert gap["rain_0_1h_full"].notna().any()


def test_folga_do_split_e_sem_lat_lon(cont):
    assert "latitude" not in C.FEATURES and "longitude" not in C.FEATURES
    tr, va = cont[cont.part == "train"], cont[cont.part == "validation"]
    assert (va.timestamp.min() - tr.timestamp.max()) >= pd.Timedelta(hours=C.GUARD_H)


def test_episodios_e_correcao_de_prevalencia():
    assert episodes(np.array([1, 2, 3, 20, 21, 40]), 6) == [(1, 3), (20, 21), (40, 40)]
    assert prior_shift(0.5, 0.5, 0.01) == pytest.approx(0.01)
    assert prior_shift(0.2, 0.2, 0.2) == pytest.approx(0.2)
