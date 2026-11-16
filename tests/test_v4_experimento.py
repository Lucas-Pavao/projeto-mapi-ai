"""v4: rótulo do balde sem zero inventado; a chuva futura verdadeira só existe no braço do teto."""
import numpy as np
import pytest

from mapi_ai.v4 import config as C

pytestmark = pytest.mark.skipif(not (C.TERRAIN_POINT.exists() and C.OOF.exists()), reason="sem terreno da F1 ou previsão fora da amostra")


@pytest.fixture(scope="module")
def df():
    from mapi_ai.v4.experiment import build
    return build(C.F_CENTRAL)


def test_so_o_braco_do_teto_ve_a_chuva_futura():
    assert not set(C.FUTURE) & set(C.ARMS["sem_previsao"])
    assert not set(C.FUTURE) & set(C.ARMS["com_previsao"])
    assert set(C.FUTURE) <= set(C.ARMS["chuva_futura_verdadeira"])
    assert set(C.ARMS["com_previsao"]) - set(C.ARMS["sem_previsao"]) == set(C.RAIN_FC)


def test_chuva_futura_e_a_da_hora_seguinte_e_as_janelas_olham_para_tras(df):
    g = df[df.slug == C.POINTS[0]].reset_index(drop=True)
    i = int(np.flatnonzero((g.rain > 1).to_numpy() & g.rain.shift(-1).notna().to_numpy())[5])
    assert g.fut_1h[i] == pytest.approx(g.rain[i + 1])
    assert g.rain_0_1h[i] == pytest.approx(g.rain[i])


def test_hora_sem_chuva_medida_fica_sem_rotulo_e_nao_vira_seco(df):
    g = df[df.slug == "AV_RECIFE_IBURA"].reset_index(drop=True)
    gap = np.flatnonzero(g.rain.isna().to_numpy())
    assert len(gap) > 0
    assert g.flood_h0[gap].isna().all()
    assert np.isnan(g.flood_h0[gap[0] + C.GAP_MASK_H])            # e as 6 h seguintes: estado do balde desconhecido


def test_jaboatao_fica_fora(df):
    assert set(df.slug) == set(C.POINTS) and "JABOATAO_CENTRO" not in set(df.slug)
