"""POST /v1/simulate/bucket (ADR 0014): regressão contra simular_ponto, faixas (422), t_cross_h e hmax.

Entradas reais: AVALIACAO_F1_DIR (padrão ../avaliacao-f1, como em avaliacao/tests). Sem elas, os casos reais são
pulados e só os sintéticos rodam.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from avaliacao.balde import Params, Ponto, simular_ponto
from avaliacao.entradas import PONTOS, carregar_chuva, carregar_terreno
from mapi_ai import app as app_module

def _base():
    if os.environ.get("AVALIACAO_F1_DIR"):
        return Path(os.environ["AVALIACAO_F1_DIR"])
    # irmã do repositório (também funciona num git worktree dentro de .claude/worktrees)
    for d in Path(__file__).resolve().parents:
        if (d / "avaliacao-f1").is_dir():
            return d / "avaliacao-f1"
    return Path("avaliacao-f1")


BASE = _base()
ENT = BASE / "entradas"
client = TestClient(app_module.app)


def ponto_sintetico(area=1000.0, rua_rel=0.10):
    z = np.linspace(0, 2.0, 41)
    return Ponto("SINT", {10: area}, z, 100.0 * z, rua_rel)


def corpo(P, pt, **params):
    return {
        "P_mm_h": [float(x) for x in P],
        "point": {"slug": pt.slug, "area_m2": pt.area_m2[Params().snap_m], "z_curva_m": pt.z_curva_m.tolist(),
                  "v_curva_m3": pt.v_curva_m3.tolist(), "rua_rel_m": pt.rua_rel_m},
        "params": params,
    }


def valido():
    return corpo(np.full(6, 80.0), ponto_sintetico())


# (a) regressão ------------------------------------------------------------------------------------------------

def _confere(P, pt, **params):
    r = client.post("/v1/simulate/bucket", json=corpo(P, pt, **params))
    assert r.status_code == 200, r.text
    j = r.json()
    h, fora = simular_ponto(np.asarray(P, float), pt, Params().com(**params))
    assert np.array_equal(np.array(j["h_mm"]), h)
    assert j["n_fora_da_curva"] == fora
    assert j["model"] == "M1a" and j["illustrative"] is True
    return j


@pytest.mark.parametrize("f,cs", [(1.0, 1.0), (0.6, 1.0), (1.0, 0.7), (0.2, 1.3)])
def test_regressao_sintetica(f, cs):
    P = np.r_[np.zeros(3), np.full(10, 120.0), np.zeros(5), np.full(60, 30.0)][:72]
    _confere(P, ponto_sintetico(), f=f, curve_scale=cs)


@pytest.mark.skipif(not (ENT / "curva_cota_volume.csv").exists(), reason=f"entradas reais ausentes em {ENT}")
@pytest.mark.parametrize("slug", PONTOS)
def test_regressao_entradas_reais(slug):
    chuva = carregar_chuva(ENT / "chuva_open_meteo_archive.csv")[slug]
    pts = carregar_terreno(ENT / "ponto_terreno.csv", ENT / "curva_cota_volume.csv")
    # três janelas de 72 h: as de maior chuva acumulada, mais uma fixa (início da série)
    v = chuva.to_numpy(float)
    ok = ~np.isnan(v)
    jan = [np.nan_to_num(v[i:i + 72]) for i in (0, len(v) // 2)]
    s = np.convolve(np.nan_to_num(v), np.ones(72), "valid")
    jan.append(np.nan_to_num(v[int(s.argmax()):int(s.argmax()) + 72]))
    assert ok.any()
    for P in jan:
        _confere(P, pts[slug])
        _confere(P, pts[slug], f=0.6, curve_scale=1.3)


# (b) 422 ------------------------------------------------------------------------------------------------------

def _muda(fn):
    b = valido()
    fn(b)
    return client.post("/v1/simulate/bucket", json=b).status_code


@pytest.mark.parametrize("fn", [
    lambda b: b.update(P_mm_h=[]),
    lambda b: b.update(P_mm_h=[1.0] * 73),
    lambda b: b.update(P_mm_h=[-1.0]),
    lambda b: b["params"].update(f=0.5),
    lambda b: b["params"].update(f=1.2),
    lambda b: b["params"].update(f=-0.2),
    lambda b: b["params"].update(curve_scale=0.69),
    lambda b: b["params"].update(curve_scale=1.31),
    lambda b: b["params"].update(C=0.5),
    lambda b: b["params"].update(iT_mm_h=10),
    lambda b: b["params"].update(hcrit_mm=10),
    lambda b: b["params"].update(snap_m=5),
    lambda b: b["point"].update(v_curva_m3=b["point"]["v_curva_m3"][:-1]),
    lambda b: b["point"].update(z_curva_m=b["point"]["z_curva_m"][::-1]),
    lambda b: b["point"].update(v_curva_m3=b["point"]["v_curva_m3"][::-1]),
    lambda b: b["point"].update(area_m2=0),
    lambda b: b["point"].pop("rua_rel_m"),
], ids=range(17))
def test_422(fn):
    assert _muda(fn) == 422


def test_422_nan_no_json():
    b = valido()
    txt = __import__("json").dumps(b).replace("80.0", "NaN", 1)
    r = client.post("/v1/simulate/bucket", content=txt, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_faixas_limite_aceitas():
    for f in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        assert _muda(lambda b: b["params"].update(f=f)) == 200
    for cs in (0.7, 1.3):
        assert _muda(lambda b: b["params"].update(curve_scale=cs)) == 200
    assert _muda(lambda b: b.update(P_mm_h=[0.0] * 72)) == 200


# (c) t_cross_h, hmax e params_usados --------------------------------------------------------------------------

def test_tcross_hmax_e_params():
    # 1000 m2, C=0,86: Q_in = 0,86*P m3/h; q_p = 51,686 m3/h; V = 100*z. P=200 -> excesso 120,314 m3/h -> z=1,203 m/h
    P = [0, 0, 200, 200, 0, 0]
    j = client.post("/v1/simulate/bucket", json=corpo(P, ponto_sintetico())).json()
    h = np.array(j["h_mm"])
    assert h[0] == h[1] == 0
    assert j["t_cross_h"] == int(np.flatnonzero(h >= 150.0)[0]) == 2
    assert j["hmax_mm"] == h.max() == max(h)
    assert j["params_usados"] == {"C": 0.86, "iT_mm_h": 60.1, "f": 1.0, "hcrit_mm": 150.0, "snap_m": 10,
                                  "curve_scale": 1.0}


def test_tcross_null_sem_cruzar():
    j = client.post("/v1/simulate/bucket", json=corpo(np.zeros(5), ponto_sintetico())).json()
    assert j["t_cross_h"] is None and j["hmax_mm"] == 0.0


def test_tcross_tardio():
    # chuva moderada cruza só no 2º/3º passo
    P = [70] * 12
    j = client.post("/v1/simulate/bucket", json=corpo(P, ponto_sintetico())).json()
    h = np.array(j["h_mm"])
    esperado = next((i for i, x in enumerate(h) if x >= 150.0), None)
    assert j["t_cross_h"] == esperado and esperado not in (None, 0)
