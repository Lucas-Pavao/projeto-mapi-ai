"""Balde (M1) conforme docs/MODELAGEM-IA.md §2.1 e PRE-REGISTRO-AVALIACAO.md §3-§4.

Grade horária em UTC; cada hora é rotulada pelo fim do intervalo (ADR 0006). Só chuva entra: sem rótulos,
sem maré (M1b não roda, ADR 0009). Cotas só relativas.

    Q_in  = C * P * A / 1000                  (m3/h; P em mm/h, A em m2)
    q_p   = f * C * i_T * A / 1000            (m3/h)
    Q_out = min(q_p * g, V_prev + Q_in)       (g = 1 na versão M1a)
    V     = max(0, V_prev + Q_in - Q_out)
    z     = curva^-1(V)                       (m acima do fundo; satura em +2 m)
    h     = 1000 * max(0, z - (cota_rua - fundo))   (mm)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

FICHA_MM = 5.0


@dataclass(frozen=True)
class Params:
    C: float = 0.86
    iT_mm_h: float = 60.1      # i(TR=5, 60 min), Silva e Araujo (2013), Eq. 16
    f: float = 1.0
    hcrit_mm: float = 150.0    # 15 cm = 30 fichas
    snap_m: int = 10
    curve_scale: float = 1.0   # fator sobre o volume por cm (+-30%)

    def com(self, **kw) -> "Params":
        return replace(self, **kw)


@dataclass(frozen=True)
class Ponto:
    slug: str
    area_m2: dict            # {snap_m: area}
    z_curva_m: np.ndarray    # altura acima do fundo (m), crescente
    v_curva_m3: np.ndarray   # volume acumulado (m3), nao decrescente
    rua_rel_m: float         # cota_rua - fundo (m), so diferenca de cotas


def simular_ponto(P_mm_h: np.ndarray, pt: Ponto, p: Params):
    """Devolve (h_mm[t], n_fora_da_curva). P_mm_h em ordem cronologica, sem NaN."""
    P = np.asarray(P_mm_h, dtype=float)
    if np.isnan(P).any():
        raise ValueError("P com NaN: dia incompleto deve ser tratado antes (MODELAGEM §2.4)")
    A = pt.area_m2[p.snap_m]
    vc = pt.v_curva_m3 * p.curve_scale
    vmax = vc[-1]
    q = p.f * p.C * p.iT_mm_h * A / 1000.0
    qin = p.C * P * A / 1000.0
    n = len(P)
    h = np.zeros(n)
    V = 0.0
    fora = 0
    zmax = float(pt.z_curva_m[-1])
    for t in range(n):
        v_disp = V + qin[t]
        qout = q if q < v_disp else v_disp
        V = v_disp - qout
        if V <= 0.0:
            V = 0.0
            continue            # z = 0 < rua_rel: h = 0
        if V > vmax:
            fora += 1
            z = zmax
        else:
            z = float(np.interp(V, vc, pt.z_curva_m))
        d = z - pt.rua_rel_m
        if d > 0:
            h[t] = 1000.0 * d
    return h, fora
