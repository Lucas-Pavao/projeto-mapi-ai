"""
Balde M1a a 10 min (balde-M1a-10min), para o nível ALAGAMENTO_PROVAVEL do alerta (prompt do alerta de 20 min, item 1.4).

Mesma estrutura de avaliacao/balde.py (ADR 0014, congelado na avaliação F1), mudando só o passo e a duração da
capacidade. Por passo de dt minutos:

    Q_in  = C * P * A / 1000                       (m3 no passo; P em mm no passo, A em m2)
    q_p   = C * i(TR, Te) * A / 1000 * dt / 60     (m3 no passo; i em mm/h da IDF na duração Te)
    Q_out = min(q_p, V_prev + Q_in)
    V     = max(0, V_prev + Q_in - Q_out)
    z     = curva^-1(V)                            (m acima do fundo; satura no topo da curva)
    h     = 1000 * max(0, z - (cota_rua - fundo))  (mm)

A capacidade é a mesma do nível ALERTA: intensidade da IDF de Silva e Araújo (2013, Eq. 16) na duração do tempo de
entrada Te (item 10 de docs/PARAMETROS-MODELO-FISICO.md, central 10 min), TR configurável (padrão 5 anos).
Nenhum parâmetro é ajustado para acertar frequência de alagamento. Sem estado escondido: cada chamada roda a série
inteira que recebe.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

VERSION = "balde-M1a-10min"


def idf_silva_araujo(tr_anos: float, t_min: float) -> float:
    """i em mm/h. Silva e Araújo (2013), Eq. 16: I = 1380,2176 TR^0,19369 / (t + 22)^0,78201."""
    return 1380.2176 * tr_anos ** 0.19369 / (t_min + 22.0) ** 0.78201


@dataclass(frozen=True)
class Params10:
    C: float = 0.86
    tr_anos: float = 5.0
    te_min: float = 10.0
    dt_min: float = 10.0
    hcrit_mm: float = 150.0
    curve_scale: float = 1.0

    @property
    def i_cap_mm_h(self) -> float:
        return idf_silva_araujo(self.tr_anos, self.te_min)

    def com(self, **kw) -> "Params10":
        return replace(self, **kw)


@dataclass(frozen=True)
class Terreno:
    area_m2: float
    z_curva_m: np.ndarray    # altura acima do fundo (m), crescente
    v_curva_m3: np.ndarray   # volume acumulado (m3), não decrescente
    rua_rel_m: float         # cota da rua - fundo (m)


def fator_refluxo(mare_nr_m: np.ndarray, soleira_nr_m: float, diametro_m: float) -> np.ndarray:
    """
    Refluxo de maré (variante M1b, ADR 0009 emenda 3): fração da capacidade da galeria livre por passo.
    1 com a maré abaixo da soleira da saída; 0 com o tubo todo submerso (maré >= soleira + D); linear no meio.
    Simplificação declarada: ignora a carga hidráulica entre a rua e o canal.
    """
    m = np.asarray(mare_nr_m, dtype=float)
    if diametro_m <= 0:
        raise ValueError("diametro_m deve ser > 0")
    return np.clip((soleira_nr_m + diametro_m - m) / diametro_m, 0.0, 1.0)


def simular(P_mm_passo: np.ndarray, t: Terreno, p: Params10, i_cap_mm_h: float | None = None, V0: float = 0.0,
            f_passo: np.ndarray | None = None):
    """
    Devolve (h_mm[k], V_m3[k], n_fora_da_curva). P em mm por passo, sem NaN.
    i_cap_mm_h: capacidade em mm/h; padrão = IDF na duração Te (só o teste de equivalência com o horário a troca).
    f_passo: fração da capacidade por passo (refluxo de maré); None = 1 em todos os passos.
    """
    P = np.asarray(P_mm_passo, dtype=float)
    if np.isnan(P).any():
        raise ValueError("P com NaN: a série precisa estar completa (ausente nunca vira zero)")
    i_cap = p.i_cap_mm_h if i_cap_mm_h is None else i_cap_mm_h
    A = t.area_m2
    vc = t.v_curva_m3 * p.curve_scale
    vmax = float(vc[-1])
    zmax = float(t.z_curva_m[-1])
    q = p.C * i_cap * A / 1000.0 * p.dt_min / 60.0
    qin = p.C * P * A / 1000.0
    n = len(P)
    h = np.zeros(n)
    Vs = np.zeros(n)
    V = float(V0)
    fora = 0
    fp = np.ones(n) if f_passo is None else np.asarray(f_passo, dtype=float)
    if len(fp) != n:
        raise ValueError("f_passo precisa ter um valor por passo")
    for k in range(n):
        v_disp = V + qin[k]
        qk = q * fp[k]
        qout = qk if qk < v_disp else v_disp
        V = v_disp - qout
        if V <= 0.0:
            V = 0.0
            Vs[k] = 0.0
            continue
        Vs[k] = V
        if V > vmax:
            fora += 1
            z = zmax
        else:
            z = float(np.interp(V, vc, t.z_curva_m))
        d = z - t.rua_rel_m
        if d > 0:
            h[k] = 1000.0 * d
    return h, Vs, fora


def simular_passos(P_mm_passo: np.ndarray, t: Terreno, p: Params10, V0: float = 0.0, f: float = 1.0) -> dict:
    """
    Mesma conta de `simular`, devolvendo a decomposição por passo (alerta por regressão, Etapa 3.1):
    entrada e saída em m3/h, capacidade da galeria em m3/h, volume em m3 e lâmina em mm.
    f: fração da capacidade da galeria (1 = projeto; 0 = obstruída), só para a sensibilidade (faixa "sem fonte" de
    docs/PARAMETROS-MODELO-FISICO.md). Com f = 1, h e V são idênticos aos de `simular`.
    """
    P = np.asarray(P_mm_passo, dtype=float)
    if np.isnan(P).any():
        raise ValueError("P com NaN: a série precisa estar completa (ausente nunca vira zero)")
    A = t.area_m2
    vc = t.v_curva_m3 * p.curve_scale
    vmax, zmax = float(vc[-1]), float(t.z_curva_m[-1])
    por_hora = 60.0 / p.dt_min
    q = f * p.C * p.i_cap_mm_h * A / 1000.0 * p.dt_min / 60.0
    qin = p.C * P * A / 1000.0
    n = len(P)
    h, Vs, qout_s = np.zeros(n), np.zeros(n), np.zeros(n)
    V, fora = float(V0), 0
    for k in range(n):
        v_disp = V + qin[k]
        qout = q if q < v_disp else v_disp
        qout_s[k] = qout
        V = max(0.0, v_disp - qout)
        Vs[k] = V
        if V <= 0.0:
            continue
        if V > vmax:
            fora += 1
            z = zmax
        else:
            z = float(np.interp(V, vc, t.z_curva_m))
        d = z - t.rua_rel_m
        if d > 0:
            h[k] = 1000.0 * d
    return {"h_mm": h, "V_m3": Vs, "inflow_m3_h": qin * por_hora, "outflow_m3_h": qout_s * por_hora,
            "capacity_m3_h": q * por_hora, "n_fora_da_curva": fora}


def avaliar(P_medido: np.ndarray, t: Terreno, p: Params10, chuva_prevista_1h_q50: float | None,
            chuva_prevista_1h_q80: float | None, passos_projecao: int = 6, i_cap_mm_h: float | None = None,
            f_medido: np.ndarray | None = None, f_projecao: np.ndarray | None = None) -> dict:
    """
    Lâmina medida (última posição da série) e projeção dos próximos 60 min com a chuva prevista distribuída por igual
    nos 6 passos (declarado), uma vez com q50 e outra com q80. A projeção parte do volume do fim da série medida.
    """
    h, V, fora = simular(P_medido, t, p, i_cap_mm_h=i_cap_mm_h, f_passo=f_medido)
    h_now = float(h[-1]) if len(h) else 0.0
    V_end = float(V[-1]) if len(V) else 0.0
    out = {"version": VERSION, "h_now_mm": h_now, "h_series_mm": h.tolist(), "n_fora_da_curva": int(fora),
           "i_cap_mm_h": p.i_cap_mm_h if i_cap_mm_h is None else i_cap_mm_h, "tr_anos": p.tr_anos, "te_min": p.te_min, "dt_min": p.dt_min,
           "hcrit_mm": p.hcrit_mm, "projection": {}}
    for nome, q in (("q50", chuva_prevista_1h_q50), ("q80", chuva_prevista_1h_q80)):
        if q is None or not np.isfinite(q):
            out["projection"][nome] = None
            continue
        Pp = np.full(passos_projecao, float(q) / passos_projecao)
        hp, _, fp = simular(Pp, t, p, i_cap_mm_h=i_cap_mm_h, V0=V_end, f_passo=f_projecao)
        cruz = np.flatnonzero(hp >= p.hcrit_mm)
        out["projection"][nome] = {"h_mm": hp.tolist(), "rain_1h_mm": float(q),
                                    "minutes_to_hcrit": int((cruz[0] + 1) * p.dt_min) if cruz.size else None,
                                    "h_max_mm": float(hp.max()) if len(hp) else 0.0}
    return out
