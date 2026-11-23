"""
Fatores por ponto do alerta único (ADR 0030 do projeto-mapi-api): refluxo de maré e limpeza (Central 156).

Refluxo (ADR 0009 emenda 3, a mesma conta de `bucket10.fator_refluxo`): fração da capacidade da galeria livre em cada passo,
com a maré prevista no fim de cada passo da série medida e dos 36 passos do horizonte (a previsão de 6 h usa a maré das
próximas 6 h). soleira_NR = soleira_MDT - viés_MDT + Δ. O cenário central (viés 0) decide o nível; o cenário de viés
(MDT - RN = -0,369 m) só é devolvido como sensibilidade.

Limpeza (docs/PRE-REGISTRO-ENTUPIMENTO-156.md e emenda 1): o balde roda com a galeria livre e com a capacidade f_E;
P(lâmina >= 15 cm) = (1 - p_E)·1[livre >= 15] + p_E·1[entupida >= 15] (mistura, não média de capacidade).

Modos: OFF (não calcula), INFO (calcula e devolve a contribuição, sem mudar o nível), NIVEL (muda o nível).
Nada aqui é ajustado para acertar alagamento.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mapi_ai import bucket10 as B10
from mapi_ai import entupimento as ENT
from mapi_ai.flood import config as C

TZ = "America/Recife"


@dataclass(frozen=True)
class Refluxo:
    mode: str = "OFF"
    tide_nr_m: list | None = None          # um valor por passo da série medida + C.HORIZON_STEPS; None = sem maré
    sill_mdt_m: float | None = None        # soleira (fundo do tubo de saída) no referencial do MDT
    diameter_m: float | None = None
    delta_m: float | None = None           # NR <-> SGB (ADR 0009 emenda 1: 0,821 m)
    mdt_bias_m: float = 0.0                # MDT - RN no cenário central
    mdt_bias_scenario_m: float | None = C.MDT_BIAS_SCENARIO_M

    def sill_nr(self, bias: float) -> float:
        return float(self.sill_mdt_m) - float(bias) + float(self.delta_m)


@dataclass(frozen=True)
class Limpeza:
    mode: str = "OFF"
    days_since_cleaning: float | None = None
    lambda_re_per_day: float | None = None
    regime: str | None = None
    mean_clogged_days: float | None = None
    f_clogged: float = ENT.F_ENTUPIDA
    prob_on: float = 0.5


def prepare_refluxo(r: Refluxo | None, n_obs: int) -> dict:
    """Estado do refluxo antes de rodar: DESLIGADO, NAO_SE_APLICA, SEM_DADO (maré ausente) ou OK com os fatores por passo."""
    if r is None or r.mode == "OFF":
        return {"status": "DESLIGADO", "mode": "OFF" if r is None else r.mode, "reason": "fator desligado na configuração"}
    if r.sill_mdt_m is None or r.diameter_m is None or r.delta_m is None:
        return {"status": "NAO_SE_APLICA", "mode": r.mode, "reason": "ponto sem saída de galeria cadastrada (soleira e diâmetro)"}
    need = n_obs + C.HORIZON_STEPS
    tide = r.tide_nr_m
    if tide is None or len(tide) != need or any(v is None or not np.isfinite(v) for v in tide):
        faltam = need if tide is None else need - sum(1 for v in tide if v is not None and np.isfinite(v))
        return {"status": "SEM_DADO", "mode": r.mode, "reason": "MARE_AUSENTE",
                "reason_detail": f"maré prevista ausente em {faltam} de {need} passos"}
    m = np.asarray(tide, dtype=float)
    f = B10.fator_refluxo(m, r.sill_nr(r.mdt_bias_m), r.diameter_m)
    out = {"status": "OK", "mode": r.mode, "tide": m, "f_obs": f[:n_obs], "f_fut": f[n_obs:]}
    if r.mdt_bias_scenario_m is not None:
        fb = B10.fator_refluxo(m, r.sill_nr(r.mdt_bias_scenario_m), r.diameter_m)
        out.update({"f_obs_bias": fb[:n_obs], "f_fut_bias": fb[n_obs:]})
    return out


def prepare_limpeza(lz: Limpeza | None) -> dict:
    if lz is None or lz.mode == "OFF":
        return {"status": "DESLIGADO", "mode": "OFF" if lz is None else lz.mode, "reason": "fator desligado na configuração"}
    pe = ENT.p_entupida_regime(lz.regime, lz.days_since_cleaning, lz.lambda_re_per_day, lz.mean_clogged_days)
    if pe is None:
        return {"status": "SEM_DADO", "mode": lz.mode, "reason": "SEM_TAXA_OU_DATA",
                "reason_detail": "sem taxa de reentupimento do ponto (menos de 10 limpezas casadas) ou sem data da última limpeza"}
    return {"status": "OK", "mode": lz.mode, "p_e": float(pe), "f_e": float(lz.f_clogged), "prob_on": float(lz.prob_on)}


def run_point(observed: np.ndarray, scen: dict, terr: B10.Terreno, p: B10.Params10, f_obs=None, f_fut=None,
              f_state: float = 1.0) -> dict:
    """Balde na série medida e nos cenários de cada horizonte. sheets[s][h] = lâmina máxima (cm), incluindo a de agora."""
    obs = B10.simular_passos(observed, terr, p, f=f_state, f_passo=f_obs)
    V_end = float(obs["V_m3"][-1])
    h_now = float(obs["h_mm"][-1]) / 10.0
    sheets = {s: {} for s in C.SCENARIOS}
    for s in C.SCENARIOS:
        for h in C.HORIZONS:
            n = int(h * 60 / C.DT_MIN)
            P = np.full(n, float(scen[s][h]) / n)
            fut = None if f_fut is None else np.asarray(f_fut[:n], dtype=float)
            sim = B10.simular_passos(P, terr, p, V0=V_end, f=f_state, f_passo=fut)
            sheets[s][h] = max(h_now, float(sim["h_mm"].max()) / 10.0)
    return {"obs": obs, "V_end": V_end, "h_now_cm": h_now, "sheets": sheets}


def level_free(q50_cm: float, q95_cm: float) -> str:
    return "ALERTA" if q50_cm >= C.HCRIT_CM else "ATENCAO" if q95_cm >= C.HCRIT_CM else "SEM_ALERTA"


def mixture(free: dict, clog: dict, pe: float) -> dict:
    """prob[s][h] = (1 - p_E)·1[livre >= 15] + p_E·1[entupida >= 15]."""
    return {s: {h: (1 - pe) * (free["sheets"][s][h] >= C.HCRIT_CM) + pe * (clog["sheets"][s][h] >= C.HCRIT_CM)
                for h in C.HORIZONS} for s in C.SCENARIOS}


def level_mixture(prob: dict, h: int, prob_on: float) -> str:
    return "ALERTA" if prob["q50"][h] >= prob_on else "ATENCAO" if prob["q95"][h] >= prob_on else "SEM_ALERTA"


def evaluate(observed, scen, terr, p, f_obs, f_fut, limp: dict) -> dict:
    """Lâminas com a galeria livre, e (limpeza OK) entupida, probabilidade da mistura e o nível de cada horizonte."""
    free = run_point(observed, scen, terr, p, f_obs, f_fut)
    out = {"free": free, "clog": None, "prob": None}
    if limp["status"] == "OK":
        out["clog"] = run_point(observed, scen, terr, p, f_obs, f_fut, f_state=limp["f_e"])
        out["prob"] = mixture(free, out["clog"], limp["p_e"])
    use_mix = limp["status"] == "OK" and limp["mode"] == "NIVEL"
    out["levels"] = {h: (level_mixture(out["prob"], h, limp["prob_on"]) if use_mix
                         else level_free(free["sheets"]["q50"][h], free["sheets"]["q95"][h])) for h in C.HORIZONS}
    return out


def worst(levels: dict) -> tuple[str, int]:
    order = {"ALERTA": 0, "ATENCAO": 1, "SEM_ALERTA": 2}
    h = min(C.HORIZONS, key=lambda k: (order[levels[k]], k if levels[k] != "SEM_ALERTA" else -k))
    return levels[h], h


def _num(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def _hhmm(t: pd.Timestamp) -> str:
    return t.tz_convert(TZ).strftime("%Hh%M")


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def periods(f_fut: np.ndarray, t0: pd.Timestamp) -> list[dict]:
    """Trechos do horizonte em que a maré reduz a saída (fator < 1). Passo k vai de t0 + 10k min a t0 + 10(k+1) min."""
    red = 1.0 - np.asarray(f_fut, dtype=float)
    on = red > 1e-9
    out, k = [], 0
    dt = pd.Timedelta(minutes=C.DT_MIN)
    while k < len(on):
        if not on[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(on) and on[j + 1]:
            j += 1
        seg = red[k:j + 1]
        out.append({"from": _iso(t0 + k * dt), "to": _iso(t0 + (j + 1) * dt),
                    "from_local": _hhmm(t0 + k * dt), "to_local": _hhmm(t0 + (j + 1) * dt),
                    "mean_reduction_pct": round(100 * float(seg.mean()), 1),
                    "max_reduction_pct": round(100 * float(seg.max()), 1)})
        k = j + 1
    return out


def _deltas(a: dict, b: dict) -> dict:
    return {s: {str(h): round(a["sheets"][s][h] - b["sheets"][s][h], 1) for h in C.HORIZONS} for s in C.SCENARIOS}


def _sheets(r: dict) -> dict:
    return {s: {str(h): round(r["sheets"][s][h], 1) for h in C.HORIZONS} for s in C.SCENARIOS}


def refluxo_block(r: Refluxo | None, prep: dict, observed, scen, terr, p, limp: dict, t0: pd.Timestamp,
                  served: tuple[str, int]) -> dict:
    """Contribuição do refluxo para a tela: períodos, diferença de lâmina com e sem maré e cenário de viés."""
    base = {"mode": prep["mode"], "status": prep["status"], "applied_to_level": False,
            "reason": prep.get("reason"), "reason_detail": prep.get("reason_detail")}
    if prep["status"] != "OK":
        base["text"] = {"DESLIGADO": "maré não considerada (fator desligado)",
                        "NAO_SE_APLICA": "maré não considerada: o ponto não tem saída de galeria cadastrada",
                        "SEM_DADO": "maré prevista indisponível"}[prep["status"]]
        return base
    scenario, wh = served
    with_tide = run_point(observed, scen, terr, p, prep["f_obs"], prep["f_fut"])
    without = run_point(observed, scen, terr, p, None, None)
    per = periods(prep["f_fut"], t0)
    now_red = round(100 * (1 - float(prep["f_obs"][-1])), 1)
    d = _deltas(with_tide, without)
    out = dict(base)
    out.update({"applied_to_level": prep["mode"] == "NIVEL", "sill_mdt_m": r.sill_mdt_m, "diameter_m": r.diameter_m,
                "delta_m": r.delta_m, "mdt_bias_m": r.mdt_bias_m, "sill_nr_m": round(r.sill_nr(r.mdt_bias_m), 3),
                "tide_now_nr_m": round(float(prep["tide"][len(observed) - 1]), 3),
                "tide_max_horizon_nr_m": round(float(prep["tide"][len(observed):].max()), 3),
                "reduction_now_pct": now_red, "periods": per, "delta_sheet_cm": d,
                "sheet_with_tide_cm": _sheets(with_tide), "sheet_without_tide_cm": _sheets(without),
                "served_delta_cm": d[scenario][str(wh)], "rule": "capacidade x clamp((soleira_NR + D - maré_NR)/D, 0, 1)"})
    if "f_obs_bias" in prep:
        ev = evaluate(observed, scen, terr, p, prep["f_obs_bias"], prep["f_fut_bias"], limp)
        lvl, h = worst(ev["levels"])
        out["bias_scenario"] = {"mdt_bias_m": r.mdt_bias_scenario_m, "sill_nr_m": round(r.sill_nr(r.mdt_bias_scenario_m), 3),
                                "sheet_cm": _sheets(ev["free"]), "level": lvl, "worst_horizon_hours": h,
                                "note": "cenário de viés do MDT (MDT - RN = -0,369 m em terra): só sensibilidade, não decide"}
    if per:
        p0 = per[0]
        quando = f"até {p0['to_local']}" if p0["from"] == _iso(t0) else f"entre {p0['from_local']} e {p0['to_local']}"
        txt = f"a maré reduz a saída da galeria em {_num(p0['mean_reduction_pct'], 0)}% {quando}"
        if len(per) > 1:
            txt += f" (mais {len(per) - 1} período(s) nas próximas 6 h)"
    else:
        txt = "a maré fica abaixo da saída da galeria nas próximas 6 h"
    dd = out["served_delta_cm"]
    if dd > 0:
        txt += f"; sem a maré, a lâmina do cenário servido seria {_num(dd)} cm menor"
    out["text"] = txt
    return out


def limpeza_block(lz: Limpeza | None, prep: dict, ev: dict, served: tuple[str, int]) -> dict:
    base = {"mode": prep["mode"], "status": prep["status"], "applied_to_level": False,
            "reason": prep.get("reason"), "reason_detail": prep.get("reason_detail")}
    if prep["status"] != "OK":
        base["text"] = ("limpeza não considerada (fator desligado)" if prep["status"] == "DESLIGADO"
                        else "sem dado de limpeza para este ponto")
        return base
    scenario, wh = served
    pe = prep["p_e"]
    lv = {h: level_mixture(ev["prob"], h, prep["prob_on"]) for h in C.HORIZONS}
    lvl, h_mix = worst(lv)
    out = dict(base)
    out.update({"applied_to_level": prep["mode"] == "NIVEL", "regime": lz.regime or "TEMPO_DESDE_LIMPEZA",
                "days_since_cleaning": None if lz.regime == "PERMANENTE" else lz.days_since_cleaning,
                "mean_clogged_days": lz.mean_clogged_days, "lambda_re_per_day": lz.lambda_re_per_day,
                "p_clogged": round(pe, 4), "f_clogged": prep["f_e"], "prob_on": prep["prob_on"],
                "sheet_clogged_cm": _sheets(ev["clog"]),
                "prob_hcrit": {s: {str(h): round(ev["prob"][s][h], 4) for h in C.HORIZONS} for s in C.SCENARIOS},
                "mixture_level": lvl, "mixture_worst_horizon_hours": h_mix,
                "f_clogged_source": "sem fonte (faixa 0 a 0,5); por isso o padrão é INFO (ADR 0030)"})
    regime = ("limpeza desconhecida, regime permanente" if lz.regime == "PERMANENTE"
              else f"{_num(lz.days_since_cleaning or 0, 0)} dias desde a última limpeza")
    cl = ev["clog"]["sheets"][scenario][wh]
    txt = (f"chance de a galeria estar entupida: {_num(100 * pe, 0)}% ({regime}); se estiver, a lâmina do cenário servido "
           f"chegaria a {_num(cl)} cm em {wh} h")
    if prep["mode"] != "NIVEL":
        txt += " (não muda o nível: a capacidade da galeria entupida não tem fonte)"
    out["text"] = txt
    return out
