"""
Explicação física do alerta por regressão (Etapa 3): SHAP local do rain_v1, contrafactual e sensibilidade pelo balde.

SHAP local: contribuições exatas de árvore do XGBoost (pred_contribs, o mesmo TreeSHAP de `shap.TreeExplainer`) do
regressor de quantis que alimentou o cenário servido. Unidade: mm de chuva prevista. A soma das contribuições mais o
termo de base é a saída bruta do regressor (antes do corte em zero e da ordenação dos quantis feitos em `GB.predict`).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import xgboost as xgb

from mapi_ai import bucket10 as B10
from mapi_ai.flood import config as C
from mapi_ai.rain import features as F
from mapi_ai.rain import models as RM

SECTOR_NAME = {"N": "ao norte", "S": "ao sul", "L": "a leste", "O": "a oeste"}
RING_NAME = {"r5_15": "a 5–15 km", "r15_40": "a 15–40 km"}


def _num(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def _label(feat: str, v) -> tuple[str, str | None]:
    """(texto em unidade física, unidade) de uma entrada do rain_v1 com o valor desta previsão."""
    na = v is None or (isinstance(v, float) and math.isnan(v))
    if feat.startswith("p_") and feat != "p_trend":
        n = feat.split("_")[1]
        return (f"chuva no ponto (até 5 km) nos últimos {n} min: " + ("sem dado" if na else f"{_num(v)} mm"), "mm")
    if feat == "p_trend":
        return ("tendência no ponto (30 min mais recentes menos os 30 anteriores): "
                + ("sem dado" if na else f"{_num(v)} mm"), "mm")
    s, _, k = feat.partition("_")
    if s in SECTOR_NAME:
        lado = SECTOR_NAME[s]
        if k.startswith("max_"):
            n = k.split("_")[1]
            return (f"maior chuva nas estações {lado} (5–40 km) nos últimos {n} min: "
                    + ("sem estação" if na else f"{_num(v)} mm"), "mm")
        if k == "frac_wet_60":
            return (f"estações {lado} (5–40 km) com chuva na última hora: "
                    + ("sem estação" if na else f"{_num(100 * v, 0)}%"), "%")
        if k == "min_since_onset":
            return (f"desde que começou a chover na estação mais próxima {lado}: "
                    + ("não começou nas últimas 3 h" if na else f"{_num(v, 0)} min"), "min")
    for r, nome in RING_NAME.items():
        if feat.startswith(r + "_"):
            k = feat[len(r) + 1:]
            what = {"max_60": "maior chuva", "mean_60": "chuva média", "frac_wet_60": "fração com chuva"}[k]
            if k == "frac_wet_60":
                return (f"estações {nome} com chuva na última hora: " + ("sem estação" if na else f"{_num(100 * v, 0)}%"), "%")
            return (f"{what} nas estações {nome} na última hora: " + ("sem estação" if na else f"{_num(v)} mm"), "mm")
    if feat == "net_valid_frac":
        return (f"estações da rede com leitura: {'sem dado' if na else _num(100 * v, 0) + '%'}", "%")
    if feat == "n_near_valid":
        return (f"estações com leitura a até 5 km: {'0' if na else int(v)}", None)
    if feat.startswith("om_"):
        h = feat[3:-1]
        return (f"previsão do Open-Meteo (rodada da véspera) para as próximas {h} h: "
                + ("sem dado" if na else f"{_num(v)} mm"), "mm")
    return (feat, None)


def local_shap(model, x: pd.DataFrame, horizon: int, quantile: str, k: int = 3) -> dict:
    """
    SHAP local do quantil `quantile` (q50|q95) no horizonte `horizon`. hora_sin/cos e mês_sin/cos somam juntos
    (uma entrada física cada). Devolve as k de maior |contribuição| em texto físico, a base e a saída bruta.
    """
    if not hasattr(model, "models"):
        return {"available": False, "note": "modelo sem árvore no horizonte: não há SHAP local", "top": []}
    reg = model.models[("q", horizon)]
    X = x[model.features]
    # a previsão servida usa as árvores até best_iteration (parada antecipada): as contribuições usam o mesmo corte
    best = getattr(reg, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    contribs = reg.get_booster().predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=rng)
    qi = RM.QN.index(quantile)
    c = contribs[0, qi, :] if contribs.ndim == 3 else contribs[0, :]
    raw = reg.predict(X)
    raw_q = float(raw[0, qi] if np.ndim(raw) == 2 else raw[0])
    base = float(c[-1])
    s = pd.Series(c[:-1], index=model.features)
    row = X.iloc[0]
    groups: dict[str, float] = {}
    for f, v in s.items():
        g = "hora" if f in ("hour_sin", "hour_cos") else "mes" if f in ("month_sin", "month_cos") else f
        groups[g] = groups.get(g, 0.0) + float(v)
    t = pd.Timestamp(x["t"].iloc[0]) if "t" in x else None
    local = t.tz_convert("America/Recife") if t is not None else None
    items = []
    for g, val in sorted(groups.items(), key=lambda kv: -abs(kv[1]))[:k]:
        if g == "hora":
            label, unit, value = (f"hora do dia: {local.hour:02d}h" if local is not None else "hora do dia"), "h", None
        elif g == "mes":
            label, unit, value = (f"mês: {local.month:02d}" if local is not None else "mês"), None, None
        else:
            value = None if pd.isna(row[g]) else float(row[g])
            label, unit = _label(g, value)
        sinal = "+" if val >= 0 else "−"
        items.append({"feature": g, "label": label, "value": value, "unit": unit,
                      "contribution_mm": round(val, 3),
                      "text": f"{label} ({sinal}{_num(abs(val), 1)} mm na previsão)"})
    return {"available": True, "method": "TreeSHAP exato do XGBoost (pred_contribs)", "horizon_hours": horizon,
            "quantile": quantile, "base_mm": round(base, 4), "raw_prediction_mm": round(raw_q, 4),
            "sum_check_mm": round(base + float(s.sum()), 4), "top": items}


def _scenario_sheet_cm(V_end: float, h_now_cm: float, rain_mm: float, horizon: int, terr: B10.Terreno,
                       p: B10.Params10, f: float = 1.0) -> float:
    n = int(horizon * 60 / C.DT_MIN)
    P = np.full(n, float(rain_mm) / n)
    out = B10.simular_passos(P, terr, p, V0=V_end, f=f)
    return max(h_now_cm, float(out["h_mm"].max()) / 10.0 if n else 0.0)


def counterfactual_mm(V_end: float, h_now_cm: float, q50_1h_mm: float, terr: B10.Terreno, p: B10.Params10) -> dict:
    """Quantos mm a mais em 1 h (sobre o q50 de 1 h, por igual nos 6 passos) levam a lâmina a 15 cm. Bissecção no balde."""
    sheet = lambda extra: _scenario_sheet_cm(V_end, h_now_cm, q50_1h_mm + extra, 1, terr, p)
    if sheet(0.0) >= C.HCRIT_CM:
        return {"extra_mm": 0.0, "reached": True, "text": "a lâmina provável já chega a 15 cm na próxima hora"}
    if sheet(C.CF_MAX_MM) < C.HCRIT_CM:
        return {"extra_mm": None, "reached": False,
                "text": f"nem {_num(C.CF_MAX_MM, 0)} mm a mais em 1 h levariam a lâmina a 15 cm neste terreno"}
    lo, hi = 0.0, C.CF_MAX_MM
    while hi - lo > C.CF_TOL_MM:
        mid = (lo + hi) / 2
        if sheet(mid) >= C.HCRIT_CM:
            hi = mid
        else:
            lo = mid
    return {"extra_mm": round(hi, 2), "reached": True, "base_rain_1h_mm": round(q50_1h_mm, 2),
            "text": f"mais {_num(hi)} mm de chuva na próxima hora, além dos {_num(q50_1h_mm)} mm previstos (q50), "
                    "levariam a lâmina a 15 cm"}


def _level(q50_cm: float, q95_cm: float) -> str:
    return "ALERTA" if q50_cm >= C.HCRIT_CM else "ATENCAO" if q95_cm >= C.HCRIT_CM else "SEM_ALERTA"


def sensitivity(observed: np.ndarray, terr: B10.Terreno, scenarios: dict, horizon: int, scenario: str,
                served_level: str) -> dict:
    """
    Lâmina máxima do cenário servido no horizonte servido com f, C e i_T (TR, Te) nas faixas; o nível muda?
    scenarios: {"q50": {h: mm}, "q95": {h: mm}}. Grade completa + uma variação por vez (qual parâmetro pesa).
    """
    def run(f, c, tr, te):
        p = B10.Params10(C=c, tr_anos=tr, te_min=te)
        obs = B10.simular_passos(observed, terr, p, f=f)
        V_end = float(obs["V_m3"][-1]) if len(observed) else 0.0
        h_now = float(obs["h_mm"][-1]) / 10.0 if len(observed) else 0.0
        cm = {s: _scenario_sheet_cm(V_end, h_now, scenarios[s][horizon], horizon, terr, p, f) for s in C.SCENARIOS}
        return cm[scenario], _level(cm["q50"], cm["q95"])

    S = C.SENSITIVITY
    grid = [run(*combo) for combo in itertools.product(S["f"], S["C"], S["tr_anos"], S["te_min"])]
    vals = [g[0] for g in grid]
    levels = sorted({g[1] for g in grid})
    one_at_a_time = {}
    for k, values in S.items():
        cms = []
        for v in values:
            kw = dict(C.CENTRAL, **{k: v})
            cms.append(run(kw["f"], kw["C"], kw["tr_anos"], kw["te_min"])[0])
        one_at_a_time[k] = {"values": list(values), "min_cm": round(min(cms), 1), "max_cm": round(max(cms), 1),
                            "source": C.SENSITIVITY_SOURCES[k]}
    heaviest = max(one_at_a_time, key=lambda k: one_at_a_time[k]["max_cm"] - one_at_a_time[k]["min_cm"])
    return {"horizon_hours": horizon, "scenario": scenario, "min_cm": round(min(vals), 1), "max_cm": round(max(vals), 1),
            "levels_seen": levels, "level_changes": levels != [served_level],
            "n_combinations": len(grid), "ranges": {k: list(v) for k, v in S.items()},
            "one_at_a_time": one_at_a_time, "heaviest_parameter": heaviest,
            "text": (f"lâmina máxima entre {_num(min(vals))} e {_num(max(vals))} cm com f, C e i_T na faixa; "
                     + ("o nível muda" if levels != [served_level] else "o nível não muda"))}
