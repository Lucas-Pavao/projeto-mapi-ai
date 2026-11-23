"""
Alerta de alagamento por regressão (Etapa 1): um único caminho.

1. chuva prevista pelo rain_v1 em (t, t+1h], (t, t+3h], (t, t+6h] (quantis q50 e q95);
2. balde M1a a 10 min partindo do estado do fim da série MEDIDA do ponto (como em /v1/bucket/point10), uma vez por
   cenário e horizonte, com o acumulado previsto espalhado por igual nos passos (declarado em config.DISTRIBUTION);
3. lâmina máxima de cada horizonte, em cm;
4. regra fixa (config.RULE): ALERTA (q50 >= 15 cm), ATENCAO (só q95 >= 15 cm), SEM_ALERTA.

Fatores por ponto (ADR 0030, alerta único): refluxo de maré a cada passo da série e do horizonte e limpeza (Central 156),
cada um em OFF / INFO / NIVEL, com a contribuição devolvida em explanation.factors (mapi_ai/flood/fatores.py).

Falta de dado devolve SEM_DADO com motivo (config.REASONS), nunca SEM_ALERTA e nunca chuva zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mapi_ai import bucket10 as B10
from mapi_ai.feature_engineering import MissingFeaturesError
from mapi_ai.flood import config as C
from mapi_ai.flood import explain as X
from mapi_ai.flood import capacidade as CAP
from mapi_ai.flood import fatores as FT
from mapi_ai.rain import config as RC
from mapi_ai.version import code_version

ORDER = {"ALERTA": 0, "ATENCAO": 1, "SEM_ALERTA": 2}


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def model_versions(rain_meta: dict | None) -> dict:
    return {"flood": C.VERSION,
            "rain": {"version": (rain_meta or {}).get("model_version", RC.MODEL_VERSION),
                     "training_data": (rain_meta or {}).get("training_data", "MEASURED"),
                     "target": (rain_meta or {}).get("target", "MEASURED")},
            "bucket": {"version": B10.VERSION, "training_data": "NONE_PHYSICAL", "calibrated": False}}


def _base(rain_meta, issued_at=None) -> dict:
    cv = code_version()
    return {"version": C.VERSION, "model_versions": model_versions(rain_meta), "code_version": cv["commit"],
            "code_dirty": cv["dirty"], "issued_at": issued_at, "hcrit_cm": C.HCRIT_CM, "rule": C.RULE,
            "rule_text": C.RULE_TEXT, "distribution": C.DISTRIBUTION, "sheet_max": C.SHEET_MAX,
            "experimental": True, "experimental_note": C.EXPERIMENTAL_NOTE, "validated": False,
            "attribution": RC.ATTRIBUTION}


def sem_dado(reason: str, detail: str | None = None, rain_meta: dict | None = None, **extra) -> dict:
    out = _base(rain_meta, extra.pop("issued_at", None))
    out.update({"level": "SEM_DADO", "reason": reason, "reason_text": C.REASONS[reason], "reason_detail": detail,
                "worst_horizon_hours": None, "horizons": [], "explanation": None, "sources": [], "missing_inputs": []})
    out.update(extra)
    return out


def forecast_point(*, latitude: float, longitude: float, readings: list, rain_10min_mm: list | None,
                   terrain: B10.Terreno | None, terrain_status: str = "OK", open_meteo: list | None = None,
                   issued_at: str | None = None, rain_model=None, now: str | None = None,
                   tr_anos: float = 5.0, te_min: float = 10.0, slug: str | None = None,
                   refluxo: FT.Refluxo | None = None, limpeza: FT.Limpeza | None = None) -> dict:
    meta = getattr(rain_model, "meta", None)
    extra = {"slug": slug}
    if terrain_status == "EM_REVISAO":
        return sem_dado("TERRENO_EM_REVISAO", "área de contribuição do ponto em revisão", meta, **extra)
    if terrain is None or terrain_status == "SEM_TERRENO":
        return sem_dado("SEM_TERRENO", "o ponto não tem terreno (curva cota-volume) carregado", meta, **extra)
    if not rain_10min_mm:
        return sem_dado("CHUVA_AUSENTE", "série de chuva medida do ponto vazia", meta, **extra)
    gaps = [i for i, v in enumerate(rain_10min_mm) if v is None or not np.isfinite(v)]
    if gaps:
        return sem_dado("SERIE_INCOMPLETA", f"{len(gaps)} passo(s) de 10 min sem leitura na série do ponto "
                        f"(o mais recente a {(len(rain_10min_mm) - 1 - gaps[-1]) * 10} min de agora)", meta, **extra)

    faixa = CAP.faixa(slug)
    p_min, p_max = CAP.params(faixa, tr_anos, te_min)
    observed = np.asarray(rain_10min_mm, dtype=float)
    ref = FT.prepare_refluxo(refluxo, len(observed))
    limp = FT.prepare_limpeza(limpeza)
    if ref["status"] == "SEM_DADO" and ref["mode"] == "NIVEL":
        return sem_dado("MARE_AUSENTE", ref["reason_detail"], meta, **extra)
    ref_on = ref["status"] == "OK" and ref["mode"] == "NIVEL"
    f_obs = ref["f_obs"] if ref_on else None
    f_fut = ref["f_fut"] if ref_on else None

    # estado de agora com a galeria no melhor estado com fonte (Q_max) e no pior (Q_min)
    obs = B10.simular_passos(observed, terrain, p_max, f_passo=f_obs)
    obs_min = B10.simular_passos(observed, terrain, p_min, f_passo=f_obs)
    V_end = float(obs["V_m3"][-1])
    h_now_cm = round(float(obs["h_mm"][-1]) / 10.0, 2)
    extra.update({"h_now_cm": h_now_cm, "h_now_cm_qmin": round(float(obs_min["h_mm"][-1]) / 10.0, 2),
                  "rain_now_10min_mm": float(observed[-1]),
                  "rain_last_60min_measured_mm": round(float(observed[-6:].sum()), 2)})

    if rain_model is None:
        return sem_dado("PREVISAO_INDISPONIVEL", "modelo de previsão de chuva não carregado", meta, **extra)
    try:
        rain, x = rain_model.forecast_with_inputs(latitude, longitude, readings, open_meteo, issued_at, now)
    except MissingFeaturesError:
        return sem_dado("CHUVA_AUSENTE", "nenhuma estação com leitura válida a até 5 km do ponto", meta, **extra)
    except Exception as e:  # rain_v1 quebrado conta como fora do ar, nunca como "sem alerta"
        return sem_dado("PREVISAO_INDISPONIVEL", f"falha no rain_v1: {type(e).__name__}", meta, **extra)

    try:
        peaks = rain_model.peaks(x) if hasattr(rain_model, "peaks") else None
    except Exception:
        peaks = None   # sem o pico, o acumulado vai por igual (declarado em explanation.profile)
    perfil = CAP.Perfil(peaks)
    t = pd.Timestamp(rain["issued_at"])
    scen = {s: {h["horizon_hours"]: float(h["quantiles_mm"][s]) for h in rain["horizons"]} for s in C.SCENARIOS}
    ev_max = FT.evaluate(observed, scen, terrain, p_max, f_obs, f_fut, limp, perfil)
    ev_min = ev_max if p_min == p_max else FT.evaluate(observed, scen, terrain, p_min, f_obs, f_fut, limp, perfil)
    horizons = []
    for h in C.HORIZONS:
        # regra do pré-registro da capacidade: ALERTA com Q_max no q50; ATENÇÃO com Q_min no q95
        lvl = ("ALERTA" if ev_max["levels"][h] == "ALERTA"
               else "ATENCAO" if ev_min["levels"][h] in ("ALERTA", "ATENCAO") else "SEM_ALERTA")
        horizons.append({"hours": h, "level": lvl, "rain_q50_mm": round(scen["q50"][h], 2),
                         "rain_q95_mm": round(scen["q95"][h], 2),
                         "sheet_q50_cm": round(ev_max["free"]["sheets"]["q50"][h], 1),
                         "sheet_q95_cm": round(ev_min["free"]["sheets"]["q95"][h], 1),
                         "sheet_q50_cm_qmin": round(ev_min["free"]["sheets"]["q50"][h], 1),
                         "sheet_q95_cm_qmax": round(ev_max["free"]["sheets"]["q95"][h], 1),
                         "peak10_q50_mm": None if not peaks else round(min(peaks["q50"][h], scen["q50"][h]), 2),
                         "peak10_q95_mm": None if not peaks else round(min(peaks["q95"][h], scen["q95"][h]), 2),
                         "valid_from": _iso(t), "valid_until": _iso(t + pd.Timedelta(hours=h))})
    worst = min(horizons, key=lambda r: (ORDER[r["level"]], r["hours"] if r["level"] != "SEM_ALERTA" else -r["hours"]))
    level, wh = worst["level"], worst["hours"]
    scenario = "q95" if level == "ATENCAO" else "q50"
    p_srv, ev_srv, obs_srv = (p_min, ev_min, obs_min) if scenario == "q95" else (p_max, ev_max, obs)
    V_srv = float(obs_srv["V_m3"][-1])

    # decomposição: 3 h medidas + o horizonte servido previsto, no cenário e na ponta da capacidade servidos
    n = int(wh * 60 / C.DT_MIN)
    serie = perfil.serie(scen[scenario][wh], wh, scenario)
    proj = B10.simular_passos(serie, terrain, p_srv, V0=V_srv, f_passo=None if f_fut is None else f_fut[:n])
    tide = ref.get("tide")
    k0 = max(0, len(observed) - C.DECOMP_HISTORY_STEPS)
    decomposition = []
    for i in range(k0, len(observed)):
        m = (i - (len(observed) - 1)) * int(C.DT_MIN)
        decomposition.append(_step(t, m, "medido", observed[i], obs_srv, i, None if tide is None else tide[i]))
    for i in range(n):
        decomposition.append(_step(t, (i + 1) * int(C.DT_MIN), "previsto", serie[i], proj, i,
                                   None if tide is None else tide[len(observed) + i]))

    model = rain_model.models[rain_model.chosen[wh]]
    area = terrain.area_m2
    cap = {"base": faixa.base if faixa else "capacidade de projeto: IDF de Silva e Araújo (2013), Te 10 min",
           "q_min_m3s": round(p_min.q_passo_m3(area) / (p_min.dt_min * 60), 4),
           "q_max_m3s": round(p_max.q_passo_m3(area) / (p_max.dt_min * 60), 4),
           "kappa_calibrado": faixa.kappa if faixa else None,
           "rule": "ALERTA: lâmina q50 com Q_max >= 15 cm; ATENÇÃO: lâmina q95 com Q_min >= 15 cm"}
    cap["i_eq_min_mm_h"] = round(cap["q_min_m3s"] / (p_max.C * area) * 3.6e6, 1)
    cap["i_eq_max_mm_h"] = round(cap["q_max_m3s"] / (p_max.C * area) * 3.6e6, 1)
    explanation = {
        "served_scenario": scenario, "served_horizon_hours": wh,
        "served_capacity": "q_min" if scenario == "q95" else "q_max",
        "shap": X.local_shap(model, x, wh, scenario),
        "counterfactual": X.counterfactual_mm(V_end, h_now_cm, scen["q50"][1], terrain, p_max, f_fut=f_fut, perfil=perfil),
        "sensitivity": X.sensitivity(observed, terrain, scen, wh, scenario, level, f_obs=f_obs, f_fut=f_fut, perfil=perfil,
                                     q_range=None if faixa is None else (cap["q_min_m3s"], cap["q_max_m3s"])),
        "decomposition": decomposition,
        "capacity_m3_h": round(float(obs_srv["capacity_m3_h"]), 2),
        "capacity": cap,
        "profile": {"method": perfil.metodo, "model": "rain_v1p" if peaks else None},
        "initial_state": {"volume_m3": round(V_end, 2), "h_now_cm": h_now_cm, "steps_measured": int(len(observed)),
                          "n_fora_da_curva": int(obs["n_fora_da_curva"])},
        "factors": {"refluxo": FT.refluxo_block(refluxo, ref, observed, scen, terrain, p_srv, limp, t, (scenario, wh), perfil),
                    "limpeza": FT.limpeza_block(limpeza, limp, ev_srv, (scenario, wh)), "rule": C.FACTOR_NOTE},
    }
    out = _base(meta, rain["issued_at"])
    out.update(extra)
    out.update({"level": level, "worst_horizon_hours": wh, "reason": None, "reason_text": None, "reason_detail": None,
                "horizons": horizons, "explanation": explanation, "sources": rain["sources"],
                "missing_inputs": rain["missing_inputs"], "latest_reading_utc": rain["latest_reading_utc"],
                "latest_reading_age_minutes": rain["latest_reading_age_minutes"],
                "stations_valid_near": rain["stations_valid_near"], "bucket_params": {
                    "C": p_max.C, "tr_anos": p_max.tr_anos, "te_min": p_max.te_min, "i_cap_mm_h": round(p_max.i_cap_mm_h, 2),
                    "f": 1.0, "q_cap_min_m3s": cap["q_min_m3s"], "q_cap_max_m3s": cap["q_max_m3s"]}})
    return out


def _step(t: pd.Timestamp, minutes: int, kind: str, rain_mm: float, sim: dict, i: int, tide_nr_m=None) -> dict:
    return {"t": _iso(t + pd.Timedelta(minutes=minutes)), "minutes_from_now": minutes, "kind": kind,
            "rain_mm": round(float(rain_mm), 3), "inflow_m3_h": round(float(sim["inflow_m3_h"][i]), 3),
            "outflow_m3_h": round(float(sim["outflow_m3_h"][i]), 3),
            "capacity_m3_h": round(float(sim["capacity_m3_h"]), 3),
            "capacity_step_m3_h": round(float(sim["capacity_step_m3_h"][i]), 3),
            "outflow_factor": round(float(sim["outflow_factor"][i]), 4),
            "tide_nr_m": None if tide_nr_m is None else round(float(tide_nr_m), 3),
            "volume_m3": round(float(sim["V_m3"][i]), 3), "sheet_cm": round(float(sim["h_mm"][i]) / 10.0, 2)}
