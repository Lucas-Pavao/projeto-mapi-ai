"""
Experimento que responde P1: com tudo o mais igual, quanto de antecedência a previsão de chuva dá ao modelo de alagamento?
Três braços: sem previsão, com a previsão fora da amostra do rain_v1, e com a chuva futura verdadeira (o teto).

Uso: python -m mapi_ai.v4.experiment            -> só o cenário CENTRAL do balde (f = 1,0)
     python -m mapi_ai.v4.experiment --sensibilidade   -> também a faixa pré-registrada de capacidade da galeria

A opção --sensibilidade foi rodada em 2026-11-16 com o ok do Lucas (faixa inteira reportada; nenhum f escolhido pela frequência).
"""
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from mapi_ai.rain import config as CR
from mapi_ai.v3 import dataset as D3
from mapi_ai.v3 import evaluate as E3
from mapi_ai.v3 import train as T3
from mapi_ai.v4 import config as C
from mapi_ai.v4 import labels as L

RNG = np.random.default_rng(C.SEED)


def build(f: float) -> pd.DataFrame:
    """Série contínua horária dos 4 pontos: rótulo do balde, entradas do v3, terreno, previsão de chuva e chuva futura."""
    s = L.simulate(f)
    T = L.terrain()
    pt = pd.read_csv(C.TERRAIN_POINT).set_index("slug")
    oof = pd.read_csv(C.OOF)
    oof["hour_end_utc"] = pd.to_datetime(oof.t, utc=True)
    parts = []
    for slug, g in s.groupby("slug", sort=False):
        g = g.reset_index(drop=True)
        acc = D3.accumulate(g.rain)                                   # janelas para trás; hora ausente propaga
        g[C.WINDOWS] = D3.windows_from_accumulated(acc)
        g["y"] = D3.horizon_label(g.flood_h0, C.HORIZON)
        # teto: chuva futura VERDADEIRA em (t, t+h] e a maior hora dentro dela (só o terceiro braço usa)
        rev = g.rain[::-1]
        for h in (1, 3, 6):
            g[f"fut_{h}h"] = rev.rolling(h, min_periods=h).sum()[::-1].shift(-1)
        for h in (3, 6):
            g[f"futpeak_{h}h"] = rev.rolling(h, min_periods=h).max()[::-1].shift(-1)
        g["rua_rel_m"] = T[slug].rua_rel_m
        g["profundidade_depressao_m"] = float(pt.loc[slug, "profundidade_depressao_m"])
        g["area_contribuinte_m2"] = T[slug].area_m2[10]
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    df = df.merge(oof[["slug", "hour_end_utc"] + C.RAIN_FC + ["fold"]], on=["slug", "hour_end_utc"], how="left")
    local = df.hour_end_utc.dt.tz_convert("America/Recife")
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * local.dt.hour / 24), np.cos(2 * np.pi * local.dt.hour / 24)
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * local.dt.month / 12), np.cos(2 * np.pi * local.dt.month / 12)
    df = df.rename(columns={"hour_end_utc": "timestamp"})
    df["part"] = D3.part_of(df.timestamp)
    df["stratum"] = np.where(df.rain.fillna(0) > 0, "CHUVA", "SECO")
    df["labeled"] = df.y.notna() & df.tide_level.notna() & df.fut_6h.notna()
    return df


def count_events(df: pd.DataFrame) -> dict:
    out = {}
    for part in ("train", "validation", "test"):
        n = 0
        for _, g in df[df.part == part].groupby("slug"):
            n += len(E3.episodes(E3.hour_index(g.timestamp[g.flood_h0 == 1]), 6))
        out[part] = n
    return out


def event_stats(cont: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
    """Eventos (horas positivas a menos de 6 h), primeiro aviso em [início - H, fim], antecedência em minutos."""
    H = C.HORIZON
    leads, detected, false_ep, alert_days, years = [], 0, 0, 0, 0.0
    cont = cont.assign(_p=p, _t=E3.hour_index(cont.timestamp),
                       _day=cont.timestamp.dt.tz_convert("America/Recife").dt.strftime("%Y-%m-%d"))
    for _, g in cont.groupby("slug", sort=False):
        years += len(g) / 8766.0
        al = g.loc[g._p >= thr, "_t"].to_numpy()
        alert_days += g.loc[g._p >= thr, "_day"].nunique()
        evs = E3.episodes(g.loc[g.flood_h0 == 1, "_t"].to_numpy(), 6)
        eps = E3.episodes(al, 6)
        used = np.zeros(len(eps), bool)
        for (s, e) in evs:
            lo, hi = np.searchsorted(al, s - H, "left"), np.searchsorted(al, e, "right")
            if hi > lo:
                detected += 1
                leads.append(60.0 * max(0, s - al[lo]))
                for k, (a, b) in enumerate(eps):
                    if a <= e and b >= s - H:
                        used[k] = True
            else:
                leads.append(0.0)
        false_ep += int((~used).sum())
    n = len(leads)
    if n == 0:
        return {"eventos": 0}
    lv = np.array(leads)
    bs = np.array([np.median(RNG.choice(lv, n)) for _ in range(C.N_BOOT)])                    # bootstrap por evento
    return {"limiar": float(thr), "eventos": n, "detectados": detected, "detectados_pct": 100 * detected / n,
            "antecedencia_mediana_min": float(np.median(lv)),
            "antecedencia_ic95_min": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "antecedencia_p10_min": float(np.quantile(lv, 0.10)), "antecedencia_media_min": float(lv.mean()),
            "com_1h_ou_mais_pct": float(100 * (lv >= 60).mean()), "falsos": false_ep, "falsos_por_evento": false_ep / n,
            "dias_de_alerta_por_ponto_ano": alert_days / years}


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-12
    idx = np.clip(np.searchsorted(edges, p, "right") - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def run_scenario(f: float) -> dict:
    df = build(f)
    ev = count_events(df)
    res = {"f": f, "capacidade_mm_h": round(60.1 * f, 1), "eventos": ev, "prevalencia_y": float(df.y.mean()),
           "treinavel": all(ev[k] >= v for k, v in C.MIN_EVENTS.items())}
    if not res["treinavel"]:
        res["nota"] = "eventos insuficientes: sem positivos não há o que treinar nem antecedência a medir"
        return res
    lab = df[df.labeled]
    tr, va, te = lab[lab.part == "train"], lab[lab.part == "validation"], lab[lab.part == "test"]
    fitted, pv = {}, {}
    for arm, feats in C.ARMS.items():
        cand = T3.fit_candidates(tr[feats].astype(float), tr.y.astype(int), va[feats].astype(float), va.y.astype(int), feats)
        fitted[arm] = {k: cand[k] for k in ("logistica", "xgboost")}
        pv[arm] = {k: m.predict_proba(va[feats].astype(float))[:, 1] for k, m in fitted[arm].items()}
    # família: escolhida uma vez, no braço com previsão, na validação
    yv = va.y.to_numpy().astype(int)
    ap = {k: float(average_precision_score(yv, p)) for k, p in pv["com_previsao"].items()}
    family = max(ap, key=ap.get)
    res["selecao"] = {"pr_auc_validacao": ap}
    if family == "xgboost":
        d, lo, hi = T3.paired_bootstrap(yv, pv["com_previsao"]["xgboost"], pv["com_previsao"]["logistica"])
        family = "logistica" if lo <= 0 else "xgboost"
        res["selecao"].update({"diff_xgb_menos_logistica": d, "ic95": [lo, hi]})
    res["selecao"]["familia"] = family
    res["bracos"] = {}
    tag = f"f{int(round(f * 100)):03d}"
    for arm, feats in C.ARMS.items():
        m = fitted[arm][family]
        p_va, p_te = pv[arm][family], m.predict_proba(te[feats].astype(float))[:, 1]
        r = {"entradas": feats, "pr_auc_validacao": float(average_precision_score(yv, p_va)),
             "pr_auc_teste": float(average_precision_score(te.y.astype(int), p_te))}
        # calibração (regra do v3): Platt na primeira metade da validação, adotado se ECE e Brier melhoram na segunda
        half = (va.timestamp <= va.timestamp.quantile(0.5)).to_numpy()
        cal = E3.Platt().fit(p_va[half], yv[half])
        pb, pc, yb = p_va[~half], cal.transform(p_va[~half]), yv[~half]
        adopt = bool(ece(yb, pc) < ece(yb, pb) and brier_score_loss(yb, pc) < brier_score_loss(yb, pb))
        r["calibracao"] = {"ece_antes": ece(yb, pb), "ece_depois": ece(yb, pc), "brier_antes": float(brier_score_loss(yb, pb)),
                           "brier_depois": float(brier_score_loss(yb, pc)), "adotada": adopt}
        # limiares por orçamento de dias de alerta, na série contínua da VALIDAÇÃO (Platt é monotônico: não muda o aviso)
        for level, budget in C.ALERT_BUDGET.items():
            thr, _ = E3.budget_threshold(va, p_va, budget, C.HORIZON)
            r[level] = {"orcamento_dias_por_ponto_ano": budget, "validacao": event_stats(va, p_va, thr),
                        "teste": event_stats(te, p_te, thr)}
        res["bracos"][arm] = r
        joblib.dump({"model": m, "calibrator": cal if adopt else None, "features": feats},
                    C.MODELS_DIR / f"{tag}_{arm}_{family}.joblib", compress=3)
    return res


def run(sensitivity: bool = False):
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = C.F_GRID if sensitivity else [C.F_CENTRAL]
    out = {"label_tag": C.LABEL_TAG, "attribution": C.ATTRIBUTION, "horizon_h": C.HORIZON, "selection": C.SELECTION,
           "calibration_rule": C.CALIBRATION_RULE, "alert_budget": C.ALERT_BUDGET, "min_events": C.MIN_EVENTS,
           "rain_forecast": {"model_version": CR.MODEL_VERSION, "out_of_fold": "data/v4/rain_forecast_oof.csv.gz"},
           "frequencia_real_declarada": C.REAL_FREQUENCY, "seed": C.SEED,
           "split": {"train_end": str(C.TRAIN_END), "valid_end": str(C.VALID_END), "guard_h": C.GUARD_H},
           "sensibilidade_rodada": sensitivity,
           "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.ROOT).stdout.strip(),
           "cenarios": {}}
    for f in grid:
        out["cenarios"][f"f={f}"] = run_scenario(f)
        print("f", f, out["cenarios"][f"f={f}"]["eventos"], out["cenarios"][f"f={f}"]["treinavel"], flush=True)
    (C.REPORTS_DIR / "experiment_p1.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    meta = {k: v for k, v in out.items() if k != "cenarios"}
    meta.update({"model_generation": "v4", "training_data": "PHYSICAL_MODEL_LABEL", "training_data_note": C.LABEL_TAG,
                 "status": "experimento; não servido (o modelo servido continua o logreg_v3)",
                 "cenarios": {k: {"eventos": v["eventos"], "treinavel": v["treinavel"],
                                  **({"familia": v["selecao"]["familia"]} if v["treinavel"] else {})}
                              for k, v in out["cenarios"].items()}})
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = run(sensitivity="--sensibilidade" in sys.argv)
    for k, s in o["cenarios"].items():
        if not s["treinavel"]:
            print(k, "não treinável:", s["eventos"])
            continue
        print(k, s["selecao"])
        for arm, r in s["bracos"].items():
            for lvl in ("medium", "high"):
                t = r[lvl]["teste"]
                print(f"  {arm:26s} {lvl:6s} PR {r['pr_auc_teste']:.3f} ev {t['eventos']} det {t['detectados']} "
                      f"med {t['antecedencia_mediana_min']:.0f} {t['antecedencia_ic95_min']} p10 {t['antecedencia_p10_min']:.0f} "
                      f">=1h {t['com_1h_ou_mais_pct']:.0f}% falsos/ev {t['falsos_por_evento']:.2f} dias {t['dias_de_alerta_por_ponto_ano']:.1f}")
