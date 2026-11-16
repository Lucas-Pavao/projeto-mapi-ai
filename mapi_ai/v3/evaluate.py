"""
Avaliação do v3 (etapas B2.4, B3.3, B6 e B7). Toda escolha é feita na VALIDAÇÃO; o teste é lido uma vez, no fim.

1. Métricas por hora na amostra (como no v2, para comparar): PR-AUC, ROC-AUC, Brier, BSS e ECE, com e sem lacunas.
2. Calibração (B6.1): Platt ajustado na primeira metade (no tempo) da validação e conferido na segunda; adotado só se
   ECE e Brier melhoram lá.
3. Correção de prevalência (B6.2): logit(p') = logit(p) + logit(pi_real) - logit(pi_sint). A probabilidade SERVIDA é a
   corrigida; a não corrigida fica no relatório.
4. Limiares por orçamento de alertas (B6.3): na série contínua da validação, o menor limiar cujo número de alertas por
   de DIAS com alerta por ponto por ano cabe no orçamento (MEDIUM <= 50, HIGH <= 10). Sobre a probabilidade servida (B6.4).
5. Série contínua e eventos (B7): alertas por ano, fração de eventos detectados, antecedência, alarmes falsos.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mapi_ai import in1165_config as C2
from mapi_ai.evaluate import ece
from mapi_ai.v3 import config as C
from mapi_ai.v3 import dataset as D
from mapi_ai.v3.train import X_of, parts

EPS = 1e-9


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def prior_shift(p, pi_from: float, pi_to: float):
    """Correção de prevalência no logit: p calibrada em pi_from vira a probabilidade num mundo com prevalência pi_to."""
    return sigmoid(logit(p) + logit(pi_to) - logit(pi_from))


class Platt:
    """Calibrador sigmoide de uma entrada (o logit da probabilidade bruta)."""

    def fit(self, p_raw, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_raw).reshape(-1, 1), y)
        return self

    def transform(self, p_raw):
        return self.lr_.predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]

    def params(self):
        return {"a": float(self.lr_.coef_[0, 0]), "b": float(self.lr_.intercept_[0])}


def hourly_metrics(y, p, clim: float) -> dict:
    brier = brier_score_loss(y, p)
    ref = brier_score_loss(y, np.full(len(y), clim))
    return {"n": int(len(y)), "positives": int(y.sum()), "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)), "brier": float(brier), "bss": float(1 - brier / ref),
            "ece": ece(y, p)}


# ------------------------------------------------------------------------------------------- alertas e eventos

def episodes(hours: np.ndarray, gap_h: int) -> list[tuple[int, int]]:
    """Agrupa índices de hora (inteiros crescentes) em episódios; novo episódio se o intervalo passa de gap_h."""
    if len(hours) == 0:
        return []
    out, start, prev = [], hours[0], hours[0]
    for t in hours[1:]:
        if t - prev > gap_h:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return out


def hour_index(ts: pd.Series) -> np.ndarray:
    return ((ts - pd.Timestamp("2021-01-01", tz="UTC")) / pd.Timedelta(hours=1)).astype(int).to_numpy()


def alert_stats(cont: pd.DataFrame, p: np.ndarray, thr: float, horizon: int) -> dict:
    """Alertas e eventos de uma série contínua (várias séries de ponto) para um limiar."""
    years_points = 0.0
    n_alerts = hours_alert = n_events = detected = false_alerts = false_hours = 0
    n_rain_events = detected_rain = 0
    leads = []
    cont = cont.assign(_p=p, _t=hour_index(cont["timestamp"]),
                       _day=cont["timestamp"].dt.tz_convert("America/Recife").dt.strftime("%Y-%m-%d"))
    alert_days = 0
    for _, g in cont.groupby("slug", sort=False):
        years_points += len(g) / C.HOURS_PER_YEAR
        al = g.loc[g._p >= thr, "_t"].to_numpy()
        hours_alert += len(al)
        alert_days += g.loc[g._p >= thr, "_day"].nunique()
        eps = episodes(al, C.ALERT_GAP_H)
        n_alerts += len(eps)
        pos = g.loc[g["flood_h0"] == 1, "_t"].to_numpy()
        evs = episodes(pos, C.EVENT_GAP_H)
        wet = set(g.loc[g["stratum"] != "SECO", "_t"].to_numpy())
        used = np.zeros(len(eps), dtype=bool)
        ep_start = np.array([a for a, _ in eps]) if eps else np.array([], dtype=int)
        for (s, e) in evs:
            n_events += 1
            in_rain = any(t in wet for t in range(s, e + 1))
            n_rain_events += in_rain
            # detectado = há HORA de alerta na janela [início - H, fim] (não basta o episódio atravessar a janela)
            lo, hi = np.searchsorted(al, s - horizon, "left"), np.searchsorted(al, e, "right")
            if hi > lo:
                detected += 1
                detected_rain += in_rain
                leads.append(max(0, s - al[lo]))
                for t in (al[lo], al[hi - 1]):
                    used[np.searchsorted(ep_start, t, "right") - 1] = True
                k0, k1 = np.searchsorted(ep_start, al[lo], "right") - 1, np.searchsorted(ep_start, al[hi - 1], "right") - 1
                used[k0:k1 + 1] = True
        for k, (a, b) in enumerate(eps):
            if not used[k]:
                false_alerts += 1
                false_hours += b - a + 1
    return {"threshold": float(thr), "alert_days_per_point_year": alert_days / years_points,
            "alerts_per_point_year": n_alerts / years_points,
            "alert_hours_fraction": hours_alert / max(len(cont), 1),
            "events": n_events, "events_detected": detected, "event_recall": detected / max(n_events, 1),
            "rain_events": n_rain_events, "rain_event_recall": detected_rain / max(n_rain_events, 1),
            "lead_median_h": float(np.median(leads)) if leads else None,
            "lead_p10_h": float(np.percentile(leads, 10)) if leads else None,
            "false_alerts": false_alerts, "false_alerts_per_event": false_alerts / max(n_events, 1),
            "false_alert_mean_hours": false_hours / max(false_alerts, 1), "point_years": years_points}


def budget_threshold(cont: pd.DataFrame, p: np.ndarray, budget: float, horizon: int) -> tuple[float, pd.DataFrame]:
    """
    Menor limiar cujo número de DIAS com alerta por ponto por ano cabe no orçamento, e a curva completa.

    O orçamento conta dias com alerta (dia local com alguma hora acima do limiar), não episódios: a contagem de
    episódios não é monótona no limiar (limiar baixo funde episódios vizinhos e a contagem CAI), e um limiar quase zero
    passaria no orçamento com um alerta só, ligado o ano inteiro. Dias com alerta só diminuem quando o limiar sobe.
    """
    # Grade fina no topo: 10 dias de alerta por ano são ~0,3% das horas; quantis em passo logarítmico até 0,003%.
    grid = np.unique(np.quantile(p, 1 - np.logspace(np.log10(0.7), -4.5, 220)))
    curve = pd.DataFrame([alert_stats(cont, p, float(t), horizon) for t in grid])
    ok = curve[curve.alert_days_per_point_year <= budget]
    thr = float(ok.threshold.min()) if len(ok) else float(curve.threshold.max())
    return thr, curve


def consulted(df: pd.DataFrame) -> pd.Series:
    """Horas em que o modelo é consultado em produção: a regra de degradação tira as sem chuva da hora e sem a de 3 h."""
    return ~(df["current_rainfall"].isna() & df["rainfall_3h_accumulated"].isna())


# ------------------------------------------------------------------------------------------------------- execução

def run(tag: str = "v3", features=None, cont: pd.DataFrame | None = None, smp: pd.DataFrame | None = None,
        horizons=C.HORIZONS) -> dict:
    features = features or C.FEATURES
    models_dir = C.MODELS_DIR if tag == "v3" else C.MODELS_DIR / tag
    reports = C.REPORTS_DIR if tag == "v3" else C.REPORTS_DIR / tag
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    meta = json.loads((models_dir / "metadata.json").read_text())
    smp = D.load() if smp is None else smp
    cont = D.load("continuous") if cont is None else cont
    tr, va, te = parts(smp)
    cont = cont[cont.labeled & cont.part.isin(["validation", "test"])].copy()

    rows, served, out = [], {}, {"horizons": {}}
    for h in horizons:
        hk = f"h{h}"
        y = {n: d[C.label(h)].astype(int).to_numpy() for n, d in (("train", tr), ("validation", va), ("test", te))}
        clim = float(y["train"].mean())
        chosen = meta["horizons"][hk]["selection"]["chosen"]
        loaded = {m: joblib.load(models_dir / f"{m}_h{h}.joblib") for m in C.SIMPLICITY}

        # 1. métricas por hora (probabilidade bruta), com lacunas e sem lacunas
        for m, clf in loaded.items():
            for part, d in (("validation", va), ("test", te)):
                p = clf.predict_proba(X_of(d, features))[:, 1]
                rows.append({"generation": tag, "horizon_h": h, "model": m, "part": part, "inputs": "com lacunas",
                             "probability": "bruta", **hourly_metrics(y[part], p, clim)})
            p_full = clf.predict_proba(X_of(te, features, full=True))[:, 1]
            rows.append({"generation": tag, "horizon_h": h, "model": m, "part": "test", "inputs": "sem lacunas",
                         "probability": "bruta", **hourly_metrics(y["test"], p_full, clim)})

        # 2. calibração do modelo escolhido: metade A ajusta, metade B confere
        clf = loaded[chosen]
        va_sorted = va.sort_values("timestamp")
        half = len(va_sorted) // 2
        A, B = va_sorted.iloc[:half], va_sorted.iloc[half:]
        pA, pB = clf.predict_proba(X_of(A, features))[:, 1], clf.predict_proba(X_of(B, features))[:, 1]
        yA, yB = A[C.label(h)].astype(int).to_numpy(), B[C.label(h)].astype(int).to_numpy()
        platt = Platt().fit(pA, yA)
        pB_cal = platt.transform(pB)
        check = {"raw": {"ece": ece(yB, pB), "brier": float(brier_score_loss(yB, pB))},
                 "calibrated": {"ece": ece(yB, pB_cal), "brier": float(brier_score_loss(yB, pB_cal))}}
        use = check["calibrated"]["ece"] < check["raw"]["ece"] and check["calibrated"]["brier"] < check["raw"]["brier"]
        pi_sint = float(yA.mean()) if use else clim
        pi_real = C.pi_real(h)

        def serve(X, _clf=clf, _platt=platt, _use=use, _a=pi_sint, _b=pi_real, corrected=True):
            p = _clf.predict_proba(X)[:, 1]
            p = _platt.transform(p) if _use else p
            return prior_shift(p, _a, _b) if corrected else p

        served[h] = serve
        for part, d in (("validation", va), ("test", te)):
            p_cal = serve(X_of(d, features), corrected=False)
            rows.append({"generation": tag, "horizon_h": h, "model": chosen, "part": part, "inputs": "com lacunas",
                         "probability": "calibrada" if use else "bruta (calibração não adotada)",
                         **hourly_metrics(y[part], p_cal, clim)})
        joblib.dump(platt, models_dir / f"calibrator_h{h}.joblib")

        # 3. série contínua: probabilidade servida, só nas horas em que o modelo é consultado
        cv, ct = cont[cont.part == "validation"], cont[cont.part == "test"]
        cv_c, ct_c = cv[consulted(cv)], ct[consulted(ct)]
        p_cv, p_ct = serve(X_of(cv_c, features)), serve(X_of(ct_c, features))

        # 4. limiares por orçamento, na validação contínua
        thr, curves = {}, {}
        for level, budget in C.ALERT_BUDGET.items():
            thr[level], curves[level] = budget_threshold(cv_c, p_cv, budget, h)
        curve = curves["medium"]
        curve.to_csv(reports / f"alert_curve_h{h}.csv", index=False)

        # 5. eventos e alertas, validação e teste, nos dois níveis
        ev = {part: {level: alert_stats(d, p, thr[level], h) for level in thr}
              for part, d, p in (("validation", cv_c, p_cv), ("test", ct_c, p_ct))}
        cont_metrics = {part: {"hours": int(len(d)), "unknown_hours": int(len(full) - len(d)),
                               "prevalence": float(d[C.label(h)].mean()),
                               "pr_auc": float(average_precision_score(d[C.label(h)].astype(int), p))}
                        for part, d, p, full in (("validation", cv_c, p_cv, cv), ("test", ct_c, p_ct, ct))}

        # 6. o modelo serve sem a chuva da hora? (informa a regra de degradação da API)
        miss = cv[cv["rain_0_1h"].isna()]
        degr = None
        if len(miss) and miss[C.label(h)].nunique() == 2:
            pm = clf.predict_proba(X_of(miss, features))[:, 1]
            pf = clf.predict_proba(X_of(miss, features, full=True))[:, 1]
            ym = miss[C.label(h)].astype(int).to_numpy()
            degr = {"hours_without_current_rain": int(len(miss)), "positives": int(ym.sum()),
                    "pr_auc_without_current": float(average_precision_score(ym, pm)),
                    "pr_auc_same_hours_if_complete": float(average_precision_score(ym, pf))}

        out["horizons"][hk] = {
            "model": chosen, "calibration": {"rule": C.CALIBRATION_RULE, "check_second_half": check, "adopted": bool(use),
                                             "platt": platt.params()},
            "prior": {"pi_sint": pi_sint, "pi_sint_source": "taxa de positivos da metade da validação que ajusta o calibrador"
                      if use else "taxa de positivos do treino", "pi_real": pi_real,
                      "pi_real_source": "parâmetro declarado (mapi_ai/v3/config.py): ~0,1% das horas, 18 dias com "
                                        "alagamento em 5 anos; duração por dia [sem fonte]"},
            "thresholds": {"medium": thr["medium"], "high": thr["high"], "probability": "servida (calibrada e corrigida "
                           "por prevalência)" if use else "servida (bruta corrigida por prevalência)",
                           "rule": f"menor limiar com dias de alerta por ponto por ano <= orçamento ({C.ALERT_BUDGET}), na "
                                   "série contínua da validação", "chosen_on": "validation"},
            "continuous": cont_metrics, "alerts_and_events": ev, "degradation_check": degr,
        }

        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.plot(curve.alert_days_per_point_year, curve.event_recall, label="eventos detectados")
        ax.plot(curve.alert_days_per_point_year, curve.rain_event_recall, label="eventos em chuva detectados", ls="--")
        for level, budget in C.ALERT_BUDGET.items():
            ax.axvline(budget, color="gray", lw=0.8)
            ax.text(budget, 0.02, f" {level} <= {budget}", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("dias com alerta por ponto por ano (validação, série contínua)")
        ax.set_ylabel("fração de eventos detectados")
        ax.set_title(f"{C.SYNTHETIC_TAG}\nH = {h} h, {chosen}", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"alert_curve_h{h}.png", dpi=120)
        plt.close(fig)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(reports / "metrics.csv", index=False)

    # B3.3: correlação entre as entradas de chuva, antes (acumulados aninhados) e depois (janelas disjuntas)
    corr = {"acumulados_v2": tr[C.ACCUMULATED].corr().round(3).to_dict(), "janelas_v3": tr[C.WINDOWS].corr().round(3).to_dict()}
    off = lambda m: float(np.abs(m.values[np.triu_indices(len(m), 1)]).mean())
    corr["media_abs_fora_da_diagonal"] = {"acumulados_v2": off(tr[C.ACCUMULATED].corr()), "janelas_v3": off(tr[C.WINDOWS].corr())}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, cols, title in ((axes[0], C.ACCUMULATED, "v2: acumulados aninhados"), (axes[1], C.WINDOWS, "v3: janelas disjuntas")):
        m = tr[cols].corr()
        im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(cols)), [c.replace("rainfall_", "").replace("_accumulated", "").replace("rain_", "") for c in cols], rotation=45, fontsize=7)
        ax.set_yticks(range(len(cols)), [c.replace("rainfall_", "").replace("_accumulated", "").replace("rain_", "") for c in cols], fontsize=7)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{m.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title, fontsize=9)
    fig.suptitle(C.SYNTHETIC_TAG, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "correlation.png", dpi=120)
    plt.close(fig)
    out["correlation"] = corr

    # metadata: o que a produção serve
    hk = f"h{C.PRIMARY_HORIZON}"
    prod = out["horizons"][hk]
    meta["calibration"] = prod["calibration"]
    meta["calibrator"] = {"artifact": f"{models_dir.relative_to(C.ROOT)}/calibrator_h{C.PRIMARY_HORIZON}.joblib"} if prod["calibration"]["adopted"] else None
    meta["prior_correction"] = {**prod["prior"], "formula": "logit(p') = logit(p) + logit(pi_real) - logit(pi_sint)"}
    meta["thresholds"] = prod["thresholds"]
    meta["probability_served"] = "calibrada (se adotada) e corrigida por prevalência; os limiares usam esta"
    meta["alert_budget_per_point_year"] = C.ALERT_BUDGET
    meta["status"] = "avaliado"
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (reports / "evaluation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = run()
    for hk, v in o["horizons"].items():
        t = v["alerts_and_events"]["test"]
        print(hk, v["model"], "cal:", v["calibration"]["adopted"], "thr:", round(v["thresholds"]["medium"], 5), round(v["thresholds"]["high"], 5))
        for lvl in ("medium", "high"):
            s = t[lvl]
            print("  teste", lvl, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items() if k not in ("point_years",)})
        print("  degradação:", v["degradation_check"])
