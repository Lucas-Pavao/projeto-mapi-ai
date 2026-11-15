"""
Etapa 4 do IN1165: avaliação no TESTE, calibração e limiares LOW/MEDIUM/HIGH.

Regras que este script segue (e por quê):
- O teste é usado uma vez, aqui. Hiperparâmetros (Etapa 2), calibrador e limiares são escolhidos na VALIDAÇÃO.
- Classe rara (~7% de positivos): acurácia engana. O classificador "nunca alaga" acerta ~93% e não serve para nada;
  por isso a tabela traz precision/recall/F1 da classe positiva, PR-AUC e o paradoxo da acurácia lado a lado.
- Probabilidade precisa ser calibrada para virar faixa de risco: se o modelo diz 30%, deveriam alagar ~30% desses
  casos. Medimos com curva de confiabilidade, Brier e ECE em 10 bins. O scale_pos_weight do XGBoost infla as
  probabilidades de propósito (para aprender a classe rara), então a calibração é esperada.
- Ganho "significativo": bootstrap PAREADO no teste (mesmas linhas reamostradas para os dois modelos), IC 95% da
  diferença de PR-AUC. Se o IC cruza o zero, o ganho não é significativo.

Saídas: reports/metrics.csv, reports/significance.csv, reports/figures/{confusion_*,pr_curve,roc_curve,calibration}.png,
e models/metadata.json atualizado (limiares, calibrador, métricas).
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

from mapi_ai import in1165_config as C
from mapi_ai.trainer import feature_matrix, load_dataset, temporal_split

HIGH_MIN_RECALL = 0.80     # perder alagamento é pior que alarme falso (critério do prompt)
MEDIUM_MIN_RECALL = 0.95   # "atenção": pega quase todos, aceitando mais alarmes falsos
N_BOOT = 1000


def ece(y, p, bins=10) -> float:
    """Expected Calibration Error: média, ponderada pelo tamanho do bin, de |freq. observada − prob. média|."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def threshold_for_recall(y, p, min_recall) -> float:
    """
    Maior limiar cuja recall na validação ainda é >= min_recall. Entre os limiares que garantem a recall, o maior é o
    que tem menos alarmes falsos (melhor precisão).
    """
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= min_recall)[0]
    best = ok[np.argmax(prec[ok])] if len(ok) else 0
    return float(thr[best])


class Trivial:
    """'Nunca alaga': probabilidade = taxa de positivos do treino (climatologia) para toda linha."""
    def __init__(self, rate):
        self.rate = rate

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.rate), np.full(len(X), self.rate)])


class RainRule:
    """'Alaga se a chuva de 3 h passa de X mm'. X escolhido na validação (maior F1). Probabilidade = 0/1."""
    def __init__(self, x_mm):
        self.x = x_mm

    def predict_proba(self, X):
        hit = (X["rainfall_3h_accumulated"].fillna(0) > self.x).astype(float).to_numpy()
        return np.column_stack([1 - hit, hit])


def models(tr, va):
    """Todos os modelos comparados, com a função que monta o X de cada um."""
    X_va, y_va = feature_matrix(va)
    grid = np.arange(1, 60, 0.5)
    x_best = max(grid, key=lambda x: f1_score(y_va, (X_va["rainfall_3h_accumulated"].fillna(0) > x).astype(int)))
    return {
        "nunca_alaga": (Trivial(float(tr[C.LABEL].mean())), lambda d: feature_matrix(d)),
        f"regra_chuva3h>{x_best:g}mm": (RainRule(float(x_best)), lambda d: feature_matrix(d)),
        "logistica": (joblib.load(C.MODELS_DIR / "baseline_logreg.joblib"), lambda d: feature_matrix(d)),
        "arvore_rasa": (joblib.load(C.MODELS_DIR / "baseline_tree.joblib"), lambda d: feature_matrix(d)),
        "xgb_v2": (joblib.load(C.MODELS_DIR / "xgb_v2.joblib"), lambda d: feature_matrix(d)),
        "xgb_v2_sem_local": (joblib.load(C.MODELS_DIR / "xgb_v2_sem_local.joblib"),
                             lambda d: feature_matrix(d, drop_location=True)),
        "xgb_v2_reanalise": (joblib.load(C.MODELS_DIR / "xgb_v2_reanalise.joblib"),
                             lambda d: feature_matrix(d, rain_source="reanalysis")),
    }


def paired_bootstrap(y, pa, pb, seed=C.SEED):
    """IC 95% de PR-AUC(a) − PR-AUC(b), reamostrando as mesmas linhas do teste para os dois."""
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        diffs.append(average_precision_score(y[i], pa[i]) - average_precision_score(y[i], pb[i]))
    d = np.array(diffs)
    return float(average_precision_score(y, pa) - average_precision_score(y, pb)), float(np.percentile(d, 2.5)), \
        float(np.percentile(d, 97.5))


def run() -> pd.DataFrame:
    df = load_dataset(synthetic=True)
    tr, va, te = temporal_split(df)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    clim = float(tr[C.LABEL].mean())

    # --- calibração do XGBoost na VALIDAÇÃO (FrozenEstimator = o antigo cv="prefit": o modelo não é retreinado) ---
    xgb = joblib.load(C.MODELS_DIR / "xgb_v2.joblib")
    X_va, y_va = feature_matrix(va)
    p_va_raw = xgb.predict_proba(X_va)[:, 1]
    cal = CalibratedClassifierCV(FrozenEstimator(xgb), method="sigmoid").fit(X_va, y_va)
    # Critério para usar o calibrador, escrito antes de olhar o teste: ECE na validação acima de 0,05.
    use_cal = ece(y_va, p_va_raw) > 0.05
    cal_info = {"method": "sigmoid (Platt)", "fit_on": "validation", "val_ece_raw": ece(y_va, p_va_raw),
                "val_ece_calibrated": ece(y_va, cal.predict_proba(X_va)[:, 1]), "used": bool(use_cal)}
    if use_cal:
        joblib.dump(cal, C.MODELS_DIR / "xgb_v2_calibrator.joblib")
        cal_info["artifact"] = "models/xgb_v2_calibrator.joblib"

    all_models = models(tr, va)
    # A versão calibrada entra sempre na comparação (curva antes/depois); é servida só se o critério acima pedir.
    all_models["xgb_v2_calibrado"] = (cal, lambda d: feature_matrix(d))

    rows, probs = [], {}
    y_te = te[C.LABEL].to_numpy()
    thresholds = {}
    for name, (m, fx) in all_models.items():
        Xv, yv = fx(va)
        Xt, yt = fx(te)
        pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
        probs[name] = pt
        thr = threshold_for_recall(yv, pv, HIGH_MIN_RECALL) if np.unique(pv).size > 2 else 0.5
        thresholds[name] = thr
        yhat = (pt >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
        brier = brier_score_loss(yt, pt)
        rows.append({
            "modelo": name, "limiar_high_val": round(thr, 4),
            "precision": precision_score(yt, yhat, zero_division=0), "recall": recall_score(yt, yhat, zero_division=0),
            "f1": f1_score(yt, yhat, zero_division=0), "acuracia": accuracy_score(yt, yhat),
            "roc_auc": roc_auc_score(yt, pt) if np.unique(pt).size > 1 else 0.5,
            "pr_auc": average_precision_score(yt, pt),
            "brier": brier, "bss": 1 - brier / brier_score_loss(yt, np.full(len(yt), clim)), "ece_10": ece(yt, pt),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
        if name in ("logistica", "arvore_rasa", "xgb_v2", "xgb_v2_calibrado", "xgb_v2_reanalise", "nunca_alaga"):
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            cm = np.array([[tn, fp], [fn, tp]])
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=12)
            ax.set_xticks([0, 1], ["prev. não", "prev. alaga"])
            ax.set_yticks([0, 1], ["real não", "real alaga"])
            ax.set_title(f"{name}\nlimiar {thr:.3f} — sintético", fontsize=9)
            fig.savefig(C.FIGURES_DIR / f"confusion_{name}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(C.REPORTS_DIR / "metrics.csv", index=False)

    # --- significância (bootstrap pareado da PR-AUC) ---
    comps = [("xgb_v2", "logistica"), ("xgb_v2", "arvore_rasa"), ("xgb_v2", "xgb_v2_reanalise"),
             ("xgb_v2", "xgb_v2_sem_local"), ("logistica", "arvore_rasa")]
    sig = []
    for a, b in comps:
        d, lo, hi = paired_bootstrap(y_te, probs[a], probs[b])
        sig.append({"a": a, "b": b, "delta_pr_auc": round(d, 4), "ic95_lo": round(lo, 4), "ic95_hi": round(hi, 4),
                    "significativo": bool(lo > 0 or hi < 0)})
    pd.DataFrame(sig).to_csv(C.REPORTS_DIR / "significance.csv", index=False)

    # --- curvas ---
    curve_models = ["logistica", "arvore_rasa", "xgb_v2", "xgb_v2_sem_local", "xgb_v2_reanalise"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in curve_models:
        p, r, _ = precision_recall_curve(y_te, probs[n])
        ax.plot(r, p, label=f"{n} (AP {average_precision_score(y_te, probs[n]):.2f})")
    ax.axhline(y_te.mean(), ls="--", c="gray", label=f"nunca alaga / acaso ({y_te.mean():.2f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(f"Curva PR no teste — {C.SYNTHETIC_TAG}", fontsize=9)
    ax.legend(fontsize=7); fig.savefig(C.FIGURES_DIR / "pr_curve.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for n in curve_models:
        f, t, _ = roc_curve(y_te, probs[n])
        ax.plot(f, t, label=f"{n} (AUC {roc_auc_score(y_te, probs[n]):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray")
    ax.set_xlabel("taxa de falso positivo"); ax.set_ylabel("recall")
    ax.set_title(f"Curva ROC no teste — {C.SYNTHETIC_TAG}", fontsize=9)
    ax.legend(fontsize=7); fig.savefig(C.FIGURES_DIR / "roc_curve.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for n in ["xgb_v2", "xgb_v2_calibrado", "logistica"]:
        if n in probs:
            fr, mp = calibration_curve(y_te, probs[n], n_bins=10, strategy="quantile")
            ax.plot(mp, fr, marker="o", label=f"{n} (ECE {ece(y_te, probs[n]):.3f}, Brier {brier_score_loss(y_te, probs[n]):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="perfeito")
    ax.set_xlabel("probabilidade prevista"); ax.set_ylabel("frequência observada")
    ax.set_title(f"Confiabilidade no teste (10 bins por quantil) — {C.SYNTHETIC_TAG}", fontsize=9)
    ax.legend(fontsize=7); fig.savefig(C.FIGURES_DIR / "calibration.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- limiares do modelo servido, escolhidos na VALIDAÇÃO com as probabilidades que a API vai emitir ---
    served = cal if use_cal else xgb
    p_served_va = served.predict_proba(X_va)[:, 1]
    high = threshold_for_recall(y_va, p_served_va, HIGH_MIN_RECALL)
    medium = threshold_for_recall(y_va, p_served_va, MEDIUM_MIN_RECALL)
    medium = min(medium, high)
    rv = lambda t: float(recall_score(y_va, (p_served_va >= t).astype(int)))
    pv = lambda t: float(precision_score(y_va, (p_served_va >= t).astype(int), zero_division=0))
    meta["thresholds"] = {
        "medium": round(medium, 4), "high": round(high, 4),
        "criteria": {"high": f"maior limiar com recall >= {HIGH_MIN_RECALL} na validação (melhor precisão)",
                     "medium": f"maior limiar com recall >= {MEDIUM_MIN_RECALL} na validação: nível de atenção, "
                               "aceita mais alarmes falsos para quase não perder alagamento"},
        "validation": {"high": {"recall": rv(high), "precision": pv(high)},
                       "medium": {"recall": rv(medium), "precision": pv(medium)}},
        "chosen_on": "validation", "probabilities": "calibrated" if use_cal else "raw",
    }
    meta["calibrator"] = cal_info if use_cal else None
    meta["calibration"] = cal_info
    srv = "xgb_v2_calibrado" if use_cal else "xgb_v2"
    meta["test_metrics"] = metrics.set_index("modelo").loc[srv].to_dict()
    yhat_m = (served.predict_proba(feature_matrix(te)[0])[:, 1] >= medium).astype(int)
    meta["test_metrics"]["medium_recall"] = float(recall_score(y_te, yhat_m))
    meta["test_metrics"]["medium_precision"] = float(precision_score(y_te, yhat_m, zero_division=0))
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float))

    print(f"== {C.SYNTHETIC_TAG} ==")
    print(metrics[["modelo", "limiar_high_val", "precision", "recall", "f1", "acuracia", "roc_auc", "pr_auc", "brier",
                   "bss", "ece_10"]].to_string(index=False))
    print("\nSignificância (bootstrap pareado, PR-AUC):")
    print(pd.DataFrame(sig).to_string(index=False))
    print(f"\nCalibração: {cal_info}")
    print(f"Limiares servidos: MEDIUM {medium:.4f}, HIGH {high:.4f}")
    return metrics


if __name__ == "__main__":
    run()
