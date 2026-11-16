"""
Explicações do v3 (etapa B8): a Etapa 6 do IN1165 refeita sobre o modelo de produção, lado a lado com o v2.

Pergunta: as janelas disjuntas aumentaram a concordância SHAP x LIME e a estabilidade do LIME?

Para responder sem confundir com a troca de modelo (o v2 era XGBoost com 12 entradas; o v3 é logística com 10), a
comparação limpa é feita DENTRO do v3: a mesma logística, no mesmo dataset, treinada duas vezes,
- com os acumulados aninhados (chuva de 1, 3, 6, 12 e 24 h, como no v2) e
- com as janelas disjuntas (0-1, 1-3, 3-6, 6-12 e 12-24 h).
Só as entradas de chuva mudam. Os números do v2 (reports/explanation_eval.json) entram na tabela como referência.

Medidas, nas mesmas ~50 linhas do teste (25 com chuva de 3 h > 1 mm e 25 quaisquer, sem entrada ausente):
1. concordância SHAP x LIME: Jaccard do top-5 e Spearman do ranking completo;
2. estabilidade do LIME: a mesma linha com 10 sementes, Jaccard médio do top-5 entre pares de sementes;
3. fidelidade por deleção: queda da probabilidade ao trocar as k entradas mais importantes pelo fundo, contra k ao acaso;
4. recuperação da fórmula (só H = 0, em que o rótulo é a própria fórmula): Spearman entre o ranking global e a
   atribuição verdadeira. Em janelas, a fórmula do v2 tem coeficiente exato: P1h = janela 0-1; P3h = 0-1 + 1-3;
   P24h = soma das cinco.
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mapi_ai import in1165_config as C2
from mapi_ai.v3 import config as C
from mapi_ai.v3 import dataset as D
from mapi_ai.v3.train import base_feature, logistic_pipeline, parts

N_ROWS, N_SEEDS, K_MAX, N_RANDOM, N_BG = 50, 10, 5, 20, 50
LIME_SAMPLES = 5000
ACC_FEATURES = C.ACCUMULATED + ["tide_level"] + C.TIME


def shap_logit(model, X_bg: pd.DataFrame, X: pd.DataFrame, features: list) -> np.ndarray:
    """SHAP exato da logística no logit, com o indicador de ausência somado à sua entrada. Linhas x entradas."""
    import shap
    pre = model[:-1]
    phi = np.asarray(shap.LinearExplainer(model.named_steps["clf"], pre.transform(X_bg)).shap_values(pre.transform(X)))
    out = pd.DataFrame(0.0, index=range(len(X)), columns=features)
    for j, name in enumerate(pre.get_feature_names_out()):
        out[base_feature(name)] += phi[:, j]
    return out.to_numpy()


def lime_weights(model, X_train: pd.DataFrame, x: np.ndarray, features: list, seed: int, num_features: int) -> dict:
    from lime.lime_tabular import LimeTabularExplainer
    ex = LimeTabularExplainer(X_train.to_numpy(), feature_names=features, class_names=["não alaga", "alaga"],
                              discretize_continuous=True, random_state=seed, mode="classification")
    e = ex.explain_instance(x, lambda a: model.predict_proba(pd.DataFrame(a, columns=features)), num_features=num_features,
                            num_samples=LIME_SAMPLES)
    w = {}
    for rule, weight in e.as_list(label=1):
        f = max((f for f in features if f in rule), key=len)
        w[f] = max(w.get(f, 0.0), abs(weight))
    return w


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def deletion_drop(model, x: pd.Series, feats: list, bg: pd.DataFrame, features: list, cut: float) -> float:
    """Queda da probabilidade da classe prevista ao trocar `feats` pelos valores do fundo (média sobre o fundo)."""
    base = float(model.predict_proba(x.to_frame().T[features])[0, 1])
    Z = pd.DataFrame([x.to_numpy()] * len(bg), columns=features)
    for f in feats:
        Z[f] = bg[f].to_numpy()
    sign = 1.0 if base >= cut else -1.0
    return sign * (base - float(model.predict_proba(Z)[:, 1].mean()))


def true_attribution(X: pd.DataFrame, features: list) -> pd.Series:
    """|termo - média| de cada entrada na fórmula do v2 (logit), na base de entradas dada. Só vale para H = 0."""
    t = json.loads((C2.REPORTS_DIR / "synthetic_ground_truth.json").read_text())
    a, k = t["anchors"], t["coefficients"]
    c1, c3, c24 = k["current_rainfall"] / a["R1_mm"], k["rainfall_3h_accumulated"] / a["R3_mm"], k["rainfall_24h_accumulated"] / a["R24_mm"]
    if "rain_0_1h" in features:
        coef = {"rain_0_1h": c1 + c3 + c24, "rain_1_3h": c3 + c24, "rain_3_6h": c24, "rain_6_12h": c24, "rain_12_24h": c24}
        p3 = X["rain_0_1h"] + X["rain_1_3h"]
    else:
        coef = {"current_rainfall": c1, "rainfall_3h_accumulated": c3, "rainfall_24h_accumulated": c24}
        p3 = X["rainfall_3h_accumulated"]
    out = pd.Series(0.0, index=features)
    for f, c in coef.items():
        term = c * X[f]
        out[f] = float((term - term.mean()).abs().mean())
    tide = k["tide_x_rain"] * np.clip(X["tide_level"] - a["T_hi_m"], 0, None) / a["T_span_m"] * (p3 > a["wet3_mm"])
    out["tide_level"] = float((tide - tide.mean()).abs().mean())
    return out


def evaluate(model, X_tr: pd.DataFrame, X_te: pd.DataFrame, features: list, with_truth: bool) -> dict:
    rng = np.random.default_rng(C.SEED)
    X_tr_f = X_tr.fillna(X_tr.median())                      # o LIME precisa de estatísticas sem NaN
    complete = X_te.dropna().reset_index(drop=True)
    rain3 = complete[features[0]] + complete[features[1]] if "rain_0_1h" in features else complete["rainfall_3h_accumulated"]
    wet = np.flatnonzero(rain3.to_numpy() > 1.0)
    a = rng.choice(wet, size=min(N_ROWS // 2, len(wet)), replace=False)
    b = rng.choice(np.setdiff1d(np.arange(len(complete)), a), size=N_ROWS - len(a), replace=False)
    Xs = complete.iloc[np.sort(np.concatenate([a, b]))].reset_index(drop=True)
    bg_shap = X_tr_f.sample(200, random_state=C.SEED)
    bg_del = X_tr_f.sample(N_BG, random_state=C.SEED + 1)
    phi = shap_logit(model, bg_shap, Xs, features)
    cut = float(np.quantile(model.predict_proba(X_tr_f)[:, 1], 0.9))

    jac, rho, stab = [], [], []
    drops = {"SHAP": np.zeros((len(Xs), K_MAX)), "LIME": np.zeros((len(Xs), K_MAX)), "aleatório": np.zeros((len(Xs), K_MAX))}
    lime_global = pd.Series(0.0, index=features)
    for r in range(len(Xs)):
        x = Xs.iloc[r]
        shap_rank = [features[j] for j in np.argsort(-np.abs(phi[r]))]
        w = lime_weights(model, X_tr_f, x.to_numpy(), features, C.SEED, len(features))
        lime_rank = sorted(features, key=lambda f: -w.get(f, 0.0))
        lime_global += pd.Series(w).reindex(features).fillna(0.0)
        jac.append(jaccard(shap_rank[:5], lime_rank[:5]))
        rho.append(spearmanr([shap_rank.index(f) for f in features], [lime_rank.index(f) for f in features])[0])
        tops = []
        for s in range(N_SEEDS):
            ws = lime_weights(model, X_tr_f, x.to_numpy(), features, s, 5)
            tops.append(sorted(ws, key=lambda f: -ws[f])[:5])
        stab.append(np.mean([jaccard(p, q) for p, q in itertools.combinations(tops, 2)]))
        for k in range(1, K_MAX + 1):
            drops["SHAP"][r, k - 1] = deletion_drop(model, x, shap_rank[:k], bg_del, features, cut)
            drops["LIME"][r, k - 1] = deletion_drop(model, x, lime_rank[:k], bg_del, features, cut)
            drops["aleatório"][r, k - 1] = np.mean([deletion_drop(model, x, list(rng.choice(features, k, replace=False)),
                                                                  bg_del, features, cut) for _ in range(N_RANDOM)])
    out = {
        "n_rows": int(len(Xs)), "n_features": len(features),
        "jaccard_top5_shap_lime": float(np.mean(jac)), "spearman_shap_lime": float(np.nanmean(rho)),
        "lime_stability_10_seeds": float(np.mean(stab)),
        "deletion_k1": {name: float(d[:, 0].mean()) for name, d in drops.items()},
        "deletion_curve": {name: [float(v) for v in d.mean(axis=0)] for name, d in drops.items()},
        "shap_global": pd.Series(np.abs(phi).mean(axis=0), index=features).sort_values(ascending=False).round(4).to_dict(),
    }
    if with_truth:
        truth = true_attribution(complete, features)
        sg = pd.Series(np.abs(phi).mean(axis=0), index=features)
        out["formula_recovery"] = {"true_attribution": truth.sort_values(ascending=False).round(4).to_dict(),
                                   "spearman_shap_vs_truth": float(spearmanr(sg, truth)[0]),
                                   "spearman_lime_vs_truth": float(spearmanr(lime_global / len(Xs), truth)[0])}
    return out


def run() -> dict:
    df = D.load()
    tr, va, te = parts(df)
    out = {"tag": C.SYNTHETIC_TAG, "v2_reference": None, "v3": {}}
    v2_path = C2.REPORTS_DIR / "explanation_eval.json"
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text())
        out["v2_reference"] = {"source": "reports/explanation_eval.json (xgb_v2, 12 entradas, H = 0)", "raw": v2}
    for h in (0, C.PRIMARY_HORIZON):
        y_tr = tr[C.label(h)].astype(int).to_numpy()
        for name, feats in (("acumulados_aninhados", ACC_FEATURES), ("janelas_disjuntas", C.FEATURES)):
            X_tr, X_te = tr[feats].astype(float), te[feats].astype(float)
            model = logistic_pipeline(feats).fit(X_tr, y_tr)
            res = evaluate(model, X_tr, X_te, feats, with_truth=(h == 0))
            out["v3"][f"h{h}_{name}"] = res
            print(f"H={h} {name}: Jaccard {res['jaccard_top5_shap_lime']:.2f} Spearman {res['spearman_shap_lime']:.2f} "
                  f"LIME estável {res['lime_stability_10_seeds']:.2f} deleção SHAP/LIME/acaso "
                  f"{res['deletion_k1']['SHAP']:.3f}/{res['deletion_k1']['LIME']:.3f}/{res['deletion_k1']['aleatório']:.3f}"
                  + (f" fórmula {res['formula_recovery']['spearman_shap_vs_truth']:.2f}" if h == 0 else ""))
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "explanation_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    from mapi_ai.v3 import explain as _e
    _e.run()
