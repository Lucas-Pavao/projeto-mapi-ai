"""
Etapa 6 do IN1165: as explicações merecem confiança?

Em ~50 linhas do teste (25 com chuva de 3 h > 1 mm e 25 quaisquer, sorteadas com a semente; só ao acaso, quase todas
seriam horas secas e qualquer explicação ficaria trivial):

1. Concordância SHAP × LIME: Jaccard do top-5 (quantas das 5 mais importantes coincidem) e Spearman dos rankings
   completos das 12 features. Duas explicações de natureza diferente que concordam reforçam uma à outra.
2. Estabilidade do LIME: a mesma linha com 10 sementes; Jaccard médio entre os top-5 de cada par de sementes.
   SHAP de árvore é determinístico; LIME sorteia vizinhos e pode mudar de resposta.
3. Fidelidade por deleção: troca as k features mais importantes (k = 1..5, pela ordem do SHAP e do LIME) por valores
   do fundo (média sobre 50 linhas do treino) e mede a queda da probabilidade. Explicação fiel aponta features que,
   removidas, derrubam a previsão; deve cair mais que remover k features ao acaso.
4. Plano B: como a fórmula geradora é conhecida, calculamos a atribuição VERDADEIRA de cada feature na escala do
   logit (|termo − média do termo|, exata para os termos lineares; o termo maré × chuva vai para a maré) e comparamos
   com o ranking global do SHAP e do LIME.

Saídas: reports/explanation_eval.csv, reports/explanation_eval.json, reports/figures/deletion_curve.png.
"""
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mapi_ai import in1165_config as C
from mapi_ai.synthetic import COEF
from mapi_ai.xai.common import load, proba
from mapi_ai.xai.lime_explain import explain, feature_of_rule, make_explainer
from mapi_ai.xai.shap_explain import shap_values

N_ROWS = 50
N_SEEDS = 10
K_MAX = 5
N_RANDOM = 20
N_BG_DELETE = 50


def rows_to_explain(X_te: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(C.SEED)
    wet = np.flatnonzero(X_te["rainfall_3h_accumulated"].fillna(0).to_numpy() > 1.0)
    a = rng.choice(wet, size=min(N_ROWS // 2, len(wet)), replace=False)
    rest = np.setdiff1d(np.arange(len(X_te)), a)
    b = rng.choice(rest, size=N_ROWS - len(a), replace=False)
    return np.sort(np.concatenate([a, b]))


def lime_ranking(model, ex, x) -> list:
    """Ranking completo das 12 features pelo |peso| do LIME (uma regra por feature)."""
    e = explain(model, ex, x, num_features=len(C.FEATURES))
    w = {}
    for rule, weight in e.as_list(label=1):
        f = feature_of_rule(rule)
        w[f] = max(w.get(f, 0.0), abs(weight))
    return sorted(C.FEATURES, key=lambda f: -w.get(f, 0.0)), w


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def deletion_drop(model, x: pd.Series, feats: list, bg: pd.DataFrame, threshold: float) -> float:
    """
    Queda da probabilidade da CLASSE PREVISTA quando `feats` são trocadas pelos valores do fundo (média sobre o fundo).
    Para linha prevista "alaga", mede p(alaga) − p'(alaga); para "não alaga", p(não alaga) − p'(não alaga). Medir só
    p(alaga) em todas as linhas mistura sinais: numa hora seca, remover a chuva (que explica o "não") AUMENTA p(alaga).
    """
    base = float(proba(model, x.to_frame().T)[0])
    Z = pd.DataFrame([x.to_numpy()] * len(bg), columns=C.FEATURES)
    for f in feats:
        Z[f] = bg[f].to_numpy()
    sign = 1.0 if base >= threshold else -1.0
    return sign * (base - float(proba(model, Z).mean()))


def true_attribution(df_te: pd.DataFrame, X_te: pd.DataFrame) -> pd.Series:
    """|termo − E[termo]| médio de cada feature na fórmula geradora (escala logit). Zero para quem não entra."""
    a = json.loads((C.REPORTS_DIR / "synthetic_ground_truth.json").read_text())["anchors"]
    tide_ex = np.clip(X_te["tide_level"] - a["T_hi_m"], 0, None) / a["T_span_m"]
    wet = (X_te["rainfall_3h_accumulated"] > a["wet3_mm"]).astype(float)
    terms = {
        "current_rainfall": COEF["current_rainfall"] * X_te["current_rainfall"] / a["R1_mm"],
        "rainfall_3h_accumulated": COEF["rainfall_3h_accumulated"] * X_te["rainfall_3h_accumulated"] / a["R3_mm"],
        "rainfall_24h_accumulated": COEF["rainfall_24h_accumulated"] * X_te["rainfall_24h_accumulated"] / a["R24_mm"],
        "tide_level": COEF["tide_x_rain"] * tide_ex * wet,
    }
    out = pd.Series(0.0, index=C.FEATURES)
    for f, t in terms.items():
        out[f] = float((t - t.mean()).abs().mean())
    return out.sort_values(ascending=False)


def run(ctx=None) -> dict:
    ctx = ctx or load()
    model, X_te, y_te = ctx["model"], ctx["X_test"], ctx["y_test"]
    idx = rows_to_explain(X_te)
    Xs = X_te.iloc[idx].reset_index(drop=True)
    sv = shap_values(model, ctx["background"], Xs)
    ex = make_explainer(ctx["X_train"])
    bg = ctx["X_train"].sample(N_BG_DELETE, random_state=C.SEED + 1)
    thr = ctx["meta"]["thresholds"]["high"]
    rng = np.random.default_rng(C.SEED)

    jac, rho, stab = [], [], []
    drops = {"SHAP": np.zeros((len(Xs), K_MAX)), "LIME": np.zeros((len(Xs), K_MAX)), "aleatório": np.zeros((len(Xs), K_MAX))}
    lime_global = pd.Series(0.0, index=C.FEATURES)
    for r in range(len(Xs)):
        x = Xs.iloc[r]
        shap_rank = [C.FEATURES[j] for j in np.argsort(-np.abs(sv.values[r]))]
        lime_rank, lime_w = lime_ranking(model, ex, x.to_numpy())
        lime_global += pd.Series(lime_w).reindex(C.FEATURES).fillna(0.0)
        jac.append(jaccard(shap_rank[:5], lime_rank[:5]))
        rho.append(spearmanr([shap_rank.index(f) for f in C.FEATURES], [lime_rank.index(f) for f in C.FEATURES])[0])
        # estabilidade: 10 sementes do LIME na mesma linha
        tops = []
        for s in range(N_SEEDS):
            e = explain(model, make_explainer(ctx["X_train"], seed=s), x.to_numpy(), num_features=5)
            tops.append([feature_of_rule(rule) for rule, _ in e.as_list(label=1)])
        stab.append(np.mean([jaccard(a, b) for a, b in itertools.combinations(tops, 2)]))
        for k in range(1, K_MAX + 1):
            drops["SHAP"][r, k - 1] = deletion_drop(model, x, shap_rank[:k], bg, thr)
            drops["LIME"][r, k - 1] = deletion_drop(model, x, lime_rank[:k], bg, thr)
            drops["aleatório"][r, k - 1] = np.mean([deletion_drop(model, x, list(rng.choice(C.FEATURES, k, replace=False)), bg, thr)
                                                    for _ in range(N_RANDOM)])

    # --- Plano B: ranking global contra a fórmula ---
    truth = true_attribution(ctx["df_test"], X_te)
    true_feats = [f for f in truth.index if truth[f] > 0]
    shap_global = pd.Series(np.abs(sv.values).mean(axis=0), index=C.FEATURES).sort_values(ascending=False)
    lime_global = (lime_global / len(Xs)).sort_values(ascending=False)
    planb = {
        "true_ranking": truth.round(4).to_dict(),
        "shap_global_ranking": shap_global.round(5).to_dict(),
        "lime_global_ranking": lime_global.round(5).to_dict(),
        "shap_top4_contains_true": len(set(shap_global.index[:4]) & set(true_feats)),
        "lime_top4_contains_true": len(set(lime_global.index[:4]) & set(true_feats)),
        # spearmanr sobre os VALORES: empates (as 8 features com atribuição verdadeira zero) recebem a posição média.
        "spearman_shap_vs_truth": float(spearmanr(shap_global.reindex(C.FEATURES), truth.reindex(C.FEATURES))[0]),
        "spearman_lime_vs_truth": float(spearmanr(lime_global.reindex(C.FEATURES), truth.reindex(C.FEATURES))[0]),
        "shap_order_of_true_features": [f for f in shap_global.index if f in true_feats],
        "lime_order_of_true_features": [f for f in lime_global.index if f in true_feats],
        "true_features": true_feats,
    }

    def ms(v):
        v = np.asarray(v, dtype=float)
        return round(float(np.nanmean(v)), 4), round(float(np.nanstd(v)), 4)

    rows = [
        {"metrica": "concordancia_jaccard_top5_shap_lime", **dict(zip(("media", "desvio"), ms(jac))), "n": len(jac)},
        {"metrica": "concordancia_spearman_shap_lime", **dict(zip(("media", "desvio"), ms(rho))), "n": len(rho)},
        {"metrica": "estabilidade_lime_jaccard_top5_10_sementes", **dict(zip(("media", "desvio"), ms(stab))), "n": len(stab)},
    ]
    for name, d in drops.items():
        for k in range(K_MAX):
            rows.append({"metrica": f"delecao_queda_prob_{name}_k{k + 1}", "media": round(float(d[:, k].mean()), 4),
                         "desvio": round(float(d[:, k].std()), 4), "n": len(Xs)})
    rows += [{"metrica": "planoB_spearman_shap_vs_formula", "media": round(planb["spearman_shap_vs_truth"], 4), "desvio": None, "n": 12},
             {"metrica": "planoB_spearman_lime_vs_formula", "media": round(planb["spearman_lime_vs_truth"], 4), "desvio": None, "n": 12},
             {"metrica": "planoB_shap_top4_verdadeiras_de_4", "media": planb["shap_top4_contains_true"], "desvio": None, "n": 4},
             {"metrica": "planoB_lime_top4_verdadeiras_de_4", "media": planb["lime_top4_contains_true"], "desvio": None, "n": 4}]
    table = pd.DataFrame(rows)
    table.to_csv(C.REPORTS_DIR / "explanation_eval.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ks = np.arange(1, K_MAX + 1)
    for name, d in drops.items():
        m, s = d.mean(axis=0), d.std(axis=0) / np.sqrt(len(d))
        ax.errorbar(ks, m, yerr=1.96 * s, marker="o", capsize=3, label=name)
    ax.set_xlabel("k features trocadas por valores do fundo"); ax.set_ylabel("queda média da prob. da classe prevista")
    ax.set_title(f"Teste de deleção ({len(Xs)} linhas; barras = IC 95%) — {C.SYNTHETIC_TAG}", fontsize=8)
    ax.legend(); fig.savefig(C.FIGURES_DIR / "deletion_curve.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    out = {"tag": C.SYNTHETIC_TAG, "rows_explained": idx.tolist(), "n_rows": int(len(Xs)),
           "positives_among_rows": int(y_te[idx].sum()), "table": rows, "plan_b": planb}
    (C.REPORTS_DIR / "explanation_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return out


if __name__ == "__main__":
    o = run()
    print(pd.DataFrame(o["table"]).to_string(index=False))
    print(json.dumps({k: v for k, v in o["plan_b"].items() if "ranking" not in k}, indent=1, ensure_ascii=False))
    print("Verdade:", o["plan_b"]["true_ranking"])
    print("SHAP:", list(o["plan_b"]["shap_global_ranking"])[:6])
    print("LIME:", list(o["plan_b"]["lime_global_ranking"])[:6])
