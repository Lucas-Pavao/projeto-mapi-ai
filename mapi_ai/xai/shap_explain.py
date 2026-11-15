"""
SHAP do XGBoost v2 (Etapa 5).

Log-odds × probabilidade: o XGBoost soma árvores na escala de log-odds (logit), e o SHAP padrão de árvore explica essa
soma. Uma contribuição de +1,0 em log-odds pode valer +2 pontos percentuais perto de p = 1% e +25 pontos perto de
p = 50%: não se lê como "chance a mais". Com model_output="probability" o SHAP explica direto a probabilidade, que é o
número que o usuário vê. O preço: na escala de probabilidade o SHAP de árvore exige
feature_perturbation="interventional" e um conjunto de fundo (aqui, 200 linhas do treino), e fica mais lento.

"interventional" quebra a correlação entre features ao substituir valores (usa o fundo), em vez de seguir os caminhos
da árvore. Com acumulados de chuva muito correlacionados (3 h, 6 h, 12 h), isso pode atribuir crédito a combinações
pouco realistas; a avaliação da Etapa 6 olha justamente isso.

Aditividade (propriedade de eficiência do Shapley): para cada linha, base + soma das contribuições = probabilidade.
O assert abaixo verifica.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from mapi_ai import in1165_config as C
from mapi_ai.xai.common import case_record, load, pick_cases, proba


def explainer(model, background):
    # Masker explícito com max_samples=200: sem ele o SHAP reduz o fundo para 100 linhas por padrão.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return shap.TreeExplainer(model, data=masker, model_output="probability",
                              feature_perturbation="interventional")


def shap_values(model, background, X):
    ex = explainer(model, background)
    sv = ex(X)
    # Aditividade: base_value + Σφ = f(x), na escala de probabilidade.
    recon = sv.base_values + sv.values.sum(axis=1)
    assert np.allclose(recon, proba(model, X), atol=1e-4), "aditividade do SHAP falhou"
    return sv


def run(ctx=None) -> dict:
    ctx = ctx or load()
    model, X_te, y_te = ctx["model"], ctx["X_test"], ctx["y_test"]
    sv = shap_values(model, ctx["background"], X_te)
    tag = C.SYNTHETIC_TAG

    plt.figure()
    shap.summary_plot(sv.values, X_te, plot_type="bar", show=False)
    plt.title(f"SHAP global (|contribuição| média, probabilidade) — {tag}", fontsize=8)
    plt.savefig(C.FIGURES_DIR / "xai_shap_bar.png", dpi=130, bbox_inches="tight"); plt.close()

    plt.figure()
    shap.summary_plot(sv.values, X_te, show=False)
    plt.title(f"SHAP beeswarm — {tag}", fontsize=8)
    plt.savefig(C.FIGURES_DIR / "xai_shap_beeswarm.png", dpi=130, bbox_inches="tight"); plt.close()

    # Interação chuva × maré: a maré só pesa quando chove (é assim na fórmula geradora).
    shap.dependence_plot("rainfall_3h_accumulated", sv.values, X_te, interaction_index="tide_level", show=False)
    plt.title(f"SHAP da chuva de 3 h, cor = maré (m) — {tag}", fontsize=8)
    plt.savefig(C.FIGURES_DIR / "xai_shap_dependence_rain3h_tide.png", dpi=130, bbox_inches="tight"); plt.close()

    thr = ctx["meta"]["thresholds"]["high"]
    cases = pick_cases(model, X_te, y_te, thr)
    p = proba(model, X_te)
    out = {"base_value": float(np.atleast_1d(sv.base_values)[0]), "threshold_high": thr, "cases": {}}
    for name, i in cases.items():
        plt.figure()
        shap.plots.waterfall(sv[i], max_display=8, show=False)
        plt.title(f"{name}: p = {p[i]:.3f}, real = {y_te[i]} — {tag}", fontsize=8)
        plt.savefig(C.FIGURES_DIR / f"xai_shap_waterfall_{name}.png", dpi=130, bbox_inches="tight"); plt.close()
        rec = case_record(ctx["df_test"], X_te, y_te, i, p)
        rec["shap"] = dict(sorted(zip(C.FEATURES, sv.values[i].round(5).tolist()), key=lambda kv: -abs(kv[1])))
        out["cases"][name] = rec
    imp = np.abs(sv.values).mean(axis=0)
    out["global_mean_abs"] = dict(sorted(zip(C.FEATURES, imp.round(5).tolist()), key=lambda kv: -kv[1]))
    out["additivity_checked"] = True
    ctx["shap"] = sv
    ctx["cases"] = cases
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["global_mean_abs"], indent=1, ensure_ascii=False))
