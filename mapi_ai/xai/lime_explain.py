"""
LIME do XGBoost v2 (Etapa 5), nos mesmos três casos do SHAP.

LIME ajusta, em volta de UM caso, um modelo linear simples a perturbações do caso, e lê os pesos desse modelo.
- Unidades originais: o explainer recebe o treino em mm e m (sem padronizar), e as regras saem como
  "rainfall_3h_accumulated > 3.56", legíveis.
- discretize_continuous=True: cada feature contínua vira faixas por quartil do treino; o peso é de "estar nesta faixa".
- LIME não aceita NaN: só para gerar as perturbações, os NaN do treino viram a mediana (o modelo explicado continua
  recebendo o que o LIME gera). Isso é declarado aqui e no relatório.
- LIME é estocástico (amostra perturbações): a Etapa 6 mede a estabilidade com 10 sementes.
"""
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from mapi_ai import in1165_config as C
from mapi_ai.xai.common import load, pick_cases, proba

NUM_FEATURES = 6


def make_explainer(X_train: pd.DataFrame, seed: int = C.SEED) -> LimeTabularExplainer:
    train = X_train.fillna(X_train.median())
    return LimeTabularExplainer(train.to_numpy(), feature_names=C.FEATURES, class_names=["não alaga", "alaga"],
                                discretize_continuous=True, mode="classification", random_state=seed)


def explain(model, explainer, x: np.ndarray, num_features=NUM_FEATURES, num_samples=5000):
    exp = explainer.explain_instance(np.nan_to_num(x, nan=0.0) if np.isnan(x).any() else x,
                                     lambda Z: np.column_stack([1 - proba(model, Z), proba(model, Z)]),
                                     num_features=num_features, num_samples=num_samples, labels=(1,))
    return exp


def feature_of_rule(rule: str) -> str:
    """'rainfall_3h_accumulated > 3.56' -> 'rainfall_3h_accumulated' (a regra traz a faixa; a feature é o nome)."""
    for f in sorted(C.FEATURES, key=len, reverse=True):
        if f in rule:
            return f
    return rule


def run(ctx=None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctx = ctx or load()
    model, X_te, y_te = ctx["model"], ctx["X_test"], ctx["y_test"]
    cases = ctx.get("cases") or pick_cases(model, X_te, y_te, ctx["meta"]["thresholds"]["high"])
    ex = make_explainer(ctx["X_train"])
    out = {}
    for name, i in cases.items():
        e = explain(model, ex, X_te.iloc[i].to_numpy())
        rules = e.as_list(label=1)
        out[name] = {"rules": [{"rule": r, "weight": round(w, 5), "feature": feature_of_rule(r)} for r, w in rules],
                     "local_model_r2": round(float(e.score), 4), "local_prediction": float(e.local_pred[0])}
        fig = e.as_pyplot_figure(label=1)
        fig.suptitle(f"LIME {name} — {C.SYNTHETIC_TAG}", fontsize=8)
        fig.savefig(C.FIGURES_DIR / f"xai_lime_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, ensure_ascii=False))
