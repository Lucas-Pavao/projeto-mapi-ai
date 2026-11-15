"""
Etapa 3 do IN1165: modelos interpretáveis ANTES da caixa-preta.

Justificativa (vai para o relatório): um modelo que se lê por inteiro (coeficientes da regressão logística, regras da
árvore rasa) é a referência. A caixa-preta (XGBoost) só se justifica se o ganho sobre eles for significativo no teste,
porque a explicação dela é aproximada (SHAP/LIME) e a dos baselines é exata.

Mesmo split temporal do XGBoost (trainer.temporal_split) e as mesmas features em unidades físicas.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from mapi_ai import in1165_config as C
from mapi_ai.trainer import feature_matrix, load_dataset, temporal_split


def logistic_pipeline() -> Pipeline:
    """
    imputer → scaler → regressão logística, TUDO dentro do Pipeline: o fit do Pipeline ajusta a mediana do imputer e
    a média/desvio do scaler só no treino. Ajustar antes do split (como o trainer antigo) vaza informação do teste.
    A regressão linear precisa de escala comum (penalidade L2 trata todas as features igual); a árvore não.
    class_weight='balanced' faz o papel do scale_pos_weight do XGBoost.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=C.SEED)),
    ])


def shallow_tree(depth: int = 3) -> DecisionTreeClassifier:
    """Árvore de profundidade 3: no máximo 8 regras, legíveis numa linha cada. Aceita NaN nativamente (sklearn ≥ 1.3)."""
    return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", min_samples_leaf=20, random_state=C.SEED)


def train(df: pd.DataFrame) -> dict:
    tr, _, _ = temporal_split(df)
    X_tr, y_tr = feature_matrix(tr)
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    lr = logistic_pipeline().fit(X_tr, y_tr)
    joblib.dump(lr, C.MODELS_DIR / "baseline_logreg.joblib")
    coefs = dict(zip(C.FEATURES, lr.named_steps["logreg"].coef_[0].round(4).tolist()))

    tree = shallow_tree().fit(X_tr, y_tr)
    joblib.dump(tree, C.MODELS_DIR / "baseline_tree.joblib")
    rules = export_text(tree, feature_names=C.FEATURES, decimals=2)
    (C.REPORTS_DIR / "baseline_tree.txt").write_text(rules)
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(tree, feature_names=C.FEATURES, class_names=["não alaga", "alaga"], filled=True, impurity=False,
              proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Árvore rasa (profundidade 3) — {C.SYNTHETIC_TAG}")
    fig.savefig(C.FIGURES_DIR / "baseline_tree.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    info = {"logreg_standardized_coefficients": coefs, "tree_rules": rules,
            "justification": "testar o modelo interpretável antes da caixa-preta; o XGBoost só se justifica com ganho "
                             "significativo no teste (ver reports/metrics.csv e o IC por bootstrap)"}
    (C.REPORTS_DIR / "baselines.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(rules)
    print("Coeficientes padronizados da logística:", coefs)
    return info


if __name__ == "__main__":
    train(load_dataset(synthetic=True))
