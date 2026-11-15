"""
Peças compartilhadas pela explicabilidade: dados no mesmo split do treino, o modelo servido, o fundo do SHAP e a
escolha dos três casos do teste (verdadeiro positivo, falso negativo, falso positivo).
"""
import json

import joblib
import numpy as np
import pandas as pd

from mapi_ai import in1165_config as C
from mapi_ai.trainer import feature_matrix, load_dataset, temporal_split

N_BACKGROUND = 200


def load():
    """Devolve dict com X/y de treino e teste, o modelo XGBoost v2, o metadata e o fundo de 200 linhas do treino."""
    df = load_dataset(synthetic=True)
    tr, va, te = temporal_split(df)
    X_tr, y_tr = feature_matrix(tr)
    X_te, y_te = feature_matrix(te)
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    model = joblib.load(C.MODELS_DIR / "xgb_v2.joblib")
    # Fundo do SHAP: 200 linhas do TREINO (nunca do teste), sorteadas com a semente. É a "linha média" contra a qual
    # cada contribuição é medida: SHAP explica f(x) − E[f(fundo)].
    background = X_tr.sample(N_BACKGROUND, random_state=C.SEED)
    return {"df_test": te.reset_index(drop=True), "X_train": X_tr, "y_train": y_tr, "X_test": X_te.reset_index(drop=True),
            "y_test": y_te, "model": model, "meta": meta, "background": background}


def proba(model, X) -> np.ndarray:
    return model.predict_proba(pd.DataFrame(X, columns=C.FEATURES))[:, 1]


def pick_cases(model, X_test, y_test, threshold) -> dict:
    """
    Três casos do teste, escolhidos por regra fixa (não à mão), no limiar HIGH:
    - TP: verdadeiro positivo com a maior probabilidade (o caso "claro");
    - FN: alagou e o modelo não avisou; o de maior probabilidade (o quase-acerto, mais informativo);
    - FP: avisou e não alagou; o de maior probabilidade (o alarme falso mais confiante).
    """
    p = proba(model, X_test)
    yhat = p >= threshold
    out = {}
    for name, mask in (("TP", yhat & (y_test == 1)), ("FN", ~yhat & (y_test == 1)), ("FP", yhat & (y_test == 0))):
        idx = np.flatnonzero(mask)
        if len(idx):
            out[name] = int(idx[np.argmax(p[idx])])
    return out


def case_record(df_test, X_test, y_test, i, p) -> dict:
    row = df_test.iloc[i]
    return {"index_in_test": int(i), "timestamp": str(row.timestamp), "slug": row.slug, "stratum": row.stratum,
            "y_true": int(y_test[i]), "probability": float(p[i]), "p_true_generator": float(row.p_true),
            "features": {f: (None if pd.isna(X_test.iloc[i][f]) else float(X_test.iloc[i][f])) for f in C.FEATURES}}
