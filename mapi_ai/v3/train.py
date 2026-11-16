"""
Treino e escolha do modelo de produção do v3 (etapa B4).

Quatro candidatos, do mais simples ao mais complexo, no mesmo split temporal e nos mesmos dados:

- regra_chuva3h: a própria chuva de 3 h como escore CONTÍNUO. No v2 a regra entrava como saída 0/1 (um ponto de operação
  só) e a PR-AUC dela (0,460) ficava injustamente baixa; aqui ela é comparada como os outros, pela curva inteira.
- logistica: Pipeline (mediana + indicador de ausência por janela de chuva -> StandardScaler ->
  LogisticRegression(class_weight="balanced")). O imputador fica DENTRO do pipeline e é ajustado só no treino.
- arvore_rasa: árvore de profundidade 3, com NaN nativo.
- xgboost: a mesma grade pequena do v2; NaN fica NaN.

Critério de escolha, fixado ANTES de treinar e gravado no metadata antes do primeiro fit (C.SELECTION): maior PR-AUC
na validação, no horizonte principal; empate dentro do IC 95% do bootstrap pareado fica com o mais simples. O teste
não entra na escolha.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mapi_ai.models import get_xgboost_classifier
from mapi_ai.trainer import PARAM_GRID, _versions, scale_pos_weight
from mapi_ai.v3 import config as C
from mapi_ai.v3 import dataset as D


def X_of(df: pd.DataFrame, features=None, full: bool = False) -> pd.DataFrame:
    """Matriz de entradas na ordem de C.FEATURES. full=True usa as janelas SEM lacunas (para isolar o efeito delas)."""
    features = features or C.FEATURES
    X = df[features].copy()
    if full:
        for w in C.WINDOWS:
            X[w] = df[w + "_full"].to_numpy()
    return X.astype(float)


class RainRule(BaseEstimator, ClassifierMixin):
    """Escore = chuva de 3 h (janelas 0-1 h + 1-3 h), contínuo e monotônico. Ausente = 0 (não dispara)."""
    _estimator_type = "classifier"

    def fit(self, X, y=None):
        self.scale_ = float(np.nanquantile(X["rain_0_1h"] + X["rain_1_3h"], 0.999)) or 1.0
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = (X["rain_0_1h"] + X["rain_1_3h"]).fillna(0).to_numpy() / self.scale_
        p = np.clip(s, 0, 1)
        return np.column_stack([1 - p, p])


def base_feature(transformed_name: str) -> str:
    """Nome da entrada original de uma coluna do pré-processamento ('valor__x', 'ausente__missingindicator_x')."""
    name = transformed_name.split("__", 1)[1]
    return name[len("missingindicator_"):] if name.startswith("missingindicator_") else name


def logistic_pipeline(features=None) -> Pipeline:
    features = features or C.FEATURES
    missing_capable = [f for f in features if f in C.WINDOWS or f in C.PHYSICAL]
    rest = [f for f in features if f not in missing_capable]
    prep = ColumnTransformer([
        ("valor", SimpleImputer(strategy="median"), missing_capable),
        ("ausente", MissingIndicator(features="all"), missing_capable),
        ("resto", SimpleImputer(strategy="median"), rest),
    ], verbose_feature_names_out=True)
    return Pipeline([("prep", prep), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=C.SEED))])


def fit_candidates(X_tr, y_tr, X_va, y_va, features=None) -> dict:
    """Ajusta os quatro candidatos; o XGBoost escolhe hiperparâmetros pela PR-AUC de validação, como no v2."""
    out = {"regra_chuva3h": RainRule().fit(X_tr), "logistica": logistic_pipeline(features).fit(X_tr, y_tr),
           "arvore_rasa": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=C.SEED).fit(X_tr, y_tr)}
    spw, best = scale_pos_weight(y_tr), None
    for params in PARAM_GRID:
        clf = get_xgboost_classifier(scale_pos_weight=spw, random_state=C.SEED, **params).fit(X_tr, y_tr)
        ap = average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])
        if best is None or ap > best[0]:
            best = (ap, params, clf)
    out["xgboost"] = best[2]
    out["_xgb_params"] = {**best[1], "scale_pos_weight": spw}
    return out


def paired_bootstrap(y, pa, pb, n_boot=C.N_BOOT, seed=C.SEED) -> tuple[float, float, float]:
    """PR-AUC(a) - PR-AUC(b) com IC 95%, reamostrando as mesmas linhas para os dois."""
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        diffs.append(average_precision_score(y[i], pa[i]) - average_precision_score(y[i], pb[i]))
    d = np.array(diffs)
    return (float(average_precision_score(y, pa) - average_precision_score(y, pb)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def select(y_va, p_va: dict) -> dict:
    """Aplica C.SELECTION: o melhor na validação e, entre os empatados com ele, o mais simples."""
    ap = {m: float(average_precision_score(y_va, p)) for m, p in p_va.items()}
    best = max(ap, key=ap.get)
    table, chosen = [], None
    for m in C.SIMPLICITY:
        if m == best:
            d, lo, hi = 0.0, 0.0, 0.0
        else:
            d, lo, hi = paired_bootstrap(y_va, p_va[best], p_va[m])
        tie = m == best or lo <= 0.0
        table.append({"model": m, "val_pr_auc": ap[m], "diff_best_minus_model": d, "ci95_lo": lo, "ci95_hi": hi,
                      "tied_with_best": bool(tie)})
        if chosen is None and tie:
            chosen = m
    return {"best_by_pr_auc": best, "chosen": chosen, "table": table}


def parts(df: pd.DataFrame):
    return df[df.part == "train"], df[df.part == "validation"], df[df.part == "test"]


def run(features=None, tag: str = "v3", dataset: pd.DataFrame | None = None, extra_meta: dict | None = None,
        horizons=C.HORIZONS) -> dict:
    features = features or C.FEATURES
    df = D.load() if dataset is None else dataset
    tr, va, te = parts(df)
    out_dir = C.MODELS_DIR if tag == "v3" else C.MODELS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"

    # O critério vai para o metadata ANTES de qualquer treino.
    meta = {
        "model_generation": tag, "training_data": "SYNTHETIC",
        "training_data_note": C.SYNTHETIC_TAG + " — reports/v3/synthetic_ground_truth.md",
        "label": {"definition": "alaga em algum momento de t a t+H; entradas da hora t",
                  "horizons_hours": list(C.HORIZONS), "primary_horizon_hours": C.PRIMARY_HORIZON},
        "features": features, "feature_units": {f: C.UNITS[f] for f in features},
        "contract_inputs": C.ACCUMULATED + ["tide_level", "timestamp"],
        "derived_in_service": "janelas disjuntas por subtração dos acumulados; ausência se propaga; hora e mês em hora local",
        "removed_from_v2": ["latitude", "longitude"],
        "selection_criterion": C.SELECTION, "simplicity_order": C.SIMPLICITY,
        "calibration_rule": C.CALIBRATION_RULE,
        "split": {"train_end": str(C.TRAIN_END), "validation_end": str(C.VALID_END), "guard_hours": C.GUARD_H,
                  **{name: {"rows": int(len(p)), **{f"positives_h{h}": int(p[C.label(h)].sum()) for h in C.HORIZONS}}
                     for name, p in (("train", tr), ("validation", va), ("test", te))}},
        "seed": C.SEED, "versions": _versions(),
        "dataset_sha256": hashlib.sha256(C.DATASET.read_bytes()).hexdigest() if dataset is None else None,
        "rain_source": "MEASURED", "model_validation_status": "NOT_VALIDATED", "attribution": C.ATTRIBUTION,
        "status": "criterio gravado; treino em andamento", **(extra_meta or {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    X_tr, X_va = X_of(tr, features), X_of(va, features)
    meta["horizons"] = {}
    for h in horizons:
        y_tr, y_va = tr[C.label(h)].astype(int).to_numpy(), va[C.label(h)].astype(int).to_numpy()
        models = fit_candidates(X_tr, y_tr, X_va, y_va, features)
        xgb_params = models.pop("_xgb_params")
        p_va = {m: models[m].predict_proba(X_va)[:, 1] for m in C.SIMPLICITY}
        sel = select(y_va, p_va)
        for m, clf in models.items():
            joblib.dump(clf, out_dir / f"{m}_h{h}.joblib")
        meta["horizons"][f"h{h}"] = {"selection": sel, "xgboost_params": xgb_params,
                                     "train_positive_rate": float(y_tr.mean()), "validation_positive_rate": float(y_va.mean()),
                                     "artifacts": {m: f"{out_dir.relative_to(C.ROOT)}/{m}_h{h}.joblib" for m in models}}
        print(f"H={h}: melhor={sel['best_by_pr_auc']} escolhido={sel['chosen']} "
              + " ".join(f"{r['model']}={r['val_pr_auc']:.3f}" for r in sel["table"]))
    primary = meta["horizons"][f"h{C.PRIMARY_HORIZON}"]
    meta["production"] = {"model": primary["selection"]["chosen"], "horizon_hours": C.PRIMARY_HORIZON,
                          "artifact": primary["artifacts"][primary["selection"]["chosen"]],
                          "why": primary["selection"]}
    meta["model_version"] = f"{'logreg' if meta['production']['model'] == 'logistica' else meta['production']['model']}_v3"
    meta["status"] = "treinado; calibração e limiares pela avaliação (mapi_ai/v3/evaluate.py)"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


if __name__ == "__main__":
    # Pelo módulo importado: rodando como __main__, o pickle gravaria RainRule como __main__.RainRule.
    from mapi_ai.v3 import train as _train
    _train.run()
