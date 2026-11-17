"""Avaliação estratificada da Opção B (ADR 0025).

Separada da avaliação F1 congelada (`avaliacao/avaliar_h1_h2.py` e `avaliacao/HASHES.sha256` não são tocados).
Lê o CSV exportado com estratificação (`tier`, `sample_weight`, `temporal_match`, `rain_tier`, `rain_confidence`),
separa treino (`HIGH` + `MEDIUM` com `sample_weight`, chuva medida), tabela principal (`HIGH`, chuva medida),
exploratório (`LOW`) e relatório de reanálise (`REANALYSIS_FALLBACK`), sempre em split temporal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
MEASURED_TIERS = ("MEASURED_PRIMARY", "MEASURED_EXTENDED")
DEFAULT_FEATURES = (
    "weatherPrecipitation",
    "accumulated3h",
    "accumulated6h",
    "accumulated12h",
    "accumulated24h",
    "tideHeight",
)


def _stratum_summary(df: pd.DataFrame, target_col: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for tier in ("HIGH", "MEDIUM", "LOW", "DISCARD"):
        sub = df[df["tier"] == tier]
        y = pd.to_numeric(sub[target_col], errors="coerce")
        out[tier] = {
            "rows": int(len(sub)),
            "positives": int((y == 1).sum()),
            "negatives": int((y == 0).sum()),
        }
    return out


def _empty_message(stratum: str) -> str:
    return f"0 positivos no estrato {stratum} — avaliação não executável"


def evaluate_stratified(
    df: pd.DataFrame,
    target_col: str = "label_h6",
    train_frac: float = 0.70,
    seed: int = SEED,
) -> dict:
    """Executa a avaliação estratificada com split estritamente temporal."""
    required = {"timestampUtc", target_col, "tier", "sample_weight", "rain_tier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no export estratificado: {sorted(missing)}")

    work = df.copy()
    work["_ts"] = pd.to_datetime(work["timestampUtc"], utc=True)
    work = work.sort_values("_ts").reset_index(drop=True)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work["sample_weight"] = pd.to_numeric(work["sample_weight"], errors="coerce").fillna(0.0)

    counts_all = _stratum_summary(work, target_col)
    measured = work[work["rain_tier"].isin(MEASURED_TIERS)].copy()
    reanalysis = work[work["rain_tier"] == "REANALYSIS_FALLBACK"].copy()

    # Split estritamente temporal por instante único
    unique_ts = np.sort(work["_ts"].unique())
    if len(unique_ts) < 2:
        split_ts = unique_ts[0] if len(unique_ts) == 1 else None
    else:
        idx = max(1, min(len(unique_ts) - 1, int(len(unique_ts) * train_frac)))
        split_ts = unique_ts[idx]

    train_pool = measured[
        (measured["_ts"] < split_ts)
        & (measured["tier"].isin(["HIGH", "MEDIUM"]))
        & (measured[target_col].isin([0, 1]))
        & (measured["sample_weight"] > 0)
    ].copy() if split_ts is not None else measured.iloc[0:0]

    test_high = measured[
        (measured["_ts"] >= split_ts)
        & (measured["tier"] == "HIGH")
        & (measured[target_col].isin([0, 1]))
    ].copy() if split_ts is not None else measured.iloc[0:0]

    test_sens = measured[
        (measured["_ts"] >= split_ts)
        & (measured["tier"].isin(["HIGH", "MEDIUM"]))
        & (measured[target_col].isin([0, 1]))
    ].copy() if split_ts is not None else measured.iloc[0:0]

    bronze = work[(work["tier"] == "LOW") & (work[target_col].isin([0, 1]))].copy()

    report: dict = {
        "policy": "opcao_b_estratificada",
        "seed": seed,
        "target_col": target_col,
        "counts_all": counts_all,
        "temporal_day_only_rows": int((work.get("temporal_match", pd.Series(dtype=str)) == "TEMPORAL_DAY_ONLY").sum()),
        "reanalysis_report": {
            "rows": int(len(reanalysis)),
            "positives": int((reanalysis[target_col] == 1).sum()),
            "note": "Separado da métrica estrita (nunca entra no treino nem na tabela principal).",
        },
        "bronze_exploratory": {
            "rows": int(len(bronze)),
            "positives": int((bronze[target_col] == 1).sum()),
            "status": _empty_message("LOW (Bronze)") if int((bronze[target_col] == 1).sum()) == 0 else "disponivel_apenas_exploratorio",
        },
    }

    train_pos = int((train_pool[target_col] == 1).sum())
    if train_pos == 0 or len(train_pool) == 0 or train_pool[target_col].nunique() < 2:
        report["status"] = "nao_executavel"
        report["main_table_high"] = {"status": _empty_message("HIGH/MEDIUM (treino)")}
        report["sensitivity_high_medium"] = {"status": _empty_message("HIGH/MEDIUM (treino)")}
        return report

    feat_cols = [c for c in DEFAULT_FEATURES if c in work.columns]
    if not feat_cols:
        raise ValueError("Nenhuma feature numérica encontrada para avaliação.")

    X_tr = train_pool[feat_cols].apply(pd.to_numeric, errors="coerce")
    med = X_tr.median().fillna(0.0)
    X_tr = X_tr.fillna(med)
    y_tr = train_pool[target_col].astype(int).to_numpy()
    w_tr = train_pool["sample_weight"].to_numpy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
    ])
    model.fit(X_tr, y_tr, clf__sample_weight=w_tr)

    def _eval_subset(sub: pd.DataFrame, label: str) -> dict:
        pos = int((sub[target_col] == 1).sum())
        if pos == 0 or len(sub) == 0:
            return {"status": _empty_message(label), "rows": int(len(sub)), "positives": pos}
        X_te = sub[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(med)
        y_te = sub[target_col].astype(int).to_numpy()
        p_te = model.predict_proba(X_te)[:, 1]
        return {
            "status": "ok",
            "rows": int(len(sub)),
            "positives": pos,
            "pr_auc": float(average_precision_score(y_te, p_te)),
            "brier": float(brier_score_loss(y_te, p_te)),
        }

    report["status"] = "ok"
    report["features_used"] = feat_cols
    report["main_table_high"] = _eval_subset(test_high, "HIGH (teste oficial)")
    report["sensitivity_high_medium"] = _eval_subset(test_sens, "HIGH+MEDIUM (sensibilidade)")
    if len(reanalysis) > 0 and int((reanalysis[target_col] == 1).sum()) > 0:
        report["reanalysis_report"]["eval"] = _eval_subset(reanalysis, "REANALYSIS_FALLBACK")
    return report


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print("Uso: python -m avaliacao.opcao_b.avaliar_estratificado DATASET.csv SAIDA.json [TARGET]", file=sys.stderr)
        return 2
    csv_path, out_path = Path(args[0]), Path(args[1])
    target = args[2] if len(args) > 2 else "label_h6"
    df = pd.read_csv(csv_path)
    res = evaluate_stratified(df, target_col=target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
