"""
Verificação da Etapa 1: `python -m mapi_ai.dataset_report`.

Imprime o resumo dos rótulos reais (por que o Plano B), do dataset usado no treino e da origem de cada arquivo
(SHA-256), e grava reports/dataset_summary.json.
"""
import hashlib
import json

import pandas as pd

from mapi_ai import in1165_config as C
from mapi_ai.build_labels import summary as real_summary


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_bounds(df: pd.DataFrame) -> dict:
    n = len(df)
    a, b = int(n * C.SPLIT[0]), int(n * (C.SPLIT[0] + C.SPLIT[1]))
    out = {}
    for name, part in (("train", df.iloc[:a]), ("validation", df.iloc[a:b]), ("test", df.iloc[b:])):
        out[name] = {"rows": int(len(part)), "positives": int(part[C.LABEL].sum()),
                     "from": str(part.timestamp.min()), "to": str(part.timestamp.max())}
    return out


def build_summary() -> dict:
    df = pd.read_csv(C.DATASET_CSV, parse_dates=["timestamp"])
    truth = json.loads((C.REPORTS_DIR / "synthetic_ground_truth.json").read_text())
    return {
        "tag": C.SYNTHETIC_TAG,
        "real_labels": real_summary(),
        "dataset": {
            "path": str(C.DATASET_CSV.relative_to(C.ROOT)), "sha256": sha(C.DATASET_CSV), "rows": int(len(df)),
            "positives": int(df[C.LABEL].sum()), "positive_rate": round(float(df[C.LABEL].mean()), 4),
            "points": int(df.slug.nunique()), "period": [str(df.timestamp.min()), str(df.timestamp.max())],
            "by_stratum": df.groupby("stratum")[C.LABEL].agg(["count", "sum"]).rename(
                columns={"count": "rows", "sum": "positives"}).to_dict("index"),
            "by_point": df.groupby("slug")[C.LABEL].agg(["count", "sum"]).rename(
                columns={"count": "rows", "sum": "positives"}).to_dict("index"),
            "missing_rate": {f: round(float(df[f].isna().mean()), 4) for f in C.FEATURES},
            "split": split_bounds(df.sort_values("timestamp")),
        },
        "generator": {"formula": truth["formula"], "b0": truth["b0"], "true_features_ranked": truth["true_features_ranked"]},
        "sources_sha256": {p.name: sha(p) for p in (C.CEMADEN_HOURLY, C.REANALYSIS_HOURLY, C.TIDE_HOURLY,
                                                     C.OCCURRENCES, C.POINTS, C.RAIN_EVENTS)},
        "rain_source": "MEASURED (Cemaden PED, point_rain_accum_v chuva-ponto-v4); reanálise só na ablação",
        "attribution": C.ATTRIBUTION,
    }


if __name__ == "__main__":
    s = build_summary()
    C.REPORTS_DIR.mkdir(exist_ok=True)
    (C.REPORTS_DIR / "dataset_summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, default=str))
    r = s["real_labels"]
    d = s["dataset"]
    print(f"== {C.SYNTHETIC_TAG} ==")
    print(f"Rótulos reais: flood_scenario_labels = {r['flood_scenario_labels_rows']} linhas; CKAN {r['occurrences']} "
          f"ocorrências em {r['occurrence_days']} dias, {r['with_hour']} com hora.")
    for k, v in r["by_radius"].items():
        print(f"  a {k}: {v['positive_point_days']} pontos-dia positivos")
    print(f"Decisão: {r['decision']}")
    print(f"Dataset: {d['rows']} linhas, {d['positives']} positivas ({d['positive_rate']:.1%}), {d['points']} pontos, "
          f"{d['period'][0]} a {d['period'][1]}")
    for k, v in d["split"].items():
        print(f"  {k:10s} {v['rows']:5d} linhas, {v['positives']:4d} positivas, {v['from'][:10]} a {v['to'][:10]}")
    print("Salvo em reports/dataset_summary.json")
