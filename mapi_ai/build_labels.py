"""
Etapa 1 do IN1165: tentar construir o rótulo is_flooded a partir de registros REAIS.

Regra (configurável): uma ocorrência de alagamento do CKAN Recife vira positivo para um ponto crítico se estiver a até
`radius_m` dele; no tempo, com precisão HOUR casa as horas em ±`hour_tolerance_h`; com precisão DAY casa o dia local
inteiro (não há hora). Os negativos são as mesmas horas/dias nos mesmos pontos sem ocorrência.

AVISO DE RÓTULO (vale para qualquer uso destes dados): alagamento que ninguém registrou vira NEGATIVO. O rótulo mede o
que foi comunicado à Prefeitura, não o que aconteceu; o "recall" medido é recall sobre o que foi notificado.
Também não se usa a hora de pico da chuva para "posicionar" a ocorrência no dia: isso faria o rótulo depender da
própria chuva que o modelo usa como feature (rótulo circular, ADR 0006 do projeto-mapi-api).

Este módulo NÃO grava no banco: produz um resumo e um CSV local. A decisão de seguir para o Plano B (sintético) é
tomada aqui, com o piso de C.MIN_REAL_POSITIVES positivos.
"""
import argparse
import json

import numpy as np
import pandas as pd

from mapi_ai import in1165_config as C
from mapi_ai.feature_engineering import haversine_distance


def match(radius_m: float = 500.0, hour_tolerance_h: int = 1) -> pd.DataFrame:
    """Uma linha por (ponto, ocorrência) casados no espaço, com a janela temporal do rótulo."""
    occ = pd.read_csv(C.OCCURRENCES, parse_dates=["occurred_from", "occurred_to"])
    pts = pd.read_csv(C.POINTS)
    rows = []
    for _, o in occ.iterrows():
        if pd.isna(o.geocoded_lat):
            continue
        for _, p in pts.iterrows():
            d = haversine_distance(o.geocoded_lat, o.geocoded_lon, p.latitude, p.longitude) * 1000.0
            if d > radius_m:
                continue
            if o.temporal_precision == "HOUR":
                start = o.occurred_from - pd.Timedelta(hours=hour_tolerance_h)
                end = o.occurred_to + pd.Timedelta(hours=hour_tolerance_h)
            else:  # DAY: o dia local inteiro (occurred_from/to já cobrem o dia em UTC)
                start, end = o.occurred_from, o.occurred_to
            rows.append({"slug": p.slug, "occurrence_id": o.id, "distance_m": round(d, 1),
                         "temporal_precision": o.temporal_precision, "window_start_utc": start, "window_end_utc": end,
                         "local_day": pd.Timestamp(o.occurred_from).tz_convert("America/Recife").date().isoformat()})
    return pd.DataFrame(rows)


def summary(radii=(500, 1000, 2000, 3000)) -> dict:
    occ = pd.read_csv(C.OCCURRENCES, parse_dates=["occurred_from"])
    out = {
        "flood_scenario_labels_rows": 0,  # medido no banco em 2026-11-15 (handoff etapa0.json)
        "occurrences": int(len(occ)),
        "occurrence_days": int(occ.occurred_from.dt.tz_convert("America/Recife").dt.date.nunique()),
        "with_hour": int((occ.temporal_precision == "HOUR").sum()),
        "period": [str(occ.occurred_from.min()), str(occ.occurred_from.max())],
        "by_radius": {},
    }
    for r in radii:
        m = match(radius_m=r)
        out["by_radius"][f"{r}m"] = {
            "occurrence_point_pairs": int(len(m)),
            "positive_point_days": int(m[["slug", "local_day"]].drop_duplicates().shape[0]) if len(m) else 0,
            "points": sorted(m.slug.unique().tolist()) if len(m) else [],
        }
    pos = out["by_radius"]["500m"]["positive_point_days"]
    out["min_real_positives"] = C.MIN_REAL_POSITIVES
    out["decision"] = ("REAL" if pos >= C.MIN_REAL_POSITIVES else
                       f"PLANO_B: {pos} positivos reais a 500 m (< {C.MIN_REAL_POSITIVES}); nenhum com hora")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rótulos reais (Etapa 1 do IN1165)")
    ap.add_argument("--radius-m", type=float, default=500.0)
    ap.add_argument("--hour-tolerance-h", type=int, default=1)
    a = ap.parse_args()
    s = summary()
    m = match(a.radius_m, a.hour_tolerance_h)
    m.to_csv(C.DATA_DIR / "real_label_matches.csv", index=False)
    print(json.dumps(s, indent=2, ensure_ascii=False))
