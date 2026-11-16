"""
Rótulo pelo balde M1a (avaliacao/balde.py, ADR 0014) sobre a chuva medida do ponto, hora a hora, 5 anos.
Positivo: lâmina >= 15 cm. O M1a não tem maré (g = 1; o M1b não roda, ADR 0009): a maré não tem papel no rótulo.
Uso: python -m mapi_ai.v4.labels
"""
import json

import numpy as np
import pandas as pd

from avaliacao.balde import Params, simular_ponto
from avaliacao.entradas import carregar_terreno
from mapi_ai.sources import load_hourly
from mapi_ai.v3.evaluate import episodes
from mapi_ai.v4 import config as C


def terrain():
    return carregar_terreno(C.TERRAIN_POINT, C.TERRAIN_CURVE)


def simulate(f: float) -> pd.DataFrame:
    """slug, hour_end_utc, rain, h_mm, flood (NaN onde o estado do balde é desconhecido)."""
    T = terrain()
    df = load_hourly("cemaden")
    out = []
    for slug in C.POINTS:
        g = df[df.slug == slug].set_index("hour_end_utc").sort_index()
        g = g[~g.index.duplicated()].asfreq("h")
        P = g.current_rainfall
        h, _ = simular_ponto(P.fillna(0.0).to_numpy(), T[slug], Params(f=f))
        unknown = P.isna().astype(float).rolling(C.GAP_MASK_H + 1, min_periods=1).max().astype(bool).to_numpy()
        flood = np.where(unknown, np.nan, (h >= C.HCRIT_MM).astype(float))
        out.append(pd.DataFrame({"slug": slug, "hour_end_utc": g.index, "rain": P.to_numpy(), "tide_level": g.tide_level.to_numpy(),
                                 "h_mm": np.where(unknown, np.nan, h), "flood_h0": flood}))
    return pd.concat(out, ignore_index=True)


def frequency() -> dict:
    """Eventos por ponto e por ano, para cada f da faixa pré-registrada. Evento = horas positivas a menos de 6 h."""
    res = {}
    for f in C.F_GRID:
        s = simulate(f)
        rows = {}
        for slug, g in s.groupby("slug"):
            years = g.flood_h0.notna().sum() / 8766.0
            idx = np.flatnonzero((g.flood_h0 == 1).to_numpy())
            ev = episodes(idx, 6)
            rows[slug] = {"horas_positivas": int(len(idx)), "eventos": len(ev), "eventos_por_ano": round(len(ev) / years, 2),
                          "lamina_maxima_mm": round(float(np.nanmax(g.h_mm)), 1), "anos": round(years, 2),
                          "horas_sem_rotulo_pct": round(100 * g.flood_h0.isna().mean(), 2)}
        res[f"f={f}"] = rows
    return res


if __name__ == "__main__":
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fr = frequency()
    out = {"label_tag": C.LABEL_TAG, "attribution": C.ATTRIBUTION, "hcrit_mm": C.HCRIT_MM,
           "parametros_centrais": {"C": 0.86, "iT_mm_h": 60.1, "f": C.F_CENTRAL, "snap_m": 10},
           "frequencia_real_declarada": C.REAL_FREQUENCY, "faixa_f": "PARAMETROS-MODELO-FISICO.md, capacidade da galeria [sem fonte]",
           "frequencia_por_f": fr,
           "nota": "Nenhum parâmetro do balde foi ajustado para acertar a frequência. A diferença fica registrada."}
    (C.REPORTS_DIR / "bucket_label_frequency.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(pd.DataFrame({f: {s: f"{r['eventos']} ({r['eventos_por_ano']}/ano)" for s, r in v.items()} for f, v in fr.items()}).to_string())
    print(pd.DataFrame({f: {s: r['lamina_maxima_mm'] for s, r in v.items()} for f, v in fr.items()}).to_string())
