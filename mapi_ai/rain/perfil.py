"""
Perfil da chuva prevista no tempo (onda 16, Fase 4.2; projeto-mapi-api/docs/PRE-REGISTRO-PERFIL-CHUVA.md, escrito antes).

Alvo: peak10_{h}h = maior chuva de 10 min do ponto em (t, t+h] (maior valor entre as estações a até 5 km válidas no início e no
fim, como y_{h}h). Opções: (u) uniforme; (a) regressor de quantis do pico (XGBoost, mesmas entradas do modelo servido no
horizonte, mesmos cortes e parada antecipada do rain_v1); (b) bloco alternado da IDF de Silva e Araújo. Avaliação só no teste,
só com y_h >= 1 mm. Uso: python -m mapi_ai.rain.perfil
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from mapi_ai import bucket10 as B10
from mapi_ai.rain import config as C
from mapi_ai.rain import dataset as D
from mapi_ai.rain import features as F

OUT = C.ROOT / "reports" / "perfil_chuva"
MODEL_PATH = C.MODELS_DIR / "rain_v1p_pico10.joblib"
CHOSEN = {1: "gb_rede", 3: "gb_rede_om", 6: "gb_rede_om"}
QS = (0.5, 0.95)
CHUVA_MIN_MM = 1.0
N_BOOT = 1000
VERSION = "rain_v1p"


def peak10_targets(readings, stations, lat, lon, times, horizons=C.HORIZONS) -> pd.DataFrame:
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    geo = F.station_geometry(stations, lat, lon)
    near = list(geo.station_id[geo.near])
    m, valid = F.grid(readings, near, times.min() - F.LOOKBACK, times.max() + pd.Timedelta(hours=max(horizons)))
    vals = m.fillna(0.0)
    out = pd.DataFrame(index=times)
    for h in horizons:
        n = h * 6
        fwd = vals[::-1].rolling(n, min_periods=1).max()[::-1].shift(-1)    # máximo dos passos t+1 .. t+n
        ok = valid.shift(-n, fill_value=False) & valid
        out[f"peak10_{h}h"] = fwd.where(ok).max(axis=1).reindex(times)
    return out


def alt_block_fraction(h: int, tr: float = 5.0) -> float:
    """Fração do maior bloco de 10 min no bloco alternado da IDF para a duração h (não depende do TR)."""
    P = lambda d: B10.idf_silva_araujo(tr, d) * d / 60.0
    return P(10.0) / P(60.0 * h)


def alt_block_profile(h: int, tr: float = 5.0) -> np.ndarray:
    """Pesos dos n passos (somam 1): maior bloco no meio, os outros alternando dos dois lados."""
    n = h * 6
    P = np.array([B10.idf_silva_araujo(tr, 10.0 * k) * 10.0 * k / 60.0 for k in range(1, n + 1)])
    blocks = np.diff(np.r_[0.0, P])
    w = np.zeros(n)
    mid = (n - 1) // 2
    order = [mid]
    for k in range(1, n):
        order.append(mid + (k + 1) // 2 if k % 2 else mid - k // 2)
    for b, pos in zip(sorted(blocks, reverse=True), order):
        w[pos] = b
    return w / w.sum()


def peak_profile(acc_mm: float, peak_mm: float, h: int) -> np.ndarray:
    """Série da opção (a): pico no passo do meio, limitado ao acumulado; o resto por igual nos outros passos."""
    n = h * 6
    pk = min(max(peak_mm, 0.0), max(acc_mm, 0.0))
    rest = (max(acc_mm, 0.0) - pk) / (n - 1) if n > 1 else 0.0
    s = np.full(n, rest)
    s[(n - 1) // 2] = pk
    return s


def _features(h):
    return list(F.BASE_FEATURES if CHOSEN[h] == "gb_rede" else F.ALL_FEATURES)


def train_peak(train: pd.DataFrame) -> dict:
    origin = pd.Timestamp("2021-01-04", tz="UTC")
    week = (train.t - origin) // pd.Timedelta(days=7)
    hold = (week % 7 == 3)
    into = (train.t - origin) % pd.Timedelta(days=7)
    near = ((week % 7 == 2) & (into >= pd.Timedelta(days=7) - C.GUARD)) | ((week % 7 == 4) & (into < C.GUARD))
    a, b = train[~hold & ~near], train[hold]
    models = {}
    for h in C.HORIZONS:
        y = f"peak10_{h}h"
        da, db = a.dropna(subset=[y]), b.dropna(subset=[y])
        r = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QS), **C.XGB)
        r.fit(da[_features(h)], da[y], eval_set=[(db[_features(h)], db[y])], verbose=False)
        models[h] = r
    return models


def predict_peak(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for h, r in models.items():
        q = np.sort(np.clip(r.predict(df[_features(h)]), 0, None), axis=1)
        out[f"pk_q50_{h}h"], out[f"pk_q95_{h}h"] = q[:, 0], q[:, 1]
    return pd.DataFrame(out, index=df.index)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    ds = D.load()
    readings, stations, points = D.load_readings(), pd.read_csv(C.STATIONS), pd.read_csv(C.POINTS)
    parts = []
    for p in points.itertuples():
        g = ds[ds.slug == p.slug]
        tg = peak10_targets(readings, stations, p.latitude, p.longitude, g.t)
        tg.index = g.index
        parts.append(tg)
    ds = ds.join(pd.concat(parts))
    train = ds[ds.part == "train"].iloc[::C.TRAIN_STEP]
    models = train_peak(train)
    joblib.dump({"models": models, "quantiles": QS, "chosen_features": {h: _features(h) for h in C.HORIZONS},
                 "version": VERSION}, MODEL_PATH, compress=3)
    test = ds[ds.part == "test"].copy()
    served = {h: joblib.load(C.MODELS_DIR / f"{CHOSEN[h]}.joblib").predict(test) for h in C.HORIZONS}
    pk = predict_peak(models, test)
    res = {"pre_registro": "projeto-mapi-api/docs/PRE-REGISTRO-PERFIL-CHUVA.md", "versao_a": VERSION,
           "criterio": "menor MAE médio do pico de 10 min (q50) nos 3 horizontes entre (a) e (b); empate (IC95 da diferença "
                       "com zero) fica (b)", "horizontes": {}, "fracao_bloco_alternado": {}}
    days = test.t.dt.tz_convert("America/Recife").dt.date.astype(str)
    diffs = []
    for h in C.HORIZONS:
        n = h * 6
        acc50 = served[h][f"q50_{h}h"].to_numpy()
        y = test[f"peak10_{h}h"].to_numpy()
        sel = (test[f"y_{h}h"].to_numpy() >= CHUVA_MIN_MM) & np.isfinite(y)
        fb = alt_block_fraction(h)
        res["fracao_bloco_alternado"][f"{h}h"] = round(fb, 4)
        pred = {"u": acc50 / n, "b": fb * acc50, "a": np.minimum(pk[f"pk_q50_{h}h"].to_numpy(), acc50)}
        err = {k: np.abs(v[sel] - y[sel]) for k, v in pred.items()}
        bias = {k: float(np.mean(v[sel] - y[sel])) for k, v in pred.items()}
        dd = days.to_numpy()[sel]
        ud = np.unique(dd)
        idx = {d: np.flatnonzero(dd == d) for d in ud}
        boots = []
        for _ in range(N_BOOT):
            pick = np.concatenate([idx[d] for d in rng.choice(ud, size=len(ud))])
            boots.append(err["a"][pick].mean() - err["b"][pick].mean())
        diffs.append(np.array(boots))
        y95 = np.minimum(pk[f"pk_q95_{h}h"].to_numpy(), served[h][f"q95_{h}h"].to_numpy())
        res["horizontes"][f"{h}h"] = {
            "linhas_com_chuva": int(sel.sum()), "dias": int(len(ud)),
            "pico_medido_mm": {"mediana": round(float(np.median(y[sel])), 2), "p95": round(float(np.quantile(y[sel], 0.95)), 2)},
            "mae_mm": {k: round(float(v.mean()), 3) for k, v in err.items()},
            "vies_mm": {k: round(v, 3) for k, v in bias.items()},
            "diferenca_mae_a_menos_b": {"valor": round(float(err["a"].mean() - err["b"].mean()), 3),
                                        "ic95": [round(float(np.quantile(boots, 0.025)), 3), round(float(np.quantile(boots, 0.975)), 3)]},
            "cobertura_q95_pico_a": round(float(np.mean(y[sel] <= y95[sel])), 3),
            "best_iteration_a": int(models[h].best_iteration)}
    mae_a = np.mean([res["horizontes"][f"{h}h"]["mae_mm"]["a"] for h in C.HORIZONS])
    mae_b = np.mean([res["horizontes"][f"{h}h"]["mae_mm"]["b"] for h in C.HORIZONS])
    dmean = np.mean(np.stack(diffs), axis=0)
    ic = [float(np.quantile(dmean, 0.025)), float(np.quantile(dmean, 0.975))]
    empate = ic[0] <= 0 <= ic[1]
    escolha = "b" if empate else ("a" if mae_a < mae_b else "b")
    res["resultado"] = {"mae_medio": {"a": round(float(mae_a), 3), "b": round(float(mae_b), 3)},
                        "diferenca_media_a_menos_b_ic95": [round(ic[0], 3), round(ic[1], 3)], "empate": bool(empate),
                        "escolha": escolha}
    (OUT / "perfil_chuva.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, ensure_ascii=False))
