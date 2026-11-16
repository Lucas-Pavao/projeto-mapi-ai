"""
Os cinco modelos da previsão de chuva, com a mesma interface: fit(treino) e predict(df) -> DataFrame com
q50_{h}h, q80_{h}h, q95_{h}h (mm em (t, t+h]) e pMEDIA_{h}h, pFORTE_{h}h (probabilidade do pico horário passar do limiar).
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from mapi_ai.rain import config as C
from mapi_ai.rain import features as F

QN = [f"q{int(q * 100)}" for q in C.QUANTILES]


def out_columns(h):
    return [f"{q}_{h}h" for q in QN] + [f"p{k}_{h}h" for k in C.THRESHOLDS]


class Binned:
    """Distribuição empírica do alvo no treino, condicionada a uma chave discreta. Chave nunca vista = distribuição global."""
    MIN_ROWS = 200

    def __init__(self, name, key):
        self.name, self.key = name, key           # key(df, h) -> Series de rótulos

    def fit(self, train: pd.DataFrame, valid=None):
        self.table, self.glob = {}, {}
        for h in C.HORIZONS:
            d = train.dropna(subset=[f"y_{h}h"])
            cols = {f"{q}_{h}h": d[f"y_{h}h"] for q in QN}
            ev = {f"p{k}_{h}h": d[f"ev_{k}_{h}h"].astype(float) for k in C.THRESHOLDS}
            g = pd.DataFrame({**cols, **ev}).groupby(self.key(d, h).to_numpy())
            t = g.agg({**{f"{q}_{h}h": (lambda s, a=a: s.quantile(a)) for q, a in zip(QN, C.QUANTILES)},
                       **{c: "mean" for c in ev}})
            t = t[g.size() >= self.MIN_ROWS]
            self.table[h] = t
            self.glob[h] = pd.Series({**{f"{q}_{h}h": d[f"y_{h}h"].quantile(a) for q, a in zip(QN, C.QUANTILES)},
                                      **{c: v.mean() for c, v in ev.items()}})
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for h in C.HORIZONS:
            k = self.key(df, h)
            p = self.table[h].reindex(k.to_numpy())
            p.index = df.index
            out.append(p.fillna(self.glob[h]))
        return pd.concat(out, axis=1)


def _local(df):
    return df.t.dt.tz_convert("America/Recife")


def key_clim(df, h):
    lt = _local(df)
    return (lt.dt.month * 100 + lt.dt.hour).astype(str)


def key_persist(df, h):
    b = pd.cut(df.p_60, [-np.inf, 0.0, 1.0, 5.0, np.inf], labels=["seco", "0-1", "1-5", ">=5"]).astype(object)
    return b.where(df.p_60.notna(), "sem_dado").astype(str)


def key_om(df, h):
    b = pd.cut(df[f"om_{h}h"], [-np.inf, 0.0, 0.2, 1.0, 3.0, 10.0, np.inf], labels=list("abcdef")).astype(object)
    return b.where(df[f"om_{h}h"].notna(), "sem_dado").astype(str)


class GB:
    """XGBoost: um regressor de três quantis e dois classificadores por horizonte. NaN nativo, sem imputação."""

    def __init__(self, name, features):
        self.name, self.features = name, list(features)

    def fit(self, train: pd.DataFrame, valid=None):
        # parada antecipada em semanas inteiras tiradas do TREINO (1 a cada 7), com folga de 6 h nas bordas: a validação
        # fica só para a escolha do modelo, e o período com Open-Meteo (2024 em diante) entra no ajuste
        origin = pd.Timestamp("2021-01-04", tz="UTC")
        week = (train.t - origin) // pd.Timedelta(days=7)
        hold = (week % 7 == 3)
        into_week = (train.t - origin) % pd.Timedelta(days=7)
        near_hold = ((week % 7 == 2) & (into_week >= pd.Timedelta(days=7) - C.GUARD)) | ((week % 7 == 4) & (into_week < C.GUARD))
        a, b = train[~hold & ~near_hold], train[hold]
        self.models = {}
        for h in C.HORIZONS:
            da, db = a.dropna(subset=[f"y_{h}h"]), b.dropna(subset=[f"y_{h}h"])
            r = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(C.QUANTILES), **C.XGB)
            r.fit(da[self.features], da[f"y_{h}h"], eval_set=[(db[self.features], db[f"y_{h}h"])], verbose=False)
            self.models[("q", h)] = r
            for k in C.THRESHOLDS:
                c = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **C.XGB)
                c.fit(da[self.features], da[f"ev_{k}_{h}h"].astype(int),
                      eval_set=[(db[self.features], db[f"ev_{k}_{h}h"].astype(int))], verbose=False)
                self.models[(k, h)] = c
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        out = {}
        for h in C.HORIZONS:
            q = np.sort(np.clip(self.models[("q", h)].predict(X), 0, None), axis=1)   # sem cruzamento, sem negativo
            for i, n in enumerate(QN):
                out[f"{n}_{h}h"] = q[:, i]
            for k in C.THRESHOLDS:
                out[f"p{k}_{h}h"] = self.models[(k, h)].predict_proba(X)[:, 1]
        return pd.DataFrame(out, index=df.index)


def make(name):
    return {"climatologia": lambda: Binned(name, key_clim),
            "persistencia": lambda: Binned(name, key_persist),
            "open_meteo": lambda: Binned(name, key_om),
            "gb_rede": lambda: GB(name, F.BASE_FEATURES),
            "gb_rede_om": lambda: GB(name, F.ALL_FEATURES)}[name]()
