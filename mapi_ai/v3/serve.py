"""
Serviço do modelo v3 (etapa B4.4): o que /v1/predict/flood faz quando o v3 está em produção.

O contrato de entrada não muda: a rota recebe current_rainfall, rainfall_{3,6,12,24}h_accumulated, tide_level e
timestamp. Aqui:

- as janelas disjuntas saem dos acumulados por subtração (a mesma função do treino), com a ausência propagada;
- chuva ausente (null) é ACEITA: o pipeline imputa e marca o indicador de ausência, e a resposta declara as entradas
  ausentes em imputed_features. Só a maré continua obrigatória (o modelo nunca viu maré ausente): sem ela, 422 com
  missing_features, como sempre;
- a probabilidade servida é a calibrada (Platt) e corrigida por prevalência; os limiares do metadata valem para ela;
- top_contributors: SHAP LinearExplainer sobre a logística (contribuições em logit), com o indicador de ausência
  somado à sua janela e o total rateado para pontos de probabilidade servida (soma = p - p de referência).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mapi_ai.v3 import config as C
from mapi_ai.v3.dataset import windows_from_accumulated
from mapi_ai.v3.evaluate import prior_shift

BACKGROUND = "shap_background.joblib"
SYNTHETIC_WARNING = ("Modelo treinado com rótulo SINTÉTICO (experimento controlado): a probabilidade não foi validada "
                     "contra alagamento real.")


class MissingRequired(Exception):
    def __init__(self, missing):
        super().__init__(f"Entradas obrigatórias ausentes: {missing}")
        self.missing_features = missing


def features_from_payload(d: dict, features: list) -> pd.DataFrame:
    """Uma linha com as entradas do modelo, a partir do corpo do contrato. None vira NaN (ausente), nunca zero."""
    acc = pd.DataFrame([{c: (np.nan if d.get(c) is None else float(d[c])) for c in C.ACCUMULATED}])
    row = windows_from_accumulated(acc)
    if d.get("tide_level") is None:
        raise MissingRequired(["tide_level"])
    row["tide_level"] = float(d["tide_level"])
    ts = pd.Timestamp(d["timestamp"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts   # sem fuso: o contrato é UTC (ADR 0001 do backend)
    local = ts.tz_convert("America/Recife")
    row["hour_sin"], row["hour_cos"] = np.sin(2 * np.pi * local.hour / 24), np.cos(2 * np.pi * local.hour / 24)
    row["month_sin"], row["month_cos"] = np.sin(2 * np.pi * local.month / 12), np.cos(2 * np.pi * local.month / 12)
    for f in C.PHYSICAL:
        if f in features:
            row[f] = np.nan if d.get(f) is None else float(d[f])
    return row[features].astype(float)


def write_background(models_dir: Path = C.MODELS_DIR, n: int = 200) -> None:
    """Guarda 200 linhas do treino (entradas brutas) como fundo do SHAP, para o serviço não depender do CSV."""
    from mapi_ai.v3 import dataset as D
    meta = json.loads((models_dir / "metadata.json").read_text())
    tr = D.load()
    tr = tr[tr.part == "train"]
    joblib.dump(tr[meta["features"]].astype(float).sample(n, random_state=C.SEED).reset_index(drop=True),
                models_dir / BACKGROUND)


class V3Model:
    def __init__(self, models_dir: Path = C.MODELS_DIR):
        self.meta = json.loads((models_dir / "metadata.json").read_text())
        if self.meta.get("status") != "avaliado" or not self.meta.get("thresholds"):
            raise RuntimeError("metadata do v3 sem avaliação (limiares): rode mapi_ai.v3.evaluate")
        self.features = self.meta["features"]
        self.horizon = int(self.meta["production"]["horizon_hours"])
        self.model = joblib.load(C.ROOT / self.meta["production"]["artifact"])
        cal = self.meta.get("calibrator")
        self.platt = joblib.load(C.ROOT / cal["artifact"]) if cal else None
        self.pi_sint = float(self.meta["prior_correction"]["pi_sint"])
        self.pi_real = float(self.meta["prior_correction"]["pi_real"])
        self.thresholds = self.meta["thresholds"]
        bg_path = models_dir / BACKGROUND
        self.background = joblib.load(bg_path) if bg_path.exists() else None
        self._explainer = None

    # ---------------------------------------------------------------------------------------------- probabilidade
    def calibrated(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X)[:, 1]
        return self.platt.transform(p) if self.platt is not None else p

    def served(self, X: pd.DataFrame) -> np.ndarray:
        return prior_shift(self.calibrated(X), self.pi_sint, self.pi_real)

    def risk(self, p: float) -> str:
        return "HIGH" if p >= self.thresholds["high"] else "MEDIUM" if p >= self.thresholds["medium"] else "LOW"

    # ------------------------------------------------------------------------------------------------ explicação
    def _linear(self):
        """(explainer, pré-processamento) da logística; None se o modelo de produção não for um Pipeline linear."""
        if self._explainer is None:
            import shap
            if self.background is None or not hasattr(self.model, "named_steps"):
                return None
            pre = self.model[:-1]
            self._explainer = (shap.LinearExplainer(self.model.named_steps["clf"], pre.transform(self.background)), pre)
        return self._explainer

    def contributions(self, X: pd.DataFrame) -> pd.Series:
        """
        Contribuição de cada entrada para UMA previsão, em pontos de probabilidade servida.

        O SHAP da logística é exato no logit. O indicador de ausência de uma janela é somado à própria janela (para o
        usuário é uma entrada só). A passagem para probabilidade é um rateio proporcional: cada entrada leva a sua
        fração do logit total vezes (p - p de referência), de modo que as contribuições somam a diferença.
        """
        lin = self._linear()
        if lin is None:
            return pd.Series(dtype=float)
        expl, pre = lin
        names = list(pre.get_feature_names_out())
        phi = np.asarray(expl.shap_values(pre.transform(X)))[0]
        by = {}
        for name, v in zip(names, phi):
            base = name.split("__", 1)[1]
            by[base] = by.get(base, 0.0) + float(v)
        s = pd.Series(by)[self.features]
        p = float(self.served(X)[0])
        p_ref = float(np.mean(self.served(self.background)))
        total = float(s.sum())
        return s * ((p - p_ref) / total) if abs(total) > 1e-12 else s * 0.0

    def top_contributors(self, X: pd.DataFrame, k: int = 3) -> list:
        try:
            c = self.contributions(X)
        except Exception as e:   # sem shap ou sem fundo: não derruba a previsão
            print(f"top_contributors indisponível: {e}")
            return []
        units = self.meta.get("feature_units", {})
        order = c.abs().sort_values(ascending=False).index[:k]
        return [{"feature_id": f, "contribution": round(float(c[f]), 5),
                 "value": None if np.isnan(X.iloc[0][f]) else float(X.iloc[0][f]), "unit": units.get(f)} for f in order]

    # --------------------------------------------------------------------------------------------------- resposta
    def predict(self, payload: dict) -> dict:
        X = features_from_payload(payload, self.features)
        p_cal = float(self.calibrated(X)[0])
        p = float(self.served(X)[0])
        t = self.thresholds
        return {
            "flood_probability": p,
            "risk_level": self.risk(p),
            "model_version": self.meta["model_version"],
            "model_validation_status": self.meta.get("model_validation_status", "NOT_VALIDATED"),
            "station_id": payload.get("station_id"),
            "timestamp": payload.get("timestamp"),
            "horizon_hours": self.horizon,
            "thresholds": {"medium": t["medium"], "high": t["high"], "source": "metadata"},
            "probability_kind": "PRIOR_CORRECTED",
            "flood_probability_uncorrected": p_cal,
            "prior_correction": {"pi_real": self.pi_real, "pi_sint": self.pi_sint},
            "calibrated": self.platt is not None,
            "rain_source": self.meta.get("rain_source"),
            "training_data": self.meta.get("training_data"),
            "synthetic_warning": SYNTHETIC_WARNING if self.meta.get("training_data") == "SYNTHETIC" else None,
            "imputed_features": [f for f in self.features if np.isnan(X.iloc[0][f])],
            "top_contributors": self.top_contributors(X),
        }
