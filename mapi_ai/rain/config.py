"""
Configuração única da previsão de chuva (Fase 3 do plano de 16/11). Grava em models/v4/rain e reports/v4/rain.
O alvo é chuva MEDIDA: nada aqui leva a etiqueta de dado sintético.
"""
from pathlib import Path

import pandas as pd

from mapi_ai.v3 import config as C3

SEED = 42
ROOT = C3.ROOT
DATA_DIR = ROOT / "data" / "rain"
MODELS_DIR = ROOT / "models" / "v4" / "rain"
REPORTS_DIR = ROOT / "reports" / "v4" / "rain"
FIGURES_DIR = REPORTS_DIR / "figures"
READINGS = DATA_DIR / "ped_10min.csv.gz"            # observation_current, fonte CEMADEN_PED: station_id, epoch, mm
STATIONS = DATA_DIR / "ped_stations.csv"
OM_HIST = DATA_DIR / "open_meteo_forecast_hist.csv.gz"   # historical-forecast-api: rodadas emendadas, 0 a ~6 h
OM_PREV = DATA_DIR / "open_meteo_forecast_prev.csv.gz"   # previous-runs-api: rodada da véspera (emitida >= 24 h antes)
POINTS = ROOT / "data" / "in1165" / "flood_points.csv"
DATASET = DATA_DIR / "dataset.pkl.gz"               # derivado; não versionado (regenera com python -m mapi_ai.rain.dataset)
ATTRIBUTION = "DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC"
MODEL_VERSION = "rain_v1"

HORIZONS = (1, 3, 6)                                # horas
QUANTILES = (0.5, 0.8, 0.95)
# Limiares de evento do ADR 0017 (docs/EVENTOS-DE-CHUVA.md): pico horário, mm/h.
THRESHOLDS = {"MEDIA": 5.03, "FORTE": 25.15}

# Split temporal: os mesmos cortes do v2 e do v3. Folga de 6 h (maior horizonte) entre as partes.
TRAIN_END = C3.TRAIN_END
VALID_END = C3.VALID_END
GUARD = pd.Timedelta(hours=max(HORIZONS))
DATA_START = pd.Timestamp("2021-09-01T06:00:00Z")
DATA_END = pd.Timestamp("2026-10-31T00:00:00Z")     # a PED de novembro ainda está chegando

RAINING_MM = 0.2                                    # regime "já chove": chuva do ponto nos últimos 60 min >= 0,2 mm

# Do mais simples ao mais complexo. O critério foi fixado antes de olhar o teste.
MODELS = ["climatologia", "persistencia", "open_meteo", "gb_rede", "gb_rede_om"]
SELECTION = ("por horizonte: menor erro de quantil (pinball médio de 50%, 80% e 95%) na validação; fica o modelo mais "
             "simples cuja diferença para o melhor tem IC 95% contendo zero (bootstrap pareado por bloco de dia, "
             "1000 reamostras)")
N_BOOT = 1000
# Aviso de evento FORTE: limiar de probabilidade escolhido na validação pelo maior CSI.
ALERT_RULE = "limiar de probabilidade com o maior CSI por linha na validação"

XGB = dict(n_estimators=400, learning_rate=0.05, max_depth=5, min_child_weight=20, subsample=0.8,
           colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=8, early_stopping_rounds=30)
TRAIN_STEP = 3                                      # treino usa 1 instante a cada 30 min (os de 10 min são quase iguais)


def part(t: pd.Series) -> pd.Series:
    """train / valid / test / guard, pelo instante t (UTC)."""
    out = pd.Series("test", index=t.index)
    out[t <= VALID_END - GUARD] = "valid"
    out[t <= TRAIN_END - GUARD] = "train"
    out[(t > TRAIN_END - GUARD) & (t <= TRAIN_END)] = "guard"
    out[(t > VALID_END - GUARD) & (t <= VALID_END)] = "guard"
    return out
