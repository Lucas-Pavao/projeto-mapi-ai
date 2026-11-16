"""
Configuração única do modelo v3. O v2 (mapi_ai/in1165_config.py, models/, reports/) não é tocado: o v3 grava em
models/v3/, reports/v3/ e data/v3/.

Tudo que define o experimento fica aqui e é gravado no metadata ANTES do treino: horizonte, entradas, cortes do split,
critério de escolha do modelo, regra de calibração, prevalência real e orçamento de alarmes.
"""
from pathlib import Path

import pandas as pd

from mapi_ai import in1165_config as C2

SEED = C2.SEED
ROOT = C2.ROOT
DATA_DIR = ROOT / "data" / "v3"
MODELS_DIR = ROOT / "models" / "v3"
REPORTS_DIR = ROOT / "reports" / "v3"
FIGURES_DIR = REPORTS_DIR / "figures"
HANDOFF_API = ROOT / "docs" / "handoff" / "v3-api.json"
GAPS_JSON = ROOT / "docs" / "handoff" / "v3-lacunas.json"      # padrão real de lacunas (api, etapa A2)
PHYSICAL_HOURLY = DATA_DIR / "physical_point_hourly.csv.gz"    # point_physical_inputs_v (api, V27)
CONTINUOUS = DATA_DIR / "continuous.csv.gz"                    # série contínua com rótulo e entradas
DATASET = DATA_DIR / "dataset.csv.gz"                          # amostra de treino/validação/teste

SYNTHETIC_TAG = C2.SYNTHETIC_TAG
ATTRIBUTION = C2.ATTRIBUTION
GZIP = C2.GZIP

# --- B1: horizonte. Entradas da hora t; rótulo = "alaga em algum momento de t a t+H". H=0 é o nowcast do v2.
HORIZONS = (0, 3, 6)
PRIMARY_HORIZON = 6          # pré-registrado em projeto-mapi-api/docs/MODELAGEM-IA.md (6 h principal, 3 h secundário)


def label(h: int) -> str:
    return f"flood_h{h}"


# --- B3: janelas disjuntas de chuva (mm), derivadas dos acumulados do contrato por subtração.
WINDOWS = ["rain_0_1h", "rain_1_3h", "rain_3_6h", "rain_6_12h", "rain_12_24h"]
ACCUMULATED = C2.RAIN_FEATURES   # nomes do contrato: current_rainfall, rainfall_{3,6,12,24}h_accumulated
TIME = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
# --- B4: entradas do v3. Latitude e longitude saem (decisão do Lucas; no v2 a variante sem local foi melhor).
FEATURES = WINDOWS + ["tide_level"] + TIME
# --- B5: variante com entradas físicas (gerador próprio; não se compara com o v2 como ganho de modelo).
PHYSICAL = ["rain_max_10min_mm", "tide_trend_m_h", "soil_moisture"]
FEATURES_PHYSICAL = FEATURES + PHYSICAL
UNITS = {**{w: "mm" for w in WINDOWS}, "tide_level": "m (NR DHN)", **{t: "adimensional" for t in TIME},
         "rain_max_10min_mm": "mm", "tide_trend_m_h": "m/h", "soil_moisture": "m3/m3"}

# --- Split temporal: os MESMOS cortes do v2 (models/metadata.json, split), para comparar. Entre as partes fica uma
# folga de 6 h (o maior horizonte): o rótulo de t olha até t+6 h e não pode atravessar o corte.
TRAIN_END = pd.Timestamp("2024-07-03T14:00:00Z")
VALID_END = pd.Timestamp("2025-07-31T00:00:00Z")
GUARD_H = max(HORIZONS)

# --- Amostra por estrato, como no v2 (mesmos tamanhos).
STRATA = {"FORTE": 900, "MEDIA": 1300, "FRACA": 1300, "SECO": 1500}
AFTER_EVENT_H = 3

# --- B4: critério de escolha do modelo de produção, fixado antes de treinar.
SIMPLICITY = ["regra_chuva3h", "logistica", "arvore_rasa", "xgboost"]   # do mais simples ao mais complexo
SELECTION = ("maior PR-AUC na validação (amostra, horizonte principal); empate = IC 95% do bootstrap pareado da "
             "diferença para o melhor contém zero; no empate fica o mais simples (regra < logística < árvore < XGBoost)")
N_BOOT = 1000

# --- B6: calibração e prevalência.
CALIBRATION_RULE = ("Platt ajustado na primeira metade (no tempo) da validação; adotado se ECE e Brier melhoram na "
                    "segunda metade")
# Prevalência real por hora, parâmetro declarado. Estimativa do Lucas (prompt do v3): ~0,1% das horas, a partir de 18
# dias com alagamento registrado (CKAN Recife, 2021-05-13 a 2024-06-25; reports/dataset_summary.json) em 5 anos.
# A duração do alagamento em cada dia não é conhecida [sem fonte]: 0,1% equivale a ~2,4 h por dia de alagamento.
# Para o horizonte H, a hora é positiva se alaga em [t, t+H]: cada episódio de d horas vira d+H horas positivas.
PI_REAL_HOURLY = 0.001
FLOOD_DAYS_PER_YEAR = 18 / 5
HOURS_PER_YEAR = 8766.0


def pi_real(h: int) -> float:
    d = PI_REAL_HOURLY * HOURS_PER_YEAR / FLOOD_DAYS_PER_YEAR      # duração implícita por episódio (h)
    return FLOOD_DAYS_PER_YEAR * (d + h) / HOURS_PER_YEAR


# --- B6: limiares por orçamento de alertas (por ponto por ano), na série contínua da validação.
ALERT_BUDGET = {"medium": 50, "high": 10}
ALERT_GAP_H = 6     # dois alertas separados por menos de 6 h sem alerta são um só (mesmo intervalo seco do ADR 0017)
EVENT_GAP_H = 6     # idem para agrupar horas positivas em eventos
