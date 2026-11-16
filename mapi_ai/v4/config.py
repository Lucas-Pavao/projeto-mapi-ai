"""
Modelo v4 (Fase 4 do plano de 16/11): o rótulo vem do simulador do balde M1a sobre a chuva medida de 5 anos, e a
previsão de chuva (rain_v1, fora da amostra) entra como entrada. Grava em models/v4, reports/v4 e data/v4.
"""
from mapi_ai.v3 import config as C3

SEED = 42
ROOT = C3.ROOT
DATA_DIR = ROOT / "data" / "v4"
MODELS_DIR = ROOT / "models" / "v4"
REPORTS_DIR = ROOT / "reports" / "v4"
TERRAIN_POINT = ROOT.parent / "avaliacao-f1" / "entradas" / "ponto_terreno.csv"     # congelado (F1): só leitura
TERRAIN_CURVE = ROOT.parent / "avaliacao-f1" / "entradas" / "curva_cota_volume.csv"
OOF = DATA_DIR / "rain_forecast_oof.csv.gz"
LABEL_TAG = "RÓTULO DO MODELO FÍSICO (balde M1a), não alagamento observado"
ATTRIBUTION = C3.ATTRIBUTION

POINTS = ["AV_RECIFE_IBURA", "CIN_UFPE", "AGAMENON_DERBY", "MASCARENHAS_IMBIRIBEIRA"]   # Jaboatão fora: sem terreno
HCRIT_MM = 150.0
HORIZON = 6
# Capacidade da galeria: [sem fonte]. PARAMETROS-MODELO-FISICO.md (linha "Capacidade da galeria por ponto") pré-registra
# a faixa de 0 à chuva de projeto TR 5 (60,1 mm/h) em passos de 20%. f = fração dessa capacidade. Central: f = 1,0.
# f = 0 (galeria totalmente obstruída) não é simulado: o balde nunca esvazia.
F_GRID = [1.0, 0.8, 0.6, 0.4, 0.2]
F_CENTRAL = 1.0
# Critério de treinabilidade (estatístico, não de frequência desejada), fixado antes de treinar: o cenário só entra no
# experimento se tiver eventos em todas as partes do split.
MIN_EVENTS = {"train": 20, "validation": 5, "test": 5}
GAP_MASK_H = 6      # hora sem chuva medida e as 6 h seguintes ficam sem rótulo (o estado do balde é desconhecido)
REAL_FREQUENCY = {"valor": "cerca de 4 por ano", "calculo": "18 dias com alagamento registrado em 5 anos = 3,6 por ano",
                  "fonte": "CKAN Recife, 2021-05-13 a 2024-06-25 (reports/dataset_summary.json; "
                           "projeto-mapi-api/docs/REVISAO-INGESTAO-2026-11.md)",
                  "ressalva": "é a cidade inteira, não um ponto; nenhum registro tem hora; a 500 m dos pontos há 0 registros"}

WINDOWS = C3.WINDOWS
TIME = C3.TIME
TERRAIN = ["rua_rel_m", "profundidade_depressao_m", "area_contribuinte_m2"]
RAIN_FC = [f"{q}_{h}h" for h in (1, 3, 6) for q in ("q50", "q80", "q95", "pMEDIA", "pFORTE")]
FUTURE = ["fut_1h", "fut_3h", "fut_6h", "futpeak_3h", "futpeak_6h"]
BASE = WINDOWS + ["tide_level"] + TIME + TERRAIN
ARMS = {"sem_previsao": BASE, "com_previsao": BASE + RAIN_FC, "chuva_futura_verdadeira": BASE + FUTURE}
ARM_LABEL = {"sem_previsao": "v4 sem previsão de chuva", "com_previsao": "v4 com previsão de chuva (rain_v1, fora da amostra)",
             "chuva_futura_verdadeira": "v4 com a chuva futura verdadeira (teto)"}
TRAIN_END, VALID_END, GUARD_H = C3.TRAIN_END, C3.VALID_END, 6
ALERT_BUDGET = C3.ALERT_BUDGET
N_BOOT = 1000
SELECTION = ("família escolhida uma vez, no braço com previsão, pela PR-AUC de validação; empate (IC 95% do bootstrap "
             "pareado contém zero) fica com a logística; a mesma família vale para os três braços")
CALIBRATION_RULE = C3.CALIBRATION_RULE
