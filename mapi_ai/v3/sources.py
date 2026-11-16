"""
Extração das entradas físicas do v3 (etapa B5) do banco do projeto-mapi-api para data/v3/, versionada com SHA-256.

Origem: view point_physical_inputs_v (V27, projeto-mapi-api/docs/INGESTAO.md), na mesma grade da chuva do ponto.
- rain_max_10min_mm: maior bloco de 10 min da hora, da estação que deu o valor horário. Ausente quando a hora tem bloco
  longo com chuva (o pico não é conhecido): falta em 40% a 80% das horas com chuva, e a ausência DEPENDE da chuva.
- soil_moisture: ERA5-Land 0-7 cm, hora anterior. É REANÁLISE e chega com ~2 semanas de atraso (ausente ao vivo).
- tide_trend_m_h: calculada aqui da maré horária (nível da hora menos o da hora anterior), como na view.
- river_level_m: NÃO entra. A régua do Duas Unas só tem leituras desde 2026-11-09; não há histórico para treinar.
"""
import hashlib

import pandas as pd
from sqlalchemy import text

from mapi_ai import in1165_config as C2
from mapi_ai.sources import engine
from mapi_ai.v3 import config as C

LAST_HOUR = "2026-10-08T00:00:00Z"   # mesma janela dos arquivos do v2 (data/in1165)


def extract() -> str:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with engine().connect() as cx:
        df = pd.read_sql(text("""
            SELECT slug, hour_end_utc, rain_max_10min_mm, soil_moisture
            FROM point_physical_inputs_v
            WHERE rule_version = 'chuva-ponto-v4' AND hour_end_utc <= :last
            ORDER BY slug, hour_end_utc"""), cx, params={"last": LAST_HOUR})
    df["hour_end_utc"] = pd.to_datetime(df["hour_end_utc"], utc=True)
    df.to_csv(C.PHYSICAL_HOURLY, index=False, compression=C.GZIP)
    return hashlib.sha256(C.PHYSICAL_HOURLY.read_bytes()).hexdigest()


def load_physical() -> pd.DataFrame:
    df = pd.read_csv(C.PHYSICAL_HOURLY, parse_dates=["hour_end_utc"])
    df["hour_end_utc"] = pd.to_datetime(df["hour_end_utc"], utc=True)
    tide = pd.read_csv(C2.TIDE_HOURLY, parse_dates=["hour_end_utc"])
    tide["hour_end_utc"] = pd.to_datetime(tide["hour_end_utc"], utc=True)
    tide = tide.sort_values("hour_end_utc").set_index("hour_end_utc").asfreq("1h")
    tide["tide_trend_m_h"] = tide["tide_level"].diff()
    return df.merge(tide[["tide_trend_m_h"]].reset_index(), on="hour_end_utc", how="left")


if __name__ == "__main__":
    print(extract())
    d = load_physical()
    print(len(d), d[["rain_max_10min_mm", "soil_moisture", "tide_trend_m_h"]].notna().mean().round(4).to_dict(),
          d[["rain_max_10min_mm", "soil_moisture", "tide_trend_m_h"]].describe().round(3).to_dict())
