"""
Extração das séries reais do banco do projeto-mapi-api para arquivos locais (data/in1165/).

Por que extrair em vez de ler o banco a cada execução: o notebook do IN1165 precisa rodar "do zero" em ambiente
limpo, sem o banco. Os arquivos ficam versionados com SHA-256, e o banco só é necessário para regenerá-los.

Origem de cada série (handoff projeto-mapi-api/docs/handoff/etapa0.json):
- chuva MEDIDA por ponto: view point_rain_accum_v, rule_version chuva-ponto-v4 (Cemaden PED, maior valor entre
  estações válidas a até 5 km; acumulado com qualquer hora faltando = NULL, nunca zero);
- chuva de REANÁLISE por ponto: observation_hourly_v, OPEN_METEO_ARCHIVE, na coordenada do ponto;
- maré: observation_hourly_v, TIDE_PE02_HOURLY (m sobre o NR da DHN; previsão de tábua, conhecida antes do evento).
Todas as horas são rotuladas pelo FIM do intervalo, em UTC.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text

from mapi_ai import in1165_config as C


def engine():
    # Mesmas variáveis do serviço mapi-ai no docker-compose; fora do compose, o banco está em 127.0.0.1:5433.
    url = "postgresql://{u}:{p}@{h}:{port}/{db}".format(
        u=os.getenv("DB_USER", "mapi_user"), p=os.getenv("DB_PASS", "mapi123"),
        h=os.getenv("DB_HOST", "127.0.0.1"), port=os.getenv("DB_PORT", "5433"), db=os.getenv("DB_NAME", "mapi_db"))
    return create_engine(url)


def _accumulate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Acumulados de 3/6/12/24 h para trás, por ponto. Janela com qualquer hora sem valor fica NaN (lacuna não é zero)."""
    out = []
    for slug, g in df.sort_values("hour_end_utc").groupby("slug"):
        # Reindexa para a grade horária completa: hora sem linha vira NaN explícito.
        g = g.set_index("hour_end_utc").asfreq("1h")
        g["slug"] = slug
        r = g[col]
        g["current_rainfall"] = r
        for n in (3, 6, 12, 24):
            s = r.rolling(n, min_periods=n).sum()   # min_periods=n: qualquer NaN na janela => NaN
            g[f"rainfall_{n}h_accumulated"] = s
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def extract() -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    eng = engine()
    with eng.connect() as cx:
        points = pd.read_sql(text("SELECT id, slug, name, latitude, longitude FROM flood_points ORDER BY id"), cx)
        points.to_csv(C.POINTS, index=False)

        cem = pd.read_sql(text("""
            SELECT slug, hour_end_utc, rain_1h_mm AS current_rainfall, rain_3h_mm AS rainfall_3h_accumulated,
                   rain_6h_mm AS rainfall_6h_accumulated, rain_12h_mm AS rainfall_12h_accumulated,
                   rain_24h_mm AS rainfall_24h_accumulated, station_id, station_distance_m, n_valid
            FROM point_rain_accum_v WHERE rule_version = 'chuva-ponto-v4' ORDER BY slug, hour_end_utc"""), cx)
        cem.to_csv(C.CEMADEN_HOURLY, index=False)

        rea = pd.read_sql(text("""
            SELECT p.slug, h.hour_end_utc, h.value AS precipitation
            FROM observation_hourly_v h
            JOIN station s ON s.source = h.source AND s.station_id = h.station_id
            JOIN flood_points p ON p.name = s.name
            WHERE h.source = 'OPEN_METEO_ARCHIVE' AND h.variable = 'precipitation'"""), cx)
        rea["hour_end_utc"] = pd.to_datetime(rea["hour_end_utc"], utc=True)
        rea = _accumulate(rea, "precipitation").drop(columns=["precipitation"])
        rea.to_csv(C.REANALYSIS_HOURLY, index=False)

        tide = pd.read_sql(text("""
            SELECT hour_end_utc, value AS tide_level FROM observation_hourly_v
            WHERE source = 'TIDE_PE02_HOURLY' AND variable = 'tide_height_hourly' ORDER BY hour_end_utc"""), cx)
        tide.to_csv(C.TIDE_HOURLY, index=False)

        occ = pd.read_sql(text("""
            SELECT id, occurred_from, occurred_to, temporal_precision, cobrade_text, geocoded_lat, geocoded_lon,
                   geocode_level, neighborhood, event_type
            FROM flood_occurrence ORDER BY occurred_from"""), cx)
        occ.to_csv(C.OCCURRENCES, index=False)
    print(f"Extraído: {len(cem)} horas Cemaden, {len(rea)} horas reanálise, {len(tide)} horas de maré, "
          f"{len(occ)} ocorrências, {len(points)} pontos -> {C.DATA_DIR}")


def load_points() -> pd.DataFrame:
    return pd.read_csv(C.POINTS)


def load_hourly(kind: str) -> pd.DataFrame:
    """kind = 'cemaden' (medida) ou 'reanalysis'. Junta a maré e as coordenadas do ponto."""
    path = C.CEMADEN_HOURLY if kind == "cemaden" else C.REANALYSIS_HOURLY
    df = pd.read_csv(path, parse_dates=["hour_end_utc"])
    df["hour_end_utc"] = pd.to_datetime(df["hour_end_utc"], utc=True)
    tide = pd.read_csv(C.TIDE_HOURLY, parse_dates=["hour_end_utc"])
    tide["hour_end_utc"] = pd.to_datetime(tide["hour_end_utc"], utc=True)
    df = df.merge(tide, on="hour_end_utc", how="left")
    pts = load_points()[["slug", "latitude", "longitude"]]
    return df.merge(pts, on="slug", how="left")


if __name__ == "__main__":
    extract()
