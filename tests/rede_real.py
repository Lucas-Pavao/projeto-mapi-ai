"""Estações reais da rede do treino (data/rain/ped_stations.csv) para os testes: desde a onda 16 (Fase 7.2) o rain_v1 ignora
estação que não está na rede do treino, como as que só existem ao vivo."""
import numpy as np
import pandas as pd

from mapi_ai.rain import config as C


def estacoes(lat: float, lon: float) -> dict:
    """{"perto": (id, lat, lon) a até 5 km, "norte": a 5–40 km ao norte, "oeste": a 5–40 km a oeste}."""
    s = pd.read_csv(C.STATIONS)
    dn, de = (s.lat - lat) * 111.0, (s.lon - lon) * 110.0
    km = np.hypot(dn, de)
    perto = s.loc[km.idxmin()]
    norte = s.loc[(dn - de.abs()).where((km > 6) & (km < 40) & (dn > de.abs())).idxmax()]
    oeste = s.loc[(-de - dn.abs()).where((km > 6) & (km < 40) & (-de > dn.abs())).idxmax()]
    return {k: (r.station_id, float(r.lat), float(r.lon)) for k, r in (("perto", perto), ("norte", norte), ("oeste", oeste))}
