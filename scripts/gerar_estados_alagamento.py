"""
Gera as respostas de POST /v1/flood/forecast para os 10 estados da prancha "Nível 6" (contrato da API e mocks do front).
Roda o código de verdade (mapi_ai.flood.forecast.forecast_point) com entradas controladas: um rain_v1 de mentira com
quantis fixos (o SHAP local usa o modelo real sobre leituras sintéticas) e um terreno raso de teste. NÃO são previsões:
são formas de resposta para testes. Uso: python -I scripts/gerar_estados_alagamento.py <pasta de saída>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mapi_ai import bucket10 as B  # noqa: E402
from mapi_ai.flood.forecast import forecast_point  # noqa: E402
from mapi_ai.rain.serve import RainForecaster  # noqa: E402

LAT, LON = -8.118123, -34.904945          # Mascarenhas/Imbiribeira
T = pd.Timestamp("2026-11-20T17:00:00Z")
REAL = RainForecaster()
TERRAIN = B.Terreno(20000.0, np.linspace(0, 2, 41), 100.0 * np.linspace(0, 2, 41), 0.10)


class Fixed:
    meta = REAL.meta
    models, chosen = REAL.models, REAL.chosen

    def __init__(self, q50, q95):
        self.q = {"q50": q50, "q95": q95}

    def forecast_with_inputs(self, *a, **k):
        rain, x = REAL.forecast_with_inputs(*a, **k)
        for h in rain["horizons"]:
            hh = h["horizon_hours"]
            h["quantiles_mm"] = {"q50": self.q["q50"][hh], "q80": (self.q["q50"][hh] + self.q["q95"][hh]) / 2,
                                 "q95": self.q["q95"][hh]}
        return rain, x


class Down:
    meta = REAL.meta

    def forecast_with_inputs(self, *a, **k):
        raise ConnectionError("rain_v1 fora do ar")


def readings(mm, missing=False):
    ts = pd.date_range(T - pd.Timedelta("6h") + pd.Timedelta("10min"), T, freq="10min")
    st = [("260960001A", LAT + 0.01, LON), ("261160614A", LAT + 0.08, LON), ("261160609A", LAT, LON - 0.09)]
    return [{"station_id": s, "latitude": a, "longitude": b, "ts": t.isoformat(), "mm": None if missing else mm}
            for s, a, b in st for t in ts]


def q(a, b, c):
    return {1: a, 3: b, 6: c}


STATES = {
    "sem_alerta_seco": dict(series=[0.0] * 144, rd=readings(0.0), rain=Fixed(q(0, 0, 0.2), q(0.4, 1.0, 2.5))),
    "chuva_forte_sem_encher": dict(series=[0.0] * 132 + [8.0] * 12, rd=readings(8.0), rain=Fixed(q(12, 20, 26), q(40, 70, 95))),
    "atencao_3h": dict(series=[0.0] * 138 + [3.0] * 6, rd=readings(3.0), rain=Fixed(q(8, 20, 30), q(30, 520, 600))),
    "alerta_1h": dict(series=[0.0] * 138 + [6.0] * 6, rd=readings(6.0), rain=Fixed(q(170, 200, 220), q(260, 300, 330))),
    "alerta_agora": dict(series=[0.0] * 126 + [30.0] * 18, rd=readings(30.0), rain=Fixed(q(60, 90, 110), q(120, 160, 200))),
    "atencao_6h": dict(series=[0.0] * 140 + [1.0] * 4, rd=readings(1.0), rain=Fixed(q(4, 10, 20), q(20, 60, 1100))),
    "sem_dado_chuva_ausente": dict(series=[0.0] * 144, rd=readings(0.0, missing=True), rain=Fixed(q(0, 0, 0), q(0, 0, 0))),
    "sem_dado_previsao_fora": dict(series=[0.0] * 144, rd=readings(0.0), rain=Down()),
    "sem_dado_sem_terreno": dict(series=[0.0] * 144, rd=readings(0.0), rain=Fixed(q(0, 0, 0), q(0, 0, 0)), terrain=None,
                                 status="SEM_TERRENO"),
    "sem_dado_terreno_em_revisao": dict(series=[0.0] * 144, rd=readings(0.0), rain=Fixed(q(0, 0, 0), q(0, 0, 0)),
                                        status="EM_REVISAO"),
}


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, s in STATES.items():
        r = forecast_point(latitude=LAT, longitude=LON, readings=s["rd"], rain_10min_mm=s["series"],
                           terrain=s.get("terrain", TERRAIN), terrain_status=s.get("status", "OK"),
                           issued_at=T.isoformat(), rain_model=s["rain"], now=T.isoformat(), slug="MASCARENHAS_IMBIRIBEIRA")
        r["code_version"] = "0000000000000000000000000000000000000000"   # fixo: o arquivo não muda a cada commit
        r["code_dirty"] = "false"
        (out / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        summary[name] = (r["level"], r.get("worst_horizon_hours"), r.get("reason"))
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
