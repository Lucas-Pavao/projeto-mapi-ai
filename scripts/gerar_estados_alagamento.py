"""
Gera as respostas de POST /v1/flood/forecast para os 10 estados da prancha "Nível 6" (contrato da API e mocks do front).
Roda o código de verdade (mapi_ai.flood.forecast.forecast_point) com entradas controladas: um rain_v1 de mentira com
quantis fixos (o SHAP local usa o modelo real sobre leituras sintéticas) e um terreno raso de teste. NÃO são previsões:
são formas de resposta para testes. As chuvas dos cenários saem de lâminas-alvo (by_sheet), por bissecção no próprio balde. Uso: python -I scripts/gerar_estados_alagamento.py <pasta de saída>
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
TERRAIN = B.Terreno(20000.0, np.linspace(0, 2, 41), 3000.0 * np.linspace(0, 2, 41), 0.10)


def rain_for(cm: float, h: int, series) -> float:
    """Chuva acumulada em h horas (espalhada por igual) que leva a lâmina máxima a `cm`, partindo do estado da série."""
    from mapi_ai.flood import explain as X
    obs = B.simular_passos(np.asarray(series, float), TERRAIN, B.Params10())
    V0, h0 = float(obs["V_m3"][-1]), float(obs["h_mm"][-1]) / 10
    if cm <= 0:
        return 0.0
    lo, hi = 0.0, 3000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if X._scenario_sheet_cm(V0, h0, mid, h, TERRAIN, B.Params10()) >= cm:
            hi = mid
        else:
            lo = mid
    return round(hi, 1)


def by_sheet(series, q50cm, q95cm, dry=(0.0, 0.0, 0.0)):
    """Quantis que dão as lâminas alvo por horizonte; lâmina 0 usa a chuva `dry` (abaixo da capacidade)."""
    q50 = {h: rain_for(c, h, series) if c > 0 else d for h, c, d in zip((1, 3, 6), q50cm, dry)}
    q95 = {h: max(q50[h], rain_for(c, h, series) if c > 0 else d * 2) for h, c, d in zip((1, 3, 6), q95cm, dry)}
    return Fixed(q50, q95)


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


SECO = [0.0] * 144
CHUVA = [0.0] * 132 + [7.5] * 12            # 45 mm/h na última hora, abaixo da capacidade da galeria
STATES = {
    "sem_alerta_seco": dict(series=SECO, rd=readings(0.0), rain=Fixed(q(0, 0, 0.2), q(0.4, 1.0, 2.5))),
    "chuva_forte_sem_encher": dict(series=CHUVA, rd=readings(7.5), rain=by_sheet(CHUVA, (0, 0, 0), (0, 0, 0), (12, 20, 26))),
    "atencao_3h": dict(series=CHUVA, rd=readings(7.5), rain=by_sheet(CHUVA, (4, 7, 8), (10, 19, 21), (12, 20, 26))),
    "alerta_1h": dict(series=CHUVA, rd=readings(7.5), rain=by_sheet(CHUVA, (17.7, 12, 9), (33, 26, 20))),
    "alerta_agora": dict(series=[0.0] * 138 + [30.0] * 6, rd=readings(30.0),
                         rain=by_sheet([0.0] * 138 + [30.0] * 6, (24, 22, 20), (38, 35, 30))),
    "atencao_6h": dict(series=CHUVA, rd=readings(7.5), rain=by_sheet(CHUVA, (0, 2, 6), (0, 9, 17), (12, 20, 26))),
    "sem_dado_chuva_ausente": dict(series=SECO, rd=readings(0.0, missing=True), rain=Fixed(q(0, 0, 0), q(0, 0, 0))),
    "sem_dado_previsao_fora": dict(series=CHUVA, rd=readings(7.5), rain=Down()),
    "sem_dado_sem_terreno": dict(series=SECO, rd=readings(0.0), rain=Fixed(q(0, 0, 0), q(0, 0, 0)), terrain=None,
                                 status="SEM_TERRENO"),
    "sem_dado_terreno_em_revisao": dict(series=SECO, rd=readings(0.0), rain=Fixed(q(0, 0, 0), q(0, 0, 0)),
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
