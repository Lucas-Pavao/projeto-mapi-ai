"""
MAPI AI: serviço de previsão. Rotas servidas (decisão de 20/11/2026: os classificadores treinados com rótulo de
alagamento saíram do serviço; os artefatos e relatórios ficam no git e em reports/ para a dissertação):

- POST /v1/forecast/rain   rain_v1, chuva MEDIDA prevista em 1, 3 e 6 h (validado, reports/v4/rain/RELATORIO.md);
- POST /v1/flood/forecast  alerta de alagamento por regressão + física (mapi_ai/flood), experimental;
- POST /v1/bucket/point10  balde M1a a 10 min (físico, sem treino);
- POST /v1/simulate/bucket balde M1a horário, ilustrativo (ADR 0014);
- GET  /health             versões (rain_v1, balde, alerta) e commit do código.

Removidos em 21/11: POST /v1/predict/flood (logreg_v3; xgb_v2 e o XGBoost legado como reserva; a LSTM do nível do
rio, que a rota nunca alcançava).
"""
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from mapi_ai.feature_engineering import MissingFeaturesError

# Previsão de chuva (models/v4/rain): alvo MEDIDO, sem etiqueta de sintético.
rain_model = None
try:
    from mapi_ai.rain.serve import RainForecaster
    rain_model = RainForecaster()
    print(f"Previsão de chuva {rain_model.meta['model_version']} carregada ({rain_model.chosen}).")
except Exception as e:
    rain_model = None
    print(f"previsão de chuva não carregada ({e}).")

app = FastAPI(title="MAPI AI", version="2.0")


from dataclasses import asdict
from typing import Literal
from avaliacao.balde import Params, Ponto, simular_ponto


class BucketPoint(BaseModel):
    slug: str = Field(min_length=1)
    area_m2: float = Field(gt=0, allow_inf_nan=False)  # area contribuinte no snap_m padrao de Params
    z_curva_m: list[float] = Field(min_length=2)
    v_curva_m3: list[float] = Field(min_length=2)
    rua_rel_m: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _curva_valida(self):
        z = np.asarray(self.z_curva_m, dtype=float)
        v = np.asarray(self.v_curva_m3, dtype=float)
        if len(z) != len(v):
            raise ValueError("z_curva_m e v_curva_m3 devem ter o mesmo tamanho")
        if not (np.isfinite(z).all() and np.isfinite(v).all()):
            raise ValueError("curva com valor nao finito")
        if np.any(np.diff(z) <= 0):
            raise ValueError("z_curva_m deve ser estritamente crescente")
        if np.any(np.diff(v) < 0) or v[0] < 0:
            raise ValueError("v_curva_m3 deve ser nao negativa e nao decrescente")
        return self


class BucketParams(BaseModel):
    model_config = ConfigDict(extra="forbid")  # C, iT_mm_h, hcrit_mm e snap_m sao fixos: enviar => 422
    f: Literal[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] = 1.0
    curve_scale: float = Field(default=1.0, ge=0.7, le=1.3)


class BucketInput(BaseModel):
    P_mm_h: list[float] = Field(min_length=1, max_length=72)
    point: BucketPoint
    params: BucketParams = BucketParams()

    @field_validator("P_mm_h")
    @classmethod
    def _p_valida(cls, v):
        a = np.asarray(v, dtype=float)
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("P_mm_h deve ser finito e >= 0")
        return v


@app.exception_handler(RequestValidationError)
async def _validacao(request: Request, exc: RequestValidationError):
    if request.url.path not in ("/v1/simulate/bucket", "/v1/bucket/point10"):
        return await request_validation_exception_handler(request, exc)
    # NaN/Inf no corpo quebram a serializacao do campo "input" no handler padrao: omite input e ctx.
    erros = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": erros})


@app.post("/v1/simulate/bucket")
def simulate_bucket(data: BucketInput):
    """Simulacao ilustrativa (nao e previsao). Sem mare, sem escrita (ADR 0014)."""
    base = Params()
    p = base.com(f=data.params.f, curve_scale=data.params.curve_scale)
    pt = Ponto(
        slug=data.point.slug,
        area_m2={p.snap_m: data.point.area_m2},
        z_curva_m=np.asarray(data.point.z_curva_m, dtype=float),
        v_curva_m3=np.asarray(data.point.v_curva_m3, dtype=float),
        rua_rel_m=data.point.rua_rel_m,
    )
    h, fora = simular_ponto(np.asarray(data.P_mm_h, dtype=float), pt, p)
    cruz = np.flatnonzero(h >= p.hcrit_mm)
    return {
        "h_mm": h.tolist(),
        "n_fora_da_curva": int(fora),
        "hmax_mm": float(h.max()),
        "t_cross_h": int(cruz[0]) if cruz.size else None,
        "params_usados": asdict(p),
        "model": "M1a",
        "illustrative": True,
    }


# --- Balde M1a a 10 min para o nível ALAGAMENTO_PROVAVEL do alerta (mapi_ai/bucket10.py) ---
from mapi_ai import bucket10 as B10


class Bucket10Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rain_10min_mm: list[float] = Field(min_length=1, max_length=288, description="chuva medida por passo de 10 min, em ordem; sem lacuna")
    point: BucketPoint
    tr_anos: float = Field(default=5.0, ge=2.0, le=10.0)
    te_min: float = Field(default=10.0, ge=5.0, le=15.0)
    forecast_1h_q50_mm: Optional[float] = Field(default=None, ge=0)
    forecast_1h_q80_mm: Optional[float] = Field(default=None, ge=0)
    # limpeza recente (mapi_ai/entupimento.py): opcionais; sem eles a resposta é a de sempre e "limpeza" vem nulo
    days_since_cleaning: Optional[float] = Field(default=None, ge=0, le=36500)
    lambda_re_per_day: Optional[float] = Field(default=None, gt=0, le=10)
    f_clogged: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("rain_10min_mm")
    @classmethod
    def _p10_valida(cls, v):
        a = np.asarray(v, dtype=float)
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("rain_10min_mm deve ser finito e >= 0 (ausente não vira zero: mande a série sem lacuna)")
        return v


@app.post("/v1/bucket/point10")
def bucket_point10(data: Bucket10Input):
    """Lâmina do balde M1a a 10 min (agora) e projeção dos próximos 60 min com a chuva prevista q50 e q80."""
    import time as _t
    t0 = _t.perf_counter()
    p = B10.Params10(tr_anos=data.tr_anos, te_min=data.te_min)
    terr = B10.Terreno(area_m2=data.point.area_m2, z_curva_m=np.asarray(data.point.z_curva_m, dtype=float),
                       v_curva_m3=np.asarray(data.point.v_curva_m3, dtype=float), rua_rel_m=data.point.rua_rel_m)
    out = B10.avaliar(np.asarray(data.rain_10min_mm, dtype=float), terr, p, data.forecast_1h_q50_mm, data.forecast_1h_q80_mm)
    from mapi_ai import entupimento as ENT
    out["limpeza"] = ENT.avaliar_com_limpeza(np.asarray(data.rain_10min_mm, dtype=float), terr, p, data.forecast_1h_q50_mm,
                                             data.forecast_1h_q80_mm, data.days_since_cleaning, data.lambda_re_per_day,
                                             data.f_clogged)
    out["slug"] = data.point.slug
    out["projection_rule"] = "chuva prevista de 1 h distribuída por igual nos 6 passos de 10 min"
    out["validated"] = False
    out["runtime_ms"] = round((_t.perf_counter() - t0) * 1000, 3)
    return out


class RainReading(BaseModel):
    station_id: str
    latitude: float
    longitude: float
    ts: datetime = Field(..., description="fim do intervalo da leitura, UTC")
    mm: Optional[float] = Field(None, description="chuva do intervalo; null = sem valor (nunca zero)")


class OpenMeteoHour(BaseModel):
    hour_end_utc: datetime
    mm: Optional[float] = None


class RainForecastInput(BaseModel):
    latitude: float
    longitude: float
    readings: List[RainReading] = Field(..., description="leituras de 10 min das últimas 6 h, estações a até 40 km")
    open_meteo: Optional[List[OpenMeteoHour]] = Field(None, description="previsão da rodada da véspera, por hora")
    issued_at: Optional[datetime] = Field(None, description="instante t da previsão; padrão = a leitura mais recente")


@app.post("/v1/forecast/rain")
def forecast_rain(data: RainForecastInput):
    """Chuva prevista no ponto em (t, t+1h], (t, t+3h] e (t, t+6h]: quantis e probabilidade de evento MEDIA e FORTE."""
    if rain_model is None:
        raise HTTPException(status_code=503, detail="modelo de previsão de chuva não carregado")
    try:
        return rain_model.forecast(
            data.latitude, data.longitude,
            [{"station_id": r.station_id, "latitude": r.latitude, "longitude": r.longitude, "ts": r.ts, "mm": r.mm} for r in data.readings],
            [{"hour_end_utc": o.hour_end_utc, "mm": o.mm} for o in data.open_meteo] if data.open_meteo else None,
            data.issued_at.isoformat() if data.issued_at else None)
    except MissingFeaturesError as e:
        return JSONResponse(status_code=422, content={"detail": str(e), "missing_features": e.missing_features})


class FloodForecastInput(BaseModel):
    """Alerta de alagamento por regressão (mapi_ai/flood). Falta de dado vira SEM_DADO com motivo (HTTP 200), nunca zero."""
    model_config = ConfigDict(extra="forbid")
    latitude: float
    longitude: float
    readings: List[RainReading] = Field(..., description="leituras de 10 min das últimas 6 h, estações a até 40 km (para o rain_v1)")
    open_meteo: Optional[List[OpenMeteoHour]] = None
    issued_at: Optional[datetime] = None
    rain_10min_mm: List[Optional[float]] = Field(..., max_length=288,
                                                 description="chuva MEDIDA do ponto por passo de 10 min, terminando em t; null = lacuna")
    point: Optional[BucketPoint] = Field(None, description="terreno do ponto; ausente = SEM_TERRENO")
    terrain_status: Literal["OK", "SEM_TERRENO", "EM_REVISAO"] = "OK"
    tr_anos: float = Field(default=5.0, ge=2.0, le=10.0)
    te_min: float = Field(default=10.0, ge=5.0, le=15.0)

    @field_validator("rain_10min_mm")
    @classmethod
    def _sem_negativo(cls, v):
        if any(x is not None and (not np.isfinite(x) or x < 0) for x in v):
            raise ValueError("rain_10min_mm deve ser >= 0 e finito (use null para lacuna)")
        return v


@app.post("/v1/flood/forecast")
def flood_forecast(data: FloodForecastInput):
    """Lâmina prevista (q50 e q95) em 1, 3 e 6 h e nível ALERTA / ATENCAO / SEM_ALERTA / SEM_DADO. Experimental."""
    from mapi_ai.flood.forecast import forecast_point
    terr = None
    if data.point is not None:
        terr = B10.Terreno(area_m2=data.point.area_m2, z_curva_m=np.asarray(data.point.z_curva_m, dtype=float),
                           v_curva_m3=np.asarray(data.point.v_curva_m3, dtype=float), rua_rel_m=data.point.rua_rel_m)
    return forecast_point(
        latitude=data.latitude, longitude=data.longitude,
        readings=[{"station_id": r.station_id, "latitude": r.latitude, "longitude": r.longitude, "ts": r.ts, "mm": r.mm}
                  for r in data.readings],
        rain_10min_mm=data.rain_10min_mm, terrain=terr, terrain_status=data.terrain_status,
        open_meteo=[{"hour_end_utc": o.hour_end_utc, "mm": o.mm} for o in data.open_meteo] if data.open_meteo else None,
        issued_at=data.issued_at.isoformat() if data.issued_at else None, rain_model=rain_model,
        tr_anos=data.tr_anos, te_min=data.te_min, slug=data.point.slug if data.point else None)


@app.get("/health")
async def health():
    from mapi_ai.flood.config import VERSION as FLOOD_VERSION
    from mapi_ai.version import code_version
    cv = code_version()
    return {
        "status": "ok",
        "rain_model_version": rain_model.meta["model_version"] if rain_model is not None else None,
        "bucket_version": B10.VERSION,
        "flood_version": FLOOD_VERSION,
        "code_version": cv["commit"],
        "code_dirty": cv["dirty"],
        "code_version_source": cv["source"],
    }
