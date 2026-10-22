import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from mapi_ai.config import MODEL_PATH_XGB, MODEL_PATH_LSTM, SCALER_PATH, TIDE_THRESHOLD
from mapi_ai.feature_engineering import prepare_scenario_features, SpatialMetadataCache, MissingFeaturesError

# Load models at startup
try:
    model = joblib.load(MODEL_PATH_XGB)
    scaler = joblib.load(SCALER_PATH)
    print("XGBoost Classifier and Scaler loaded successfully.")
except Exception as e:
    model = None
    scaler = None
    print(f"Warning: XGBoost models not found ({e}). Train the model first.")

try:
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH_LSTM):
        lstm_model = load_model(MODEL_PATH_LSTM, compile=False)
        print("LSTM Regressor loaded successfully (without compile for inference).")
    else:
        lstm_model = None
        print("Warning: LSTM model file not found.")
except Exception as e:
    lstm_model = None
    print(f"Warning: Could not load LSTM model ({e}).")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize cache from DB at startup
    print("Pre-loading Spatial Metadata Cache from database...")
    cache = SpatialMetadataCache()
    cache.load_from_db()
    yield

app = FastAPI(title="MAPI Flood Prediction API", version="1.0", lifespan=lifespan)

class NearbySensorInput(BaseModel):
    sensor_id: str
    latitude: float = None
    longitude: float = None
    lat: float = None
    lon: float = None
    value: float
    # O backend repassa leituras de sensor com unidade nula (ex.: régua de rio da APAC sem unidade cadastrada);
    # a unidade não entra em nenhuma feature, então não pode derrubar a previsão com 422 de validação.
    unit: Optional[str] = None
    type: str
    distanceKm: float = None
    distance_km: float = None

class PredictionInput(BaseModel):
    station_id: str
    lat: float = None
    lon: float = None
    latitude: float = None
    longitude: float = None
    current_rainfall: float
    rainfall_3h_accumulated: float
    rainfall_6h_accumulated: float
    rainfall_12h_accumulated: float
    rainfall_24h_accumulated: float
    tide_level: float
    river_level: Optional[float] = None
    timestamp: str
    
    # Campos meteorologicos sem valor padrao: ausentes => 422 com missing_features (sem imputacao, ADR 0007 par. 3)
    wind_speed: Optional[float] = None
    temperature: Optional[float] = None
    apparent_temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wave_height: Optional[float] = None
    wave_period: Optional[float] = None
    wave_direction: Optional[float] = None
    solar_radiation: Optional[float] = None
    
    # Proximity sensor readings
    nearby_sensors: list[NearbySensorInput] = []

def prepare_features_for_inference(data: PredictionInput):
    """
    Converts input prediction payload to a DataFrame and routes it
    through the feature engineering pipeline.
    """
    data_dict = data.dict()
    
    # Resolve latitude/longitude aliases
    data_dict['latitude'] = data_dict['latitude'] if data_dict['latitude'] is not None else data_dict['lat']
    data_dict['longitude'] = data_dict['longitude'] if data_dict['longitude'] is not None else data_dict['lon']
    
    if data_dict['latitude'] is None or data_dict['longitude'] is None:
        missing = [k for k in ('latitude', 'longitude') if data_dict[k] is None]
        raise MissingFeaturesError(missing)
        
    df = pd.DataFrame([data_dict])
    X = prepare_scenario_features(df, is_training=False)
    return X

def predict_river_level_lstm(data: PredictionInput, X_current: pd.DataFrame):
    """
    Predicts river level using LSTM. Combines DB history with current prediction
    input to build a sequence.
    """
    if lstm_model is None or scaler is None:
        return None
        
    try:
        window_size = 12
        latitude = data.latitude if data.latitude is not None else data.lat
        longitude = data.longitude if data.longitude is not None else data.lon
        
        # Try loading historical records from PostgreSQL
        from mapi_ai.data_engineering import get_engine
        engine = get_engine()
        
        query = f"""
            SELECT * FROM flood_scenario_labels 
            WHERE latitude BETWEEN {latitude - 0.05} AND {latitude + 0.05}
              AND longitude BETWEEN {longitude - 0.05} AND {longitude + 0.05}
            ORDER BY timestamp DESC
            LIMIT {window_size - 1}
        """
        df_hist = pd.read_sql(query, engine)
        
        if len(df_hist) < window_size - 1:
            # Generate padding using current row if history is insufficient
            padding_needed = (window_size - 1) - len(df_hist)
            X_pad = pd.concat([X_current] * padding_needed, ignore_index=True)
            if len(df_hist) > 0:
                X_hist = prepare_scenario_features(df_hist, is_training=False)
                X_seq_df = pd.concat([X_pad, X_hist, X_current], ignore_index=True)
            else:
                X_seq_df = pd.concat([X_pad, X_current], ignore_index=True)
        else:
            df_hist = df_hist.iloc[::-1]  # Chronological order
            X_hist = prepare_scenario_features(df_hist, is_training=False)
            X_seq_df = pd.concat([X_hist, X_current], ignore_index=True)
            
        X_scaled = scaler.transform(X_seq_df)
        X_input = np.expand_dims(X_scaled, axis=0)
        
        pred = lstm_model.predict(X_input, verbose=0)
        return float(pred[0][0])
    except Exception as e:
        print(f"LSTM Prediction Error: {e}")
        return None

@app.post("/v1/predict/flood")
def predict_flood(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model.")
    
    try:
        try:
            X = prepare_features_for_inference(data)
        except MissingFeaturesError as e:
            return JSONResponse(
                status_code=422,
                content={"detail": str(e), "missing_features": e.missing_features},
            )
        X_scaled = scaler.transform(X)
        
        prob = model.predict_proba(X_scaled)[0][1]
        
        risk_level = "LOW"
        if prob > 0.8: 
            risk_level = "HIGH"
        elif prob > 0.5: 
            risk_level = "MEDIUM"
        
        response = {
            "flood_probability": float(prob),
            "risk_level": risk_level,
            # ADR 0007 §3 (projeto-mapi-api): sem artefato versionado até a Fase 4, a versão é "unversioned".
            "model_version": "unversioned",
            # ADR 0007 Emenda 1: os artefatos atuais nao tem avaliacao pre-registrada positiva.
            "model_validation_status": "NOT_VALIDATED",
            "station_id": data.station_id,
            "timestamp": data.timestamp
        }
        
        # Predict river level using LSTM if available
        lstm_val = predict_river_level_lstm(data, X)
        if lstm_val is not None:
            response["predicted_river_level"] = lstm_val
            
        return response
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Simulacao do balde M1a (ADR 0014): a formula vive so em avaliacao/balde.py ---
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
    if request.url.path != "/v1/simulate/bucket":
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


@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "xgboost_loaded": model is not None,
        "lstm_loaded": lstm_model is not None
    }
