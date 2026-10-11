import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
    unit: str
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

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "xgboost_loaded": model is not None,
        "lstm_loaded": lstm_model is not None
    }
