from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from mapi_ai.config import MODEL_PATH_XGB, SCALER_PATH, TIDE_THRESHOLD

app = FastAPI(title="MAPI Flood Prediction API", version="1.0")

# Load models at startup
try:
    model = joblib.load(MODEL_PATH_XGB)
    scaler = joblib.load(SCALER_PATH)
    print("Models loaded successfully.")
except Exception as e:
    model = None
    scaler = None
    print(f"Warning: Models not found ({e}). Train the model first.")

class PredictionInput(BaseModel):
    station_id: str
    lat: float
    lon: float
    current_rainfall: float
    rainfall_3h_accumulated: float
    rainfall_6h_accumulated: float
    rainfall_12h_accumulated: float
    rainfall_24h_accumulated: float
    tide_level: float
    river_level: float = 0.0
    timestamp: str

def prepare_features_for_inference(data: PredictionInput):
    """
    Manually creates the feature vector for a single point, 
    matching the training features exactly.
    """
    ts = pd.to_datetime(data.timestamp)
    
    # 1. Cyclic time features
    hour_sin = np.sin(2 * np.pi * ts.hour / 24)
    hour_cos = np.cos(2 * np.pi * ts.hour / 24)
    month_sin = np.sin(2 * np.pi * ts.month / 12)
    month_cos = np.cos(2 * np.pi * ts.month / 12)
    
    # 2. Tide features
    is_high_tide = 1 if data.tide_level > TIDE_THRESHOLD else 0
    
    # 3. Create Feature Dictionary (Order must match training columns)
    # Note: Training features: [rainfall, rainfall_lag_1h, ..., hour_sin, ..., is_high_tide]
    # WE MUST ENSURE THIS ORDER MATCHES WHAT THE MODEL EXPECTS.
    features = {
        'rainfall': data.current_rainfall,
        'rainfall_lag_1h': data.rainfall_3h_accumulated / 3, # Approximation if lag 1h is missing
        'rainfall_lag_3h': data.rainfall_3h_accumulated,
        'rainfall_lag_6h': data.rainfall_6h_accumulated,
        'rainfall_lag_12h': data.rainfall_12h_accumulated,
        'rainfall_lag_24h': data.rainfall_24h_accumulated,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_high_tide': is_high_tide
    }
    
    return pd.DataFrame([features])

@app.post("/v1/predict/flood")
async def predict_flood(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model.")
    
    try:
        X = prepare_features_for_inference(data)
        X_scaled = scaler.transform(X)
        
        prob = model.predict_proba(X_scaled)[0][1]
        
        risk_level = "LOW"
        if prob > 0.8: risk_level = "HIGH"
        elif prob > 0.5: risk_level = "MEDIUM"
        
        return {
            "flood_probability": float(prob),
            "risk_level": risk_level,
            "station_id": data.station_id,
            "timestamp": data.timestamp
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
