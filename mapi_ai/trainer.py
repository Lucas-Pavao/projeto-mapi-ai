import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from mapi_ai.data_engineering import get_engine, load_sensor_data
from mapi_ai.feature_engineering import prepare_scenario_features, FEATURE_COLUMNS
from mapi_ai.models import get_xgboost_classifier, get_lstm_model, create_sequences
from mapi_ai.config import MODEL_PATH_XGB, MODEL_PATH_LSTM, SCALER_PATH

def load_scenario_data(engine):
    """Loads scenario labels from flood_scenario_labels table."""
    query = "SELECT * FROM flood_scenario_labels ORDER BY timestamp ASC"
    return pd.read_sql(query, engine)

def generate_dummy_scenario_data(n_samples=200):
    """Generates synthetic flood scenario dataset for testing/bootstrapping."""
    np.random.seed(42)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='15min')
    
    # Coordinates around Recife area
    latitudes = np.random.uniform(-8.12, -8.04, size=n_samples)
    longitudes = np.random.uniform(-34.95, -34.88, size=n_samples)
    
    current_rainfall = np.random.exponential(scale=2.0, size=n_samples)
    # Accumulated precipitation
    rainfall_3h = current_rainfall * 1.5 + np.random.normal(0, 1, size=n_samples)
    rainfall_6h = rainfall_3h * 1.8 + np.random.normal(0, 2, size=n_samples)
    rainfall_12h = rainfall_6h * 1.5 + np.random.normal(0, 4, size=n_samples)
    rainfall_24h = rainfall_12h * 1.3 + np.random.normal(0, 8, size=n_samples)
    
    current_rainfall = np.clip(current_rainfall, 0, None)
    rainfall_3h = np.clip(rainfall_3h, 0, None)
    rainfall_6h = np.clip(rainfall_6h, 0, None)
    rainfall_12h = np.clip(rainfall_12h, 0, None)
    rainfall_24h = np.clip(rainfall_24h, 0, None)
    
    tide_level = np.random.uniform(0.0, 2.8, size=n_samples)
    river_level = np.random.uniform(0.5, 4.0, size=n_samples)
    
    # Make flooding probability depend on rainfall and tide
    flood_prob = (current_rainfall * 0.15 + rainfall_6h * 0.05 + (tide_level > 2.0) * 0.3 + (river_level > 3.0) * 0.4)
    is_flooded = (flood_prob > 0.6).astype(bool)
    
    data = {
        'id': range(1, n_samples + 1),
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes,
        'is_flooded': is_flooded,
        'current_rainfall': current_rainfall,
        'rainfall_3h_accumulated': rainfall_3h,
        'rainfall_6h_accumulated': rainfall_6h,
        'rainfall_12h_accumulated': rainfall_12h,
        'rainfall_24h_accumulated': rainfall_24h,
        'tide_level': tide_level,
        'river_level': river_level,
        'wind_speed': np.random.uniform(0, 25, size=n_samples),
        'temperature': np.random.uniform(22, 32, size=n_samples),
        'apparent_temperature': np.random.uniform(24, 35, size=n_samples),
        'humidity': np.random.uniform(60, 95, size=n_samples),
        'pressure': np.random.uniform(1008, 1016, size=n_samples),
        'wave_height': np.random.uniform(0.5, 2.5, size=n_samples),
        'wave_period': np.random.uniform(4, 12, size=n_samples),
        'wave_direction': np.random.uniform(0, 360, size=n_samples),
        'solar_radiation': np.random.uniform(0, 800, size=n_samples)
    }
    
    return pd.DataFrame(data)

def load_dataset(csv_path=None):
    """
    Loads dataset either from CSV or PostgreSQL database.
    If both fail or are empty, creates a synthetic dataset for demonstration and testing.
    """
    df = None
    if csv_path:
        if os.path.exists(csv_path):
            print(f"Loading data from CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            print(f"Warning: CSV file {csv_path} not found.")
            
    if df is None:
        try:
            print("Connecting to database...")
            engine = get_engine()
            print("Loading data from table 'flood_scenario_labels'...")
            df = load_scenario_data(engine)
            print(f"Loaded {len(df)} records from database.")
        except Exception as e:
            print(f"Warning: Could not load data from database: {e}")
            
    if df is None or len(df) == 0:
        print("Warning: No data loaded. Generating synthetic/dummy dataset for fallback/testing...")
        df = generate_dummy_scenario_data()
        
    return df

def train_xgboost(df):
    """Trains XGBoost Classifier for is_flooded prediction."""
    print("--- XGBoost Training Pipeline ---")
    
    # 1. Feature Engineering
    print("Preparing features...")
    X = prepare_scenario_features(df, is_training=True)
    y = df['is_flooded'].astype(int)
    
    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train/Test Split
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 4. Train Model
    clf = get_xgboost_classifier()
    clf.fit(X_train, y_train)
    
    # 5. Evaluate Model
    preds = clf.predict(X_test)
    print("XGBoost Classifier Evaluation:")
    print(classification_report(y_test, preds))
    
    # 6. Save Model and Scaler
    os.makedirs(os.path.dirname(MODEL_PATH_XGB), exist_ok=True)
    joblib.dump(clf, MODEL_PATH_XGB)
    joblib.dump(scaler, SCALER_PATH)
    print(f"XGBoost model saved to {MODEL_PATH_XGB}")
    print(f"StandardScaler saved to {SCALER_PATH}")
    
    return clf, scaler

def train_lstm(df):
    """Trains LSTM model for water level (river_level) regression."""
    print("--- LSTM Training Pipeline ---")
    
    # 1. Feature Engineering
    print("Preparing features...")
    X = prepare_scenario_features(df, is_training=True)
    
    # For LSTM, we target 'river_level'
    target_col = 'river_level'
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset. LSTM training aborted.")
        return
        
    y = df[target_col].values
    
    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Create Sequences
    # We use a standard window size (e.g. 12 or 24 timesteps, let's use 12 if data is small, 24 if larger)
    window_size = min(12, len(df) // 10)
    if window_size < 1:
        window_size = 1
        
    print(f"Creating sequences with window_size={window_size}...")
    X_seq, y_seq = create_sequences(X_scaled, y, window_size)
    
    if len(X_seq) < 5:
        print("Error: Not enough sequences for LSTM training. Aborted.")
        return
        
    # 4. Train/Test Split
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # 5. Build and Train Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = get_lstm_model(input_shape)
    
    print("Training LSTM...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=16,
        verbose=1
    )
    
    # 6. Evaluate Model
    preds = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"LSTM Evaluation: MSE={mse:.4f}, MAE={mae:.4f}")
    
    # 7. Save Model
    os.makedirs(os.path.dirname(MODEL_PATH_LSTM), exist_ok=True)
    model.save(MODEL_PATH_LSTM)
    print(f"LSTM model saved to {MODEL_PATH_LSTM}")

def train_pipeline(csv_path=None, model_type="all"):
    """Facade for triggering training from other scripts/code."""
    df = load_dataset(csv_path)
    if model_type in ["all", "xgb"]:
        train_xgboost(df)
    if model_type in ["all", "lstm"]:
        train_lstm(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPI AI Model Trainer")
    parser.add_argument("--csv", type=str, help="Path to scenario labels CSV file")
    parser.add_argument("--model", type=str, choices=["all", "xgb", "lstm"], default="all",
                        help="Model to train: xgb, lstm, or all (default: all)")
    args = parser.parse_args()
    
    # Load dataset
    df = load_dataset(args.csv)
    
    # Train
    if args.model in ["all", "xgb"]:
        train_xgboost(df)
    if args.model in ["all", "lstm"]:
        train_lstm(df)
