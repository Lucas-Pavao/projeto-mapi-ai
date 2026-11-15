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

class NoLabelsError(RuntimeError):
    """Não há rótulo real e o modo sintético não foi pedido explicitamente."""


def load_dataset(csv_path=None, synthetic=False):
    """
    Carrega o dataset de treino. NÃO existe mais fallback silencioso.

    Antes, com flood_scenario_labels vazia, esta função gerava 200 linhas com is_flooded por fórmula fixa e treinava
    sobre elas sem avisar: as explicações (SHAP/LIME) apenas devolviam a fórmula — eram circulares.

    Agora:
    - `csv_path`: lê o CSV indicado;
    - sem CSV, lê `flood_scenario_labels` do banco; vazia => NoLabelsError com a explicação;
    - `synthetic=True` (flag `--synthetic`): lê o dataset do Plano B (mapi_ai/synthetic.py), com aviso explícito.
    """
    from mapi_ai import in1165_config as C
    if synthetic:
        print("=" * 72)
        print(f"AVISO: {C.SYNTHETIC_TAG}.")
        print("Rótulo gerado por fórmula conhecida (reports/synthetic_ground_truth.md) sobre horas reais.")
        print("Nenhuma métrica deste treino é desempenho em alagamento real.")
        print("=" * 72)
        if not C.DATASET_CSV.exists():
            raise FileNotFoundError(f"{C.DATASET_CSV} não existe: rode `python -m mapi_ai.synthetic` antes.")
        return pd.read_csv(C.DATASET_CSV, parse_dates=["timestamp"])
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV {csv_path} não encontrado.")
        return pd.read_csv(csv_path)
    df = load_scenario_data(get_engine())
    if len(df) == 0:
        raise NoLabelsError("flood_scenario_labels está vazia e não há rótulo real suficiente "
                            "(python -m mapi_ai.build_labels). Para o experimento controlado, use --synthetic.")
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
