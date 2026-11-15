import numpy as np
import xgboost as xgb
from mapi_ai.config import PREDICTION_WINDOW_HOURS

def get_xgboost_classifier(**params):
    """XGBoost para is_flooded. Sem padronização: árvores dividem por limiar e não dependem de escala, e manter as
    unidades físicas (mm, m) faz SHAP, LIME e contrafactuais saírem em mm e m, não em z-scores."""
    base = dict(n_estimators=100, max_depth=6, learning_rate=0.1, objective="binary:logistic",
                random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=1)
    base.update(params)
    return xgb.XGBClassifier(**base)


def get_lstm_model(input_shape):
    """Returns a compiled LSTM model for regression."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1) # Predicting water level
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_sequences(data, target, window_size):
    """Utility to create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)
