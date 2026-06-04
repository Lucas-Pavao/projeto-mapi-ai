import numpy as np
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from mapi_ai.config import PREDICTION_WINDOW_HOURS

def get_xgboost_classifier():
    """Returns an initialized XGBoost Classifier."""
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

def get_lstm_model(input_shape):
    """Returns a compiled LSTM model for regression."""
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
