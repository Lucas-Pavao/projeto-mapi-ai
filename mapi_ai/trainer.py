import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from mapi_ai.data_engineering import get_engine, load_sensor_data, preprocess_time_series, merge_datasets
from mapi_ai.feature_engineering import generate_features
from mapi_ai.models import get_xgboost_classifier, get_lstm_model
from mapi_ai.config import MODEL_PATH_XGB, MODEL_PATH_LSTM, SCALER_PATH

def train_pipeline():
    # 1. Load Data
    engine = get_engine()
    # Assuming tables: sensor_data, weather_data, tide_table, flood_labels
    # This is a placeholder for actual table names
    try:
        df_sensors = load_sensor_data("sensor_data", engine)
        df_weather = load_sensor_data("weather_data", engine)
        df_tide = load_sensor_data("tide_table", engine)
        df_labels = load_sensor_data("flood_labels", engine)
    except Exception as e:
        print(f"Error loading data: {e}. Ensure database is configured.")
        return

    # 2. Preprocess
    df_sensors = preprocess_time_series(df_sensors)
    df_weather = preprocess_time_series(df_weather)
    df_tide = preprocess_time_series(df_tide)
    df_labels = preprocess_time_series(df_labels)

    # 3. Merge and Feature Engineering
    df = merge_datasets(df_sensors, df_weather, df_tide)
    df = df.join(df_labels, how='left').fillna(0) # 0 for no flood
    df = generate_features(df)

    # 4. Prepare Features and Targets
    target_clf = 'is_flood'  # Binary classification
    target_reg = 'water_level' # Regression
    
    features = [c for c in df.columns if c not in [target_clf, target_reg]]
    X = df[features]
    y_clf = df[target_clf]

    # 5. TimeSeriesSplit Validation
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_clf.iloc[train_index], y_clf.iloc[test_index]

        # Scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train XGBoost
        clf = get_xgboost_classifier()
        clf.fit(X_train_scaled, y_train)
        
        preds = clf.predict(X_test_scaled)
        print("XGBoost Evaluation:")
        print(classification_report(y_test, preds))

    # Save final model and scaler
    joblib.dump(clf, MODEL_PATH_XGB)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Models saved to {MODEL_PATH_XGB} and {SCALER_PATH}")

if __name__ == "__main__":
    train_pipeline()
