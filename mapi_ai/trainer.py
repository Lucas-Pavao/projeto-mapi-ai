import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from mapi_ai.data_engineering import get_engine, load_sensor_data, preprocess_time_series, merge_datasets
from mapi_ai.feature_engineering import generate_features
from mapi_ai.models import get_xgboost_classifier
from mapi_ai.config import MODEL_PATH_XGB, SCALER_PATH

def train_pipeline():
    # 1. Load Data
    engine = get_engine()
    
    # Using direct SQL as requested. 
    # Table names updated to match the database schema created by the Java API and SQL script.
    try:
        print("Loading data from database...")
        df_sensors = load_sensor_data("sensor_data", engine)
        df_weather = load_sensor_data("weather_data", engine)
        # Note: Java API uses plural names 'tide_tables' and 'flood_events'
        df_tide = load_sensor_data("tide_tables", engine)
        df_labels = load_sensor_data("flood_events", engine)
        print(f"Data loaded: Sensors({len(df_sensors)}), Weather({len(df_weather)}), Tide({len(df_tide)}), Labels({len(df_labels)})")
    except Exception as e:
        print(f"Error loading data: {e}. Ensure database is configured and tables exist.")
        return

    # 2. Preprocess
    # This assumes tables have a 'timestamp' column or similar as handled in data_engineering.py
    df_sensors = preprocess_time_series(df_sensors)
    df_weather = preprocess_time_series(df_weather)
    df_tide = preprocess_time_series(df_tide)
    df_labels = preprocess_time_series(df_labels)

    # 3. Merge and Feature Engineering
    df = merge_datasets(df_sensors, df_weather, df_tide)
    # Join with labels - assumes labels are already resampled/aligned
    df = df.join(df_labels, how='left').fillna(0) 
    
    # We need to identify the correct target column from the joined labels
    # In 'flood_events', this might be 'severity' or a generated 'is_flood' column
    # For now, we assume 'is_flood' exists after preprocessing/merging or we use a fallback
    if 'is_flood' not in df.columns:
        if 'severity' in df.columns:
            df['is_flood'] = (df['severity'].notnull()).astype(int)
        else:
            # Fallback if no explicit label column is found
            df['is_flood'] = 0 

    df = generate_features(df, rainfall_col='accumulated_precipitation' if 'accumulated_precipitation' in df.columns else 'precipitation')

    # 4. Prepare Features and Targets
    target_clf = 'is_flood'
    
    # Selected features based on the available columns
    available_features = [c for c in df.columns if c not in [target_clf, 'severity', 'id', 'latitude', 'longitude']]
    X = df[available_features]
    y_clf = df[target_clf].astype(int)

    if len(X) < 10:
        print("Not enough data to train. Ingest more data first.")
        return

    # 5. Training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Simple split (could use TimeSeriesSplit for better validation)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_clf[:train_size], y_clf[train_size:]

    print(f"Training XGBoost on {len(X_train)} samples...")
    clf = get_xgboost_classifier()
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print("XGBoost Evaluation:")
    print(classification_report(y_test, preds))

    # Save final model and scaler
    joblib.dump(clf, MODEL_PATH_XGB)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Models saved to {MODEL_PATH_XGB} and {SCALER_PATH}")

if __name__ == "__main__":
    train_pipeline()
