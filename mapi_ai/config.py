import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mapi_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Model Configuration
PREDICTION_WINDOW_HOURS = 6
RESAMPLE_INTERVAL = "15min"
TIDE_THRESHOLD = 2.0  # Meters

# Paths
MODEL_PATH_XGB = "models/flood_classifier.joblib"
MODEL_PATH_LSTM = "models/water_level_lstm.h5"
SCALER_PATH = "models/scaler.joblib"
