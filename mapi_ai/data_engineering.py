import pandas as pd
from sqlalchemy import create_engine
from mapi_ai.config import DATABASE_URL, RESAMPLE_INTERVAL

def get_engine():
    """Returns a SQLAlchemy engine for the PostgreSQL database."""
    return create_engine(DATABASE_URL)

def load_sensor_data(table_name: str, engine):
    """Loads raw data from a specified table."""
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, engine)

def preprocess_time_series(df: pd.DataFrame, time_col='timestamp'):
    """
    Standardizes the time series:
    - Sets timestamp as index
    - Converts to datetime
    - Resamples to a fixed interval
    - Interpolates missing values
    """
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    
    # Resample and interpolate
    df = df.resample(RESAMPLE_INTERVAL).mean()
    df = df.interpolate(method='linear')
    
    return df

def merge_datasets(sensor_df, weather_df, tide_df):
    """Merges all datasets on the timestamp index."""
    merged = sensor_df.join(weather_df, how='outer', rsuffix='_weather')
    merged = merged.join(tide_df, how='outer', rsuffix='_tide')
    return merged.interpolate(method='linear').dropna()
