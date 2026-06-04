import pandas as pd
import numpy as np
from mapi_ai.config import TIDE_THRESHOLD

def create_lags(df: pd.DataFrame, col: str, lags: list):
    """Creates lag features for a specific column."""
    for lag in lags:
        df[f'{col}_lag_{lag}h'] = df[col].shift(periods=int(lag * 4)) # Assuming 15min intervals (4 per hour)
    return df

def extract_cyclic_features(df: pd.DataFrame):
    """Extracts cyclic components (sine/cosine) for hour and month."""
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df.drop(columns=['hour', 'month'])

def add_tide_features(df: pd.DataFrame, tide_col='tide_level'):
    """Adds tide-related features like boolean high tide and normalized tide."""
    df['is_high_tide'] = (df[tide_col] > TIDE_THRESHOLD).astype(int)
    # Simple normalization (min-max or similar can be applied later with StandardScaler)
    return df

def generate_features(df: pd.DataFrame, rainfall_col='rainfall'):
    """Main pipeline for feature engineering."""
    df = create_lags(df, rainfall_col, [1, 3, 6, 12, 24])
    df = extract_cyclic_features(df)
    df = add_tide_features(df)
    return df.dropna()
