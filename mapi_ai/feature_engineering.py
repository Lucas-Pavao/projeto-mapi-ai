import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine
from mapi_ai.config import DATABASE_URL, TIDE_THRESHOLD

logger = logging.getLogger(__name__)

# Standard ordered feature list for model training and inference
FEATURE_COLUMNS = [
    'latitude',
    'longitude',
    'current_rainfall',
    'rainfall_3h_accumulated',
    'rainfall_6h_accumulated',
    'rainfall_12h_accumulated',
    'rainfall_24h_accumulated',
    'tide_level',
    'river_level',
    'altitude',
    'distance_to_channel',
    'distance_to_flood_point',
    'distance_to_rain_sensor',
    'distance_to_river_sensor',
    'wind_speed',
    'temperature',
    'apparent_temperature',
    'humidity',
    'pressure',
    'wave_height',
    'wave_period',
    'wave_direction',
    'solar_radiation',
    'hour_sin',
    'hour_cos',
    'month_sin',
    'month_cos',
    'is_high_tide'
]

class SpatialMetadataCache:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SpatialMetadataCache, cls).__new__(cls, *args, **kwargs)
            cls._instance.initialized = False
        return cls._instance
        
    def __init__(self):
        if self.initialized:
            return
        self.sensors = []       # [{'sensor_id': ..., 'latitude': ..., 'longitude': ..., 'type': ...}]
        self.flood_points = []  # [{'slug': ..., 'lat': ..., 'lon': ..., 'altitude': ..., 'distance_to_channel': ...}]
        self.initialized = True
        
    def load_from_db(self):
        try:
            engine = create_engine(DATABASE_URL)
            
            # Load unique sensors from sensor_data
            query_sensors = """
                SELECT DISTINCT sensor_id, latitude, longitude, type 
                FROM sensor_data 
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """
            df_s = pd.read_sql(query_sensors, engine)
            self.sensors = df_s.to_dict('records')
            logger.info(f"Loaded {len(self.sensors)} unique sensors from database.")
        except Exception as e:
            logger.warning(f"Could not load sensor data metadata: {e}")
            self.sensors = []

        try:
            engine = create_engine(DATABASE_URL)
            # Load flood points
            query_fp = "SELECT * FROM flood_points"
            df_fp = pd.read_sql(query_fp, engine)
            
            self.flood_points = []
            for _, row in df_fp.iterrows():
                # Handle altitude column name variations
                alt = 0.0
                for col in ['altitude_m', 'altitudem', 'altitude_m_elevation', 'elevation', 'altitudeM']:
                    if col in df_fp.columns and pd.notna(row[col]):
                        alt = float(row[col])
                        break
                else:
                    if 'altitude' in df_fp.columns and pd.notna(row['altitude']):
                        alt = float(row['altitude'])
                
                # Handle distance to channel
                dist_chan = 100.0
                for col in ['distance_to_channel_m', 'distancetochannelm', 'dist_canal_m', 'distance_to_channel', 'distanceToChannelM']:
                    if col in df_fp.columns and pd.notna(row[col]):
                        dist_chan = float(row[col])
                        break
                
                lat = float(row['latitude']) if 'latitude' in df_fp.columns else 0.0
                lon = float(row['longitude']) if 'longitude' in df_fp.columns else 0.0
                slug = str(row['slug']) if 'slug' in df_fp.columns else (str(row['id_ponto']) if 'id_ponto' in df_fp.columns else '')
                
                self.flood_points.append({
                    'slug': slug,
                    'lat': lat,
                    'lon': lon,
                    'altitude': alt,
                    'distance_to_channel': dist_chan
                })
            logger.info(f"Loaded {len(self.flood_points)} flood points from database.")
        except Exception as e:
            logger.warning(f"Could not load flood points metadata: {e}")
            self.flood_points = []

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_proximity_features(lat, lon, nearby_sensors=None):
    cache = SpatialMetadataCache()
    
    # 1. Proximity to sensors
    dist_rain = 99.0
    dist_river = 99.0
    
    # Try using nearby_sensors from API payload first
    if nearby_sensors:
        for s in nearby_sensors:
            s_lat = s.get('latitude') or s.get('lat')
            s_lon = s.get('longitude') or s.get('lon')
            s_type = s.get('type')
            s_dist = s.get('distanceKm') or s.get('distance_km')
            
            if s_lat is not None and s_lon is not None:
                dist = s_dist if s_dist is not None else haversine_distance(lat, lon, s_lat, s_lon)
                if s_type == 'PRECIPITATION':
                    dist_rain = min(dist_rain, dist)
                elif s_type == 'RIVER_LEVEL':
                    dist_river = min(dist_river, dist)
    
    # Fallback to database cached sensors if none found or nearby_sensors not provided
    if dist_rain == 99.0 and cache.sensors:
        rain_sensors = [s for s in cache.sensors if s.get('type') == 'PRECIPITATION']
        if rain_sensors:
            dists = [haversine_distance(lat, lon, s['latitude'], s['longitude']) for s in rain_sensors]
            dist_rain = min(dists)
            
    if dist_river == 99.0 and cache.sensors:
        river_sensors = [s for s in cache.sensors if s.get('type') == 'RIVER_LEVEL']
        if river_sensors:
            dists = [haversine_distance(lat, lon, s['latitude'], s['longitude']) for s in river_sensors]
            dist_river = min(dists)
            
    # Keep the initial value of 99.0 km to represent "no sensor nearby" rather than resetting to 0.0
    # (which would indicate that a sensor is directly overlapping the coordinates).
    pass
    
    # 2. Proximity to flood points, altitude and distance to channel
    dist_fp = 0.0
    altitude = 10.0      # default low elevation in Recife
    dist_channel = 100.0 # default distance to channel in meters
    
    if cache.flood_points:
        dists = [haversine_distance(lat, lon, fp['lat'], fp['lon']) for fp in cache.flood_points]
        min_idx = np.argmin(dists)
        dist_fp = dists[min_idx]
        altitude = cache.flood_points[min_idx]['altitude']
        dist_channel = cache.flood_points[min_idx]['distance_to_channel']
        
    return {
        'altitude': altitude,
        'distance_to_channel': dist_channel,
        'distance_to_flood_point': dist_fp,
        'distance_to_rain_sensor': dist_rain,
        'distance_to_river_sensor': dist_river
    }

def extract_cyclic_features_from_time(dt_series):
    """Computes sin/cos values for hour and month."""
    hours = dt_series.dt.hour
    months = dt_series.dt.month
    
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    
    return hour_sin, hour_cos, month_sin, month_cos

def create_lags(df: pd.DataFrame, col: str, lags: list):
    """Creates lag features for a specific column (legacy compatibility)."""
    if col not in df.columns:
        return df
    for lag in lags:
        df[f'{col}_lag_{lag}h'] = df[col].shift(periods=int(lag * 4))
    return df

def extract_cyclic_features(df: pd.DataFrame):
    """Extracts cyclic components for hour and month (legacy compatibility)."""
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df.drop(columns=['hour', 'month'])

def add_tide_features(df: pd.DataFrame, tide_col='tide_level'):
    """Adds tide-related features like boolean high tide and normalized tide (legacy compatibility)."""
    if tide_col in df.columns:
        df['is_high_tide'] = (df[tide_col] > TIDE_THRESHOLD).astype(int)
    else:
        df['is_high_tide'] = 0
    return df

def generate_features(df: pd.DataFrame, rainfall_col='rainfall'):
    """Legacy feature generation pipeline (keeps backward compatibility)."""
    df = create_lags(df, rainfall_col, [1, 3, 6, 12, 24])
    df = extract_cyclic_features(df)
    df = add_tide_features(df)
    return df.dropna()

def prepare_scenario_features(df: pd.DataFrame, is_training=True):
    """
    Enhanced feature engineering pipeline for scenario labels.
    Ensures spatial, proximity, temporal/cyclic, and environmental features 
    are properly generated and aligned.
    """
    df = df.copy()
    
    # 1. Clean column names / Ensure mapping matches our feature columns
    mapping = {
        'lat': 'latitude',
        'lon': 'longitude',
        'precipitation': 'current_rainfall',
    }
    for old_col, new_col in mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
            
    # 2. Fill missing columns with defaults
    required_cols = [
        'latitude', 'longitude', 'current_rainfall', 
        'rainfall_3h_accumulated', 'rainfall_6h_accumulated', 
        'rainfall_12h_accumulated', 'rainfall_24h_accumulated', 
        'tide_level', 'river_level',
        'wind_speed', 'temperature', 'apparent_temperature', 
        'humidity', 'pressure', 'wave_height', 
        'wave_period', 'wave_direction', 'solar_radiation'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # 3. Add tide threshold feature
    df['is_high_tide'] = (df['tide_level'] > TIDE_THRESHOLD).astype(int)
    
    # 4. Generate cyclic temporal features from timestamp
    if 'timestamp' in df.columns:
        dt_series = pd.to_datetime(df['timestamp'])
    else:
        dt_series = pd.to_datetime(df.index)
        
    df['hour_sin'], df['hour_cos'], df['month_sin'], df['month_cos'] = extract_cyclic_features_from_time(dt_series)
    
    # 5. Add spatial/proximity features (altitude, distance to channel, distance to rain/river sensors)
    cache = SpatialMetadataCache()
    # Lazy load spatial cache on the first run of training or inference
    if is_training and not cache.sensors and not cache.flood_points:
        cache.load_from_db()
        
    altitudes = []
    dist_channels = []
    dist_fps = []
    dist_rains = []
    dist_rivers = []
    
    for _, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        
        # Pass nearby sensors list if it exists in the row (e.g. from prediction input)
        nearby = row.get('nearby_sensors') if 'nearby_sensors' in df.columns else None
        
        feats = compute_proximity_features(lat, lon, nearby)
        altitudes.append(feats['altitude'])
        dist_channels.append(feats['distance_to_channel'])
        dist_fps.append(feats['distance_to_flood_point'])
        dist_rains.append(feats['distance_to_rain_sensor'])
        dist_rivers.append(feats['distance_to_river_sensor'])
        
    df['altitude'] = altitudes
    df['distance_to_channel'] = dist_channels
    df['distance_to_flood_point'] = dist_fps
    df['distance_to_rain_sensor'] = dist_rains
    df['distance_to_river_sensor'] = dist_rivers
    
    # Return DataFrame containing exactly FEATURE_COLUMNS in correct order
    return df[FEATURE_COLUMNS]
