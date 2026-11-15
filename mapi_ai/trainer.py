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
from mapi_ai.config import MODEL_PATH_LSTM

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

# ---------------------------------------------------------------------------------------------------------------
# Etapa 2 do IN1165: treino do XGBoost sem vazamento.
# ---------------------------------------------------------------------------------------------------------------

def temporal_split(df):
    """
    Split temporal em treino 60% / validação 20% / teste 20%, SEM embaralhar.

    Por quê: em série temporal, embaralhar deixa o modelo ver o futuro (horas vizinhas de um mesmo evento caem no
    treino e no teste), e a métrica fica otimista. Aqui o passado treina, o futuro próximo escolhe hiperparâmetros e
    limiares (validação) e o futuro mais distante é usado UMA vez, no fim (teste).
    """
    from mapi_ai import in1165_config as C
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    a, b = int(n * C.SPLIT[0]), int(n * (C.SPLIT[0] + C.SPLIT[1]))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def feature_matrix(df, rain_source="cemaden", drop_location=False):
    """
    X com as features na ordem de C.FEATURES. NaN continua NaN: o XGBoost aprende para que lado mandar o ausente
    (na árvore, cada divisão tem um "caminho padrão" para o NaN). Trocar NaN por 0.0 fabricaria "não choveu".

    rain_source='reanalysis' troca as 5 features de chuva pela reanálise na MESMA hora e ponto (colunas *_rea):
    é a ablação que mede quanto vale a qualidade da forçante.
    """
    from mapi_ai import in1165_config as C
    X = df[C.FEATURES].copy()
    if rain_source == "reanalysis":
        for f in C.RAIN_FEATURES:
            X[f] = df[f + "_rea"].to_numpy()
    if drop_location:
        X = X.drop(columns=["latitude", "longitude"])
    return X.astype(float), df[C.LABEL].astype(int).to_numpy()


def scale_pos_weight(y_train):
    """#negativos / #positivos, SÓ no treino: compensa a classe rara sem olhar validação ou teste."""
    pos = int(y_train.sum())
    return (len(y_train) - pos) / max(pos, 1)


# Grade pequena de propósito: com ~3 000 linhas de treino, uma busca grande sobreajusta a própria validação.
PARAM_GRID = [
    {"max_depth": d, "n_estimators": n, "learning_rate": lr}
    for d in (2, 3, 4) for n in (150, 400) for lr in (0.03, 0.1)
]


def fit_xgb(X_tr, y_tr, X_va, y_va):
    """Escolhe hiperparâmetros pela PR-AUC na validação (a métrica certa para classe rara) e devolve o melhor."""
    from sklearn.metrics import average_precision_score
    from mapi_ai import in1165_config as C
    spw = scale_pos_weight(y_tr)
    best = None
    for params in PARAM_GRID:
        clf = get_xgboost_classifier(scale_pos_weight=spw, random_state=C.SEED, **params)
        clf.fit(X_tr, y_tr)
        ap = average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])
        if best is None or ap > best[0]:
            best = (ap, params, clf)
    ap, params, clf = best
    return clf, {**params, "scale_pos_weight": spw, "val_pr_auc": ap}


VARIANTS = {
    # nome: (fonte da chuva, sem lat/lon?)
    "xgb_v2": ("cemaden", False),
    "xgb_v2_sem_local": ("cemaden", True),       # ablação 1: o modelo decora o local?
    "xgb_v2_reanalise": ("reanalysis", False),   # ablação 2: chuva medida × reanálise, resto igual
}


def _versions():
    import sklearn, xgboost, numpy, pandas, platform, subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(__file__)).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return {"python": platform.python_version(), "xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
            "numpy": numpy.__version__, "pandas": pandas.__version__, "code_commit": commit}


def train_xgboost(df):
    """Treina o modelo principal e as duas ablações; salva os artefatos versionados e o metadata.json."""
    import hashlib, json
    from mapi_ai import in1165_config as C
    tr, va, te = temporal_split(df)
    C.MODELS_DIR.mkdir(exist_ok=True)
    meta = {
        "model_version": "xgb_v2",
        "artifact": "models/xgb_v2.joblib",
        "features": C.FEATURES,
        "feature_units": C.UNITS,
        "label": C.LABEL,
        "training_data": "SYNTHETIC",
        "training_data_note": C.SYNTHETIC_TAG + " — reports/synthetic_ground_truth.md",
        "rain_source": "MEASURED",
        "rain_source_detail": "Cemaden PED, point_rain_accum_v chuva-ponto-v4 (projeto-mapi-api ADRs 0021-0023)",
        "dataset_sha256": hashlib.sha256(C.DATASET_CSV.read_bytes()).hexdigest() if C.DATASET_CSV.exists() else None,
        "seed": C.SEED,
        "split": {name: {"rows": int(len(p)), "positives": int(p[C.LABEL].sum()),
                         "from": str(p.timestamp.min()), "to": str(p.timestamp.max())}
                  for name, p in (("train", tr), ("validation", va), ("test", te))},
        "versions": _versions(),
        "variants": {},
        "thresholds": None,        # escolhidos na validação pela Etapa 4 (mapi_ai/evaluate.py)
        "calibrator": None,        # idem, se a calibração melhorar o Brier na validação
        "model_validation_status": "NOT_VALIDATED",
        "attribution": C.ATTRIBUTION,
    }
    for name, (rain, no_loc) in VARIANTS.items():
        X_tr, y_tr = feature_matrix(tr, rain, no_loc)
        X_va, y_va = feature_matrix(va, rain, no_loc)
        clf, info = fit_xgb(X_tr, y_tr, X_va, y_va)
        path = C.MODELS_DIR / f"{name}.joblib"
        joblib.dump(clf, path)
        meta["variants"][name] = {"artifact": f"models/{name}.joblib", "rain_source": rain,
                                  "features": list(X_tr.columns), **info}
        print(f"{name}: {info}")
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Artefatos em {C.MODELS_DIR}; metadata.json gravado.")
    return meta


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

def train_pipeline(csv_path=None, model_type="all", synthetic=False):
    """Fachada usada pelo main.py e pelo notebook."""
    df = load_dataset(csv_path, synthetic=synthetic)
    if model_type in ["all", "xgb"]:
        train_xgboost(df)
    if model_type == "lstm":  # LSTM fora do escopo do IN1165; caminho legado mantido
        train_lstm(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPI AI Model Trainer")
    parser.add_argument("--csv", type=str, help="CSV com rótulos")
    parser.add_argument("--synthetic", action="store_true",
                        help="usa o dataset do Plano B (rótulo sintético, experimento controlado)")
    parser.add_argument("--model", type=str, choices=["all", "xgb", "lstm"], default="xgb")
    args = parser.parse_args()
    train_pipeline(args.csv, args.model, synthetic=args.synthetic)
