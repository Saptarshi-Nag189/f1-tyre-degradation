"""GPU-accelerated training pipeline for F1 tyre model.

This is the main training script per Implementation Plan.
Implements all Training Pipeline tasks (8.1-8.5).

Exit codes:
- 0: Success
- 1: Empty features - feature engineering failed
- 2: No sequences created
- 3: Model training failed
"""
import sys
import os
from pathlib import Path

# Anchor all paths to project root regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src to path for imports
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import pickle
import numpy as np
import pandas as pd
import torch
import logging
from sklearn.preprocessing import StandardScaler

from utils.setup import setup_environment, load_config
from utils.gpu_processing import GPUDataProcessor
from modeling.gpu_models import GPUTyreModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# RUNTIME INSTRUMENTATION (Logging Only - No Behavior Changes)
# =============================================================================

def log_startup_device_report():
    """Log comprehensive GPU/CUDA availability report at startup.
    
    Instrumentation requirement 1: Startup Device Report.
    Logs unconditionally at beginning of execution.
    """
    logger.info("=" * 60)
    logger.info("DEVICE AVAILABILITY REPORT")
    logger.info("=" * 60)
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"torch.cuda.is_available(): {cuda_available}")
    
    # CUDA version
    cuda_version = torch.version.cuda if torch.version.cuda else "N/A"
    logger.info(f"torch.version.cuda: {cuda_version}")
    
    # Device count and names
    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA device count: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            logger.info(f"  Device {i}: {device_name}")
    else:
        logger.info("CUDA device count: 0")
        logger.info("  No CUDA devices detected")
    
    logger.info("=" * 60)


def log_training_device_confirmation(model_device: str, xgb_available: bool):
    """Log explicit device confirmation before training begins.
    
    Instrumentation requirement 2: Training Device Confirmation.
    Clearly states USING GPU or USING CPU and WHY.
    
    Args:
        model_device: The device string ('cuda' or 'cpu') for PyTorch.
        xgb_available: Whether XGBoost model is available.
    """
    logger.info("=" * 60)
    logger.info("TRAINING DEVICE CONFIRMATION")
    logger.info("=" * 60)
    
    # PyTorch LSTM device
    if model_device == 'cuda':
        logger.info("PyTorch (LSTM): USING GPU")
        logger.info("  Reason: CUDA available and selected")
    else:
        logger.info("PyTorch (LSTM): USING CPU")
        if not torch.cuda.is_available():
            logger.info("  Reason: CUDA not available on this system")
        else:
            logger.info("  Reason: CPU mode selected or GPU fallback triggered")
    
    # XGBoost device
    if xgb_available:
        if torch.cuda.is_available():
            logger.info("XGBoost: USING GPU (tree_method='gpu_hist')")
            logger.info("  Reason: CUDA available, GPU acceleration enabled")
        else:
            logger.info("XGBoost: USING CPU")
            logger.info("  Reason: CUDA not available, CPU fallback active")
    else:
        logger.info("XGBoost: NOT AVAILABLE")
        logger.info("  Reason: XGBoost library not installed or import failed")
    
    logger.info("=" * 60)


def log_gpu_memory_sanity_check():
    """Log GPU memory usage if CUDA is available.
    
    Instrumentation requirement 3: GPU Memory Sanity Check.
    Informational only - does NOT fail based on this.
    """
    if not torch.cuda.is_available():
        return
    
    logger.info("GPU Memory Status (informational):")
    allocated = torch.cuda.memory_allocated() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info(f"  memory_allocated: {allocated:.3f} GB")
    logger.info(f"  max_memory_allocated: {max_allocated:.3f} GB")


def check_gpu_config_mismatch(config: dict):
    """Check for GPU config vs availability mismatch.
    
    Instrumentation requirement 4: Early Misconfiguration Guard.
    Logs WARNING if GPU requested but unavailable. Does NOT exit.
    
    Args:
        config: The main configuration dictionary.
    """
    gpu_requested = config.get('feature_engineering', {}).get('enable_gpu_processing', False)
    cuda_available = torch.cuda.is_available()
    
    if gpu_requested and not cuda_available:
        logger.warning("=" * 60)
        logger.warning("CONFIGURATION MISMATCH DETECTED")
        logger.warning("=" * 60)
        logger.warning("Config requests GPU (enable_gpu_processing=true)")
        logger.warning("But torch.cuda.is_available() == False")
        logger.warning("CPU FALLBACK WILL BE USED - Execution continues")
        logger.warning("=" * 60)


def load_data():
    """Load cleaned F1 data.
    
    Task 8.1: Load data from hardcoded path.
    
    Returns:
        dict: Nested dictionary with year keys.
        
    Raises:
        FileNotFoundError: If data file doesn't exist.
        SystemExit: If data file is missing.
    """
    logger.info("Loading data...")
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'f1_cleaned_data.pkl'
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run 'python scripts/download_data.py' first")
        sys.exit(1)
    
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def engineer_features(data, config):
    """Feature engineering with GPU acceleration.
    
    Task 8.2: Implements GPU-accelerated feature engineering.
    
    Features computed per Feature Contract Section 1:
    - acceleration: Speed.diff()
    - speed_ma_5: 5-sample rolling mean
    - lateral_acceleration: curvature * speed^2
    
    Args:
        data: Nested dictionary of cleaned F1 data.
        config: Configuration dictionary.
        
    Returns:
        pd.DataFrame: Concatenated DataFrame with all features.
    """
    logger.info("Engineering features...")
    
    processor = GPUDataProcessor(
        batch_size=config['feature_engineering']['gpu_batch_size'],
        use_gpu=config['feature_engineering']['enable_gpu_processing']
    )
    
    all_features = []
    
    for year, year_data in data.items():
        logger.info(f"Processing {year}...")
        
        for event_name, event_data in year_data.items():
            for session_type, session_data in event_data.items():
                if not isinstance(session_data, dict) or 'telemetry' not in session_data:
                    continue
                
                # Features to calculate per Feature Contract (DEFINED only)
                features_to_calc = [
                    'speed_acceleration',
                    'speed_moving_avg',
                    'lateral_acceleration'
                ]
                
                processed = processor.process_telemetry_batch(
                    session_data['telemetry'],
                    features_to_calc
                )
                
                if not processed.empty:
                    # Tag with event/session identity (needed for per-session target)
                    processed['Event'] = event_name
                    processed['Session'] = session_type
                    
                    # Merge lap-level data: TyreLife, LapTime, Compound, Stint
                    if 'laps' in session_data and session_data['laps'] is not None:
                        laps = session_data['laps']
                        merge_cols = ['Driver', 'LapNumber']
                        data_cols = []
                        for col in ['TyreLife', 'Compound', 'Stint']:
                            if col in laps.columns:
                                merge_cols.append(col)
                                data_cols.append(col)
                        if 'LapTime' in laps.columns:
                            laps = laps.copy()
                            laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
                            merge_cols.append('LapTimeSec')
                            data_cols.append('LapTimeSec')
                        
                        if len(merge_cols) > 2 and 'LapNumber' in processed.columns:
                            lap_info = laps[merge_cols].drop_duplicates()
                            # Drop columns that will come from merge
                            for col in data_cols:
                                if col in processed.columns:
                                    processed = processed.drop(columns=[col])
                            processed = processed.merge(
                                lap_info, on=['Driver', 'LapNumber'], how='left'
                            )
                            if 'TyreLife' in processed.columns:
                                processed['TyreLife'] = processed['TyreLife'].fillna(0).astype(np.float32)
                    
                    all_features.append(processed)
    
    if not all_features:
        return pd.DataFrame()
    
    return pd.concat(all_features, ignore_index=True)


def prepare_sequences(df, sequence_length=5):
    """Prepare lap-level sequential data for LSTM.
    
    Aggregates telemetry per lap into statistical features, then creates
    sequences across consecutive laps. This avoids the target redundancy
    problem where 300-500 telemetry rows per lap all share the same target.
    
    Features per lap (13 total):
        mean/std Speed, mean/std acceleration, mean lateral_accel,
        mean/std Throttle, mean/std Brake, TyreLife, max Speed, max acceleration
    
    Target: lap_time_delta = LapTimeSec - best clean lap per (Driver, Compound, Event, Session)
    
    Args:
        df: Features DataFrame with LapTimeSec, Compound, Stint, Event, Session.
        sequence_length: Number of consecutive laps per sequence (default 5).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (n, seq_len, 13), y (n,).
    """
    logger.info(f"Creating lap-level sequences (length={sequence_length})...")
    
    # ---- Step 1: Validate required columns ----
    if 'LapTimeSec' not in df.columns:
        logger.error("LapTimeSec not found -- cannot compute target!")
        return np.array([]), np.array([])
    
    if 'LapNumber' not in df.columns:
        logger.error("LapNumber not found -- cannot aggregate per lap!")
        return np.array([]), np.array([])
    
    # ---- Step 2: Aggregate telemetry per lap ----
    # Group columns that identify a unique lap
    lap_id_cols = ['Driver', 'LapNumber']
    for col in ['Event', 'Session', 'Compound', 'Stint']:
        if col in df.columns:
            lap_id_cols.append(col)
    
    # Telemetry columns to aggregate
    tel_cols = [c for c in ['Speed', 'acceleration', 'speed_ma_5',
                            'lateral_acceleration', 'Throttle', 'Brake'] if c in df.columns]
    
    # Build aggregation dict
    agg_dict = {}
    for col in tel_cols:
        agg_dict[col] = ['mean', 'std', 'max']
    agg_dict['TyreLife'] = 'first'
    agg_dict['LapTimeSec'] = 'first'
    
    logger.info(f"Aggregating telemetry per lap ({len(df):,} rows)...")
    lap_df = df.groupby(lap_id_cols, observed=True).agg(agg_dict).reset_index()
    
    # Flatten multi-level column names
    flat_cols = []
    for col in lap_df.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'first':
                flat_cols.append(col[0])
            else:
                flat_cols.append(f"{col[0]}_{col[1]}")
        else:
            flat_cols.append(col)
    lap_df.columns = flat_cols
    
    logger.info(f"Lap-level data: {len(lap_df):,} laps, {len(flat_cols)} columns")
    
    # ---- Step 3: Filter bad laps ----
    # Remove laps without valid lap times
    lap_df = lap_df[lap_df['LapTimeSec'].notna() & (lap_df['LapTimeSec'] > 0)].copy()
    
    # Remove outlier laps (> 150% of session median — catches out-laps, in-laps, SC)
    session_group = ['Driver']
    for col in ['Compound', 'Event', 'Session']:
        if col in lap_df.columns:
            session_group.append(col)
    
    median_time = lap_df.groupby(session_group)['LapTimeSec'].transform('median')
    lap_df = lap_df[lap_df['LapTimeSec'] < median_time * 1.5].copy()
    
    logger.info(f"After filtering outlier laps: {len(lap_df):,} laps")
    
    # ---- Step 4: Compute target ----
    # best_time per (Driver, Compound, Event, Session)
    target_group = ['Driver']
    for col in ['Compound', 'Event', 'Session']:
        if col in lap_df.columns:
            target_group.append(col)
    
    best_time = lap_df.groupby(target_group)['LapTimeSec'].transform(
        lambda x: x.quantile(0.05)
    )
    lap_df['lap_time_delta'] = (lap_df['LapTimeSec'] - best_time).clip(lower=0, upper=10)
    
    logger.info(f"Target stats: mean={lap_df['lap_time_delta'].mean():.3f}s, "
                f"std={lap_df['lap_time_delta'].std():.3f}s, "
                f"max={lap_df['lap_time_delta'].max():.3f}s, "
                f"laps={len(lap_df):,}")
    
    # ---- Step 5: Identify feature columns ----
    feature_cols = [c for c in lap_df.columns
                    if c not in lap_id_cols + ['LapTimeSec', 'lap_time_delta']]
    feature_cols = [c for c in feature_cols if lap_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    logger.info(f"Lap features ({len(feature_cols)}): {feature_cols}")
    
    # Fill NaN in features
    lap_df[feature_cols] = lap_df[feature_cols].fillna(0)
    
    # ---- Step 6: Create sequences per stint ----
    X_sequences = []
    y_targets = []
    
    seq_group_cols = ['Driver']
    for col in ['Compound', 'Stint', 'Event', 'Session']:
        if col in lap_df.columns:
            seq_group_cols.append(col)
    
    for group_key, group_data in lap_df.groupby(seq_group_cols):
        group_data = group_data.sort_values('LapNumber')
        
        if len(group_data) < sequence_length + 1:
            continue
        
        features = group_data[feature_cols].values
        targets = group_data['lap_time_delta'].values
        
        for i in range(len(features) - sequence_length):
            X_sequences.append(features[i:i+sequence_length])
            y_targets.append(targets[i+sequence_length])
    
    if not X_sequences:
        logger.error("No sequences created!")
        return np.array([]), np.array([])
    
    logger.info(f"Total sequences: {len(X_sequences):,} "
                f"(features={len(feature_cols)}, seq_len={sequence_length})")
    
    return np.array(X_sequences, dtype=np.float32), np.array(y_targets, dtype=np.float32)


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance.
    
    Task 8.4: Calculate metrics for lap time degradation prediction.
    
    Metrics:
    - RMSE: Root mean squared error (seconds)
    - MAE: Mean absolute error (seconds)
    - R2: Coefficient of determination (target > 0.50)
    - MedAE: Median absolute error — robust to outliers
    - Within_1s: % of predictions within 1 second of actual
    - Within_2s: % of predictions within 2 seconds of actual
    
    Args:
        model: Trained GPUTyreModel.
        X_test: Test features array.
        y_test: Test targets array.
        
    Returns:
        dict: Dictionary with performance metrics.
    """
    logger.info("Evaluating model...")
    
    predictions = model.predict(X_test)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
    from scipy.stats import spearmanr
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    med_ae = median_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    spearman_corr, _ = spearmanr(y_test, predictions)
    
    # Domain-specific: what % of predictions are within 1s / 2s of actual?
    abs_errors = np.abs(y_test - predictions)
    within_1s = np.mean(abs_errors < 1.0) * 100
    within_2s = np.mean(abs_errors < 2.0) * 100
    
    results = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MedAE': float(med_ae),
        'R2': float(r2),
        'Spearman': float(spearman_corr),
        'Within_1s': float(within_1s),
        'Within_2s': float(within_2s)
    }
    
    logger.info("\nModel Performance:")
    for metric, value in results.items():
        if metric.startswith('Within'):
            logger.info(f"  {metric}: {value:.1f}%")
        else:
            logger.info(f"  {metric}: {value:.4f}")
    
    return results


def build_lap_features(data):
    """Build lap-level features directly from the laps DataFrames.
    
    Uses ALL laps (not just telemetry-sampled ones), extracting sector times,
    speed traps, tyre info, and position data.
    
    Args:
        data: Nested dict {year: {event: {session: {laps, telemetry, ...}}}}.
        
    Returns:
        pd.DataFrame: One row per lap with features + metadata.
    """
    logger.info("Building lap features from laps data...")
    
    all_laps = []
    
    for year, year_data in data.items():
        for event_name, event_data in year_data.items():
            for session_type, session_data in event_data.items():
                if not isinstance(session_data, dict) or 'laps' not in session_data:
                    continue
                
                laps = session_data['laps']
                if laps is None or laps.empty:
                    continue
                
                # Build a clean lap-level DataFrame
                lap_rows = pd.DataFrame()
                lap_rows['Driver'] = laps['Driver']
                lap_rows['LapNumber'] = laps['LapNumber']
                lap_rows['Event'] = event_name
                lap_rows['Session'] = session_type
                
                # Target: raw lap time in seconds
                if 'LapTime' in laps.columns:
                    lt = laps['LapTime']
                    if hasattr(lt.iloc[0], 'total_seconds') if len(lt) > 0 else False:
                        lap_rows['LapTimeSec'] = lt.dt.total_seconds()
                    else:
                        lap_rows['LapTimeSec'] = pd.to_numeric(lt, errors='coerce')
                else:
                    continue
                
                # Tyre info
                for col in ['Compound', 'TyreLife', 'Stint']:
                    if col in laps.columns:
                        lap_rows[col] = laps[col].values
                
                # Speed traps (per-lap features from FastF1)
                for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
                    if col in laps.columns:
                        lap_rows[col] = pd.to_numeric(laps[col], errors='coerce')
                
                # Sector times → seconds
                for col in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                    if col in laps.columns:
                        sec = laps[col]
                        if hasattr(sec.iloc[0], 'total_seconds') if len(sec) > 0 else False:
                            lap_rows[col + 'Sec'] = sec.dt.total_seconds()
                        else:
                            lap_rows[col + 'Sec'] = pd.to_numeric(sec, errors='coerce')
                
                # Position
                if 'Position' in laps.columns:
                    lap_rows['Position'] = pd.to_numeric(laps['Position'], errors='coerce')
                
                all_laps.append(lap_rows)
    
    if not all_laps:
        return pd.DataFrame()
    
    df = pd.concat(all_laps, ignore_index=True)
    logger.info(f"Built {len(df):,} lap records from {len(all_laps)} sessions")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def prepare_lap_sequences(df, sequence_length=5):
    """Create sequences from lap-level features for LSTM.
    
    Target: lap_time_delta = LapTimeSec - best per (Driver, Compound, Event, Session)
    
    Args:
        df: Lap-level DataFrame from build_lap_features().
        sequence_length: Number of consecutive laps per sequence.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, list]: X, y, feature_names.
    """
    logger.info(f"Preparing lap sequences (length={sequence_length})...")
    
    if 'LapTimeSec' not in df.columns:
        logger.error("LapTimeSec not found!")
        return np.array([]), np.array([]), []
    
    # ---- Filter bad laps ----
    df = df[df['LapTimeSec'].notna() & (df['LapTimeSec'] > 0)].copy()
    
    # Remove outlier laps (>150% of session median — catches out-laps, in-laps, SC)
    session_group = ['Driver']
    for col in ['Compound', 'Event', 'Session']:
        if col in df.columns:
            session_group.append(col)
    
    median_time = df.groupby(session_group)['LapTimeSec'].transform('median')
    df = df[df['LapTimeSec'] < median_time * 1.5].copy()
    logger.info(f"After filtering outlier laps: {len(df):,}")
    
    # ---- Compute target ----
    target_group = ['Driver']
    for col in ['Compound', 'Event', 'Session']:
        if col in df.columns:
            target_group.append(col)
    
    best_time = df.groupby(target_group)['LapTimeSec'].transform(
        lambda x: x.quantile(0.05)
    )
    df['lap_time_delta'] = (df['LapTimeSec'] - best_time).clip(lower=0, upper=10)
    
    logger.info(f"Target: mean={df['lap_time_delta'].mean():.3f}s, "
                f"std={df['lap_time_delta'].std():.3f}s, "
                f"max={df['lap_time_delta'].max():.3f}s")
    
    # ---- Identify numeric feature columns ----
    exclude = ['LapTimeSec', 'lap_time_delta', 'LapNumber', 'Driver',
               'Event', 'Session', 'Compound', 'Stint', 'Position']
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # Also include TyreLife if numeric
    if 'TyreLife' in feature_cols:
        pass  # already included
    elif 'TyreLife' in df.columns:
        df['TyreLife'] = pd.to_numeric(df['TyreLife'], errors='coerce')
        feature_cols.append('TyreLife')
    
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # ---- Create sequences per stint ----
    X, y_out = [], []
    
    seq_group = ['Driver']
    for col in ['Compound', 'Stint', 'Event', 'Session']:
        if col in df.columns:
            seq_group.append(col)
    
    for _, group in df.groupby(seq_group):
        group = group.sort_values('LapNumber')
        
        if len(group) < sequence_length + 1:
            continue
        
        feats = group[feature_cols].values
        targets = group['lap_time_delta'].values
        
        for i in range(len(feats) - sequence_length):
            X.append(feats[i:i + sequence_length])
            y_out.append(targets[i + sequence_length])
    
    if not X:
        logger.error("No sequences created!")
        return np.array([]), np.array([]), []
    
    logger.info(f"Total sequences: {len(X):,} (features={len(feature_cols)}, "
                f"seq_len={sequence_length})")
    
    return (np.array(X, dtype=np.float32),
            np.array(y_out, dtype=np.float32),
            feature_cols)


def main():
    """Main training pipeline.
    
    Task 8.5: Orchestrates complete training sequence.
    
    Exit Codes:
    - 0: Success
    - 1: Empty features (feature engineering produced no data)
    - 2: No sequences created
    """
    print("GPU-Enhanced F1 Tyre Model Training")
    print("=" * 50)
    
    # =========================================================================
    # INSTRUMENTATION: Startup Device Report (unconditional)
    # =========================================================================
    log_startup_device_report()
    
    # Setup
    setup_environment(project_root=PROJECT_ROOT)
    config = load_config(str(PROJECT_ROOT / 'config' / 'config.yaml'))
    gpu_config = load_config(str(PROJECT_ROOT / 'config' / 'gpu_config.yaml'))
    
    if config is None:
        logger.error("Main config not found!")
        sys.exit(1)
    if gpu_config is None:
        logger.error("GPU config not found!")
        sys.exit(1)
    
    # =========================================================================
    # INSTRUMENTATION: Early Misconfiguration Guard
    # =========================================================================
    check_gpu_config_mismatch(config)
    
    # Load and process data
    data = load_data()
    
    # Build lap-level features directly from laps DataFrames (uses ALL laps)
    lap_df = build_lap_features(data)
    
    if lap_df.empty:
        logger.error("No lap features built!")
        sys.exit(1)
    
    # Create sequences from lap features
    X, y, feature_names = prepare_lap_sequences(lap_df, config['model']['sequence_length'])
    
    # FAILURE SEMANTICS: No sequences created
    if len(X) == 0:
        logger.error("No sequences created from features!")
        sys.exit(2)
    
    logger.info(f"Sequences: {X.shape}, Targets: {y.shape}")
    
    # Temporal split (70/15/15) - preserves time ordering, avoids leakage
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize with StandardScaler per Feature Contract
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Merge configs for model
    merged_config = {**gpu_config, 'model': config['model']}
    
    # Initialize model
    model = GPUTyreModel(merged_config, models_dir=str(PROJECT_ROOT / 'results' / 'models'))
    model.build_models(
        input_dim=X_train.shape[-1],
        sequence_length=X_train.shape[1]
    )
    
    # =========================================================================
    # INSTRUMENTATION: Training Device Confirmation
    # =========================================================================
    log_training_device_confirmation(
        model_device=model.device,
        xgb_available=(model.xgb_model is not None)
    )
    log_gpu_memory_sanity_check()
    
    # Train
    logger.info("\nStarting training...")
    model.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=config['model']['epochs']
    )
    
    # FAILURE SEMANTICS: Check training actually happened
    if not model.history['train_loss']:
        logger.error("Model training produced no history - training skipped!")
        sys.exit(3)
    
    # Evaluate
    results = evaluate_model(model, X_test_scaled, y_test)
    
    # Save models
    models_dir = PROJECT_ROOT / 'results' / 'models'
    reports_dir = PROJECT_ROOT / 'results' / 'reports'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    model.save(str(models_dir / 'gpu_ensemble'))
    
    # Save scaler
    import joblib
    joblib.dump(scaler, str(models_dir / 'scaler.pkl'))
    
    # Save results
    with open(str(reports_dir / 'training_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    logger.info("\n" + "="*50)
    logger.info("Training complete!")
    logger.info(f"Models saved to {models_dir}")
    logger.info("="*50)
    
    # Exit with success
    sys.exit(0)


if __name__ == "__main__":
    main()
