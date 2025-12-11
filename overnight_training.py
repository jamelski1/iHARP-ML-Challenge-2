#!/usr/bin/env python3
"""
Overnight Training Script for iHARP ML Challenge 2
Runs extensive hyperparameter search and model comparison for ~16 hours.

Usage:
    python overnight_training.py

Output:
    - results/ directory with all models, logs, and best submission
    - Checkpoints saved every iteration (crash-safe)
    - Final best model ready for submission
"""

import os
import sys
import json
import pickle
import time
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.io import loadmat

# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: LightGBM not installed")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
BEST_MODEL_FILE = RESULTS_DIR / "best_model.pkl"
LOG_FILE = RESULTS_DIR / "training.log"

# Data settings
HIST_DAYS = 7
FUTURE_DAYS = 14
FEATURES = ["sea_level", "sea_level_3d_mean", "sea_level_7d_mean"]

# Training settings
N_CV_FOLDS = 5
MAX_RUNTIME_HOURS = 15.5  # Stop 30 min before 16 hours for safety

# Train/test station split (matches competition)
TRAIN_STATIONS = [
    'Annapolis', 'Atlantic_City', 'Charleston', 'Washington',
    'Wilmington', 'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook'
]
TEST_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging():
    RESULTS_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# DATA LOADING
# =============================================================================
def matlab2datetime(matlab_datenum):
    """Convert MATLAB datenum to Python datetime."""
    return datetime.fromordinal(int(matlab_datenum)) \
           + timedelta(days=matlab_datenum % 1) \
           - timedelta(days=366)

def load_data():
    """Load and preprocess the .mat dataset."""
    logger.info("Loading dataset...")

    mat_file = BASE_DIR / "NEUSTG_19502020_12stations.mat"
    data = loadmat(str(mat_file))

    lat = data['lattg'].flatten()
    lon = data['lontg'].flatten()
    sea_level = data['sltg']
    station_names = [s[0] for s in data['sname'].flatten()]
    time_raw = data['t'].flatten()
    time_dt = pd.to_datetime([matlab2datetime(t) for t in time_raw])

    logger.info(f"Loaded {len(station_names)} stations, {len(time_dt)} time points")

    # Build hourly DataFrame
    records = []
    for i, stn in enumerate(station_names):
        for j, t in enumerate(time_dt):
            records.append({
                'time': t,
                'station_name': stn,
                'latitude': lat[i],
                'longitude': lon[i],
                'sea_level': sea_level[j, i]
            })

    df_hourly = pd.DataFrame(records)
    logger.info(f"Built hourly DataFrame: {len(df_hourly)} rows")

    return df_hourly, station_names

def compute_daily_features(df_hourly):
    """Convert hourly data to daily with features."""
    logger.info("Computing daily aggregates and features...")

    # Compute flood thresholds per station
    threshold_df = df_hourly.groupby('station_name')['sea_level'].agg(['mean', 'std']).reset_index()
    threshold_df['flood_threshold'] = threshold_df['mean'] + 1.5 * threshold_df['std']

    df_hourly = df_hourly.merge(
        threshold_df[['station_name', 'flood_threshold']],
        on='station_name',
        how='left'
    )

    # Daily aggregation
    df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
        'sea_level': 'mean',
        'latitude': 'first',
        'longitude': 'first',
        'flood_threshold': 'first'
    }).reset_index()

    # Daily max for flood detection
    hourly_max = df_hourly.groupby(
        ['station_name', pd.Grouper(key='time', freq='D')]
    )['sea_level'].max().reset_index()

    df_daily = df_daily.merge(hourly_max, on=['station_name', 'time'], suffixes=('', '_max'))
    df_daily['flood'] = (df_daily['sea_level_max'] > df_daily['flood_threshold']).astype(int)

    # Rolling features
    df_daily = df_daily.sort_values(['station_name', 'time']).reset_index(drop=True)
    df_daily['sea_level_3d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df_daily['sea_level_7d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    # Additional features for better models
    df_daily['sea_level_std_7d'] = df_daily.groupby('station_name')['sea_level'].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )
    df_daily['sea_level_min_7d'] = df_daily.groupby('station_name')['sea_level'].transform(
        lambda x: x.rolling(7, min_periods=1).min()
    )
    df_daily['sea_level_max_7d'] = df_daily.groupby('station_name')['sea_level'].transform(
        lambda x: x.rolling(7, min_periods=1).max()
    )
    df_daily['sea_level_range_7d'] = df_daily['sea_level_max_7d'] - df_daily['sea_level_min_7d']

    # Trend feature
    df_daily['sea_level_diff'] = df_daily.groupby('station_name')['sea_level'].diff()
    df_daily['sea_level_diff'] = df_daily['sea_level_diff'].fillna(0)

    logger.info(f"Daily DataFrame: {len(df_daily)} rows")

    return df_daily, threshold_df

def build_windows(df_daily, stations, feature_cols, use_labels=True):
    """Build training windows from daily data."""
    X, y, meta = [], [], []

    for stn in stations:
        grp = df_daily[df_daily['station_name'] == stn].sort_values('time').reset_index(drop=True)

        for i in range(len(grp) - HIST_DAYS - FUTURE_DAYS + 1):
            hist_block = grp.loc[i:i+HIST_DAYS-1, feature_cols]

            if hist_block.isna().any().any():
                continue

            X.append(hist_block.values.flatten())
            meta.append({
                'station': stn,
                'hist_start': grp.loc[i, 'time'],
                'future_start': grp.loc[i+HIST_DAYS, 'time']
            })

            if use_labels:
                fut = grp.loc[i+HIST_DAYS:i+HIST_DAYS+FUTURE_DAYS-1, 'flood']
                y.append(int(fut.max() > 0))

    return np.array(X), np.array(y) if use_labels else None, pd.DataFrame(meta)

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
def get_model_configs():
    """Define all models and their hyperparameter grids."""
    configs = []

    # XGBoost configurations
    if HAS_XGB:
        xgb_params = {
            'n_estimators': [100, 200, 400, 600, 800],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'min_child_weight': [1, 3, 5, 7],
        }
        configs.append(('XGBoost', XGBClassifier, xgb_params, {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0
        }))

    # LightGBM configurations
    if HAS_LGBM:
        lgbm_params = {
            'n_estimators': [100, 200, 400, 600, 800],
            'max_depth': [3, 5, 7, 10, 15, -1],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'num_leaves': [15, 31, 63, 127],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_lambda': [0.0, 0.1, 1.0],
            'reg_alpha': [0.0, 0.1, 1.0],
        }
        configs.append(('LightGBM', LGBMClassifier, lgbm_params, {
            'objective': 'binary',
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        }))

    # CatBoost configurations
    if HAS_CATBOOST:
        catboost_params = {
            'iterations': [100, 200, 400, 600],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7],
        }
        configs.append(('CatBoost', CatBoostClassifier, catboost_params, {
            'random_state': 42,
            'verbose': False
        }))

    # Random Forest configurations
    rf_params = {
        'n_estimators': [100, 200, 400, 600],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }
    configs.append(('RandomForest', RandomForestClassifier, rf_params, {
        'n_jobs': -1,
        'random_state': 42
    }))

    # Extra Trees
    et_params = {
        'n_estimators': [100, 200, 400],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    configs.append(('ExtraTrees', ExtraTreesClassifier, et_params, {
        'n_jobs': -1,
        'random_state': 42
    }))

    # HistGradientBoosting (fast sklearn boosting)
    hgb_params = {
        'max_iter': [100, 200, 400],
        'max_depth': [5, 10, 15, None],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_leaf': [10, 20, 30],
        'l2_regularization': [0.0, 0.1, 1.0],
    }
    configs.append(('HistGradientBoosting', HistGradientBoostingClassifier, hgb_params, {
        'random_state': 42
    }))

    # Gradient Boosting
    gb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5],
    }
    configs.append(('GradientBoosting', GradientBoostingClassifier, gb_params, {
        'random_state': 42
    }))

    # MLP Neural Network
    mlp_params = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'batch_size': [32, 64, 128],
    }
    configs.append(('MLP', MLPClassifier, mlp_params, {
        'max_iter': 500,
        'early_stopping': True,
        'random_state': 42
    }))

    # Logistic Regression
    lr_params = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
    }
    configs.append(('LogisticRegression', LogisticRegression, lr_params, {
        'max_iter': 1000,
        'random_state': 42
    }))

    return configs

def sample_hyperparams(param_grid, n_samples=None):
    """Generate random hyperparameter combinations."""
    keys = list(param_grid.keys())
    all_values = [param_grid[k] for k in keys]

    all_combos = list(product(*all_values))

    if n_samples and n_samples < len(all_combos):
        np.random.shuffle(all_combos)
        all_combos = all_combos[:n_samples]

    for combo in all_combos:
        yield dict(zip(keys, combo))

# =============================================================================
# TRAINING LOOP
# =============================================================================
def evaluate_model(model, X, y, n_folds=5):
    """Evaluate model with cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    auc_scores = []
    acc_scores = []
    f1_scores = []
    mcc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = model.predict(X_val)

        y_pred = (y_prob > 0.5).astype(int)

        auc_scores.append(roc_auc_score(y_val, y_prob))
        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        mcc_scores.append(matthews_corrcoef(y_val, y_pred))

    return {
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'acc_mean': np.mean(acc_scores),
        'f1_mean': np.mean(f1_scores),
        'mcc_mean': np.mean(mcc_scores)
    }

def save_checkpoint(state):
    """Save current training state."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def load_checkpoint():
    """Load previous training state if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def run_training():
    """Main training loop."""
    start_time = time.time()
    max_runtime_sec = MAX_RUNTIME_HOURS * 3600

    logger.info("=" * 60)
    logger.info("OVERNIGHT TRAINING STARTED")
    logger.info(f"Max runtime: {MAX_RUNTIME_HOURS} hours")
    logger.info("=" * 60)

    # Load data
    df_hourly, station_names = load_data()
    df_daily, threshold_df = compute_daily_features(df_hourly)

    # Feature sets to try
    feature_sets = {
        'basic': ['sea_level', 'sea_level_3d_mean', 'sea_level_7d_mean'],
        'extended': ['sea_level', 'sea_level_3d_mean', 'sea_level_7d_mean',
                     'sea_level_std_7d', 'sea_level_range_7d', 'sea_level_diff'],
    }

    # Build training data
    logger.info("Building training windows...")
    datasets = {}
    for name, features in feature_sets.items():
        X, y, meta = build_windows(df_daily, TRAIN_STATIONS, features)
        datasets[name] = {'X': X, 'y': y, 'features': features}
        logger.info(f"  {name}: X={X.shape}, y={y.shape}, pos_rate={y.mean():.3f}")

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint.get('completed_trials', 0)} trials completed")
        results = checkpoint.get('results', [])
        completed_keys = set(checkpoint.get('completed_keys', []))
        best_score = checkpoint.get('best_score', 0)
        best_config = checkpoint.get('best_config', None)
    else:
        results = []
        completed_keys = set()
        best_score = 0
        best_config = None

    # Get model configurations
    model_configs = get_model_configs()

    trial_count = len(completed_keys)
    total_estimated = sum(100 for _ in model_configs)  # rough estimate

    logger.info(f"Starting hyperparameter search across {len(model_configs)} model types...")

    # Main training loop
    for model_name, model_class, param_grid, fixed_params in model_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"{'='*60}")

        # Sample hyperparameters (limit per model to ensure coverage)
        n_samples_per_model = 50 if model_name in ['XGBoost', 'LightGBM', 'CatBoost'] else 30

        for params in sample_hyperparams(param_grid, n_samples=n_samples_per_model):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_runtime_sec:
                logger.info(f"Time limit reached ({elapsed/3600:.2f} hours). Stopping.")
                break

            # Try each feature set
            for feat_name, data in datasets.items():
                trial_key = f"{model_name}|{feat_name}|{json.dumps(params, sort_keys=True)}"

                if trial_key in completed_keys:
                    continue

                trial_count += 1

                try:
                    # Handle class imbalance
                    pos_count = data['y'].sum()
                    neg_count = len(data['y']) - pos_count
                    scale_weight = neg_count / max(pos_count, 1)

                    # Build model with params
                    all_params = {**fixed_params, **params}

                    # Add scale_pos_weight for boosting models
                    if model_name in ['XGBoost', 'LightGBM']:
                        all_params['scale_pos_weight'] = scale_weight
                    elif model_name == 'CatBoost':
                        all_params['scale_pos_weight'] = scale_weight
                    elif model_name in ['RandomForest', 'ExtraTrees', 'GradientBoosting']:
                        all_params['class_weight'] = 'balanced'

                    model = model_class(**all_params)

                    # Evaluate
                    scores = evaluate_model(model, data['X'], data['y'], n_folds=N_CV_FOLDS)

                    result = {
                        'trial': trial_count,
                        'model': model_name,
                        'features': feat_name,
                        'params': params,
                        **scores,
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                    completed_keys.add(trial_key)

                    # Log progress
                    logger.info(
                        f"Trial {trial_count} | {model_name} | {feat_name} | "
                        f"AUC={scores['auc_mean']:.4f}±{scores['auc_std']:.4f} | "
                        f"F1={scores['f1_mean']:.4f}"
                    )

                    # Update best
                    if scores['auc_mean'] > best_score:
                        best_score = scores['auc_mean']
                        best_config = {
                            'model_name': model_name,
                            'model_class': model_class.__name__,
                            'features': feat_name,
                            'feature_cols': data['features'],
                            'params': all_params,
                            'scores': scores
                        }
                        logger.info(f"  *** NEW BEST: AUC={best_score:.4f} ***")

                        # Train and save best model on full data
                        best_model = model_class(**all_params)
                        best_model.fit(data['X'], data['y'])

                        with open(BEST_MODEL_FILE, 'wb') as f:
                            pickle.dump({
                                'model': best_model,
                                'config': best_config,
                                'threshold_df': threshold_df
                            }, f)

                    # Save checkpoint every 10 trials
                    if trial_count % 10 == 0:
                        save_checkpoint({
                            'completed_trials': trial_count,
                            'completed_keys': list(completed_keys),
                            'results': results,
                            'best_score': best_score,
                            'best_config': best_config
                        })

                except Exception as e:
                    logger.error(f"Trial {trial_count} failed: {e}")
                    completed_keys.add(trial_key)
                    continue

        # Check time limit after each model type
        elapsed = time.time() - start_time
        if elapsed > max_runtime_sec:
            break

    # Final summary
    elapsed_hours = (time.time() - start_time) / 3600

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total runtime: {elapsed_hours:.2f} hours")
    logger.info(f"Total trials: {trial_count}")
    logger.info(f"Best AUC: {best_score:.4f}")

    if best_config:
        logger.info(f"Best model: {best_config['model_name']}")
        logger.info(f"Best features: {best_config['features']}")
        logger.info(f"Best params: {json.dumps(best_config['params'], indent=2, default=str)}")

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "all_results.csv", index=False)

    # Save final checkpoint
    save_checkpoint({
        'completed_trials': trial_count,
        'completed_keys': list(completed_keys),
        'results': results,
        'best_score': best_score,
        'best_config': best_config,
        'finished': True,
        'runtime_hours': elapsed_hours
    })

    # Generate submission model.py
    generate_submission_model(best_config)

    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info(f"Best model saved to: {BEST_MODEL_FILE}")
    logger.info(f"Submission model: {RESULTS_DIR / 'model.py'}")

    return results_df, best_config

def generate_submission_model(best_config):
    """Generate a model.py file for submission with the best config."""
    if not best_config:
        return

    features_str = str(best_config['feature_cols'])

    # Filter out non-serializable params
    clean_params = {}
    for k, v in best_config['params'].items():
        if isinstance(v, (int, float, str, bool, type(None), list, tuple)):
            clean_params[k] = v

    params_str = json.dumps(clean_params, indent=8)

    model_template = f'''#!/usr/bin/env python3
"""
Auto-generated submission model from overnight training.
Best config: {best_config['model_name']} with AUC={best_config['scores']['auc_mean']:.4f}
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

try:
    from xgboost import XGBClassifier
except:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

try:
    from lightgbm import LGBMClassifier
except:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except:
    CatBoostClassifier = None

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

HIST_DAYS = 7
FUTURE_DAYS = 14
FEATURES = {features_str}

MODEL_CLASS = "{best_config['model_class']}"
MODEL_PARAMS = {params_str}

def daily_aggregate(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.floor("D")
    daily = (df.groupby(["station_name", "date"])
               .agg(sea_level=("sea_level", "mean"),
                    sea_level_max=("sea_level", "max"),
                    latitude=("latitude", "first"),
                    longitude=("longitude", "first"))
               .reset_index())
    daily = daily.sort_values(["station_name", "date"]).reset_index(drop=True)
    daily["sea_level_3d_mean"] = daily.groupby("station_name")["sea_level"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    daily["sea_level_7d_mean"] = daily.groupby("station_name")["sea_level"].transform(
        lambda x: x.rolling(7, min_periods=1).mean())
    daily["sea_level_std_7d"] = daily.groupby("station_name")["sea_level"].transform(
        lambda x: x.rolling(7, min_periods=1).std())
    daily["sea_level_min_7d"] = daily.groupby("station_name")["sea_level"].transform(
        lambda x: x.rolling(7, min_periods=1).min())
    daily["sea_level_max_7d"] = daily.groupby("station_name")["sea_level"].transform(
        lambda x: x.rolling(7, min_periods=1).max())
    daily["sea_level_range_7d"] = daily["sea_level_max_7d"] - daily["sea_level_min_7d"]
    daily["sea_level_diff"] = daily.groupby("station_name")["sea_level"].diff().fillna(0)
    return daily

def build_windows(daily, stations, use_labels=False, thresholds=None):
    X, y, meta = [], [], []
    if use_labels:
        if thresholds is None:
            thr = (daily.groupby("station_name")["sea_level"]
                        .agg(["mean", "std"])
                        .assign(flood_threshold=lambda x: x["mean"] + 1.5*x["std"])
                        [["flood_threshold"]].reset_index())
        else:
            thr = thresholds
        daily = daily.merge(thr, on="station_name", how="left")
        daily["flood"] = (daily["sea_level_max"] > daily["flood_threshold"]).astype(int)

    for stn, grp in daily[daily["station_name"].isin(stations)].groupby("station_name"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i in range(len(grp) - HIST_DAYS - FUTURE_DAYS + 1):
            hist_block = grp.loc[i:i+HIST_DAYS-1, FEATURES]
            if hist_block.isna().any().any():
                continue
            X.append(hist_block.values.flatten())
            meta.append({{"station": stn,
                         "hist_start": grp.loc[i, "date"],
                         "future_start": grp.loc[i+HIST_DAYS, "date"]}})
            if use_labels:
                fut = grp.loc[i+HIST_DAYS:i+HIST_DAYS+FUTURE_DAYS-1, "flood"]
                y.append(int(fut.max() > 0))
    return np.array(X), (np.array(y) if use_labels else None), pd.DataFrame(meta)

def get_model():
    model_map = {{
        "XGBClassifier": XGBClassifier,
        "LGBMClassifier": LGBMClassifier,
        "CatBoostClassifier": CatBoostClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "MLPClassifier": MLPClassifier,
        "LogisticRegression": LogisticRegression,
    }}
    cls = model_map.get(MODEL_CLASS)
    if cls is None:
        raise ValueError(f"Unknown model class: {{MODEL_CLASS}}")
    return cls(**MODEL_PARAMS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", required=True)
    ap.add_argument("--test_hourly", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--predictions_out", required=True)
    args = ap.parse_args()

    train = pd.read_csv(args.train_hourly)
    test = pd.read_csv(args.test_hourly)
    index = pd.read_csv(args.test_index)

    daily_tr = daily_aggregate(train)
    daily_te = daily_aggregate(test)

    thr = (train.groupby("station_name")["sea_level"]
                .agg(["mean", "std"])
                .assign(flood_threshold=lambda x: x["mean"] + 1.5*x["std"])
                [["flood_threshold"]]
                .reset_index())

    stn_tr = daily_tr["station_name"].unique().tolist()
    X_tr, y_tr, _ = build_windows(daily_tr, stn_tr, use_labels=True, thresholds=thr)

    clf = get_model()
    clf.fit(X_tr, y_tr)

    X_te, _, meta_te = build_windows(daily_te, daily_te["station_name"].unique().tolist(), use_labels=False)
    meta_te["key"] = meta_te["station"].astype(str) + "|" + meta_te["hist_start"].astype(str) + "|" + meta_te["future_start"].astype(str)

    index["hist_start"] = pd.to_datetime(index["hist_start"])
    index["future_start"] = pd.to_datetime(index["future_start"])
    index["key"] = index["station_name"].astype(str) + "|" + index["hist_start"].astype(str) + "|" + index["future_start"].astype(str)

    if len(X_te) == 0:
        raise RuntimeError("No test windows built")

    probs = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_te)
    pred_df = pd.DataFrame({{"key": meta_te["key"], "y_prob": probs}})

    out = index.merge(pred_df, on="key", how="left")[["id", "y_prob"]]
    out["y_prob"] = out["y_prob"].fillna(0.5)
    out.to_csv(args.predictions_out, index=False)
    print(f"Wrote {{args.predictions_out}}")

if __name__ == "__main__":
    main()
'''

    with open(RESULTS_DIR / "model.py", 'w') as f:
        f.write(model_template)

    logger.info(f"Generated submission model: {RESULTS_DIR / 'model.py'}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    try:
        results_df, best_config = run_training()

        # Print top 10 results
        if len(results_df) > 0:
            print("\n" + "=" * 60)
            print("TOP 10 CONFIGURATIONS")
            print("=" * 60)
            top10 = results_df.nlargest(10, 'auc_mean')[['model', 'features', 'auc_mean', 'f1_mean', 'mcc_mean']]
            print(top10.to_string(index=False))

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
