#!/usr/bin/env python3
"""
Flood Prediction Training Pipeline - Refactored
================================================

This refactored pipeline addresses the following issues from the original:
1. ROC-AUC ~0.56 (barely better than random)
2. F1 stuck at 0.935 (model predicting all positives due to imbalance)

Key Changes:
- Proper class imbalance handling (weighted loss, focal loss)
- Comprehensive evaluation (PR-AUC, optimal thresholds, curves)
- Time-based splits (no temporal leakage)
- Enhanced features (lag, rolling, rate-of-rise)
- Strong baselines (LightGBM/XGBoost)
- Early stopping on PR-AUC

Usage:
    python flood_prediction_refactored.py --config config.yaml
    python flood_prediction_refactored.py --loss focal --split time
"""

import os
import sys
import json
import pickle
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

# ML imports
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    matthews_corrcoef, classification_report
)

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import get_linear_schedule_with_warmup
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Gradient Boosting baselines
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: LightGBM not installed")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """Configuration for the training pipeline."""

    # Data paths
    data_path: str = "NEUSTG_19502020_12stations.mat"
    output_dir: str = "results_refactored"

    # Column names (adjust these to match your data)
    timestamp_col: str = "time"
    station_col: str = "station_name"
    target_col: str = "flood"

    # Data settings
    hist_days: int = 7
    future_days: int = 14

    # Feature engineering
    rolling_windows: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])  # hours
    lag_windows: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])  # hours

    # Split settings
    split_method: str = "time"  # "time", "random", "grouped"
    val_ratio: float = 0.20
    time_split_date: Optional[str] = "2015-01-01"  # Train before, validate after

    # Model settings
    model_type: str = "transformer"  # "transformer", "lstm", "lgbm", "xgboost"
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1

    # Loss function
    loss_type: str = "weighted_bce"  # "bce", "weighted_bce", "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Training settings
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    patience: int = 10

    # Early stopping metric
    early_stop_metric: str = "pr_auc"  # "pr_auc", "roc_auc", "f1_best"

    # Evaluation thresholds
    target_recall: float = 0.90  # For recall-targeted threshold

    # Stations
    train_stations: List[str] = field(default_factory=lambda: [
        'Annapolis', 'Atlantic_City', 'Charleston', 'Washington',
        'Wilmington', 'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook'
    ])
    test_stations: List[str] = field(default_factory=lambda: [
        'Lewes', 'Fernandina_Beach', 'The_Battery'
    ])


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================
def matlab2datetime(matlab_datenum):
    """Convert MATLAB datenum to Python datetime."""
    return datetime.fromordinal(int(matlab_datenum)) \
           + timedelta(days=matlab_datenum % 1) \
           - timedelta(days=366)


def load_data(config: Config, logger) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    logger.info("Loading dataset...")

    mat_file = Path(config.data_path)
    data = loadmat(str(mat_file))

    lat = data['lattg'].flatten()
    lon = data['lontg'].flatten()
    sea_level = data['sltg']
    station_names = [s[0] for s in data['sname'].flatten()]
    time_raw = data['t'].flatten()
    time_dt = pd.to_datetime([matlab2datetime(t) for t in time_raw])

    logger.info(f"Loaded {len(station_names)} stations, {len(time_dt)} time points")

    # Build DataFrame efficiently
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

    df = pd.DataFrame(records)
    logger.info(f"Built DataFrame: {len(df):,} rows")

    return df


# =============================================================================
# FEATURE ENGINEERING (Enhanced)
# =============================================================================
def create_enhanced_features(df: pd.DataFrame, config: Config, logger) -> pd.DataFrame:
    """
    Create enhanced features for flood prediction.

    Features created:
    - Rolling statistics (mean, std, min, max) over configurable windows
    - Lag features
    - Rate-of-rise features (differences over time)
    - Flood labels
    """
    logger.info("Creating enhanced features...")

    df = df.copy()
    df = df.sort_values(['station_name', 'time']).reset_index(drop=True)

    # Compute flood thresholds per station
    threshold_df = df.groupby('station_name')['sea_level'].agg(['mean', 'std']).reset_index()
    threshold_df['flood_threshold'] = threshold_df['mean'] + 1.5 * threshold_df['std']
    df = df.merge(threshold_df[['station_name', 'flood_threshold']], on='station_name', how='left')

    # Group by station for feature engineering
    feature_dfs = []

    for stn, grp in df.groupby('station_name'):
        grp = grp.sort_values('time').reset_index(drop=True)

        # =================================================================
        # ROLLING FEATURES (over different windows)
        # =================================================================
        for window in config.rolling_windows:
            # Rolling mean
            grp[f'sea_level_roll_mean_{window}h'] = grp['sea_level'].rolling(
                window=window, min_periods=1
            ).mean()

            # Rolling std (volatility)
            grp[f'sea_level_roll_std_{window}h'] = grp['sea_level'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)

            # Rolling max (peak detection)
            grp[f'sea_level_roll_max_{window}h'] = grp['sea_level'].rolling(
                window=window, min_periods=1
            ).max()

            # Rolling min
            grp[f'sea_level_roll_min_{window}h'] = grp['sea_level'].rolling(
                window=window, min_periods=1
            ).min()

            # Rolling range (max - min)
            grp[f'sea_level_roll_range_{window}h'] = (
                grp[f'sea_level_roll_max_{window}h'] - grp[f'sea_level_roll_min_{window}h']
            )

        # =================================================================
        # LAG FEATURES
        # =================================================================
        for lag in config.lag_windows:
            grp[f'sea_level_lag_{lag}h'] = grp['sea_level'].shift(lag)

        # =================================================================
        # RATE-OF-RISE FEATURES (critical for flood prediction!)
        # =================================================================
        for window in [1, 3, 6, 12, 24]:
            # Absolute change
            grp[f'sea_level_diff_{window}h'] = grp['sea_level'].diff(window)

            # Percent change
            grp[f'sea_level_pct_change_{window}h'] = grp['sea_level'].pct_change(window)

            # Acceleration (second derivative)
            if window >= 2:
                grp[f'sea_level_accel_{window}h'] = grp[f'sea_level_diff_{window}h'].diff(1)

        # =================================================================
        # THRESHOLD-RELATIVE FEATURES
        # =================================================================
        grp['sea_level_vs_threshold'] = grp['sea_level'] - grp['flood_threshold']
        grp['sea_level_pct_of_threshold'] = grp['sea_level'] / grp['flood_threshold']

        # Hours since last above threshold
        above_threshold = (grp['sea_level'] > grp['flood_threshold']).astype(int)
        grp['hours_since_above_threshold'] = above_threshold.groupby(
            (above_threshold != above_threshold.shift()).cumsum()
        ).cumcount()

        feature_dfs.append(grp)

    df = pd.concat(feature_dfs, ignore_index=True)

    # Fill NaN from lag/diff operations
    df = df.fillna(method='bfill').fillna(0)

    logger.info(f"Created {len(df.columns)} features")

    return df


def aggregate_to_daily(df: pd.DataFrame, config: Config, logger) -> pd.DataFrame:
    """Aggregate hourly data to daily with flood labels."""
    logger.info("Aggregating to daily data...")

    # Get feature columns (excluding metadata)
    meta_cols = ['time', 'station_name', 'latitude', 'longitude', 'flood_threshold']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Daily aggregation
    agg_dict = {col: 'mean' for col in feature_cols if 'sea_level' in col}
    agg_dict['latitude'] = 'first'
    agg_dict['longitude'] = 'first'
    agg_dict['flood_threshold'] = 'first'

    df_daily = df.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg(agg_dict).reset_index()

    # Daily max for flood detection
    daily_max = df.groupby(['station_name', pd.Grouper(key='time', freq='D')])['sea_level'].max().reset_index()
    daily_max.columns = ['station_name', 'time', 'sea_level_daily_max']

    df_daily = df_daily.merge(daily_max, on=['station_name', 'time'])
    df_daily['flood'] = (df_daily['sea_level_daily_max'] > df_daily['flood_threshold']).astype(int)

    df_daily = df_daily.sort_values(['station_name', 'time']).reset_index(drop=True)

    logger.info(f"Daily DataFrame: {len(df_daily):,} rows")

    return df_daily


def create_sequences(
    df: pd.DataFrame,
    stations: List[str],
    config: Config,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create sequence windows for training."""
    sequences = []
    labels = []
    metadata = []

    for stn in stations:
        grp = df[df['station_name'] == stn].sort_values('time').reset_index(drop=True)
        features = grp[feature_cols].values
        floods = grp['flood'].values
        times = grp['time'].values

        for i in range(len(grp) - config.hist_days - config.future_days + 1):
            seq = features[i:i+config.hist_days]

            if np.isnan(seq).any():
                continue

            future_floods = floods[i+config.hist_days:i+config.hist_days+config.future_days]
            label = int(future_floods.max() > 0)

            sequences.append(seq.flatten())
            labels.append(label)
            metadata.append({
                'station': stn,
                'start_time': times[i],
                'end_time': times[i+config.hist_days-1],
                'predict_start': times[i+config.hist_days] if i+config.hist_days < len(times) else None
            })

    return np.array(sequences), np.array(labels), pd.DataFrame(metadata)


# =============================================================================
# DATA SPLITTING (Time-based, prevents leakage)
# =============================================================================
def split_data(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    config: Config,
    logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data using time-based or grouped splitting.

    Methods:
    - "time": Train on data before split_date, validate on data after
    - "grouped": GroupKFold by station
    - "random": Stratified random split (NOT recommended for time series)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA SPLITTING: {config.split_method.upper()} METHOD")
    logger.info(f"{'='*60}")

    if config.split_method == "time":
        # Time-based split (RECOMMENDED for time series)
        split_date = pd.to_datetime(config.time_split_date)

        # Convert metadata times
        if 'start_time' in metadata.columns:
            times = pd.to_datetime(metadata['start_time'])
        else:
            times = pd.to_datetime(metadata['end_time'])

        train_mask = times < split_date
        val_mask = times >= split_date

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        logger.info(f"Time split date: {split_date}")
        logger.info(f"Training: data before {split_date.date()}")
        logger.info(f"Validation: data from {split_date.date()} onwards")

    elif config.split_method == "grouped":
        # Grouped split by station (for spatial generalization)
        groups = metadata['station'].values
        gkf = GroupKFold(n_splits=5)

        # Use first fold
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            break

        logger.info("Grouped split by station")

    else:
        # Random stratified split (NOT recommended, but available)
        logger.warning("⚠️  Using random split - may cause temporal leakage!")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config.val_ratio,
            random_state=42,
            stratify=y
        )

    # Report class imbalance
    logger.info(f"\n{'='*60}")
    logger.info("CLASS IMBALANCE REPORT")
    logger.info(f"{'='*60}")

    train_pos = y_train.sum()
    train_neg = len(y_train) - train_pos
    val_pos = y_val.sum()
    val_neg = len(y_val) - val_pos

    logger.info(f"\nTraining Set:")
    logger.info(f"  Total samples:  {len(y_train):,}")
    logger.info(f"  Positive (flood): {train_pos:,} ({train_pos/len(y_train)*100:.1f}%)")
    logger.info(f"  Negative (no flood): {train_neg:,} ({train_neg/len(y_train)*100:.1f}%)")
    logger.info(f"  Positive rate: {y_train.mean():.4f}")

    logger.info(f"\nValidation Set:")
    logger.info(f"  Total samples:  {len(y_val):,}")
    logger.info(f"  Positive (flood): {val_pos:,} ({val_pos/len(y_val)*100:.1f}%)")
    logger.info(f"  Negative (no flood): {val_neg:,} ({val_neg/len(y_val)*100:.1f}%)")
    logger.info(f"  Positive rate: {y_val.mean():.4f}")

    # Compute pos_weight for weighted loss
    pos_weight = train_neg / max(train_pos, 1)
    logger.info(f"\nRecommended pos_weight for BCEWithLogitsLoss: {pos_weight:.4f}")

    return X_train, X_val, y_train, y_val, pos_weight


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma > 0, reduces the relative loss for well-classified examples,
    focusing training on hard negatives.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Compute BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_loss_function(config: Config, pos_weight: float, device: torch.device) -> nn.Module:
    """Get the appropriate loss function based on config."""

    if config.loss_type == "focal":
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    elif config.loss_type == "weighted_bce":
        # BCEWithLogitsLoss with pos_weight for class imbalance
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    else:
        # Standard BCE (not recommended for imbalanced data)
        return nn.BCEWithLogitsLoss()


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================
class ComprehensiveEvaluator:
    """
    Comprehensive evaluation for binary classification.

    Computes:
    - ROC-AUC
    - PR-AUC (Average Precision)
    - F1 at various thresholds
    - Optimal threshold for F1
    - Confusion matrices
    - Precision at target recall
    """

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        plot: bool = False,
        save_dir: Optional[Path] = None,
        epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of predictions.

        Args:
            y_true: Ground truth labels (0/1)
            y_prob: Predicted probabilities (0-1)
            plot: Whether to generate plots
            save_dir: Directory to save plots
            epoch: Current epoch (for plot titles)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # =================================================================
        # 1. ROC-AUC
        # =================================================================
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

        # =================================================================
        # 2. PR-AUC (Average Precision) - Better for imbalanced data
        # =================================================================
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)

        # =================================================================
        # 3. F1 at threshold=0.5 (standard)
        # =================================================================
        y_pred_05 = (y_prob >= 0.5).astype(int)
        metrics['f1_at_0.5'] = f1_score(y_true, y_pred_05, zero_division=0)
        metrics['precision_at_0.5'] = precision_score(y_true, y_pred_05, zero_division=0)
        metrics['recall_at_0.5'] = recall_score(y_true, y_pred_05, zero_division=0)

        # =================================================================
        # 4. Find optimal threshold for F1
        # =================================================================
        thresholds = np.arange(0.01, 1.0, 0.01)
        f1_scores = []

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)

        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]

        metrics['f1_best'] = f1_scores[best_f1_idx]
        metrics['best_threshold'] = best_threshold

        # Metrics at best threshold
        y_pred_best = (y_prob >= best_threshold).astype(int)
        metrics['precision_at_best'] = precision_score(y_true, y_pred_best, zero_division=0)
        metrics['recall_at_best'] = recall_score(y_true, y_pred_best, zero_division=0)
        metrics['accuracy_at_best'] = accuracy_score(y_true, y_pred_best)
        metrics['mcc_at_best'] = matthews_corrcoef(y_true, y_pred_best)

        # =================================================================
        # 5. Threshold for target recall
        # =================================================================
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_prob)

        # Find threshold that achieves target recall
        target_recall = self.config.target_recall
        recall_achieved_idx = np.where(recall_vals >= target_recall)[0]

        if len(recall_achieved_idx) > 0:
            # Get the highest precision at target recall
            idx = recall_achieved_idx[-1]  # Last index where recall >= target
            if idx < len(pr_thresholds):
                threshold_at_recall = pr_thresholds[idx]
                metrics[f'threshold_at_recall_{target_recall}'] = threshold_at_recall
                metrics[f'precision_at_recall_{target_recall}'] = precision_vals[idx]

        # =================================================================
        # 6. Generate plots
        # =================================================================
        if plot and save_dir:
            self._generate_plots(
                y_true, y_prob, thresholds, f1_scores,
                best_threshold, save_dir, epoch
            )

        return metrics

    def _generate_plots(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: np.ndarray,
        f1_scores: List[float],
        best_threshold: float,
        save_dir: Path,
        epoch: Optional[int]
    ):
        """Generate evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        epoch_str = f" (Epoch {epoch})" if epoch is not None else ""

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title(f'ROC Curve{epoch_str}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        baseline = y_true.mean()

        axes[0, 1].plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {pr_auc:.4f})')
        axes[0, 1].axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.4f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title(f'Precision-Recall Curve{epoch_str}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. F1 vs Threshold
        axes[1, 0].plot(thresholds, f1_scores, 'b-', linewidth=2)
        axes[1, 0].axvline(x=best_threshold, color='r', linestyle='--',
                          label=f'Best threshold = {best_threshold:.2f}')
        axes[1, 0].axvline(x=0.5, color='g', linestyle=':', label='Default (0.5)')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title(f'F1 Score vs Threshold{epoch_str}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Probability Distribution
        axes[1, 1].hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='No Flood (0)', density=True)
        axes[1, 1].hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Flood (1)', density=True)
        axes[1, 1].axvline(x=best_threshold, color='r', linestyle='--',
                          label=f'Best threshold = {best_threshold:.2f}')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title(f'Probability Distribution by Class{epoch_str}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = save_dir / f"evaluation_plots{'_epoch_' + str(epoch) if epoch else ''}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved evaluation plots to: {save_path}")

    def print_confusion_matrices(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        best_threshold: float
    ):
        """Print confusion matrices at different thresholds."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("CONFUSION MATRICES")
        self.logger.info(f"{'='*60}")

        thresholds_to_show = [
            (0.5, "Default (0.5)"),
            (best_threshold, f"Best F1 ({best_threshold:.3f})"),
        ]

        # Add target recall threshold if available
        target_recall = self.config.target_recall
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_prob)
        recall_achieved_idx = np.where(recall_vals >= target_recall)[0]

        if len(recall_achieved_idx) > 0:
            idx = recall_achieved_idx[-1]
            if idx < len(pr_thresholds):
                thresh_at_recall = pr_thresholds[idx]
                thresholds_to_show.append(
                    (thresh_at_recall, f"Target Recall {target_recall} ({thresh_at_recall:.3f})")
                )

        for thresh, name in thresholds_to_show:
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)

            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            self.logger.info(f"\n--- Threshold: {name} ---")
            self.logger.info(f"              Predicted")
            self.logger.info(f"              No Flood    Flood")
            self.logger.info(f"Actual No Flood  {tn:>7}  {fp:>7}")
            self.logger.info(f"       Flood     {fn:>7}  {tp:>7}")
            self.logger.info(f"\nPrecision: {precision:.4f}")
            self.logger.info(f"Recall:    {recall:.4f}")
            self.logger.info(f"F1 Score:  {f1:.4f}")


# =============================================================================
# PYTORCH MODELS
# =============================================================================
class FloodDataset(Dataset):
    """PyTorch Dataset for flood prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerClassifier(nn.Module):
    """Transformer for binary classification."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # No sigmoid - using BCEWithLogitsLoss
        )

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.input_projection(x)  # (batch, 1, d_model)
        x = self.encoder(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        x = self.classifier(x)  # (batch, 1)
        return x.squeeze(-1)


class LSTMClassifier(nn.Module):
    """LSTM for binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.classifier(x)
        return x.squeeze(-1)


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_neural_model(
    config: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    logger,
    evaluator: ComprehensiveEvaluator
) -> Tuple[nn.Module, Dict]:
    """Train neural network model with comprehensive evaluation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nDevice: {device}")

    # Create datasets and loaders
    train_dataset = FloodDataset(X_train, y_train)
    val_dataset = FloodDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]

    if config.model_type == "transformer":
        model = TransformerClassifier(
            input_dim=input_dim,
            d_model=config.d_model,
            nhead=config.n_heads,
            num_layers=config.n_layers,
            dropout=config.dropout
        )
    else:
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout
        )

    model = model.to(device)

    # Loss function
    criterion = get_loss_function(config, pos_weight, device)
    logger.info(f"Loss function: {config.loss_type}")
    if config.loss_type == "weighted_bce":
        logger.info(f"  pos_weight: {pos_weight:.4f}")
    elif config.loss_type == "focal":
        logger.info(f"  alpha: {config.focal_alpha}, gamma: {config.focal_gamma}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Scheduler
    if HAS_TRANSFORMERS:
        total_steps = len(train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
    else:
        scheduler = None

    # Training tracking
    best_metric = 0
    best_model_state = None
    best_roc_auc_state = None
    best_pr_auc_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'roc_auc': [], 'pr_auc': [], 'f1_best': [], 'f1_at_0.5': []
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING STARTED")
    logger.info(f"{'='*60}")
    logger.info(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'F1@0.5':>8} | {'F1 Best':>8}")
    logger.info("-" * 80)

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler:
                scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # IMPORTANT: Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)

                val_loss += loss.item()
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)

        # Comprehensive evaluation
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Verify probabilities are in valid range
        assert all_probs.min() >= 0 and all_probs.max() <= 1, \
            f"Probabilities out of range: [{all_probs.min()}, {all_probs.max()}]"

        metrics = evaluator.evaluate(
            all_labels, all_probs,
            plot=(epoch % 10 == 0),  # Plot every 10 epochs
            save_dir=output_dir,
            epoch=epoch + 1
        )

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['roc_auc'].append(metrics['roc_auc'])
        history['pr_auc'].append(metrics['pr_auc'])
        history['f1_best'].append(metrics['f1_best'])
        history['f1_at_0.5'].append(metrics['f1_at_0.5'])

        # Log progress
        logger.info(
            f"{epoch+1:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
            f"{metrics['roc_auc']:>8.4f} | {metrics['pr_auc']:>8.4f} | "
            f"{metrics['f1_at_0.5']:>8.4f} | {metrics['f1_best']:>8.4f}"
        )

        # Save best models
        current_metric = metrics[config.early_stop_metric]

        if metrics['roc_auc'] > (best_roc_auc_state[1] if best_roc_auc_state else 0):
            best_roc_auc_state = (model.state_dict().copy(), metrics['roc_auc'])
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'epoch': epoch + 1
            }, output_dir / "best_model_roc_auc.pt")

        if metrics['pr_auc'] > (best_pr_auc_state[1] if best_pr_auc_state else 0):
            best_pr_auc_state = (model.state_dict().copy(), metrics['pr_auc'])
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'epoch': epoch + 1
            }, output_dir / "best_model_pr_auc.pt")
            logger.info(f"  *** New best PR-AUC: {metrics['pr_auc']:.4f} ***")

        # Early stopping
        if current_metric > best_metric:
            best_metric = current_metric
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            logger.info(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history, all_probs, all_labels


# =============================================================================
# GRADIENT BOOSTING BASELINES
# =============================================================================
def train_lightgbm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    logger,
    evaluator: ComprehensiveEvaluator
) -> Tuple[Any, Dict]:
    """Train LightGBM baseline with proper class weighting."""

    if not HAS_LGBM:
        logger.warning("LightGBM not installed, skipping baseline")
        return None, {}

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING LIGHTGBM BASELINE")
    logger.info(f"{'='*60}")

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': pos_weight,  # Handle class imbalance
        'random_state': 42,
        'verbose': -1
    }

    logger.info(f"scale_pos_weight: {pos_weight:.4f}")

    # Train
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Predict
    y_prob = model.predict_proba(X_val)[:, 1]

    # Evaluate
    metrics = evaluator.evaluate(y_val, y_prob, plot=True)

    logger.info(f"\nLightGBM Results:")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
    logger.info(f"  F1 @ 0.5: {metrics['f1_at_0.5']:.4f}")
    logger.info(f"  F1 Best: {metrics['f1_best']:.4f} (threshold={metrics['best_threshold']:.3f})")

    return model, metrics


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    logger,
    evaluator: ComprehensiveEvaluator
) -> Tuple[Any, Dict]:
    """Train XGBoost baseline with proper class weighting."""

    if not HAS_XGB:
        logger.warning("XGBoost not installed, skipping baseline")
        return None, {}

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING XGBOOST BASELINE")
    logger.info(f"{'='*60}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric=['auc', 'aucpr'],
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0
    )

    logger.info(f"scale_pos_weight: {pos_weight:.4f}")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluator.evaluate(y_val, y_prob, plot=True)

    logger.info(f"\nXGBoost Results:")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
    logger.info(f"  F1 @ 0.5: {metrics['f1_at_0.5']:.4f}")
    logger.info(f"  F1 Best: {metrics['f1_best']:.4f} (threshold={metrics['best_threshold']:.3f})")

    return model, metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline(config: Config):
    """Run the complete training pipeline."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    evaluator = ComprehensiveEvaluator(config, logger)

    logger.info("=" * 60)
    logger.info("FLOOD PREDICTION PIPELINE - REFACTORED")
    logger.info("=" * 60)

    # 1. Load data
    df = load_data(config, logger)

    # 2. Create enhanced features
    df = create_enhanced_features(df, config, logger)

    # 3. Aggregate to daily
    df_daily = aggregate_to_daily(df, config, logger)

    # 4. Get feature columns
    exclude_cols = ['time', 'station_name', 'latitude', 'longitude',
                    'flood_threshold', 'flood', 'sea_level_daily_max']
    feature_cols = [c for c in df_daily.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # 5. Create sequences
    X, y, metadata = create_sequences(df_daily, config.train_stations, config, feature_cols)
    logger.info(f"Created {len(X)} sequences")

    # 6. Split data
    X_train, X_val, y_train, y_val, pos_weight = split_data(X, y, metadata, config, logger)

    # 7. Train models
    results = {}

    # Neural model
    if config.model_type in ["transformer", "lstm"]:
        model, history, val_probs, val_labels = train_neural_model(
            config, X_train, y_train, X_val, y_val, pos_weight, logger, evaluator
        )

        # Final evaluation
        final_metrics = evaluator.evaluate(val_labels, val_probs, plot=True, save_dir=output_dir)
        evaluator.print_confusion_matrices(val_labels, val_probs, final_metrics['best_threshold'])

        results['neural'] = final_metrics

    # LightGBM baseline
    lgbm_model, lgbm_metrics = train_lightgbm_baseline(
        X_train, y_train, X_val, y_val, pos_weight, logger, evaluator
    )
    if lgbm_metrics:
        results['lightgbm'] = lgbm_metrics

    # XGBoost baseline
    xgb_model, xgb_metrics = train_xgboost_baseline(
        X_train, y_train, X_val, y_val, pos_weight, logger, evaluator
    )
    if xgb_metrics:
        results['xgboost'] = xgb_metrics

    # 8. Print summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Model':<15} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1@0.5':<10} {'F1 Best':<10} {'Threshold':<10}")
    logger.info("-" * 65)

    for name, metrics in results.items():
        logger.info(
            f"{name:<15} {metrics['roc_auc']:<10.4f} {metrics['pr_auc']:<10.4f} "
            f"{metrics['f1_at_0.5']:<10.4f} {metrics['f1_best']:<10.4f} "
            f"{metrics['best_threshold']:<10.3f}"
        )

    # Save results
    with open(output_dir / "results_summary.json", 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Flood Prediction Training Pipeline')

    # Data
    parser.add_argument('--data', type=str, default='NEUSTG_19502020_12stations.mat')
    parser.add_argument('--output', type=str, default='results_refactored')

    # Split
    parser.add_argument('--split', type=str, default='time', choices=['time', 'random', 'grouped'])
    parser.add_argument('--split-date', type=str, default='2015-01-01')

    # Model
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'lstm'])

    # Loss
    parser.add_argument('--loss', type=str, default='weighted_bce', choices=['bce', 'weighted_bce', 'focal'])
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()

    # Create config
    config = Config(
        data_path=args.data,
        output_dir=args.output,
        split_method=args.split,
        time_split_date=args.split_date,
        model_type=args.model,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
