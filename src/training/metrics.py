"""
Credit Scoring Metrics
=======================

Custom metric calculations specific to credit scoring:
- KS Statistic: max separation between cumulative good/bad distributions
- Gini Coefficient: 2 × AUC − 1
- Decile Analysis: lift table by score decile
- PR-AUC: critical for imbalanced classes
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    recall_score,
)


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate KS (Kolmogorov-Smirnov) statistic.

    Maximum separation between cumulative distribution of good and bad borrowers
    when sorted by predicted probability. Standard credit scoring metric.

    Returns:
        KS statistic (0-1, higher = better separation).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return float(ks)


def calculate_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Gini coefficient.

    Gini = 2 × AUC − 1. Standard measure of model discriminatory power.

    Returns:
        Gini coefficient (-1 to 1, higher = better).
    """
    auc = roc_auc_score(y_true, y_prob)
    return float(2 * auc - 1)


def calculate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float = 0.5) -> dict:
    """
    Calculate all credit scoring metrics.

    Returns dict with: roc_auc, ks_statistic, gini, pr_auc, log_loss,
    recall_positive, precision, confusion_matrix.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "ks_statistic": calculate_ks_statistic(y_true, y_prob),
        "gini": calculate_gini(y_true, y_prob),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "n_total": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "bad_rate": float(y_true.mean()),
    }

    return metrics


def generate_decile_table(y_true: np.ndarray, y_prob: np.ndarray,
                           n_bins: int = 10) -> pd.DataFrame:
    """
    Generate decile analysis table.

    Sorts predictions into deciles by score and calculates:
    - Count per decile
    - Number and rate of defaults (bads)
    - Cumulative capture rate
    - Lift

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins (default 10 for decile).

    Returns:
        DataFrame with decile analysis.
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], q=n_bins, labels=False, duplicates="drop")
    df["decile"] = df["decile"].max() - df["decile"]  # Higher score = higher risk decile

    table = df.groupby("decile").agg(
        count=("y_true", "count"),
        n_bads=("y_true", "sum"),
        avg_score=("y_prob", "mean"),
    ).reset_index()

    table["bad_rate"] = table["n_bads"] / table["count"]
    table["cumulative_bads"] = table["n_bads"].cumsum()
    total_bads = table["n_bads"].sum()
    table["cumulative_capture_rate"] = table["cumulative_bads"] / total_bads
    overall_bad_rate = total_bads / table["count"].sum()
    table["lift"] = table["bad_rate"] / overall_bad_rate

    return table
