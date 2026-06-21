"""
Drift & Monitoring Module
==========================

Built from scratch (no heavy dependencies like Evidently).

- PSI (Population Stability Index): overall distribution shift
- CSI (Characteristic Stability Index): per-feature drift
- Missingness tracking: monitors missing-value rates over time
- Segment performance: AUC/KS by region, loan purpose, borrower type
- Basic fairness slicing: performance across gender, occupation type
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.training.metrics import calculate_ks_statistic

logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray,
                   n_bins: int = 10, eps: float = 1e-4) -> float:
    """
    Calculate Population Stability Index (PSI).

    Measures distribution shift between two populations.
    - PSI < 0.10: no significant shift
    - 0.10 ≤ PSI < 0.25: moderate shift (investigate)
    - PSI ≥ 0.25: significant shift (retrain recommended)

    Args:
        expected: Reference distribution (training data).
        actual: Current distribution (scoring data).
        n_bins: Number of bins for discretization.
        eps: Small constant to avoid log(0).

    Returns:
        PSI value (float ≥ 0).
    """
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Remove duplicate breakpoints
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def calculate_csi(train_df: pd.DataFrame, scoring_df: pd.DataFrame,
                   numeric_columns: list[str] | None = None,
                   n_bins: int = 10) -> pd.DataFrame:
    """
    Calculate Characteristic Stability Index (CSI) — per-feature PSI.

    Args:
        train_df: Training data.
        scoring_df: Scoring/production data.
        numeric_columns: Columns to check (default: all numeric).
        n_bins: Number of bins for PSI calculation.

    Returns:
        DataFrame with feature, psi, and status columns.
    """
    if numeric_columns is None:
        numeric_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in numeric_columns:
        if col in train_df.columns and col in scoring_df.columns:
            train_vals = train_df[col].dropna().values
            score_vals = scoring_df[col].dropna().values

            if len(train_vals) == 0 or len(score_vals) == 0:
                psi = np.nan
                status = "insufficient_data"
            else:
                psi = calculate_psi(train_vals, score_vals, n_bins)
                if psi < 0.10:
                    status = "stable"
                elif psi < 0.25:
                    status = "moderate_shift"
                else:
                    status = "significant_shift"

            results.append({"feature": col, "psi": psi, "status": status})

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("psi", ascending=False).reset_index(drop=True)

    n_shifted = len(df[df["status"] == "significant_shift"])
    n_moderate = len(df[df["status"] == "moderate_shift"])
    logger.info(f"CSI: {n_shifted} features with significant shift, "
                f"{n_moderate} with moderate shift")

    return df


def check_missingness_drift(train_df: pd.DataFrame,
                              scoring_df: pd.DataFrame,
                              threshold: float = 0.05) -> pd.DataFrame:
    """
    Compare missing-value rates between training and scoring data.

    Flags features where missingness rate changed by more than threshold.

    Returns:
        DataFrame with feature, train_missing_rate, scoring_missing_rate, delta, flagged.
    """
    results = []
    for col in train_df.columns:
        if col in scoring_df.columns:
            train_rate = train_df[col].isna().mean()
            score_rate = scoring_df[col].isna().mean()
            delta = abs(score_rate - train_rate)

            results.append({
                "feature": col,
                "train_missing_rate": round(train_rate, 4),
                "scoring_missing_rate": round(score_rate, 4),
                "delta": round(delta, 4),
                "flagged": delta > threshold,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("delta", ascending=False).reset_index(drop=True)

    n_flagged = df["flagged"].sum()
    if n_flagged > 0:
        logger.warning(f"Missingness drift: {n_flagged} features flagged "
                       f"(threshold={threshold})")

    return df


def segment_performance(y_true: np.ndarray, y_prob: np.ndarray,
                          segment_col: np.ndarray,
                          segment_name: str = "segment") -> pd.DataFrame:
    """
    Calculate model performance by segment (region, loan purpose, etc.).

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        segment_col: Segment labels (e.g., state, loan_purpose).
        segment_name: Name for the segment column.

    Returns:
        DataFrame with segment, count, bad_rate, auc, ks.
    """
    results = []
    segments = np.unique(segment_col)

    for seg in segments:
        mask = segment_col == seg
        y_t = y_true[mask]
        y_p = y_prob[mask]

        count = len(y_t)
        bad_rate = y_t.mean()

        try:
            if y_t.sum() > 0 and (y_t == 0).sum() > 0:
                auc = roc_auc_score(y_t, y_p)
                ks = calculate_ks_statistic(y_t, y_p)
            else:
                auc = np.nan
                ks = np.nan
        except Exception:
            auc = np.nan
            ks = np.nan

        results.append({
            segment_name: seg,
            "count": count,
            "bad_rate": round(bad_rate, 4),
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "ks": round(ks, 4) if not np.isnan(ks) else None,
        })

    return pd.DataFrame(results)


def fairness_slicing(y_true: np.ndarray, y_prob: np.ndarray,
                       sensitive_col: np.ndarray,
                       sensitive_name: str = "group") -> pd.DataFrame:
    """
    Basic fairness analysis — performance metrics across sensitive groups.

    This is NOT a comprehensive fairness audit, but a starting point for
    monitoring disparate impact across gender, occupation type, etc.
    """
    return segment_performance(y_true, y_prob, sensitive_col, sensitive_name)


def generate_monitoring_report(train_df: pd.DataFrame,
                                 scoring_df: pd.DataFrame,
                                 y_true: np.ndarray | None = None,
                                 y_prob: np.ndarray | None = None,
                                 segment_columns: dict[str, np.ndarray] | None = None,
                                 output_path: str | Path = "docs/monitoring_report.json") -> dict:
    """
    Generate comprehensive monitoring report.

    Args:
        train_df: Training feature DataFrame.
        scoring_df: Scoring/production feature DataFrame.
        y_true: True labels for scoring data (if available).
        y_prob: Predicted probabilities for scoring data.
        segment_columns: Dict of {name: values} for segment analysis.
        output_path: Path to save JSON report.

    Returns:
        Report dict.
    """
    report = {"generated_at": pd.Timestamp.now().isoformat()}

    # Overall PSI (on predicted scores if available)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # CSI — per-feature drift
    csi_df = calculate_csi(train_df, scoring_df, numeric_cols)
    report["csi"] = csi_df.to_dict(orient="records")
    report["overall_psi_summary"] = {
        "n_stable": len(csi_df[csi_df["status"] == "stable"]),
        "n_moderate_shift": len(csi_df[csi_df["status"] == "moderate_shift"]),
        "n_significant_shift": len(csi_df[csi_df["status"] == "significant_shift"]),
    }

    # Missingness drift
    miss_df = check_missingness_drift(train_df, scoring_df)
    report["missingness_drift"] = miss_df.to_dict(orient="records")

    # Segment performance (if labels available)
    if y_true is not None and y_prob is not None and segment_columns:
        report["segment_performance"] = {}
        for name, values in segment_columns.items():
            seg_df = segment_performance(y_true, y_prob, values, name)
            report["segment_performance"][name] = seg_df.to_dict(orient="records")

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Monitoring report saved to {output_path}")

    return report
