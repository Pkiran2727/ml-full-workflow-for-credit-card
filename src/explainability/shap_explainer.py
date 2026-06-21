"""
SHAP Explainability Module
============================

Global and local SHAP explanations for credit scoring models.
- Global: beeswarm and bar summary plots
- Local: per-prediction reason codes (top-N features driving the score)
- Stability: compares top-K feature rankings across model versions

Reason codes are critical for adverse action notices in regulated credit.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(model, X: pd.DataFrame | np.ndarray,
                         feature_names: list[str] | None = None,
                         max_samples: int = 1000) -> shap.Explanation:
    """
    Compute SHAP values for a trained tree model.

    Uses TreeExplainer for XGBoost/LightGBM (exact, fast).

    Args:
        model: Trained model.
        X: Feature matrix.
        feature_names: Feature names.
        max_samples: Max samples for SHAP computation (speed).

    Returns:
        shap.Explanation object.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or list(X.columns)
        X_array = X.values
    else:
        X_array = X

    # Subsample if too large
    if len(X_array) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_array), max_samples, replace=False)
        X_array = X_array[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_array)

    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class

    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                     else explainer.expected_value[1],
        data=X_array,
        feature_names=feature_names,
    )

    logger.info(f"Computed SHAP values for {len(X_array)} samples, "
                f"{shap_values.shape[1]} features")
    return explanation


def get_reason_codes(model, X_single: pd.DataFrame | np.ndarray,
                      feature_names: list[str],
                      top_n: int = 3) -> list[dict]:
    """
    Get top-N reason codes for a single prediction.

    Returns the features with highest absolute SHAP values that are
    pushing the score UP (towards default). This is used for adverse
    action notices in credit lending.

    Args:
        model: Trained model.
        X_single: Single borrower features (1D or 1-row 2D array).
        feature_names: Feature names.
        top_n: Number of reason codes to return.

    Returns:
        List of dicts with 'feature', 'shap_value', 'direction' keys.
    """
    if isinstance(X_single, pd.DataFrame):
        X_array = X_single.values
    else:
        X_array = np.atleast_2d(X_single)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_array)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_vals = shap_values[0]  # First (and only) sample

    # Sort by absolute value, take top-N features pushing towards default
    abs_idx = np.argsort(np.abs(shap_vals))[::-1]

    reason_codes = []
    for idx in abs_idx[:top_n]:
        reason_codes.append({
            "feature": feature_names[idx],
            "shap_value": float(shap_vals[idx]),
            "direction": "increases_risk" if shap_vals[idx] > 0 else "decreases_risk",
            "feature_value": float(X_array[0, idx]) if X_array.shape[1] > idx else None,
        })

    return reason_codes


def plot_global_shap(shap_explanation: shap.Explanation,
                      save_dir: str | Path = "docs") -> list[Path]:
    """
    Generate global SHAP summary plots.

    Creates:
        - Beeswarm plot (shows distribution of SHAP values per feature)
        - Bar plot (shows mean |SHAP| per feature)

    Args:
        shap_explanation: SHAP Explanation object.
        save_dir: Directory to save plots.

    Returns:
        List of saved file paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Beeswarm plot
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_explanation.values, shap_explanation.data,
                          feature_names=shap_explanation.feature_names,
                          show=False, max_display=20)
        path = save_dir / "shap_beeswarm.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)
        logger.info(f"Saved beeswarm plot to {path}")
    except Exception as e:
        logger.warning(f"Beeswarm plot failed: {e}")

    # Bar plot
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_explanation.values, shap_explanation.data,
                          feature_names=shap_explanation.feature_names,
                          plot_type="bar", show=False, max_display=20)
        path = save_dir / "shap_bar.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)
        logger.info(f"Saved bar plot to {path}")
    except Exception as e:
        logger.warning(f"Bar plot failed: {e}")

    return saved


def check_feature_stability(current_shap: shap.Explanation,
                              previous_top_features: list[str] | None = None,
                              top_k: int = 10) -> dict:
    """
    Check stability of top feature rankings across retrains.

    Compares current top-K features (by mean |SHAP|) with a previous model's
    top-K features. Flags if significant rank changes occur.

    Args:
        current_shap: Current model's SHAP explanation.
        previous_top_features: Previous model's top-K features (if available).
        top_k: Number of top features to compare.

    Returns:
        Dict with 'current_top', 'overlap_ratio', 'new_features', 'dropped_features'.
    """
    mean_abs_shap = np.mean(np.abs(current_shap.values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    current_top = [current_shap.feature_names[i] for i in top_indices]

    result = {
        "current_top_features": current_top,
        "mean_abs_shap": {
            current_shap.feature_names[i]: float(mean_abs_shap[i])
            for i in top_indices
        },
    }

    if previous_top_features:
        current_set = set(current_top)
        previous_set = set(previous_top_features[:top_k])
        overlap = current_set & previous_set

        result["overlap_ratio"] = len(overlap) / top_k
        result["new_features"] = list(current_set - previous_set)
        result["dropped_features"] = list(previous_set - current_set)

        if result["overlap_ratio"] < 0.7:
            logger.warning(
                f"FEATURE STABILITY WARNING: only {result['overlap_ratio']:.0%} overlap "
                f"in top-{top_k} features. New: {result['new_features']}, "
                f"Dropped: {result['dropped_features']}"
            )
        else:
            logger.info(f"Feature stability check passed: {result['overlap_ratio']:.0%} overlap")
    else:
        logger.info("No previous model for stability comparison — establishing baseline")

    return result
