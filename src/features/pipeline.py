"""
Feature Pipeline Builder
=========================

Assembles all feature transformers into a scikit-learn Pipeline.
Handles column selection, ordering, and serialization.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from .transformers import (
    LeakageGuard,
    TemporalFeatureExtractor,
    MissingnessIndicator,
    CategoricalEncoder,
    NumericImputer,
    InteractionFeatures,
)

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path) -> dict:
    """Load feature engineering YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_feature_pipeline(config_path: str | Path = "configs/features.yaml",
                            strict: bool = True) -> Pipeline:
    """
    Build the complete feature engineering pipeline.

    Pipeline order:
        1. LeakageGuard — drop leakage + non-feature columns (FIRST)
        2. TemporalFeatureExtractor — extract date features
        3. MissingnessIndicator — create _is_missing flags for bureau features
        4. CategoricalEncoder — encode all categoricals
        5. InteractionFeatures — domain-driven interactions
        6. NumericImputer — fill remaining NaNs (LAST)

    Args:
        config_path: Path to features.yaml config.
        strict: If True, LeakageGuard raises on leakage detection.

    Returns:
        Unfitted sklearn Pipeline.
    """
    config = _load_config(config_path)

    pipeline = Pipeline([
        ("leakage_guard", LeakageGuard(
            leakage_columns=config.get("leakage_columns", []),
            drop_columns=config.get("drop_columns", []),
            strict=strict,
        )),
        ("temporal", TemporalFeatureExtractor(
            date_column=config.get("date_column", "origination_date"),
        )),
        ("missingness", MissingnessIndicator(
            columns=config.get("bureau_features", []),
        )),
        ("categorical", CategoricalEncoder(
            ordinal_mappings=config.get("ordinal_features", {}),
            high_cardinality_cols=config.get("high_cardinality_features", []),
        )),
        ("interactions", InteractionFeatures()),
        ("imputer", NumericImputer()),
    ])

    logger.info(f"Built feature pipeline with {len(pipeline.steps)} steps (strict={strict})")
    return pipeline


def fit_transform_pipeline(df: pd.DataFrame, pipeline: Pipeline,
                            config_path: str | Path = "configs/features.yaml") -> pd.DataFrame:
    """
    Fit and transform data through the feature pipeline.

    Args:
        df: Raw borrower DataFrame.
        pipeline: Unfitted feature pipeline.
        config_path: Path to features config (for extracting target/metadata).

    Returns:
        Transformed feature DataFrame (no target, no ID, no date).
    """
    config = _load_config(config_path)
    target_col = config.get("target_column", "default_flag")

    # Separate target before transformation
    y = df[target_col].values if target_col in df.columns else None

    # Fit and transform
    X_transformed = pipeline.fit_transform(df)

    logger.info(f"Pipeline fit_transform: {df.shape} → {X_transformed.shape}")
    return X_transformed, y


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    """Save fitted pipeline to disk via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Saved pipeline to {path}")


def load_pipeline(path: str | Path) -> Pipeline:
    """Load fitted pipeline from disk."""
    pipeline = joblib.load(path)
    logger.info(f"Loaded pipeline from {path}")
    return pipeline
