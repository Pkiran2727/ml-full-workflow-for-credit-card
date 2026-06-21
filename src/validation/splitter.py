"""
Validation Splitters
=====================

OOT (Out-of-Time) splitting for model evaluation and stratified k-fold
for model selection. Mirrors regulated credit shop validation methodology.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path) -> dict:
    """Load validation YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class OOTSplitter:
    """
    Out-of-Time splitter for credit scoring validation.

    Splits by origination_date — train on loans originated before cutoff,
    test (OOT) on loans originated after. This mirrors how regulated
    credit shops validate their models.

    Validates:
        - No temporal overlap between train/test
        - Sufficient positive class in OOT window (warns if < min threshold)
    """

    def __init__(self, cutoff_date: str, date_column: str = "origination_date",
                 target_column: str = "default_flag", min_oot_defaults: int = 50):
        self.cutoff_date = pd.Timestamp(cutoff_date)
        self.date_column = date_column
        self.target_column = target_column
        self.min_oot_defaults = min_oot_defaults

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/validation.yaml") -> "OOTSplitter":
        """Create OOTSplitter from YAML config."""
        config = _load_config(config_path)
        return cls(
            cutoff_date=config["oot_cutoff_date"],
            min_oot_defaults=config.get("min_oot_defaults", 50),
        )

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train and OOT sets.

        Args:
            df: Full DataFrame with origination_date column.

        Returns:
            (train_df, oot_df) tuple.
        """
        dates = pd.to_datetime(df[self.date_column])

        train_mask = dates < self.cutoff_date
        oot_mask = dates >= self.cutoff_date

        train_df = df[train_mask].reset_index(drop=True)
        oot_df = df[oot_mask].reset_index(drop=True)

        # Validate no overlap
        train_max_date = pd.to_datetime(train_df[self.date_column]).max()
        oot_min_date = pd.to_datetime(oot_df[self.date_column]).min()
        assert train_max_date < oot_min_date, (
            f"Temporal overlap detected: train max={train_max_date}, OOT min={oot_min_date}"
        )

        # Check OOT default count
        oot_defaults = oot_df[self.target_column].sum()
        if oot_defaults < self.min_oot_defaults:
            logger.warning(
                f"LOW OOT DEFAULTS: only {oot_defaults} defaults in OOT set "
                f"(minimum recommended: {self.min_oot_defaults}). "
                f"OOT metrics may be statistically unreliable."
            )

        # Log split info
        train_rate = train_df[self.target_column].mean()
        oot_rate = oot_df[self.target_column].mean()
        logger.info(
            f"OOT Split (cutoff={self.cutoff_date.date()}):\n"
            f"  Train: {len(train_df)} records, bad rate={train_rate:.3f}, "
            f"dates={pd.to_datetime(train_df[self.date_column]).min().date()} to {train_max_date.date()}\n"
            f"  OOT:   {len(oot_df)} records, bad rate={oot_rate:.3f}, "
            f"dates={oot_min_date.date()} to {pd.to_datetime(oot_df[self.date_column]).max().date()}\n"
            f"  OOT defaults: {oot_defaults}"
        )

        return train_df, oot_df


class StratifiedKFoldCV:
    """
    Stratified K-Fold cross-validation within training set.

    Used for model selection (hyperparameter tuning with Optuna).
    Preserves class distribution in each fold.
    """

    def __init__(self, n_folds: int = 5, target_column: str = "default_flag",
                 random_seed: int = 42):
        self.n_folds = n_folds
        self.target_column = target_column
        self.random_seed = random_seed
        self._skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_seed
        )

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/validation.yaml") -> "StratifiedKFoldCV":
        """Create StratifiedKFoldCV from YAML config."""
        config = _load_config(config_path)
        return cls(
            n_folds=config.get("n_folds", 5),
            target_column=config.get("stratify_by", "default_flag"),
        )

    def split(self, X: pd.DataFrame | np.ndarray,
              y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified k-fold indices.

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        folds = list(self._skf.split(X, y))
        logger.info(f"Stratified {self.n_folds}-fold CV: {len(folds)} folds generated")
        return folds

    def get_sklearn_splitter(self) -> StratifiedKFold:
        """Return the underlying sklearn StratifiedKFold object for use with Optuna/cross_val_score."""
        return self._skf
