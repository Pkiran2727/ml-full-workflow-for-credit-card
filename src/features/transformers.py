"""
Leakage-Safe Feature Transformers
==================================

Scikit-learn compatible transformers for the credit scoring pipeline.
All transformers implement fit/transform with proper train/test discipline.

Key design: LeakageGuard runs FIRST and prevents any post-event features
from entering the pipeline. It uses strict mode by default (fail loud, fail early).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class LeakageError(Exception):
    """Raised when leakage columns are detected in strict mode."""
    pass


class LeakageGuard(BaseEstimator, TransformerMixin):
    """
    Hard guardrail that prevents post-event leakage columns from entering training.

    Behavior:
        fit(): reads leakage column list, records them for enforcement.
        transform(): checks incoming DataFrame for leakage columns.
            - strict=True (default, production): raises LeakageError
            - strict=False (dev/debug): silently drops + logs warning

    This is the FIRST transformer in the pipeline — it checks BEFORE dropping.
    """

    def __init__(self, leakage_columns: list[str], drop_columns: list[str] | None = None,
                 strict: bool = True):
        self.leakage_columns = leakage_columns
        self.drop_columns = drop_columns or []
        self.strict = strict

    def fit(self, X: pd.DataFrame, y=None):
        """Record leakage columns for enforcement. No fitting needed."""
        self.leakage_set_ = set(self.leakage_columns)
        self.drop_set_ = set(self.drop_columns)
        logger.info(f"LeakageGuard fitted: blocking {len(self.leakage_set_)} leakage columns, "
                     f"dropping {len(self.drop_set_)} non-feature columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check for and remove leakage + non-feature columns."""
        X = X.copy()

        # Check for leakage columns FIRST
        found_leakage = self.leakage_set_.intersection(set(X.columns))
        if found_leakage:
            if self.strict:
                raise LeakageError(
                    f"LEAKAGE DETECTED: columns {sorted(found_leakage)} are post-event "
                    f"features that must not be used in training. These columns are "
                    f"correlated with the target because they are set AFTER the default event."
                )
            else:
                logger.warning(
                    f"LeakageGuard (non-strict): dropping leakage columns {sorted(found_leakage)}"
                )
                X = X.drop(columns=list(found_leakage))

        # Drop non-feature columns (ID, date, target)
        found_drop = self.drop_set_.intersection(set(X.columns))
        if found_drop:
            X = X.drop(columns=list(found_drop))

        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from origination date.

    All features are anchored to the application date — never uses future information.
    Creates: origination_month, origination_quarter, origination_season
    """

    def __init__(self, date_column: str = "origination_date"):
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.date_column not in X.columns:
            logger.warning(f"Date column '{self.date_column}' not found — skipping temporal extraction")
            return X

        dates = pd.to_datetime(X[self.date_column])

        X["origination_month"] = dates.dt.month
        X["origination_quarter"] = dates.dt.quarter

        # Season mapping for Indian agriculture
        season_map = {
            1: "rabi", 2: "rabi", 3: "rabi",        # Jan-Mar: Rabi harvest
            4: "summer", 5: "summer",                 # Apr-May: Summer
            6: "kharif", 7: "kharif", 8: "kharif",   # Jun-Aug: Kharif sowing
            9: "kharif", 10: "kharif",                # Sep-Oct: Kharif harvest
            11: "rabi", 12: "rabi",                   # Nov-Dec: Rabi sowing
        }
        X["origination_season"] = dates.dt.month.map(season_map)

        return X


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """
    Create binary missingness indicator columns for specified features.

    Missingness in bureau features is a signal — thin-file borrowers with no
    credit history have higher risk, so we encode this explicitly.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[f"{col}_is_missing"] = X[col].isna().astype(int)
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features.

    Strategy:
        - Ordinal features: encoded by specified order (education_level)
        - High-cardinality features: frequency encoding (district)
        - Other categoricals: label encoding with unknown handling
    """

    def __init__(self, ordinal_mappings: dict[str, list] | None = None,
                 high_cardinality_cols: list[str] | None = None):
        self.ordinal_mappings = ordinal_mappings or {}
        self.high_cardinality_cols = high_cardinality_cols or []
        self.label_maps_: dict[str, dict] = {}
        self.freq_maps_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn encoding mappings from training data only."""
        # Label encoding for standard categoricals
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if col in self.ordinal_mappings or col in self.high_cardinality_cols:
                continue
            unique_vals = X[col].dropna().unique()
            self.label_maps_[col] = {val: idx for idx, val in enumerate(sorted(unique_vals))}

        # Frequency encoding for high-cardinality columns
        for col in self.high_cardinality_cols:
            if col in X.columns:
                freq = X[col].value_counts(normalize=True).to_dict()
                self.freq_maps_[col] = freq

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Ordinal encoding
        for col, order in self.ordinal_mappings.items():
            if col in X.columns:
                mapping = {val: idx for idx, val in enumerate(order)}
                X[col] = X[col].map(mapping).fillna(-1).astype(int)

        # Frequency encoding for high-cardinality
        for col in self.high_cardinality_cols:
            if col in X.columns and col in self.freq_maps_:
                X[col] = X[col].map(self.freq_maps_[col]).fillna(0).astype(float)

        # Label encoding for remaining categoricals
        for col, mapping in self.label_maps_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(-1).astype(int)

        return X


class NumericImputer(BaseEstimator, TransformerMixin):
    """
    Median imputation for numeric features.

    Fitted on training data only — transform uses stored medians.
    """

    def __init__(self):
        self.medians_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.medians_[col] = X[col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, median_val in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_val)
        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """
    Create domain-driven interaction features.

    - rainfall_x_farmer: rainfall_index × is_farmer (low rainfall hurts farmers more)
    - bureau_x_loans: bureau_score × existing_loan_count (risky combo)
    """

    def __init__(self, occupation_column: str = "occupation_type"):
        self.occupation_column = occupation_column

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Create is_farmer binary (may already exist if occupation was encoded)
        if self.occupation_column in X.columns:
            if X[self.occupation_column].dtype == object:
                X["is_farmer"] = (X[self.occupation_column] == "farmer").astype(int)
            else:
                # Already encoded — check if we stored the farmer code
                X["is_farmer"] = (X[self.occupation_column] == 0).astype(int)  # first in sorted

        # rainfall × farmer interaction
        if "district_rainfall_index" in X.columns and "is_farmer" in X.columns:
            X["rainfall_x_farmer"] = X["district_rainfall_index"] * X["is_farmer"]

        # Bureau × existing loans interaction
        # Bureau score band should be encoded by now
        if "bureau_score_band" in X.columns and "existing_loan_count" in X.columns:
            X["bureau_x_loans"] = X["bureau_score_band"].astype(float) * X["existing_loan_count"]

        return X
