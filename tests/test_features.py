"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest
import joblib

from src.features.transformers import LeakageGuard, LeakageError, MissingnessIndicator
from src.features.pipeline import build_feature_pipeline


class TestLeakageGuard:
    """Test LeakageGuard transformer — the hard guardrail."""

    def test_strict_mode_raises_on_leakage(self, sample_borrower_df):
        """In strict mode, leakage columns should raise LeakageError."""
        guard = LeakageGuard(
            leakage_columns=["collections_flag", "days_past_due_at_snapshot", "recovery_amount"],
            strict=True,
        )
        guard.fit(sample_borrower_df)

        with pytest.raises(LeakageError, match="LEAKAGE DETECTED"):
            guard.transform(sample_borrower_df)

    def test_non_strict_mode_drops_silently(self, sample_borrower_df):
        """In non-strict mode, leakage columns should be silently dropped."""
        guard = LeakageGuard(
            leakage_columns=["collections_flag", "days_past_due_at_snapshot", "recovery_amount"],
            strict=False,
        )
        guard.fit(sample_borrower_df)
        result = guard.transform(sample_borrower_df)

        assert "collections_flag" not in result.columns
        assert "days_past_due_at_snapshot" not in result.columns
        assert "recovery_amount" not in result.columns

    def test_drops_non_feature_columns(self, sample_borrower_df):
        """Should drop ID, date, and target columns."""
        guard = LeakageGuard(
            leakage_columns=["collections_flag", "days_past_due_at_snapshot", "recovery_amount"],
            drop_columns=["borrower_id", "origination_date", "default_flag"],
            strict=False,
        )
        guard.fit(sample_borrower_df)
        result = guard.transform(sample_borrower_df)

        assert "borrower_id" not in result.columns
        assert "origination_date" not in result.columns
        assert "default_flag" not in result.columns

    def test_no_leakage_columns_passes(self):
        """Clean DataFrame should pass through without error."""
        df = pd.DataFrame({"age": [25, 30], "income": [50000, 60000]})
        guard = LeakageGuard(leakage_columns=["collections_flag"], strict=True)
        guard.fit(df)
        result = guard.transform(df)
        assert list(result.columns) == ["age", "income"]


class TestMissingnessIndicator:
    """Test MissingnessIndicator transformer."""

    def test_creates_missing_flags(self, sample_borrower_df):
        """Should create _is_missing flags for bureau features."""
        indicator = MissingnessIndicator(columns=["bureau_score_band", "repayment_history_score"])
        result = indicator.transform(sample_borrower_df)

        assert "bureau_score_band_is_missing" in result.columns
        assert "repayment_history_score_is_missing" in result.columns


class TestFeaturePipeline:
    """Test the complete feature engineering pipeline."""

    def test_pipeline_removes_leakage(self, sample_borrower_df, sample_config_dir):
        """Full pipeline should remove all leakage columns."""
        pipeline = build_feature_pipeline(
            config_path=f"{sample_config_dir}/features.yaml",
            strict=False,
        )
        result = pipeline.fit_transform(sample_borrower_df)

        assert "collections_flag" not in result.columns
        assert "days_past_due_at_snapshot" not in result.columns
        assert "recovery_amount" not in result.columns
        assert "borrower_id" not in result.columns
        assert "default_flag" not in result.columns

    def test_no_nans_after_pipeline(self, sample_borrower_df, sample_config_dir):
        """All NaNs should be imputed by the pipeline."""
        pipeline = build_feature_pipeline(
            config_path=f"{sample_config_dir}/features.yaml",
            strict=False,
        )
        result = pipeline.fit_transform(sample_borrower_df)

        nan_count = result.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaNs after pipeline"

    def test_pipeline_serialization_roundtrip(self, sample_borrower_df, sample_config_dir, tmp_path):
        """Pipeline should produce same output after save/load via joblib."""
        pipeline = build_feature_pipeline(
            config_path=f"{sample_config_dir}/features.yaml",
            strict=False,
        )
        result_before = pipeline.fit_transform(sample_borrower_df)

        # Save and load
        path = tmp_path / "pipeline.joblib"
        joblib.dump(pipeline, path)
        loaded = joblib.load(path)

        result_after = loaded.transform(sample_borrower_df)

        pd.testing.assert_frame_equal(result_before, result_after)
