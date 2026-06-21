"""Tests for synthetic data generation."""

import pandas as pd
import pytest
from src.data_generation import generate_synthetic_data


class TestDataGeneration:
    """Test suite for synthetic data generator."""

    def test_generates_correct_shape(self, sample_config_dir, tmp_path):
        """Should generate the configured number of borrowers."""
        df = generate_synthetic_data(
            config_path=f"{sample_config_dir}/data_generation.yaml",
            output_path=str(tmp_path / "test.csv"),
        )
        assert len(df) == 500  # from test config
        assert "borrower_id" in df.columns

    def test_bad_rate_within_range(self, sample_config_dir, tmp_path):
        """Bad rate should be within ±3% of configured rate."""
        df = generate_synthetic_data(
            config_path=f"{sample_config_dir}/data_generation.yaml",
            output_path=str(tmp_path / "test.csv"),
        )
        actual_rate = df["default_flag"].mean()
        assert 0.07 <= actual_rate <= 0.13, f"Bad rate {actual_rate:.3f} outside ±3% of 0.10"

    def test_leakage_traps_present(self, sample_config_dir, tmp_path):
        """Leakage trap columns should be present when configured."""
        df = generate_synthetic_data(
            config_path=f"{sample_config_dir}/data_generation.yaml",
            output_path=str(tmp_path / "test.csv"),
        )
        assert "collections_flag" in df.columns
        assert "days_past_due_at_snapshot" in df.columns
        assert "recovery_amount" in df.columns

    def test_leakage_traps_correlate_with_target(self, sample_config_dir, tmp_path):
        """Leakage traps should be correlated with default_flag (that's the point)."""
        df = generate_synthetic_data(
            config_path=f"{sample_config_dir}/data_generation.yaml",
            output_path=str(tmp_path / "test.csv"),
        )
        # collections_flag should be mostly set for defaulted loans
        default_collections = df[df["default_flag"] == 1]["collections_flag"].mean()
        non_default_collections = df[df["default_flag"] == 0]["collections_flag"].mean()
        assert default_collections > non_default_collections

    def test_date_range_valid(self, sample_config_dir, tmp_path):
        """Origination dates should be within configured range."""
        df = generate_synthetic_data(
            config_path=f"{sample_config_dir}/data_generation.yaml",
            output_path=str(tmp_path / "test.csv"),
        )
        dates = pd.to_datetime(df["origination_date"])
        assert dates.min() >= pd.Timestamp("2023-01-01")
        assert dates.max() <= pd.Timestamp("2024-12-31")

    def test_bureau_missingness_rate(self, sample_config_dir, tmp_path):
        """Bureau score should have ~30% missing (thin-file borrowers)."""
        df = generate_synthetic_data(
            config_path=f"{sample_config_dir}/data_generation.yaml",
            output_path=str(tmp_path / "test.csv"),
        )
        missing_rate = df["bureau_score_band"].isna().mean()
        assert 0.20 <= missing_rate <= 0.40, f"Bureau missing rate {missing_rate:.3f} outside 20-40%"

    def test_output_csv_saved(self, sample_config_dir, tmp_path):
        """CSV should be saved to configured path."""
        output = tmp_path / "output.csv"
        generate_synthetic_data(
            config_path=f"{sample_config_dir}/data_generation.yaml",
            output_path=str(output),
        )
        assert output.exists()
        df = pd.read_csv(output)
        assert len(df) == 500
