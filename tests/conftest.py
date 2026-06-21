"""Shared pytest fixtures for yogyank-scoring tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_config_dir(tmp_path):
    """Create temporary config directory with test configs."""
    import yaml

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Data generation config (small for tests)
    data_gen = {
        "n_borrowers": 500,
        "bad_rate": 0.10,
        "missing_bureau_rate": 0.30,
        "date_range": {"start": "2023-01-01", "end": "2024-12-31"},
        "leakage_traps": True,
        "random_seed": 42,
        "demographics": {
            "age_range": [18, 70],
            "gender_ratio": 0.55,
            "states": ["Telangana", "Andhra Pradesh", "Karnataka"],
            "education_levels": ["illiterate", "primary", "secondary", "higher_secondary", "graduate"],
            "occupation_types": ["farmer", "agricultural_laborer", "small_trader", "artisan", "other"],
        },
        "loan": {
            "amount_range": [5000, 500000],
            "tenure_months": [3, 6, 12, 18, 24],
            "purposes": ["crop_input", "equipment", "livestock", "consumption", "housing"],
        },
        "output_path": str(tmp_path / "data" / "raw" / "borrowers.csv"),
    }
    (config_dir / "data_generation.yaml").write_text(yaml.dump(data_gen))

    # Features config
    features = {
        "target_column": "default_flag",
        "date_column": "origination_date",
        "id_column": "borrower_id",
        "leakage_columns": ["collections_flag", "days_past_due_at_snapshot", "recovery_amount"],
        "drop_columns": ["borrower_id", "origination_date", "default_flag"],
        "bureau_features": ["bureau_score_band", "repayment_history_score"],
        "ordinal_features": {
            "education_level": ["illiterate", "primary", "secondary", "higher_secondary", "graduate"],
        },
        "high_cardinality_features": ["district"],
    }
    (config_dir / "features.yaml").write_text(yaml.dump(features))

    # Validation config
    validation = {
        "oot_cutoff_date": "2024-07-01",
        "n_folds": 3,
        "stratify_by": "default_flag",
        "min_oot_defaults": 5,
    }
    (config_dir / "validation.yaml").write_text(yaml.dump(validation))

    # Training config
    training = {
        "models": ["xgboost"],
        "optuna": {"n_trials": 3, "direction": "maximize", "metric": "roc_auc", "timeout_seconds": 60},
        "mlflow": {"experiment_name": "test-yogyank", "tracking_uri": str(tmp_path / "mlruns")},
    }
    (config_dir / "training.yaml").write_text(yaml.dump(training))

    # Packaging config
    packaging = {
        "model_name": "test-credit-scorer",
        "version_prefix": "v",
        "metadata": {
            "intended_use": "Test",
            "known_limitations": ["Test only"],
            "ethical_considerations": ["Test only"],
        },
        "output_dir": str(tmp_path / "models"),
        "model_card_path": str(tmp_path / "docs" / "model_card.md"),
    }
    (config_dir / "packaging.yaml").write_text(yaml.dump(packaging))

    return str(config_dir)


@pytest.fixture
def sample_borrower_df():
    """Create a small sample borrower DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame({
        "borrower_id": [f"BRW_{i:06d}" for i in range(n)],
        "age": rng.integers(18, 70, n),
        "gender": rng.choice(["M", "F"], n),
        "state": rng.choice(["Telangana", "AP"], n),
        "district": rng.choice(["T_D1", "T_D2", "AP_D1"], n),
        "land_acres": rng.exponential(2.5, n).round(1),
        "household_size": rng.integers(1, 10, n),
        "education_level": rng.choice(["illiterate", "primary", "secondary", "higher_secondary", "graduate"], n),
        "occupation_type": rng.choice(["farmer", "agricultural_laborer", "small_trader"], n),
        "existing_loan_count": rng.poisson(1.5, n),
        "repayment_history_score": rng.normal(60, 20, n).round(1),
        "bureau_score_band": rng.choice(["low", "medium", "high", None], n),
        "mobile_recharge_freq": rng.poisson(8, n),
        "mobile_recharge_value": rng.lognormal(4.5, 0.8, n).round(0),
        "utility_bill_payment_score": rng.beta(5, 2, n).round(2),
        "shg_membership": rng.choice(["yes", "no"], n),
        "kcc_usage_months": rng.integers(0, 60, n),
        "agri_input_purchases": rng.poisson(3, n),
        "seasonal_income_proxy": rng.lognormal(9.5, 0.6, n).round(0),
        "district_rainfall_index": rng.beta(4, 3, n).round(3),
        "crop_yield_index": rng.beta(4, 3, n).round(3),
        "market_distance_km": rng.lognormal(2.5, 0.7, n).round(1),
        "bank_branch_density": rng.exponential(3, n).round(1),
        "loan_amount": rng.lognormal(10, 1, n).round(0),
        "tenure_months": rng.choice([3, 6, 12, 18, 24], n),
        "loan_purpose": rng.choice(["crop_input", "equipment", "livestock"], n),
        "origination_date": pd.date_range("2023-01-01", "2024-12-31", periods=n).strftime("%Y-%m-%d"),
        "default_flag": rng.choice([0, 1], n, p=[0.9, 0.1]),
        # Leakage traps
        "collections_flag": rng.choice([0, 1], n),
        "days_past_due_at_snapshot": rng.choice([np.nan, 30, 60, 90], n),
        "recovery_amount": rng.choice([0, 1000, 5000], n).astype(float),
    })

    return df
