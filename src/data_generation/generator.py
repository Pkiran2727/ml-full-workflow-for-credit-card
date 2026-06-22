"""
Synthetic Data Generator for Rural Credit Scoring
===================================================

Generates realistic borrower data with:
- Correlated feature relationships (bureau score ↔ default, rainfall ↔ agri-default)
- Configurable bad rate, missingness, and date range
- Intentional leakage trap columns for pipeline testing

Leakage Traps:
    - collections_flag: set AFTER default event (post-event leakage)
    - days_past_due_at_snapshot: only populated for defaulted loans (label leakage)
    - recovery_amount: only non-zero after default
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path) -> dict:
    """Load data generation YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _generate_demographics(n: int, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate borrower demographic features."""
    demo = config["demographics"]

    age = rng.integers(demo["age_range"][0], demo["age_range"][1] + 1, size=n)
    gender = rng.choice(["M", "F"], size=n, p=[demo["gender_ratio"], 1 - demo["gender_ratio"]])
    state = rng.choice(demo["states"], size=n)

    # Districts — generate 5-8 districts per state for realistic cardinality
    districts = {}
    for s in demo["states"]:
        n_districts = rng.integers(5, 9)
        districts[s] = [f"{s}_D{i+1}" for i in range(n_districts)]
    district = [rng.choice(districts[s]) for s in state]

    land_acres = np.round(rng.exponential(scale=2.5, size=n), 1)
    land_acres = np.clip(land_acres, 0, 50)

    household_size = rng.integers(1, 12, size=n)

    # Education — weighted towards lower levels for rural population
    edu_weights = [0.15, 0.30, 0.25, 0.20, 0.10]
    education_level = rng.choice(demo["education_levels"], size=n, p=edu_weights)

    # Occupation — weighted towards agriculture
    occ_weights = [0.40, 0.25, 0.15, 0.10, 0.10]
    occupation_type = rng.choice(demo["occupation_types"], size=n, p=occ_weights)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "state": state,
        "district": district,
        "land_acres": land_acres,
        "household_size": household_size,
        "education_level": education_level,
        "occupation_type": occupation_type,
    })


def _generate_credit_bureau(n: int, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate credit bureau features with realistic missingness for thin-file borrowers."""
    missing_rate = config["missing_bureau_rate"]

    # Bureau score band — ordinal categories
    bands = ["no_hit", "very_low", "low", "medium", "high", "very_high"]
    band_weights = [0.05, 0.10, 0.20, 0.30, 0.25, 0.10]
    bureau_score_band = rng.choice(bands, size=n, p=band_weights)

    # Make ~30% of bureau scores missing (thin-file borrowers)
    missing_mask = rng.random(n) < missing_rate
    bureau_score_band = pd.array(bureau_score_band, dtype="object")
    bureau_score_band[missing_mask] = None

    # Repayment history score (0-100, higher = better)
    repayment_history_score = rng.normal(loc=60, scale=20, size=n).astype(float)
    repayment_history_score = np.clip(repayment_history_score, 0, 100).round(1)
    # Also missing for thin-file borrowers
    repayment_history_score[missing_mask] = np.nan

    existing_loan_count = rng.poisson(lam=1.5, size=n)
    existing_loan_count = np.clip(existing_loan_count, 0, 10)

    return pd.DataFrame({
        "existing_loan_count": existing_loan_count,
        "repayment_history_score": repayment_history_score,
        "bureau_score_band": bureau_score_band,
    })


def _generate_alternative_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate alternative data features (mobile, utility, SHG, KCC, agri)."""
    mobile_recharge_freq = rng.poisson(lam=8, size=n)  # recharges per month
    mobile_recharge_value = rng.lognormal(mean=4.5, sigma=0.8, size=n).round(0)  # INR
    mobile_recharge_value = np.clip(mobile_recharge_value, 10, 5000)

    utility_bill_payment_score = rng.beta(a=5, b=2, size=n).round(2)  # 0-1, higher = better

    shg_membership = rng.choice(["yes", "no"], size=n, p=[0.35, 0.65])

    kcc_usage_months = rng.integers(0, 61, size=n)  # 0-60 months

    agri_input_purchases = rng.poisson(lam=3, size=n)  # count per season

    # Seasonal income proxy — based on crop cycle (higher in harvest months)
    seasonal_income_proxy = rng.lognormal(mean=9.5, sigma=0.6, size=n).round(0)

    return pd.DataFrame({
        "mobile_recharge_freq": mobile_recharge_freq,
        "mobile_recharge_value": mobile_recharge_value,
        "utility_bill_payment_score": utility_bill_payment_score,
        "shg_membership": shg_membership,
        "kcc_usage_months": kcc_usage_months,
        "agri_input_purchases": agri_input_purchases,
        "seasonal_income_proxy": seasonal_income_proxy,
    })


def _generate_geospatial(n: int, states: list, state_col: np.ndarray,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Generate geospatial/contextual features."""
    # Rainfall index by state (some states are drought-prone)
    state_rainfall_base = {s: rng.uniform(0.3, 1.0) for s in states}
    district_rainfall_index = np.array([
        state_rainfall_base[s] + rng.normal(0, 0.1) for s in state_col
    ])
    district_rainfall_index = np.clip(district_rainfall_index, 0, 1).round(3)

    crop_yield_index = rng.beta(a=4, b=3, size=n).round(3)

    market_distance_km = rng.lognormal(mean=2.5, sigma=0.7, size=n).round(1)
    market_distance_km = np.clip(market_distance_km, 1, 100)

    bank_branch_density = rng.exponential(scale=3, size=n).round(1)
    bank_branch_density = np.clip(bank_branch_density, 0.1, 20)

    return pd.DataFrame({
        "district_rainfall_index": district_rainfall_index,
        "crop_yield_index": crop_yield_index,
        "market_distance_km": market_distance_km,
        "bank_branch_density": bank_branch_density,
    })


def _generate_loan_features(n: int, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate loan-specific features."""
    loan_cfg = config["loan"]

    loan_amount = rng.lognormal(mean=10, sigma=1, size=n).round(0)
    loan_amount = np.clip(loan_amount, loan_cfg["amount_range"][0],
                          loan_cfg["amount_range"][1])

    tenure_months = rng.choice(loan_cfg["tenure_months"], size=n)
    loan_purpose = rng.choice(loan_cfg["purposes"], size=n)

    return pd.DataFrame({
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "loan_purpose": loan_purpose,
    })


def _generate_origination_dates(n: int, config: dict,
                                 rng: np.random.Generator) -> pd.Series:
    """Generate origination dates spanning the configured date range."""
    start = pd.Timestamp(config["date_range"]["start"])
    end = pd.Timestamp(config["date_range"]["end"])

    # Uniform distribution across the date range
    days_range = (end - start).days
    random_days = rng.integers(0, days_range + 1, size=n)
    dates = start + pd.to_timedelta(random_days, unit="D")

    return pd.Series(dates, name="origination_date")


def _generate_target(df: pd.DataFrame, config: dict,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Generate default_flag with correlated probabilities.

    Default probability is driven by a logistic model:
    - Lower bureau score → higher default
    - More existing loans → higher default
    - Low rainfall × farmer → higher default
    - Lower repayment history → higher default
    - Younger age → slightly higher default

    Noise is injected to prevent trivially separable data.
    """
    target_bad_rate = config["bad_rate"]
    n = len(df)

    # Encode bureau score band to numeric (handle missing)
    bureau_map = {"no_hit": 0, "very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}
    bureau_numeric = df["bureau_score_band"].map(bureau_map).fillna(1.5).values.astype(float)

    # Normalize features to 0-1 range for logit calculation
    def _normalize(x):
        x = np.asarray(x, dtype=float)
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if x_max - x_min == 0:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    bureau_norm = _normalize(bureau_numeric)  # 0=bad, 1=good
    repayment_norm = _normalize(
        df["repayment_history_score"].fillna(df["repayment_history_score"].median())
    )
    existing_loans_norm = _normalize(df["existing_loan_count"])
    age_norm = _normalize(df["age"])
    rainfall_norm = _normalize(df["district_rainfall_index"])
    is_farmer = (df["occupation_type"] == "farmer").astype(float).values

    # Logit score: higher = more likely to default
    logit = (
        -2.0 * bureau_norm          # lower bureau → higher default
        - 1.5 * repayment_norm      # lower repayment history → higher default
        + 1.0 * existing_loans_norm  # more loans → higher default
        - 0.5 * age_norm             # younger → slightly higher default
        - 0.8 * rainfall_norm * is_farmer  # low rainfall + farmer → higher default
        + 0.3 * _normalize(df["loan_amount"])  # larger loan → slightly higher default
    )

    # Add noise to prevent perfect separability
    logit += rng.normal(0, 0.8, size=n)

    # Convert to probability via sigmoid
    # Convert to probability via sigmoid (removed unused variable)

    # Calibrate to target bad rate by adjusting the intercept
    # Binary search for the right intercept
    def _get_bad_rate(intercept):
        p = 1 / (1 + np.exp(-(logit + intercept)))
        return (p > 0.5).mean()

    low, high = -10, 10
    for _ in range(50):
        mid = (low + high) / 2
        if _get_bad_rate(mid) < target_bad_rate:
            low = mid
        else:
            high = mid

    calibrated_prob = 1 / (1 + np.exp(-(logit + (low + high) / 2)))

    # Final binary target using calibrated threshold at 0.5
    default_flag = (calibrated_prob > 0.5).astype(int)

    actual_rate = default_flag.mean()
    logger.info(f"Generated default rate: {actual_rate:.3f} (target: {target_bad_rate:.3f})")

    # If rate is still off, randomly flip some labels to get closer
    if abs(actual_rate - target_bad_rate) > 0.02:
        n_target = int(n * target_bad_rate)
        n_current = default_flag.sum()
        if n_current < n_target:
            # Need more defaults — flip some non-defaults with highest probability
            non_default_idx = np.where(default_flag == 0)[0]
            probs_non_default = calibrated_prob[non_default_idx]
            n_flip = n_target - n_current
            flip_idx = non_default_idx[np.argsort(-probs_non_default)[:n_flip]]
            default_flag[flip_idx] = 1
        elif n_current > n_target:
            # Too many defaults — flip some defaults with lowest probability
            default_idx = np.where(default_flag == 1)[0]
            probs_default = calibrated_prob[default_idx]
            n_flip = n_current - n_target
            flip_idx = default_idx[np.argsort(probs_default)[:n_flip]]
            default_flag[flip_idx] = 0

        logger.info(f"Adjusted default rate: {default_flag.mean():.3f}")

    return default_flag


def _generate_leakage_traps(default_flag: np.ndarray,
                             rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate intentional leakage trap columns.

    These columns are correlated with the target BECAUSE they are set AFTER
    the default event. A leakage-safe pipeline must detect and drop them.

    - collections_flag: set only after default (post-event)
    - days_past_due_at_snapshot: only populated for defaulted loans
    - recovery_amount: only non-zero after default
    """
    n = len(default_flag)

    # collections_flag — set after default event
    collections_flag = np.zeros(n, dtype=int)
    default_mask = default_flag == 1
    # ~85% of defaulted loans have collections flag set
    collections_flag[default_mask] = rng.choice([0, 1], size=default_mask.sum(), p=[0.15, 0.85])

    # days_past_due_at_snapshot — only for defaulted loans
    days_past_due = np.full(n, np.nan)
    days_past_due[default_mask] = rng.integers(30, 181, size=default_mask.sum())

    # recovery_amount — only non-zero after default
    recovery_amount = np.zeros(n, dtype=float)
    recovery_amount[default_mask] = rng.lognormal(
        mean=8, sigma=1, size=default_mask.sum()
    ).round(0)

    return pd.DataFrame({
        "collections_flag": collections_flag,
        "days_past_due_at_snapshot": days_past_due,
        "recovery_amount": recovery_amount,
    })


def generate_synthetic_data(config_path: str | Path = "configs/data_generation.yaml",
                             output_path: str | Path | None = None) -> pd.DataFrame:
    """
    Generate synthetic rural credit scoring dataset.

    Args:
        config_path: Path to data generation YAML config.
        output_path: Override output CSV path (default: from config).

    Returns:
        pd.DataFrame: Generated borrower dataset with leakage traps.
    """
    config = _load_config(config_path)
    n = config["n_borrowers"]
    seed = config.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    logger.info(f"Generating {n} synthetic borrowers (seed={seed})")

    # Generate feature groups
    demographics = _generate_demographics(n, config, rng)
    credit_bureau = _generate_credit_bureau(n, config, rng)
    alt_data = _generate_alternative_data(n, rng)
    geospatial = _generate_geospatial(
        n, config["demographics"]["states"], demographics["state"].values, rng
    )
    loan_features = _generate_loan_features(n, config, rng)
    origination_dates = _generate_origination_dates(n, config, rng)

    # Combine all features
    df = pd.concat([demographics, credit_bureau, alt_data, geospatial, loan_features], axis=1)
    df["origination_date"] = origination_dates.values

    # Generate correlated target
    default_flag = _generate_target(df, config, rng)
    df["default_flag"] = default_flag

    # Generate leakage traps
    if config.get("leakage_traps", True):
        leakage_traps = _generate_leakage_traps(default_flag, rng)
        df = pd.concat([df, leakage_traps], axis=1)
        logger.info("Leakage trap columns added: collections_flag, "
                     "days_past_due_at_snapshot, recovery_amount")

    # Add borrower ID
    df.insert(0, "borrower_id", [f"BRW_{i:06d}" for i in range(n)])

    # Save to CSV
    save_path = Path(output_path or config.get("output_path", "data/raw/borrowers.csv"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved {len(df)} records to {save_path}")

    # Log summary statistics
    logger.info(f"  Default rate: {df['default_flag'].mean():.3f}")
    logger.info(f"  Bureau missing rate: {df['bureau_score_band'].isna().mean():.3f}")
    logger.info(f"  Date range: {df['origination_date'].min()} to {df['origination_date'].max()}")
    logger.info(f"  Columns: {list(df.columns)}")

    return df
