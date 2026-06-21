"""
Pydantic Schemas for API Request/Response Validation
=====================================================

Defines strict input validation for scoring requests and structured
output format including score, risk band, and reason codes.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ScoringRequest(BaseModel):
    """Borrower features for credit scoring."""

    # Demographics
    age: int = Field(..., ge=18, le=100, description="Borrower age")
    gender: str = Field(..., pattern="^(M|F)$", description="Gender (M/F)")
    state: str = Field(..., description="Indian state")
    district: str = Field(..., description="District within state")
    land_acres: float = Field(..., ge=0, le=100, description="Land owned in acres")
    household_size: int = Field(..., ge=1, le=20, description="Number in household")
    education_level: str = Field(
        ..., description="Education level",
        pattern="^(illiterate|primary|secondary|higher_secondary|graduate)$"
    )
    occupation_type: str = Field(
        ..., description="Occupation type",
        pattern="^(farmer|agricultural_laborer|small_trader|artisan|other)$"
    )

    # Credit bureau
    existing_loan_count: int = Field(..., ge=0, le=20, description="Number of existing loans")
    repayment_history_score: Optional[float] = Field(
        None, ge=0, le=100, description="Repayment history score (null for thin-file)"
    )
    bureau_score_band: Optional[str] = Field(
        None, description="Bureau score band (null for thin-file)"
    )

    # Alternative data
    mobile_recharge_freq: int = Field(..., ge=0, description="Monthly recharge frequency")
    mobile_recharge_value: float = Field(..., ge=0, description="Monthly recharge value (INR)")
    utility_bill_payment_score: float = Field(
        ..., ge=0, le=1, description="Utility payment score (0-1)"
    )
    shg_membership: str = Field(..., pattern="^(yes|no)$", description="SHG membership")
    kcc_usage_months: int = Field(..., ge=0, le=120, description="KCC usage in months")
    agri_input_purchases: int = Field(..., ge=0, description="Agri input purchase count")
    seasonal_income_proxy: float = Field(..., ge=0, description="Seasonal income proxy (INR)")

    # Geospatial
    district_rainfall_index: float = Field(..., ge=0, le=1, description="Rainfall index (0-1)")
    crop_yield_index: float = Field(..., ge=0, le=1, description="Crop yield index (0-1)")
    market_distance_km: float = Field(..., ge=0, description="Distance to market (km)")
    bank_branch_density: float = Field(..., ge=0, description="Bank branch density")

    # Loan
    loan_amount: float = Field(..., ge=1000, description="Loan amount (INR)")
    tenure_months: int = Field(..., ge=1, le=60, description="Loan tenure in months")
    loan_purpose: str = Field(
        ..., description="Loan purpose",
        pattern="^(crop_input|equipment|livestock|consumption|housing)$"
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "age": 35, "gender": "M", "state": "Telangana", "district": "Telangana_D3",
            "land_acres": 2.5, "household_size": 4, "education_level": "secondary",
            "occupation_type": "farmer", "existing_loan_count": 1,
            "repayment_history_score": 65.0, "bureau_score_band": "medium",
            "mobile_recharge_freq": 8, "mobile_recharge_value": 200,
            "utility_bill_payment_score": 0.75, "shg_membership": "no",
            "kcc_usage_months": 24, "agri_input_purchases": 3,
            "seasonal_income_proxy": 15000, "district_rainfall_index": 0.65,
            "crop_yield_index": 0.7, "market_distance_km": 12.5,
            "bank_branch_density": 3.2, "loan_amount": 50000,
            "tenure_months": 12, "loan_purpose": "crop_input"
        }]
    }}


class ReasonCode(BaseModel):
    """Single reason code explaining a score component."""
    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP contribution value")
    direction: str = Field(..., description="increases_risk or decreases_risk")
    feature_value: Optional[float] = Field(None, description="Actual feature value")


class ScoringResponse(BaseModel):
    """Credit scoring response with score, risk band, and reason codes."""
    score: float = Field(..., ge=0, le=1, description="Default probability (0-1)")
    risk_band: str = Field(..., description="Risk classification band")
    reason_codes: list[ReasonCode] = Field(
        ..., description="Top reason codes driving the score"
    )
    model_version: str = Field(..., description="Model version used for scoring")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    model_version: str = Field(default="unknown")


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    version: str
    model_type: str
    training_date: str
    oot_auc: Optional[float] = None
    oot_ks: Optional[float] = None
    oot_gini: Optional[float] = None
    n_features: int
