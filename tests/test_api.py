"""Tests for FastAPI scoring API."""

import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Test API endpoints (without a trained model loaded)."""

    def test_health_endpoint(self):
        """GET /health should return 200."""
        from src.api.app import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_score_without_model_returns_503(self):
        """POST /score without model should return 503."""
        from src.api.app import app
        client = TestClient(app)
        response = client.post("/score", json={
            "age": 35, "gender": "M", "state": "Telangana", "district": "T_D1",
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
        })
        assert response.status_code == 503

    def test_invalid_input_returns_422(self):
        """Invalid input should return 422 validation error."""
        from src.api.app import app
        client = TestClient(app)
        response = client.post("/score", json={"age": -5})  # Invalid
        assert response.status_code == 422

    def test_score_with_model_loaded(self):
        """POST /score with model loaded should return 200 and valid scoring response."""
        from src.api.app import app
        # Use context manager to trigger lifespan and load the model
        with TestClient(app) as client:
            response = client.post("/score", json={
                "age": 35, "gender": "M", "state": "Telangana", "district": "T_D1",
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
            })
            assert response.status_code == 200
            data = response.json()
            assert "score" in data
            assert 0 <= data["score"] <= 1
            assert "risk_band" in data
            assert "reason_codes" in data
            assert len(data["reason_codes"]) <= 3
            assert "model_version" in data
            assert data["model_version"] == "v1.0"

