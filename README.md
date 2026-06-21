---
title: Yogyank Rural Credit Scorer
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# yogyank-scoring 🏦

Production-grade ML pipeline for **rural credit scoring** in India. Covers the full lifecycle from synthetic data generation to model serving.

## Quick Start

```powershell
# 1. Create virtual environment (Python 3.12 required)
C:\Python312\python.exe -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python pipelines/run_pipeline.py --stage all

# 4. Start scoring API
python pipelines/run_pipeline.py --stage serve
```

## Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| A. Generate | `--stage generate` | Create 25K synthetic borrowers with leakage traps |
| B+C. Validate | `--stage validate` | OOT train/test split (cutoff: 2024-07-01) |
| D+E. Train | `--stage train` | XGBoost + LightGBM with Optuna, MLflow tracking, package best |
| F. Explain | `--stage explain` | SHAP global/local explanations + reason codes |
| G. Monitor | `--stage monitor` | PSI/CSI drift detection + fairness slicing |
| H. Serve | `--stage serve` | FastAPI at http://localhost:8000 |
| All | `--stage all` | Run generate → validate → train → explain → monitor |

## API Endpoints

```bash
# Score a borrower
POST http://localhost:8000/score
Content-Type: application/json

{
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
}

# Response
{
  "score": 0.0823,
  "risk_band": "low_risk",
  "reason_codes": [
    {"feature": "bureau_score_band", "shap_value": -0.15, "direction": "decreases_risk"},
    {"feature": "existing_loan_count", "shap_value": 0.08, "direction": "increases_risk"},
    {"feature": "district_rainfall_index", "shap_value": -0.06, "direction": "decreases_risk"}
  ],
  "model_version": "v1.0"
}

# Health check
GET http://localhost:8000/health

# Model info
GET http://localhost:8000/model-info
```

## Project Structure

```
yogyank-scoring/
├── configs/                  # YAML configs per pipeline stage
├── src/
│   ├── data_generation/      # Synthetic data generator (with leakage traps)
│   ├── features/             # Leakage-safe feature transformers + pipeline
│   ├── validation/           # OOT + stratified k-fold splitters
│   ├── training/             # Model training, Optuna HPO, credit metrics
│   ├── packaging/            # Model artifact bundling + registry
│   ├── explainability/       # SHAP explanations + reason codes
│   ├── monitoring/           # PSI/CSI drift + fairness slicing
│   └── api/                  # FastAPI scoring service
├── pipelines/                # CLI orchestrator (replaces Makefile)
├── tests/                    # pytest test suite
├── docker/                   # Dockerfile + docker-compose
├── .github/workflows/        # CI/CD
├── docs/                     # Model card, SHAP plots, monitoring reports
├── data/                     # raw → processed data (gitignored)
├── models/                   # Packaged model artifacts (gitignored)
└── mlruns/                   # MLflow tracking (gitignored)
```

## Key Design Decisions

- **Python 3.12** — stable ML library compatibility (intentionally avoids 3.13+)
- **No CatBoost** — Python wheel availability issues; XGBoost + LightGBM only
- **LeakageGuard** — hard guardrail transformer with strict mode (raises on leakage detection)
- **OOT validation** — mirrors regulated credit shop methodology
- **SHAP reason codes** — for adverse action notices (regulatory requirement)
- **PSI/CSI monitoring** — built from scratch (no heavy dependencies)

## Tests

```powershell
pytest tests/ -v --tb=short
```

## Docker

```powershell
# Build and run
docker compose -f docker/docker-compose.yaml up

# With MLflow UI
docker compose -f docker/docker-compose.yaml --profile monitoring up
```

## MLflow

```powershell
# View experiment tracking UI
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```
