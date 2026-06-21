# Model Card: yogyank-credit-scorer

## Model Description

**Type**: Binary classification (credit default prediction)
**Target Population**: Rural borrowers in India � microfinance and priority sector lending
**Generated**: 2026-06-20 21:41

## Intended Use

Rural credit scoring for Indian borrowers — microfinance and priority sector lending

## Training Data

- **Source**: Synthetic data generator (see `src/data_generation/`)
- **Size**: 18760 training records
- **Bad Rate**: 0.09930703624733475
- **Features**: 30 features

## Validation Methodology

- **Primary**: Out-of-Time (OOT) holdout � train on earlier loans, test on later loans
- **Secondary**: 5-fold stratified cross-validation within training set

## Performance Metrics

| Metric | Train | OOT |
|--------|-------|-----|
| AUC-ROC | 0.8430 | 0.8026 |
| KS Statistic | 0.5213 | 0.4568 |
| Gini | 0.6861 | 0.6051 |
| PR-AUC | 0.3935 | 0.3209 |

## Features Used

- age
- gender
- state
- district
- land_acres
- household_size
- education_level
- occupation_type
- existing_loan_count
- repayment_history_score
- bureau_score_band
- mobile_recharge_freq
- mobile_recharge_value
- utility_bill_payment_score
- shg_membership
- kcc_usage_months
- agri_input_purchases
- seasonal_income_proxy
- district_rainfall_index
- crop_yield_index
- market_distance_km
- bank_branch_density
- loan_amount
- tenure_months
- loan_purpose
- bureau_score_band_is_missing
- repayment_history_score_is_missing
- is_farmer
- rainfall_x_farmer
- bureau_x_loans

## Known Limitations

- Trained on synthetic data — requires retraining on real borrower data before production use
- Bureau features have high missingness by design (thin-file population)
- Geospatial features are proxy-based, not actual GPS/satellite data

## Ethical Considerations

- Gender and occupation type are included — monitor for disparate impact
- Model should not be sole decision-maker — human-in-the-loop required

## Model Governance

- This model should be reviewed and revalidated at minimum every 12 months
- Drift monitoring (PSI/CSI) should be active in production
- Human-in-the-loop required for all credit decisions
