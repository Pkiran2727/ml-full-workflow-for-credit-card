"""Tests for model training."""

import numpy as np

from src.training.metrics import calculate_ks_statistic, calculate_gini, calculate_all_metrics


class TestMetrics:
    """Test credit scoring metrics."""

    def test_ks_statistic_range(self):
        """KS should be between 0 and 1."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], 1000, p=[0.9, 0.1])
        y_prob = rng.random(1000)
        ks = calculate_ks_statistic(y_true, y_prob)
        assert 0 <= ks <= 1

    def test_gini_range(self):
        """Gini should be between -1 and 1."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], 1000, p=[0.9, 0.1])
        y_prob = rng.random(1000)
        gini = calculate_gini(y_true, y_prob)
        assert -1 <= gini <= 1

    def test_perfect_predictions(self):
        """Perfect predictions should give AUC=1, Gini=1."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        metrics = calculate_all_metrics(y_true, y_prob)
        assert metrics["roc_auc"] == 1.0
        assert metrics["gini"] == 1.0

    def test_all_metrics_keys(self):
        """calculate_all_metrics should return all expected keys."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], 100, p=[0.9, 0.1])
        y_prob = rng.random(100)
        metrics = calculate_all_metrics(y_true, y_prob)

        expected_keys = {"roc_auc", "ks_statistic", "gini", "pr_auc",
                         "log_loss", "recall_positive", "n_total", "n_positive",
                         "n_negative", "bad_rate"}
        assert expected_keys.issubset(set(metrics.keys()))


class TestModelTraining:
    """Test that models train and produce non-trivial results."""

    def test_xgboost_trains_and_nontrivial(self, sample_borrower_df, sample_config_dir):
        """XGBoost should train and produce non-trivial predictions (not all-zero)."""
        import xgboost as xgb
        from src.features.pipeline import build_feature_pipeline

        pipeline = build_feature_pipeline(
            config_path=f"{sample_config_dir}/features.yaml", strict=False
        )
        X = pipeline.fit_transform(sample_borrower_df)
        y = sample_borrower_df["default_flag"].values

        n_neg = (y == 0).sum()
        n_pos = max((y == 1).sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=3, scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss", random_state=42, verbosity=0
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # NON-TRIVIAL CLASSIFIER CHECK: recall on positive class > 0
        from sklearn.metrics import recall_score
        recall_pos = recall_score(y, y_pred, pos_label=1, zero_division=0)
        assert recall_pos > 0, (
            "Model is trivial (predicts all zeros). "
            "Check scale_pos_weight / class imbalance handling."
        )

        # Metrics should be in valid range
        assert 0 <= y_prob.mean() <= 1
