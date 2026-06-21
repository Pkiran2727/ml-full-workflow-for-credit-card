"""
Model Trainer
==============

Trains XGBoost and LightGBM models with Optuna hyperparameter optimization
and full MLflow tracking. Handles class imbalance via scale_pos_weight/is_unbalance.
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve

from .metrics import calculate_all_metrics, generate_decile_table

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_config(config_path: str | Path) -> dict:
    """Load training YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_xgboost_objective(X_train, y_train, cv_splitter, scale_pos_weight):
    """Create Optuna objective for XGBoost."""
    import xgboost as xgb

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }

        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_splitter, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    return objective


def _get_lightgbm_objective(X_train, y_train, cv_splitter):
    """Create Optuna objective for LightGBM."""
    import lightgbm as lgb

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance": True,
            "random_state": 42,
            "verbose": -1,
        }

        model = lgb.LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_splitter, scoring="roc_auc", n_jobs=-1
            )
        return scores.mean()

    return objective


def _plot_roc_curve(y_true, y_prob, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_pr_curve(y_true, y_prob, save_path):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, "r-", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(model, feature_names, save_path, top_n=20):
    """Plot and save feature importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color="steelblue")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def train_models(X_train: pd.DataFrame | np.ndarray,
                 y_train: np.ndarray,
                 X_oot: pd.DataFrame | np.ndarray,
                 y_oot: np.ndarray,
                 feature_names: list[str] | None = None,
                 config_path: str | Path = "configs/training.yaml") -> dict:
    """
    Train all configured models with Optuna tuning and MLflow tracking.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_oot: OOT test features.
        y_oot: OOT test labels.
        feature_names: List of feature names for importance plots.
        config_path: Path to training config.

    Returns:
        Dict with model results: {model_name: {model, params, train_metrics, oot_metrics, run_id}}.
    """
    config = _load_config(config_path)
    mlflow_config = config["mlflow"]
    optuna_config = config["optuna"]

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # Class imbalance weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    logger.info(f"Class imbalance: {n_neg} negatives, {n_pos} positives, "
                f"scale_pos_weight={scale_pos_weight:.2f}")

    # CV splitter
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    results = {}
    models_to_train = config.get("models", ["xgboost", "lightgbm"])

    for model_name in models_to_train:
        logger.info(f"\n{'='*60}\nTraining {model_name.upper()}\n{'='*60}")

        with mlflow.start_run(run_name=f"{model_name}_optuna") as run:
            # Optuna hyperparameter search
            if model_name == "xgboost":
                import xgboost as xgb
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    _get_xgboost_objective(X_train, y_train, cv_splitter, scale_pos_weight),
                    n_trials=optuna_config["n_trials"],
                    timeout=optuna_config.get("timeout_seconds", 600),
                )
                best_params = study.best_params
                best_params.update({
                    "scale_pos_weight": scale_pos_weight,
                    "eval_metric": "logloss",
                    "random_state": 42,
                    "verbosity": 0,
                })
                best_model = xgb.XGBClassifier(**best_params)

            elif model_name == "lightgbm":
                import lightgbm as lgb
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    _get_lightgbm_objective(X_train, y_train, cv_splitter),
                    n_trials=optuna_config["n_trials"],
                    timeout=optuna_config.get("timeout_seconds", 600),
                )
                best_params = study.best_params
                best_params.update({
                    "is_unbalance": True,
                    "random_state": 42,
                    "verbose": -1,
                })
                best_model = lgb.LGBMClassifier(**best_params)

            else:
                logger.warning(f"Unknown model type: {model_name}, skipping")
                continue

            # Train best model on full training set
            logger.info(f"Best CV AUC: {study.best_value:.4f}")
            logger.info(f"Best params: {best_params}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_model.fit(X_train, y_train)

            # Predict on train and OOT
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            y_oot_prob = best_model.predict_proba(X_oot)[:, 1]

            # Calculate metrics
            train_metrics = calculate_all_metrics(y_train, y_train_prob)
            oot_metrics = calculate_all_metrics(y_oot, y_oot_prob)

            logger.info(f"Train metrics: AUC={train_metrics['roc_auc']:.4f}, "
                         f"KS={train_metrics['ks_statistic']:.4f}, "
                         f"Gini={train_metrics['gini']:.4f}")
            logger.info(f"OOT metrics:   AUC={oot_metrics['roc_auc']:.4f}, "
                         f"KS={oot_metrics['ks_statistic']:.4f}, "
                         f"Gini={oot_metrics['gini']:.4f}")

            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_param("best_cv_auc", study.best_value)
            mlflow.log_param("n_optuna_trials", optuna_config["n_trials"])

            for prefix, metrics in [("train", train_metrics), ("oot", oot_metrics)]:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{prefix}_{k}", v)

            # Generate and log plots
            plots_dir = Path("plots") / model_name
            plots_dir.mkdir(parents=True, exist_ok=True)

            _plot_roc_curve(y_oot, y_oot_prob, plots_dir / "roc_curve_oot.png")
            _plot_pr_curve(y_oot, y_oot_prob, plots_dir / "pr_curve_oot.png")
            _plot_feature_importance(best_model, feature_names,
                                     plots_dir / "feature_importance.png")

            mlflow.log_artifact(str(plots_dir / "roc_curve_oot.png"))
            mlflow.log_artifact(str(plots_dir / "pr_curve_oot.png"))
            if (plots_dir / "feature_importance.png").exists():
                mlflow.log_artifact(str(plots_dir / "feature_importance.png"))

            # Log decile table
            decile_table = generate_decile_table(y_oot, y_oot_prob)
            decile_path = plots_dir / "decile_table_oot.csv"
            decile_table.to_csv(decile_path, index=False)
            mlflow.log_artifact(str(decile_path))

            # Log model
            mlflow.sklearn.log_model(
                best_model,
                artifact_path=f"{model_name}_model",
                serialization_format="cloudpickle",
            )

            results[model_name] = {
                "model": best_model,
                "params": best_params,
                "train_metrics": train_metrics,
                "oot_metrics": oot_metrics,
                "run_id": run.info.run_id,
                "cv_auc": study.best_value,
                "y_oot_prob": y_oot_prob,
            }

    # Select best model by OOT AUC
    if results:
        best_name = max(results, key=lambda k: results[k]["oot_metrics"]["roc_auc"])
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL: {best_name.upper()} "
                     f"(OOT AUC={results[best_name]['oot_metrics']['roc_auc']:.4f})")
        logger.info(f"{'='*60}")
        results["_best"] = best_name

    return results
