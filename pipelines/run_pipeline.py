"""
Pipeline Orchestrator
======================

CLI-driven orchestration of all pipeline stages.
Replaces Makefile — runs directly on Windows with no extra tooling.

Usage:
    python pipelines/run_pipeline.py --stage generate
    python pipelines/run_pipeline.py --stage features
    python pipelines/run_pipeline.py --stage train
    python pipelines/run_pipeline.py --stage all
    python pipelines/run_pipeline.py --stage serve
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Disable MLflow file store maintenance mode check
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def stage_generate(config_dir: str):
    """Stage A: Generate synthetic data."""
    logger.info("=" * 60)
    logger.info("STAGE A: Synthetic Data Generation")
    logger.info("=" * 60)

    from src.data_generation import generate_synthetic_data
    df = generate_synthetic_data(config_path=f"{config_dir}/data_generation.yaml")
    logger.info(f"Generated {len(df)} borrowers → data/raw/borrowers.csv")
    return df


def stage_features(config_dir: str):
    """Stage B: Feature engineering."""
    logger.info("=" * 60)
    logger.info("STAGE B: Feature Engineering")
    logger.info("=" * 60)

    import pandas as pd
    from src.features.pipeline import build_feature_pipeline, save_pipeline

    # Load raw data
    df = pd.read_csv("data/raw/borrowers.csv")
    logger.info(f"Loaded {len(df)} records from data/raw/borrowers.csv")

    # Build pipeline (non-strict for first run — drops leakage with warnings)
    pipeline = build_feature_pipeline(
        config_path=f"{config_dir}/features.yaml",
        strict=False,
    )

    # Fit and transform
    X = pipeline.fit_transform(df)
    logger.info(f"Feature pipeline: {df.shape} → {X.shape}")

    # Save fitted pipeline
    save_pipeline(pipeline, "models/feature_pipeline.joblib")

    # Save processed features
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X.to_csv("data/processed/features.csv", index=False)

    # Save target separately
    y = df["default_flag"]
    y.to_csv("data/processed/target.csv", index=False)

    # Save origination dates for OOT split
    dates = df["origination_date"]
    dates.to_csv("data/processed/origination_dates.csv", index=False)

    logger.info("Features saved to data/processed/")
    return X, y


def stage_validate(config_dir: str):
    """Stage C: Validation split."""
    logger.info("=" * 60)
    logger.info("STAGE C: Validation (OOT Split)")
    logger.info("=" * 60)

    import pandas as pd
    from src.validation import OOTSplitter

    # Load raw data for splitting (need origination_date)
    df = pd.read_csv("data/raw/borrowers.csv")

    splitter = OOTSplitter.from_config(f"{config_dir}/validation.yaml")
    train_df, oot_df = splitter.split(df)

    # Save splits
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_csv("data/processed/train_raw.csv", index=False)
    oot_df.to_csv("data/processed/oot_raw.csv", index=False)

    logger.info(f"Train: {len(train_df)}, OOT: {len(oot_df)}")
    return train_df, oot_df


def stage_train(config_dir: str):
    """Stage D+E: Train models, package best."""
    logger.info("=" * 60)
    logger.info("STAGE D: Model Training")
    logger.info("=" * 60)

    import pandas as pd
    from src.features.pipeline import build_feature_pipeline, save_pipeline
    from src.training import train_models
    from src.packaging import package_model

    # Load raw train/OOT splits
    train_df = pd.read_csv("data/processed/train_raw.csv")
    oot_df = pd.read_csv("data/processed/oot_raw.csv")

    # Build and fit feature pipeline on TRAINING data only
    pipeline = build_feature_pipeline(
        config_path=f"{config_dir}/features.yaml",
        strict=False,
    )

    y_train = train_df["default_flag"].values
    y_oot = oot_df["default_flag"].values

    X_train = pipeline.fit_transform(train_df)
    X_oot = pipeline.transform(oot_df)

    feature_names = list(X_train.columns)

    # Save properly fitted pipeline (fitted on train only)
    save_pipeline(pipeline, "models/feature_pipeline.joblib")

    # Train models
    results = train_models(
        X_train, y_train, X_oot, y_oot,
        feature_names=feature_names,
        config_path=f"{config_dir}/training.yaml",
    )

    # Package best model
    if results and "_best" in results:
        best_name = results["_best"]
        best = results[best_name]

        logger.info("=" * 60)
        logger.info("STAGE E: Packaging")
        logger.info("=" * 60)

        model_dir = package_model(
            model=best["model"],
            feature_pipeline=pipeline,
            feature_names=feature_names,
            train_metrics=best["train_metrics"],
            oot_metrics=best["oot_metrics"],
            config_path=f"{config_dir}/packaging.yaml",
        )
        logger.info(f"Best model ({best_name}) packaged to {model_dir}")

    return results


def stage_explain(config_dir: str):
    """Stage F: Explainability."""
    logger.info("=" * 60)
    logger.info("STAGE F: Explainability (SHAP)")
    logger.info("=" * 60)

    import pandas as pd
    from src.packaging import load_packaged_model
    from src.explainability import compute_shap_values, plot_global_shap, check_feature_stability

    # Load packaged model
    pkg = load_packaged_model("models/v1.0")
    model = pkg["model"]
    pipeline = pkg["feature_pipeline"]
    feature_names = pkg["feature_schema"]["feature_names"]

    # Load OOT data and transform
    oot_df = pd.read_csv("data/processed/oot_raw.csv")
    X_oot = pipeline.transform(oot_df)

    # Compute SHAP
    shap_explanation = compute_shap_values(model, X_oot, feature_names)

    # Global plots
    plot_global_shap(shap_explanation, save_dir="docs")

    # Feature stability (baseline — no previous model)
    stability = check_feature_stability(shap_explanation)
    logger.info(f"Top features: {stability['current_top_features'][:5]}")

    return shap_explanation


def stage_monitor(config_dir: str):
    """Stage G: Monitoring."""
    logger.info("=" * 60)
    logger.info("STAGE G: Monitoring")
    logger.info("=" * 60)

    import pandas as pd
    from src.packaging import load_packaged_model
    from src.monitoring import generate_monitoring_report

    # Load packaged model
    pkg = load_packaged_model("models/v1.0")
    model = pkg["model"]
    pipeline = pkg["feature_pipeline"]

    # Load train and OOT raw data
    train_df = pd.read_csv("data/processed/train_raw.csv")
    oot_df = pd.read_csv("data/processed/oot_raw.csv")

    # Transform both through pipeline
    X_train = pipeline.transform(train_df)
    X_oot = pipeline.transform(oot_df)

    # Get predictions for OOT
    y_oot = oot_df["default_flag"].values
    y_prob = model.predict_proba(X_oot)[:, 1]

    # Segment columns from raw OOT data
    segment_columns = {}
    for col in ["state", "loan_purpose", "occupation_type", "gender"]:
        if col in oot_df.columns:
            segment_columns[col] = oot_df[col].values

    # Generate report
    report = generate_monitoring_report(
        train_df=X_train,
        scoring_df=X_oot,
        y_true=y_oot,
        y_prob=y_prob,
        segment_columns=segment_columns,
    )

    logger.info(f"Monitoring report: {report['overall_psi_summary']}")
    return report


def stage_serve(config_dir: str):
    """Stage H: Start FastAPI server."""
    logger.info("=" * 60)
    logger.info("STAGE H: Starting Scoring API")
    logger.info("=" * 60)

    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)


STAGES = {
    "generate": stage_generate,
    "features": stage_features,
    "validate": stage_validate,
    "train": stage_train,
    "explain": stage_explain,
    "monitor": stage_monitor,
    "serve": stage_serve,
}

ALL_STAGES = ["generate", "validate", "train", "explain", "monitor"]


def main():
    parser = argparse.ArgumentParser(
        description="Yogyank Scoring Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  generate   Generate synthetic borrower data (25K records)
  features   Run feature engineering pipeline (called within train)
  validate   Create OOT train/test split (called within train)
  train      Train models + package best (includes features + validate)
  explain    Generate SHAP explanations
  monitor    Run drift/PSI monitoring report
  serve      Start FastAPI scoring server
  all        Run generate → validate → train → explain → monitor
        """,
    )
    parser.add_argument(
        "--stage", required=True,
        choices=list(STAGES.keys()) + ["all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config-dir", default="configs",
        help="Config directory (default: configs/)",
    )

    args = parser.parse_args()

    if args.stage == "all":
        for stage_name in ALL_STAGES:
            STAGES[stage_name](args.config_dir)
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
    else:
        STAGES[args.stage](args.config_dir)


if __name__ == "__main__":
    main()
