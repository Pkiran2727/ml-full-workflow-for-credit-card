"""
Model Packager & Registry
==========================

Bundles preprocessing pipeline + model + schema + metadata into a single
versioned artifact. Registers in MLflow Model Registry.
Auto-generates model card documentation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import yaml
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path) -> dict:
    """Load packaging YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _generate_model_card(config: dict, metrics: dict, feature_names: list[str],
                          save_path: str | Path) -> None:
    """Auto-generate model card markdown."""
    meta = config.get("metadata", {})
    card = f"""# Model Card: {config.get('model_name', 'yogyank-credit-scorer')}

## Model Description

**Type**: Binary classification (credit default prediction)
**Target Population**: Rural borrowers in India — microfinance and priority sector lending
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Intended Use

{meta.get('intended_use', 'Rural credit scoring')}

## Training Data

- **Source**: Synthetic data generator (see `src/data_generation/`)
- **Size**: {metrics.get('n_total_train', 'N/A')} training records
- **Bad Rate**: {metrics.get('bad_rate_train', 'N/A')}
- **Features**: {len(feature_names)} features

## Validation Methodology

- **Primary**: Out-of-Time (OOT) holdout — train on earlier loans, test on later loans
- **Secondary**: 5-fold stratified cross-validation within training set

## Performance Metrics

| Metric | Train | OOT |
|--------|-------|-----|
| AUC-ROC | {metrics.get('train_roc_auc', 'N/A'):.4f} | {metrics.get('oot_roc_auc', 'N/A'):.4f} |
| KS Statistic | {metrics.get('train_ks_statistic', 'N/A'):.4f} | {metrics.get('oot_ks_statistic', 'N/A'):.4f} |
| Gini | {metrics.get('train_gini', 'N/A'):.4f} | {metrics.get('oot_gini', 'N/A'):.4f} |
| PR-AUC | {metrics.get('train_pr_auc', 'N/A'):.4f} | {metrics.get('oot_pr_auc', 'N/A'):.4f} |

## Features Used

{chr(10).join(f'- {f}' for f in feature_names)}

## Known Limitations

{chr(10).join(f'- {l}' for l in meta.get('known_limitations', []))}

## Ethical Considerations

{chr(10).join(f'- {e}' for e in meta.get('ethical_considerations', []))}

## Model Governance

- This model should be reviewed and revalidated at minimum every 12 months
- Drift monitoring (PSI/CSI) should be active in production
- Human-in-the-loop required for all credit decisions
"""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(card)
    logger.info(f"Model card saved to {save_path}")


def package_model(model,
                   feature_pipeline,
                   feature_names: list[str],
                   train_metrics: dict,
                   oot_metrics: dict,
                   config_path: str | Path = "configs/packaging.yaml",
                   output_dir: str | Path | None = None) -> Path:
    """
    Bundle model + pipeline + metadata into a versioned artifact.

    Args:
        model: Trained model object.
        feature_pipeline: Fitted sklearn Pipeline.
        feature_names: List of feature names.
        train_metrics: Training set metrics dict.
        oot_metrics: OOT set metrics dict.
        config_path: Path to packaging config.
        output_dir: Override output directory.

    Returns:
        Path to the packaged model directory.
    """
    config = _load_config(config_path)
    model_name = config.get("model_name", "yogyank-credit-scorer")
    version = f"{config.get('version_prefix', 'v')}1.0"

    out_dir = Path(output_dir or config.get("output_dir", "models")) / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature pipeline
    pipeline_path = out_dir / "feature_pipeline.joblib"
    joblib.dump(feature_pipeline, pipeline_path)
    logger.info(f"Feature pipeline saved to {pipeline_path}")

    # Save feature schema
    schema = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    schema_path = out_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))

    # Save metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "training_date": datetime.now().isoformat(),
        "python_version": "3.12",
        "model_type": type(model).__name__,
        "train_metrics": train_metrics,
        "oot_metrics": oot_metrics,
        "intended_use": config.get("metadata", {}).get("intended_use", ""),
        "known_limitations": config.get("metadata", {}).get("known_limitations", []),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    # Generate model card
    all_metrics = {}
    for k, v in train_metrics.items():
        all_metrics[f"train_{k}"] = v
    for k, v in oot_metrics.items():
        all_metrics[f"oot_{k}"] = v
    all_metrics["n_total_train"] = train_metrics.get("n_total", "N/A")
    all_metrics["bad_rate_train"] = train_metrics.get("bad_rate", "N/A")

    model_card_path = config.get("model_card_path", "docs/model_card.md")
    _generate_model_card(config, all_metrics, feature_names, model_card_path)

    # Register with MLflow Model Registry
    try:
        mlflow_config = yaml.safe_load(
            open("configs/training.yaml", "r")
        ).get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "./mlruns"))

        with mlflow.start_run(run_name=f"package_{version}"):
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(pipeline_path))
            mlflow.log_artifact(str(schema_path))
            mlflow.log_artifact(str(metadata_path))

            model_uri = mlflow.sklearn.log_model(
                model,
                artifact_path="registered_model",
                serialization_format="cloudpickle",
            ).model_uri

            # Register model
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model '{model_name}' version {result.version}")

    except Exception as e:
        logger.warning(f"MLflow registration failed (non-blocking): {e}")

    logger.info(f"Model packaged to {out_dir}")
    return out_dir


def load_packaged_model(model_dir: str | Path) -> dict:
    """
    Load a packaged model artifact.

    Returns dict with: model, feature_pipeline, feature_schema, metadata.
    """
    model_dir = Path(model_dir)

    result = {
        "model": joblib.load(model_dir / "model.joblib"),
        "feature_pipeline": joblib.load(model_dir / "feature_pipeline.joblib"),
        "feature_schema": json.loads((model_dir / "feature_schema.json").read_text()),
        "metadata": json.loads((model_dir / "metadata.json").read_text()),
    }

    logger.info(f"Loaded packaged model from {model_dir}: "
                f"{result['metadata']['model_name']} {result['metadata']['version']}")
    return result
