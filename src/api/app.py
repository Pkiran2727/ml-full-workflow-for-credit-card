"""
FastAPI Scoring Service
========================

Loads packaged model artifact on startup and serves credit scores
via REST API with SHAP-based reason codes.

Endpoints:
    POST /score     — score a borrower, returns probability + reason codes
    GET  /health    — health check
    GET  /model-info — model version and metrics
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .schemas import (
    ScoringRequest,
    ScoringResponse,
    ReasonCode,
    HealthResponse,
    ModelInfoResponse,
)
from src.packaging.bundler import load_packaged_model
from src.explainability.shap_explainer import get_reason_codes

logger = logging.getLogger(__name__)

# Global state
_model_state = {
    "model": None,
    "pipeline": None,
    "schema": None,
    "metadata": None,
    "loaded": False,
    "start_time": None,
}

# Model artifact path — configurable via environment
MODEL_DIR = Path("models/v1.0")


def _get_risk_band(score: float) -> str:
    """Map probability score to risk band."""
    if score < 0.05:
        return "very_low_risk"
    elif score < 0.15:
        return "low_risk"
    elif score < 0.30:
        return "medium_risk"
    elif score < 0.50:
        return "high_risk"
    else:
        return "very_high_risk"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    try:
        if MODEL_DIR.exists():
            pkg = load_packaged_model(MODEL_DIR)
            _model_state["model"] = pkg["model"]
            _model_state["pipeline"] = pkg["feature_pipeline"]
            _model_state["schema"] = pkg["feature_schema"]
            _model_state["metadata"] = pkg["metadata"]
            _model_state["loaded"] = True
            _model_state["start_time"] = time.time()
            logger.info(f"Model loaded: {pkg['metadata']['model_name']} "
                         f"{pkg['metadata']['version']}")
        else:
            logger.warning(f"Model directory {MODEL_DIR} not found — API will start "
                            f"without model. Run the pipeline first.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    yield  # App runs here

    logger.info("Shutting down scoring service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(
        title="Yogyank Credit Scoring API",
        description="Rural credit scoring service with SHAP-based reason codes",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    async def get_index():
        """Serve the interactive scoring dashboard."""
        template_path = Path(__file__).parent / "templates" / "index.html"
        if template_path.exists():
            return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
        else:
            return HTMLResponse(content="<h1>Dashboard Template Not Found</h1>", status_code=404)

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if _model_state["loaded"] else "no_model",
            model_loaded=_model_state["loaded"],
            model_version=_model_state["metadata"]["version"] if _model_state["metadata"] else "unknown",
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
    async def model_info():
        """Get model metadata and performance metrics."""
        if not _model_state["loaded"]:
            raise HTTPException(status_code=503, detail="Model not loaded")

        meta = _model_state["metadata"]
        oot = meta.get("oot_metrics", {})

        return ModelInfoResponse(
            model_name=meta["model_name"],
            version=meta["version"],
            model_type=meta["model_type"],
            training_date=meta["training_date"],
            oot_auc=oot.get("roc_auc"),
            oot_ks=oot.get("ks_statistic"),
            oot_gini=oot.get("gini"),
            n_features=_model_state["schema"]["n_features"],
        )

    @app.post("/score", response_model=ScoringResponse, tags=["Scoring"])
    async def score_borrower(request: ScoringRequest):
        """
        Score a borrower and return default probability with reason codes.

        The response includes:
        - score: default probability (0-1)
        - risk_band: categorical risk classification
        - reason_codes: top-3 features driving the score (for adverse action notices)
        """
        if not _model_state["loaded"]:
            raise HTTPException(status_code=503, detail="Model not loaded — run pipeline first")

        try:
            # Convert request to DataFrame (single row)
            borrower_data = request.model_dump()
            df = pd.DataFrame([borrower_data])

            # Add a dummy origination_date for the pipeline (won't affect scoring)
            df["origination_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
            df["borrower_id"] = "API_REQUEST"
            df["default_flag"] = 0  # dummy, will be dropped by pipeline

            # Transform through feature pipeline
            pipeline = _model_state["pipeline"]
            X = pipeline.transform(df)

            # Predict
            model = _model_state["model"]
            score = float(model.predict_proba(X)[:, 1][0])
            risk_band = _get_risk_band(score)

            # Get reason codes
            feature_names = _model_state["schema"]["feature_names"]
            try:
                reasons = get_reason_codes(model, X, feature_names, top_n=3)
                reason_codes = [ReasonCode(**r) for r in reasons]
            except Exception as e:
                logger.warning(f"Reason code generation failed: {e}")
                reason_codes = []

            return ScoringResponse(
                score=round(score, 4),
                risk_band=risk_band,
                reason_codes=reason_codes,
                model_version=_model_state["metadata"]["version"],
            )

        except Exception as e:
            logger.error(f"Scoring failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

    return app


# Module-level app for uvicorn
app = create_app()
