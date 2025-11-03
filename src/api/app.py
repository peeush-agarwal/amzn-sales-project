import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

from contextlib import asynccontextmanager
from dotenv import load_dotenv
import yaml
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    QuestionAnswerRequest,
    QuestionAnswerResponse,
    PredictionRequest,
)
from api.predictor import ModelPredictor
from api.rag import ApiRagService


load_dotenv(override=True)

# Globals populated at startup
_config: Dict[str, Any] = {}
_predictor: Optional[ModelPredictor] = None
_rag_service: Optional[ApiRagService] = None


try:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "params.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        import yaml

        _config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load configuration: {e}")

APP_NAME = _config["api"]["name"]
APP_VERSION = _config["api"]["version"]

os.makedirs(_config["logs"]["path"], exist_ok=True)
logging.basicConfig(
    level=_config["logs"]["level"],
    format="%(asctime)s - %(levelname)-8s - %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                _config["logs"]["path"],
                "api.log",
            )
        ),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info(f"Starting up {APP_NAME}...")
    # Only the predictor and config are globals managed here.
    global _config, _predictor, _rag_service
    try:
        # Resolve model/transformer paths.
        def _resolve(p: Optional[str]) -> Optional[Path]:
            """Resolve a path string to an absolute Path under project_root.

            Accepts None or relative/absolute path strings. Returns None when
            input is falsy.
            """
            if not p:
                return None

            raw = Path(p)
            if raw.is_absolute():
                return raw

            return raw.resolve()

        _predictor = ModelPredictor(
            model_path=_resolve(_config.get("training", {}).get("output_model_path")),
            meta_path=_resolve(_config.get("training", {}).get("model_meta_path")),
            transformer_path=_resolve(
                _config.get("artifacts", {}).get("preprocessor_path")
            ),
        )
        _predictor.load_model()

        logger.info("Model and preprocessor loaded successfully")

        _rag_service = ApiRagService(
            collection_name=_config["rag"]["collection_name"],
            persist_directory=Path(_config["rag"]["persist_directory"]),
            embedding_model_name=_config["rag"]["embedding_model_name"],
            llm_model_name=_config["rag"]["llm_model_name"],
        )
        _rag_service.load()

        logger.info("RAG service loaded successfully")

        logger.info(
            f"API ready at http://{_config['api']['host']}:{_config['api']['port']}"
        )

    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise

    yield

    # Shutdown
    logger.info(f"Shutting down {APP_NAME}...")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{_config['api']['port']}", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")

    # Convert errors to JSON-serializable format
    errors = []
    for error in exc.errors():
        error_dict = {
            "type": error.get("type"),
            "loc": error.get("loc"),
            "msg": error.get("msg"),
            "input": str(error.get("input"))
            if error.get("input") is not None
            else None,
        }
        errors.append(error_dict)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request data",
            "error_type": "ValidationError",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error_type": type(exc).__name__},
    )


# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {APP_NAME}!",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = bool(_predictor and _predictor.is_ready())
    model_info = _predictor.get_model_info() if model_loaded else None  # type: ignore

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        rag_ready=_rag_service and _rag_service.is_ready(),
        model_info=model_info,
        version=APP_VERSION,
    )


@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """
    Get information about the loaded model.

    Returns model metadata, performance metrics, and usage statistics.
    """
    if not _predictor or not _predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    return _predictor.get_model_info()


@app.get("/metrics", response_model=Dict[str, Any])
async def metrics():
    """
    Get API and model performance metrics.

    Returns statistics about predictions and model performance.
    """

    if not _predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model predictor not initialized",
        )

    model_info = _predictor.get_model_info()

    return {
        "api_version": APP_VERSION,
        "predictions_served": _predictor.prediction_count,
        "total_prediction_time": _predictor.total_prediction_time,
        "average_prediction_time_ms": model_info.get("avg_prediction_time_ms", 0),
        "model_metrics": model_info.get("metrics", {}),
    }


@app.post(
    "/predict_discount",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Prediction Error"},
        503: {"model": ErrorResponse, "description": "Transformer Not Available"},
    },
)
async def predict_discount(payload: PredictionRequest = Body(...)):
    """Predict discount percentage for raw input features.

    Accepts a single JSON object with raw columns (see `config/params.yaml` > data.cols_raw). Returns predicted discount as decimal and percent.
    """
    try:
        if not _predictor or not _predictor.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available",
            )

        # Convert Pydantic model to dict
        raw_data = payload.model_dump()

        # Make prediction from raw data
        prediction = _predictor.predict_from_raw(raw_data)

        # Clip prediction to valid range [0.0, 1.0], then convert to percentage
        prediction = max(0.0, min(1.0, prediction))
        prediction_percentage = prediction * 100

        # Get model info
        model_info = _predictor.get_model_info()

        return PredictionResponse(
            prediction_decimal=prediction,
            prediction_percent=prediction_percentage,
            model_name=model_info.get("model_name"),
            model_version=model_info.get("run_id"),
            confidence_interval=model_info.get("confidence_interval"),
        )

    except RuntimeError as e:
        if "Transformer is not loaded" in str(e):
            logger.error(f"Transformer not available: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/answer_question",
    response_model=QuestionAnswerResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        503: {"model": ErrorResponse, "description": "Assistant Not Available"},
    },
)
async def answer_question(request: QuestionAnswerRequest):
    """Answer a product related question using the RAG assistant."""

    if not _rag_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG assistant is not available",
        )

    if not _rag_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG assistant is not ready",
        )

    try:
        result = await run_in_threadpool(
            _rag_service.answer_question,
            request.question,
            top_k=request.top_k,
        )
        answer = str(result.get("answer", ""))
        contexts_payload = result.get("contexts", [])
        return QuestionAnswerResponse(
            answer=answer,
            contexts=contexts_payload,  # type: ignore[arg-type]
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except Exception as exc:
        logger.error("RAG assistant error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG assistant is currently unavailable",
        )
