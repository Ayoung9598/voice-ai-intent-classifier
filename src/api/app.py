"""
FastAPI Application for Voice AI Intent Classification

Production-ready REST API for intent prediction with:
- Single and batch prediction endpoints
- Confidence-based fallback responses
- Health checks
- Request/response validation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import time

# Lazy imports to avoid loading model at import time
_predictor = None
_confidence_handler = None


# ============ Request/Response Models ============

class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    text: str = Field(..., description="Utterance text to classify", min_length=1)
    return_all_scores: bool = Field(
        default=False,
        description="Whether to return scores for all intents"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Ntabasha kwinjira muri account yanjye.",
                "return_all_scores": False
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    texts: List[str] = Field(
        ...,
        description="List of utterance texts to classify",
        min_length=1
    )
    return_all_scores: bool = Field(default=False)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "What are the requirements for passport?",
                    "Help me with payment"
                ],
                "return_all_scores": False
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    text: str
    intent: str
    confidence: float
    action: str = Field(description="Recommended action: execute, confirm, or escalate")
    explanation: str
    all_scores: Optional[Dict[str, float]] = None
    alternatives: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str
    version: str


# ============ App Creation ============

def create_app(
    model_path: Optional[str] = None,
    title: str = "Voice AI Intent Classifier API",
    version: str = "1.0.0",
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        model_path: Path to trained model checkpoint
        title: API title
        version: API version
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description="""
        ## Voice AI Intent Classification API
        
        A multilingual intent classifier for Rwanda's Voice AI assistant.
        
        ### Features
        - Supports Kinyarwanda, English, and mixed-language utterances
        - Confidence-based fallback recommendations
        - Single and batch prediction endpoints
        
        ### Confidence Actions
        - **execute**: High confidence (≥85%) - proceed with intent
        - **confirm**: Medium confidence (60-85%) - ask user to confirm
        - **escalate**: Low confidence (<60%) - route to human agent
        """,
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store model path for lazy loading
    app.state.model_path = model_path
    
    # ============ Endpoints ============
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize model on startup if path provided."""
        if model_path and Path(model_path).exists():
            _load_model(model_path)
    
    @app.get("/", tags=["Info"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": title,
            "version": version,
            "docs": "/docs",
            "health": "/health",
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Info"])
    async def health_check():
        """Check API health and model status."""
        global _predictor
        
        return HealthResponse(
            status="healthy",
            model_loaded=_predictor is not None,
            device=str(_predictor.device) if _predictor else "not loaded",
            version=version,
        )
    
    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict(request: PredictionRequest):
        """
        Predict intent for a single utterance.
        
        Returns the predicted intent, confidence score, and recommended action.
        """
        global _predictor, _confidence_handler
        
        if _predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please ensure the model is initialized."
            )
        
        start_time = time.time()
        
        # Get prediction
        result = _predictor.predict(
            request.text,
            return_all_scores=request.return_all_scores,
        )
        
        # Add confidence handling
        processed = _confidence_handler.process_prediction(result)
        
        # Get alternatives for medium confidence
        alternatives = None
        if processed["action"] == "confirm":
            alternatives = _predictor.predict_top_k(request.text, k=3)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            text=request.text,
            intent=processed["intent"],
            confidence=processed["confidence"],
            action=processed["action"],
            explanation=processed["explanation"],
            all_scores=result.get("all_scores"),
            alternatives=alternatives,
            processing_time_ms=round(processing_time, 2),
        )
    
    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
    async def predict_batch(request: BatchPredictionRequest):
        """
        Predict intents for multiple utterances.
        
        More efficient than multiple single predictions.
        """
        global _predictor, _confidence_handler
        
        if _predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please ensure the model is initialized."
            )
        
        start_time = time.time()
        
        # Get batch predictions
        results = _predictor.predict_batch(
            request.texts,
            return_all_scores=request.return_all_scores,
        )
        
        # Process each prediction
        predictions = []
        for result in results:
            processed = _confidence_handler.process_prediction(result)
            
            predictions.append(PredictionResponse(
                text=result["text"],
                intent=processed["intent"],
                confidence=processed["confidence"],
                action=processed["action"],
                explanation=processed["explanation"],
                all_scores=result.get("all_scores"),
                alternatives=None,  # Skip for batch
                processing_time_ms=0,  # Will be set at batch level
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=round(total_time, 2),
        )
    
    @app.get("/intents", tags=["Info"])
    async def list_intents():
        """List all supported intents."""
        from ..data.dataset import IntentLabelEncoder
        encoder = IntentLabelEncoder()
        
        return {
            "intents": encoder.intents,
            "count": encoder.num_labels,
        }
    
    @app.get("/stats", tags=["Monitoring"])
    async def get_stats():
        """Get prediction statistics."""
        global _confidence_handler
        
        if _confidence_handler is None:
            return {"message": "No predictions made yet"}
        
        return _confidence_handler.get_stats()
    
    return app


def _load_model(model_path: str):
    """Load the model and initialize handlers."""
    global _predictor, _confidence_handler
    
    from ..inference.predictor import IntentPredictor
    from ..inference.confidence import ConfidenceHandler
    
    _predictor = IntentPredictor(model_path=model_path)
    _confidence_handler = ConfidenceHandler()


def init_with_model(predictor, confidence_handler=None):
    """
    Initialize API with an existing predictor.
    
    Useful for testing or when model is already loaded.
    """
    global _predictor, _confidence_handler
    
    from ..inference.confidence import ConfidenceHandler
    
    _predictor = predictor
    _confidence_handler = confidence_handler or ConfidenceHandler()


# ============ Dev Server ============

def run_dev_server(
    model_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    Run development server.
    
    Args:
        model_path: Path to trained model
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn
    
    app = create_app(model_path=model_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # For testing without a trained model
    from ..models.intent_classifier import create_model
    from ..inference.predictor import IntentPredictor
    
    model = create_model()
    predictor = IntentPredictor(model=model)
    init_with_model(predictor)
    
    app = create_app()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
