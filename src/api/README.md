# API Module Documentation

This module provides a FastAPI REST API for intent classification inference.

---

## app.py

### Purpose
Production-ready REST API for real-time intent prediction with confidence-based responses.

### Why FastAPI?
1. **Performance**: Built on Starlette and Pydantic, very fast
2. **Async support**: Handles concurrent requests efficiently
3. **Auto-documentation**: Swagger UI at `/docs`, ReDoc at `/redoc`
4. **Type validation**: Pydantic models validate request/response
5. **Easy to use**: Simple decorator-based routing

---

### Request/Response Models

#### `PredictionRequest`
```python
class PredictionRequest(BaseModel):
    text: str = Field(
        ...,                           # Required
        description="Utterance text",
        min_length=1                   # Can't be empty
    )
    return_all_scores: bool = Field(
        default=False,
        description="Return scores for all intents"
    )
```

**Validation**:
- Pydantic automatically validates request body
- Returns 422 error if validation fails
- `min_length=1` prevents empty text

#### `BatchPredictionRequest`
```python
class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1  # At least one text
    )
    return_all_scores: bool = Field(default=False)
```

#### `PredictionResponse`
```python
class PredictionResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    action: str           # "execute" | "confirm" | "escalate"
    explanation: str      # Human-readable action reason
    all_scores: Optional[Dict[str, float]] = None
    alternatives: Optional[List[Dict]] = None  # For "confirm" action
    processing_time_ms: float
```

#### `HealthResponse`
```python
class HealthResponse(BaseModel):
    status: str         # "healthy"
    model_loaded: bool  # True if model is ready
    device: str         # "cuda" or "cpu"
    version: str
```

---

### Global State

```python
_predictor = None           # IntentPredictor instance
_confidence_handler = None  # ConfidenceHandler instance
```

**Why global?**
- Model loading is expensive (~10 seconds)
- Load once at startup, reuse for all requests
- Single instance handles concurrent requests (model is thread-safe for inference)

---

### Functions

#### `create_app(model_path, title, version)`
**Purpose**: Factory function to create configured FastAPI app.

**Flow**:
```python
def create_app(model_path=None, title="...", version="1.0.0"):
    # 1. Create FastAPI instance with metadata
    app = FastAPI(
        title=title,
        description="...",  # Shows in Swagger UI
        version=version,
        docs_url="/docs",   # Swagger UI
        redoc_url="/redoc"  # ReDoc
    )
    
    # 2. Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"]
    )
    
    # 3. Store model path for lazy loading
    app.state.model_path = model_path
    
    # 4. Register startup event
    @app.on_event("startup")
    async def startup_event():
        if model_path:
            _load_model(model_path)
    
    # 5. Register all endpoints (see below)
    ...
    
    return app
```

**Why CORS middleware?**
- Allows web frontends from different domains to call API
- Necessary for browser-based applications
- `allow_origins=["*"]` is permissive (restrict in production)

---

### Endpoints

#### `GET /`
**Purpose**: Root endpoint with API info.

**Response**:
```json
{
    "name": "Voice AI Intent Classifier API",
    "version": "1.0.0",
    "docs": "/docs",
    "health": "/health"
}
```

---

#### `GET /health`
**Purpose**: Health check for monitoring/load balancers.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda",
    "version": "1.0.0"
}
```

**Use cases**:
- Kubernetes liveness probe
- Load balancer health checks
- Monitoring dashboards

---

#### `POST /predict`
**Purpose**: Predict intent for single utterance.

**Request**:
```json
{
    "text": "Ndashaka kureba status ya application yanjye.",
    "return_all_scores": false
}
```

**Response**:
```json
{
    "text": "Ndashaka kureba status ya application yanjye.",
    "intent": "check_application_status",
    "confidence": 0.92,
    "action": "execute",
    "explanation": "High confidence - proceeding automatically",
    "all_scores": null,
    "alternatives": null,
    "processing_time_ms": 45.2
}
```

**Implementation**:
```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # 1. Get raw prediction
    result = _predictor.predict(request.text, request.return_all_scores)
    
    # 2. Add confidence handling
    processed = _confidence_handler.process_prediction(result)
    
    # 3. Get alternatives for "confirm" action
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
        processing_time_ms=round(processing_time, 2)
    )
```

**Error handling**:
- 503 if model not loaded
- 422 if request validation fails (automatic from Pydantic)

---

#### `POST /predict/batch`
**Purpose**: Predict intents for multiple utterances.

**Request**:
```json
{
    "texts": [
        "Check my application status",
        "What are the requirements for passport?"
    ],
    "return_all_scores": false
}
```

**Response**:
```json
{
    "predictions": [
        {
            "text": "...",
            "intent": "check_application_status",
            "confidence": 0.88,
            "action": "execute",
            ...
        },
        {
            "text": "...",
            "intent": "requirements_information",
            "confidence": 0.91,
            "action": "execute",
            ...
        }
    ],
    "total_processing_time_ms": 78.5
}
```

**Why batch endpoint?**
- More efficient than multiple single requests
- GPU processes batch in parallel
- Lower total latency for bulk operations

---

#### `GET /intents`
**Purpose**: List all supported intents.

**Response**:
```json
{
    "intents": [
        "check_application_status",
        "start_new_application",
        ...
    ],
    "count": 13
}
```

**Use case**: Frontend can use this to build intent selection UI.

---

#### `GET /stats`
**Purpose**: Get prediction statistics.

**Response**:
```json
{
    "total_predictions": 1000,
    "action_counts": {
        "execute": 700,
        "confirm": 200,
        "escalate": 100
    },
    "action_percentages": {
        "execute": 70.0,
        "confirm": 20.0,
        "escalate": 10.0
    }
}
```

**Use case**: Monitoring dashboard, alerting on escalation rate.

---

### Helper Functions

#### `_load_model(model_path)`
**Purpose**: Load model and initialize handlers.

```python
def _load_model(model_path):
    global _predictor, _confidence_handler
    
    from ..inference.predictor import IntentPredictor
    from ..inference.confidence import ConfidenceHandler
    
    _predictor = IntentPredictor(model_path=model_path)
    _confidence_handler = ConfidenceHandler()
```

**Why global variables?**
- FastAPI doesn't have built-in dependency injection for heavy objects
- Model needs to be loaded once and shared
- Thread-safe for inference (PyTorch handles this)

#### `init_with_model(predictor, confidence_handler)`
**Purpose**: Initialize API with existing predictor.

**Use case**: Testing, or when model is loaded elsewhere.

#### `run_dev_server(model_path, host, port)`
**Purpose**: Start development server.

```python
def run_dev_server(model_path, host="0.0.0.0", port=8000):
    import uvicorn
    app = create_app(model_path=model_path)
    uvicorn.run(app, host=host, port=port)
```

---

## API Design Decisions

### Why REST vs GraphQL?
- REST is simpler for single-purpose API
- GraphQL overhead not needed for fixed responses
- REST has better caching support

### Why include processing_time_ms?
- Monitoring latency
- SLA tracking
- Debug slow requests

### Why return alternatives only for "confirm"?
- Execute: User doesn't need to see alternatives
- Escalate: Human will handle it
- Confirm: Alternatives help user clarify intent

### Why async endpoints?
```python
async def predict(request: PredictionRequest):
```
- FastAPI runs in async event loop
- Model inference is CPU-bound, not truly async
- But async allows handling concurrent requests efficiently
- While one request is processing, others can be queued

### Why 503 for missing model?
- 503 = Service Unavailable
- Indicates temporary issue (model not loaded yet)
- Client can retry
- Different from 500 (server error) or 404 (not found)

---

## Production Considerations

### Scaling
- Run multiple instances behind load balancer
- Each instance loads its own model copy
- Stateless design allows horizontal scaling

### Monitoring
- `/health` endpoint for liveness checks
- `/stats` for business metrics
- Add logging middleware for request tracing

### Security (for production)
- Add authentication (API keys, JWT)
- Rate limiting
- Input sanitization (Pydantic helps)
- Restrict CORS origins
