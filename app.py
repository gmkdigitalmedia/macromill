"""
Sentiment Analysis REST API Service
Enterprise-grade FastAPI implementation for serving sentiment classification model
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "models/final_model")
MAX_LENGTH = 512
MAX_BATCH_SIZE = 32

# Global model and tokenizer instances
model = None
tokenizer = None
device = None


# Pydantic models for request/response validation
class ReviewInput(BaseModel):
    """Single review input schema"""
    text: str = Field(
        ...,
        description="Movie review text to analyze",
        min_length=1,
        max_length=10000,
        example="This movie was absolutely fantastic! The acting was superb."
    )

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Review text cannot be empty or whitespace only')
        return v.strip()


class BatchReviewInput(BaseModel):
    """Batch review input schema"""
    reviews: List[str] = Field(
        ...,
        description="List of movie review texts to analyze",
        min_items=1,
        max_items=MAX_BATCH_SIZE,
        example=[
            "This movie was absolutely fantastic!",
            "Terrible waste of time. Poor acting and bad script."
        ]
    )

    @validator('reviews')
    def validate_reviews(cls, v):
        if not all(text.strip() for text in v):
            raise ValueError('All review texts must be non-empty')
        return [text.strip() for text in v]


class SentimentOutput(BaseModel):
    """Sentiment prediction output schema"""
    text: str = Field(..., description="Original review text")
    sentiment: str = Field(..., description="Predicted sentiment: 'positive' or 'negative'")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution for each class")


class BatchSentimentOutput(BaseModel):
    """Batch sentiment prediction output schema"""
    predictions: List[SentimentOutput] = Field(..., description="List of sentiment predictions")
    total_count: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Device being used (cpu or cuda)")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Enterprise-grade REST API for movie review sentiment classification using transformer models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer, device

    try:
        logger.info(f"Loading model from {MODEL_PATH}...")

        # Check if model exists
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def predict_sentiment(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Predict sentiment for a list of texts

    Args:
        texts: List of review texts

    Returns:
        List of prediction dictionaries
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    # Tokenize inputs
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    ).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)

    # Format results
    results = []
    for i, text in enumerate(texts):
        pred = predictions[i].item()
        confidence = probs[i][pred].item()
        neg_prob = probs[i][0].item()
        pos_prob = probs[i][1].item()

        sentiment = 'positive' if pred == 1 else 'negative'

        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'probabilities': {
                'negative': round(neg_prob, 4),
                'positive': round(pos_prob, 4)
            }
        })

    return results


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        # Optionally, you can prevent the app from starting
        # raise


@app.get(
    "/",
    summary="Root endpoint",
    description="Returns basic API information"
)
async def root():
    """Root endpoint"""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "description": "API for movie review sentiment classification",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is running and model is loaded"
)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device is not None else "unknown",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post(
    "/predict",
    response_model=SentimentOutput,
    summary="Predict sentiment for single review",
    description="Analyze sentiment of a single movie review",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict(review: ReviewInput):
    """
    Predict sentiment for a single review

    Args:
        review: ReviewInput containing the review text

    Returns:
        SentimentOutput with prediction results
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Get prediction
        results = predict_sentiment([review.text])

        return SentimentOutput(**results[0])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchSentimentOutput,
    summary="Predict sentiment for multiple reviews",
    description=f"Analyze sentiment of multiple movie reviews (max {MAX_BATCH_SIZE} at once)",
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_batch(batch: BatchReviewInput):
    """
    Predict sentiment for multiple reviews

    Args:
        batch: BatchReviewInput containing list of review texts

    Returns:
        BatchSentimentOutput with prediction results and metadata
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Track processing time
        start_time = datetime.now()

        # Get predictions
        results = predict_sentiment(batch.reviews)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return BatchSentimentOutput(
            predictions=[SentimentOutput(**r) for r in results],
            total_count=len(results),
            processing_time_ms=round(processing_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get(
    "/model/info",
    summary="Get model information",
    description="Returns information about the loaded model"
)
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "model_path": MODEL_PATH,
        "model_type": model.config.model_type,
        "num_labels": model.config.num_labels,
        "max_length": MAX_LENGTH,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation error", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


def main():
    """Run the application"""
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
