"""FastAPI service for MLOps pipeline."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import yaml
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    texts: List[str]
    return_probabilities: bool = False


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, str]


class TrainingRequest(BaseModel):
    dataset_name: str = "imdb"
    sample_size: int = 100
    epochs: int = 1
    batch_size: int = 16


class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False

    def load_model(self, model_path: str = "models/trained_model") -> bool:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            if not Path(model_path).exists():
                logger.warning(f"Model not found at {model_path}")
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, texts: List[str], return_probabilities: bool = False) -> List[Dict[str, Any]]:
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        predictions = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, padding=True, max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()

                result = {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "predicted_class": predicted_class,
                    "prediction": "positive" if predicted_class == 1 else "negative",
                }

                if return_probabilities:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    result["probabilities"] = {
                        "negative": float(probs[0]),
                        "positive": float(probs[1]),
                    }

                predictions.append(result)

        return predictions


model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MLOps Pipeline API...")
    if Path("models/trained_model").exists():
        model_service.load_model()
    else:
        logger.warning("No trained model found. Use /train to train a model first.")
    yield


app = FastAPI(
    title="MLOps Pipeline API",
    description="Production API for the MLOps training pipeline",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "MLOps Pipeline API",
        "version": "2.0.0",
        "model_loaded": model_service.model_loaded,
        "device": str(model_service.device),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_service.model_loaded,
        "device": str(model_service.device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_service.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first using /train.",
        )
    try:
        predictions = model_service.predict(request.texts, request.return_probabilities)
        return PredictionResponse(
            predictions=predictions,
            model_info={"device": str(model_service.device), "model_path": "models/trained_model"},
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(request: TrainingRequest):
    try:
        from dagster import materialize
        from src.dagster_definitions import all_assets

        config_path = Path("config/pipeline.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        config["data"]["huggingface"]["sample_size"] = request.sample_size
        config["training"]["epochs"] = request.epochs
        config["training"]["batch_size"] = request.batch_size

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        logger.info(f"Starting training with {request.sample_size} samples...")
        result = materialize(all_assets)

        if result.success:
            model_service.load_model()
            return {
                "status": "success",
                "message": "Model trained successfully",
                "config": {
                    "dataset_name": request.dataset_name,
                    "sample_size": request.sample_size,
                    "epochs": request.epochs,
                    "batch_size": request.batch_size,
                },
            }
        raise HTTPException(status_code=500, detail="Training failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
