from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

import torch
from torchvision import transforms
from PIL import Image
import io
import os

from model import load_model

# Loguru setup
logger.add("logs.log")

# FastAPI app
app = FastAPI(
    title="Anime vs Cartoon Classifier",
    description="Predict whether an image is anime or cartoon with ResNet50",
    version="1.0.0"
)

# Available weights directory
WEIGHTS_DIR = "app/weights"

# Load default model
DEFAULT_WEIGHTS = f"{WEIGHTS_DIR}/resnet_v1.pth"
model = load_model(DEFAULT_WEIGHTS)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

labels = ["anime", "cartoon"]

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse, summary="Predict anime or cartoon")
async def predict(
    file: UploadFile = File(...),
    weights: str = Query(default="resnet_v1.pth", description="Weights file to use")
):
    logger.info(f"Received file: {file.filename} | Using weights: {weights}")

    # Load specified weights if different
    weights_path = os.path.join(WEIGHTS_DIR, weights)
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=400, detail=f"Weights file {weights} not found.")

    global model
    model = load_model(weights_path)

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

        label = labels[predicted.item()]
        logger.info(f"Prediction: {label}, Confidence: {confidence:.4f}")

        return JSONResponse(content={
            "label": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

@app.get("/health", summary="Check if API is running")
def health():
    return {"status": "ok"}

@app.get("/model-info", summary="Get current model info")
def model_info():
    return {
        "model_name": "ResNet50",
        "available_weights": os.listdir(WEIGHTS_DIR),
        "default_weights": DEFAULT_WEIGHTS
    }
