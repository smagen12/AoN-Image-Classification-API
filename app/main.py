from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
from PIL import Image
import numpy as np
import io

from app.model import load_keras_model

# Load TensorFlow model
model = load_keras_model()

# Labels
labels = ["Anime", "Cartoon"]

# Logging setup
logger.add("app/logs.log")

# FastAPI app
app = FastAPI(title="Anime vs Cartoon Classifier (TensorFlow/Keras)")

class PredictionResponse(BaseModel):
    label: str
    confidence: float

def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image)

        predictions = model.predict(input_tensor)
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[0][predicted_index])
        label = labels[predicted_index]

        logger.info(f"Predicted: {label}, Confidence: {confidence:.4f}")

        return {"label": label, "confidence": round(confidence, 4)}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": "CNN",
        "framework": "TensorFlow/Keras",
        "input_shape": "128x128x3",
        "labels": labels,
        "version": "v1"
    }
