from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import uuid
import random
import time

app = FastAPI(
    title="Oculus: Back-End",
    description="Backend faker para clasificación de monedas peruanas con una sola imagen.",
    version="0.1.0"
)

COIN_CLASSES = [
    "10_centimos",
    "20_centimos",
    "50_centimos",
    "1_sol",
    "2_soles",
    "5_soles",
]

class ClassificationResponse(BaseModel):
    id: str
    class_name: str
    confidence: float
    inference_time_ms: float
    metadata: dict


class ModelStatus(BaseModel):
    loaded: bool
    model_name: str
    version: str
    device: str
    classes: List[str]


# ============================================================
# Endpoints
# ============================================================

@app.get("/status", response_model=ModelStatus)
def status():
    """
    Información del modelo actual.
    """
    return ModelStatus(
        loaded=True,
        model_name="mobilenet_v3_large",
        version="mock-0.2",
        device="cpu",
        classes=COIN_CLASSES
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify(image: UploadFile = File(...)):
    """
    Clasifica una sola imagen. (Faker)
    """
    # Simulación de latencia de inferencia
    inference_time = round(random.uniform(30.0, 65.0), 2)

    fake_class = random.choice(COIN_CLASSES)
    fake_conf = round(random.uniform(0.82, 0.99), 2)

    return ClassificationResponse(
        id=str(uuid.uuid4()),
        class_name=fake_class,
        confidence=fake_conf,
        inference_time_ms=inference_time,
        metadata={
            "description": f"Mocked metadata for {fake_class}",
            "side": random.choice(["obverse", "reverse"]),
            "file_received": image.filename
        }
    )
