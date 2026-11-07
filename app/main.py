"""API de visão do Rua Acesa: confirma, pela foto, se o relato de poste apagado procede."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.detector import Detector
from app.schemas import RespostaDeteccao, Saude

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL_PATH = ROOT / os.getenv("MODEL_PATH", "models/best.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
VERDICT_THRESHOLD = float(os.getenv("VERDICT_THRESHOLD", "0.5"))
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

detector: Detector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    detector = Detector(MODEL_PATH, CONF_THRESHOLD, VERDICT_THRESHOLD)
    yield


app = FastAPI(
    title="Rua Acesa — Visão",
    description="Detecção de postes acesos e apagados em fotos de relatos (YOLO customizado). "
                "A imagem é processada em memória e não é armazenada.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=Saude)
def health() -> Saude:
    return Saude(status="ok", modelo=MODEL_PATH.name, classes=detector.classes)


@app.post("/detect", response_model=RespostaDeteccao)
async def detect(file: UploadFile = File(..., description="Foto do relato (JPEG, PNG ou WebP)")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Tipo não suportado: {file.content_type}. Use JPEG, PNG ou WebP.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Arquivo maior que {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "Não foi possível decodificar a imagem.")
    return detector.detect(image)
