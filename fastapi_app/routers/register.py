from fastapi import APIRouter, UploadFile, File
from fastapi_app.services.liveness_detector import LivenessDetector
from fastapi_app.db.qdrant_handler import store_user_embedding
from fastapi_app.models.liveness_schema import LivenessResult

import numpy as np
import cv2

router = APIRouter()
liveness = LivenessDetector()

@router.post("/register-liveness", response_model=LivenessResult)
async def register_liveness(name: str, image: UploadFile = File(...)):
    contents = await image.read()
    img_np = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    score = liveness.predict(img)
    status = "live" if score > 0.5 else "spoof"

    if status == "spoof":
        return LivenessResult(name=name, user_id="N/A", liveness_score=score, status=status)

    # Example embedding
    dummy_embedding = np.random.rand(512).tolist()  # Replace with real face embedding
    user_id = store_user_embedding(name, dummy_embedding, score)

    return LivenessResult(name=name, user_id=user_id, liveness_score=score, status=status)
