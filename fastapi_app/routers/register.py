from fastapi import APIRouter, UploadFile, File
from fastapi_app.services.liveness_detector import LivenessDetector
import cv2
import numpy as np

router = APIRouter()
liveness = LivenessDetector()

@router.post("/register_with_liveness")
async def register_with_liveness(name: str, image: UploadFile = File(...)):
    content = await image.read()
    npimg = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    score = liveness.predict(frame)

    if score < 0.5:
        return {"status": "failed", "reason": "Liveness check failed"}

    # TODO: Save face embedding to Qdrant here
    return {"status": "success", "liveness_score": score}
