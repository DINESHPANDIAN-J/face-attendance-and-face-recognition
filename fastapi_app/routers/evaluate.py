# This script is part of a FastAPI application that evaluates face recognition models.
# It runs a separate Python script to perform the evaluation and returns the status.
# The script is designed to be modular and can be easily integrated into a larger FastAPI application.
from fastapi import APIRouter
import subprocess

router = APIRouter()

@router.get("/evaluate")
async def evaluate_models():
    try:
        subprocess.run(["python", "scripts/evaluate_face_models.py"], check=True)
        return {"status": "success", "message": "Evaluation completed. Check the plot at scripts/evaluation_results.png"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "details": str(e)}
