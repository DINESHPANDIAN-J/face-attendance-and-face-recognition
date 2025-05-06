# ai_models/face_recognition/insightface/model.py

import numpy as np
from ai_models.face_recognition.base_model import FaceRecognitionModel

class InsightFaceModel(FaceRecognitionModel):
    def __init__(self):
        self.model = None

    def load_model(self):
        print("[InsightFace] Loading model...")
        self.model = "Loaded Dummy InsightFace Model"  # Replace with real loader

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        # Dummy embedding for now
        return np.random.rand(512)

    def get_model_name(self) -> str:
        return "InsightFace"