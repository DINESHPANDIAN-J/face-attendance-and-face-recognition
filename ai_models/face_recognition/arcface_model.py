# ai_models/face_recognition/arcface/model.py

import numpy as np
from ai_models.face_recognition.base_model import FaceRecognitionModel

class ArcFaceModel(FaceRecognitionModel):
    def __init__(self):
        self.model = None

    def load_model(self):
        print("[ArcFace] Loading model... (dummy)")
        self.model = "Dummy ArcFace"

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        return np.random.rand(512)

    def get_model_name(self) -> str:
        return "ArcFace"
