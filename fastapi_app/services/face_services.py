from ai_models.face_recognition.facenet_model import FaceNetModel
from ai_models.face_recognition.arcface_model import ArcFaceModel
from ai_models.face_recognition.insightface_model  import InsightFaceModel
import numpy as np

model_map = {
    "facenet": FaceNetModel(),
    "insightface": InsightFaceModel(),
    "arcface": ArcFaceModel()
}

def get_face_embedding(image: np.ndarray, model_name: str):
    model = model_map.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found.")
    return model.get_embedding(image)