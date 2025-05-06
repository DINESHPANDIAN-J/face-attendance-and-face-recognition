# ai_models/face_recognition/model_registry.py

from ai_models.face_recognition.insightface_model import InsightFaceModel
from ai_models.face_recognition.facenet_model import FaceNetModel
from ai_models.face_recognition.arcface_model import ArcFaceModel

def get_all_models():
    return [
        InsightFaceModel(),
        FaceNetModel(),
        ArcFaceModel(),
    ]
