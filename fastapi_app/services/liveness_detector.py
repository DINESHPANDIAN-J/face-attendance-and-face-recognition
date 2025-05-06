# fastapi_app/services/liveness_detector.py

import cv2
import numpy as np

class LivenessDetector:
    def __init__(self, model_path="models/liveness/liveness_model.onnx"):
        self.model = cv2.dnn.readNetFromONNX(model_path)

    def predict(self, face_img: np.ndarray) -> float:
        """
        Input: face image
        Output: float score where > 0.5 means live
        """
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (224, 224), (104, 117, 123), swapRB=True)
        self.model.setInput(blob)
        output = self.model.forward()
        return float(output[0][0])