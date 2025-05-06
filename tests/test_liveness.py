import numpy as np
from fastapi_app.services.liveness_detector import LivenessDetector

def test_liveness_predict():
    detector = LivenessDetector()
    fake_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    score = detector.predict(fake_img)
    assert 0.0 <= score <= 1.0
