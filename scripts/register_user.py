# scripts/register_user.py

import uuid
import cv2
import numpy as np
from ai_models.face_recognition.insightface_model import InsightFaceModel
from fastapi_app.services.qdrant_service import save_user_to_qdrant

def capture_images(count=5):
    cam = cv2.VideoCapture(0)
    captured = []
    print("Press SPACE to capture 5 face images.")

    while len(captured) < count:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1)
        if key % 256 == 32:
            print(f"Captured image {len(captured) + 1}")
            captured.append(frame)

    cam.release()
    cv2.destroyAllWindows()
    return captured

def register_user():
    name = input("Enter full name: ")
    user_id = str(uuid.uuid4())
    
    model = InsightFaceModel()
    model.load_model()

    images = capture_images()
    embeddings = [model.get_embedding(img) for img in images]
    avg_embedding = np.mean(embeddings, axis=0)

    save_user_to_qdrant(user_id, name, avg_embedding)
    print(f"✅ User '{name}' registered with ID {user_id}")

if __name__ == "__main__":
    register_user()