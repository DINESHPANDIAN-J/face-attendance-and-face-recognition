# face_utils.py
import os
import cv2
import numpy as np
from ai_models import face_recognition
  
def capture_and_store_faces(name: str, user_id: str, save_dir: str):
    # Initialize webcam
    cap = cv2.VideoCapture(1)  # External USB camera
    if not cap.isOpened():
        return {"error": "Webcam not accessible"}

    embeddings = []
    count = 0

    while count < 5:
        ret, frame = cap.read()
        if not ret:
            break

        img_path = os.path.join(save_dir, f"face_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face and compute embedding
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            embeddings.append(face_encoding)
            count += 1

        cv2.imshow("Capturing Face", frame)
        cv2.waitKey(500)  # Wait for 0.5 seconds between shots

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 5:
        return {"error": f"Only {len(embeddings)} valid face(s) detected. Try again."}

    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding
