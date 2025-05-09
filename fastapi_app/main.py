from fastapi import FastAPI, Form
from fastapi_app.routers import evaluate 
from uuid import uuid4
import os
import cv2

app = FastAPI()

app.include_router(evaluate.router, prefix="/api")

@app.post("/capture_faces/")
def capture_faces(name: str = Form(...)):
    # Generate unique ID folder
    user_id = str(uuid4())[:8]  # Short UUID
    save_dir = os.path.join("captured_faces", user_id)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        return {"error": "Webcam not accessible"}

    print(f"Capturing images for {name} in folder: {save_dir}")

    count = 0
    while count < 5:
        ret, frame = cap.read()
        if not ret:
            break

        img_path = os.path.join(save_dir, f"face_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1

        cv2.imshow("Capturing Face", frame)
        cv2.waitKey(500)  # Wait for 0.5 seconds between shots

    cap.release()
    cv2.destroyAllWindows()

    return {
        "message": f"5 images captured for {name}",
        "user_id": user_id,
        "folder": save_dir
    }
