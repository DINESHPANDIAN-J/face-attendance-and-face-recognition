import cv2

def show_result(img, score):
    label = "Live" if score > 0.5 else "Spoof"
    color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
    cv2.putText(img, f"{label}: {score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Liveness Check", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
