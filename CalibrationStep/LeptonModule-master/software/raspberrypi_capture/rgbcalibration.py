import cv2
import numpy as np

# === Initialization of the RGB camera ===
camera_index = 0
cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L)

if not cap.isOpened():
    print("Error: Could not access the RGB camera.")
    exit()

print("RGB camera opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'p' to save the image matrix using np.save
    if key == ord('p'):
        filename = "captured_image.npy"
        np.save(filename, frame)
        print(f"Image captured and saved as {filename} using np.save")

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
