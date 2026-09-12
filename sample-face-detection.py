import cv2
import mediapipe as mp
import urllib.request
import os

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
)

cap = cv2.VideoCapture(0)

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))

        for landmarks in result.face_landmarks:
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            cv2.rectangle(frame,
                (int(min(xs)), int(min(ys))),
                (int(max(xs)), int(max(ys))),
                (0, 0, 255), 2)

            nose = landmarks[4]

            nx, ny = int(nose.x * w), int(nose.y * h)

            cv2.circle(frame, (nx, ny), 6, (0, 255, 255), -1)

            cv2.putText(frame, f"({nx}, {ny})", (nx + 10, ny - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
