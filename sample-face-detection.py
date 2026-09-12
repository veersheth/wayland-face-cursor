import cv2
import mediapipe as mp
import urllib.request
import os

MODEL_PATH = "face_detector.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

if not os.path.exists(MODEL_PATH):
    print("Downloading face detection model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
)

cap = cv2.VideoCapture(0)

with FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect(mp_image)

        for det in result.detections:
            box = det.bounding_box
            cv2.rectangle(frame, (box.origin_x, box.origin_y), (box.origin_x + box.width, box.origin_y + box.height), (0, 0, 255), 2)

        cv2.imshow('feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
