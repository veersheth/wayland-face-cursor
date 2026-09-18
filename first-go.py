import cv2
import mediapipe as mp
import wayland_automation as wa
import urllib.request
import os
from screeninfo import get_monitors

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

import subprocess


def get_screen_size():
    output = subprocess.check_output(
        ["xrandr"],
        text=True
    )

    for line in output.splitlines():
        if "*" in line:
            resolution = line.split()[0]
            width, height = map(int, resolution.split("x"))
            return width, height

    raise RuntimeError("Could not determine screen size")


SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()

print(SCREEN_WIDTH, SCREEN_HEIGHT)



CAMERA_X_MIN = 0.15
CAMERA_X_MAX = 0.85
CAMERA_Y_MIN = 0.15
CAMERA_Y_MAX = 0.85

def map_range(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(value, in_max))

    return (
        (value - in_min)
        * (out_max - out_min)
        / (in_max - in_min)
        + out_min
    )


options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# -----------------------------
# Wayland mouse
# -----------------------------

mouse = wa.Mouse()

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1) # mirror cam

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image( image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(image)

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            nose = landmarks[4]

            nx = nose.x
            ny = nose.y

            screen_x = map_range(
                nx,
                CAMERA_X_MIN,
                CAMERA_X_MAX,
                0,
                SCREEN_WIDTH
            )

            screen_y = map_range(
                ny,
                CAMERA_Y_MIN,
                CAMERA_Y_MAX,
                0,
                SCREEN_HEIGHT
            )

            screen_x = int(screen_x)
            screen_y = int(screen_y)

            mouse.click(screen_x, screen_y, "nothing")

            px = int(nx * w)
            py = int(ny * h)

            cv2.circle(
                frame,
                (px, py),
                6,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                f"Screen: ({screen_x}, {screen_y})",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

        cv2.imshow("Face Cursor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
