import cv2
import mediapipe as mp
import wayland_automation as wa
import urllib.request
import os
import subprocess

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def get_screen_size():
    output = subprocess.check_output(["xrandr"], text=True)
    for line in output.splitlines():
        if "*" in line:
            width, height = map(int, line.split()[0].split("x"))
            return width, height
    raise RuntimeError("Could not determine screen size")

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()

SMOOTH     = 0.2    # 0 = frozen, 1 = no smoothing
DEAD_ZONE  = 0.02   

HEAD_X_RANGE = 0.3
HEAD_Y_MIN   = 0.1
HEAD_Y_MAX   = 0.6

def map_range(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(value, in_max))
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

mouse = wa.Mouse()
smooth_x = smooth_y = None
prev_head_x = prev_head_y = None



with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # face 
            left_eye  = landmarks[33]
            right_eye = landmarks[263]
            nose      = landmarks[4]

            cx = (left_eye.x + right_eye.x) / 2
            cy = (left_eye.y + right_eye.y) / 2
            face_w = abs(right_eye.x - left_eye.x) + 1e-6

            # relative nose
            head_x = (nose.x - cx) / face_w
            head_y = (nose.y - cy) / face_w

            # dead zone
            if prev_head_x is not None:
                dx = abs(head_x - prev_head_x)
                dy = abs(head_y - prev_head_y)
                if dx < DEAD_ZONE and dy < DEAD_ZONE:
                    head_x, head_y = prev_head_x, prev_head_y

            prev_head_x, prev_head_y = head_x, head_y

            screen_x = map_range(head_x, -HEAD_X_RANGE, HEAD_X_RANGE, 0, SCREEN_WIDTH)
            screen_y = map_range(head_y,  HEAD_Y_MIN,   HEAD_Y_MAX,   0, SCREEN_HEIGHT)

            if smooth_x is None:
                smooth_x, smooth_y = screen_x, screen_y
            else:
                smooth_x = SMOOTH * screen_x + (1 - SMOOTH) * smooth_x
                smooth_y = SMOOTH * screen_y + (1 - SMOOTH) * smooth_y

            mouse.click(int(smooth_x), int(smooth_y), "nothing")

            px, py = int(nose.x * w), int(nose.y * h)

            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)

            cv2.putText(frame, f"head=({head_x:.2f}, {head_y:.2f})  screen=({int(smooth_x)}, {int(smooth_y)})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Face Cursor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
