import cv2
import time

from config import Config
from face   import load_landmarker, detect, face_relative_nose, detect_precision_mode, detect_blink, detect_lip_roll
from cursor import get_screen_size, head_to_velocity, move, make_mouse

cfg      = Config()
SCREEN_W, SCREEN_H = get_screen_size()
mouse    = make_mouse()

cursor_x = SCREEN_W / 2
cursor_y = SCREEN_H / 2

jaw_counter   = 0
blink_left    = 0
blink_right   = 0
lip_counter   = 0
lip_last_time = 0.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

with load_landmarker(cfg.model_path, cfg.model_url) as landmarker:

    # calibration: average nose position over first CALIB_FRAMES frames
    CALIB_FRAMES = 60
    calib_samples = []
    print(f"Calibrating — hold your head in a natural position...")

    while len(calib_samples) < CALIB_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        result = detect(landmarker, frame)
        if result.face_landmarks:
            hx, hy = face_relative_nose(result.face_landmarks[0])
            calib_samples.append((hx, hy))

        remaining = CALIB_FRAMES - len(calib_samples)
        cv2.putText(frame, f"Calibrating... {remaining} frames",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("facemove", frame)
        cv2.waitKey(1)

    cfg.neutral_x = sum(s[0] for s in calib_samples) / CALIB_FRAMES
    cfg.neutral_y = sum(s[1] for s in calib_samples) / CALIB_FRAMES
    print(f"Calibrated neutral: x={cfg.neutral_x:.3f}  y={cfg.neutral_y:.3f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        result = detect(landmarker, frame)

        jaw_counter, precise = detect_precision_mode(result.face_landmarks, jaw_counter, cfg)
        blink_left, blink_right, blinked = detect_blink(result.face_blendshapes, blink_left, blink_right, cfg)
        lip_counter, lip_rolling = detect_lip_roll(result.face_blendshapes, lip_counter, cfg)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            hx, hy = face_relative_nose(landmarks)

            vx, vy = head_to_velocity(hx, hy, cfg, precise=precise)

            cursor_x = max(0, min(cursor_x + vx, SCREEN_W))
            cursor_y = max(0, min(cursor_y + vy, SCREEN_H))

            move(mouse, cursor_x, cursor_y)

            # blink while in fast mode = click
            if precise and blinked:
                mouse.click(int(cursor_x), int(cursor_y), "left")

            # lip roll = left click (1 per second max)
            now = time.monotonic()
            if lip_rolling and now - lip_last_time >= 1.0:
                mouse.click(int(cursor_x), int(cursor_y), "left")
                lip_last_time = now

            px, py = int(landmarks[4].x * w), int(landmarks[4].y * h)
            colour = (0, 100, 255) if precise else (0, 255, 255)
            cv2.circle(frame, (px, py), 6, colour, -1)
            cv2.putText(frame,
                f"{'FAST' if precise else 'normal'}  head=({hx:.2f},{hy:.2f})  cursor=({int(cursor_x)},{int(cursor_y)})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        cv2.imshow("facemove", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
