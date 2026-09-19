import cv2

from config import Config
from face   import load_landmarker, detect, face_relative_nose
from cursor import get_screen_size, apply_dead_zone, map_to_screen, apply_smooth, move, make_mouse

cfg      = Config()
SCREEN_W, SCREEN_H = get_screen_size()
mouse    = make_mouse()

smooth_x = smooth_y = None
prev_hx  = prev_hy  = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

with load_landmarker(cfg.model_path, cfg.model_url) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        result = detect(landmarker, frame)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            hx, hy = face_relative_nose(landmarks)
            hx, hy = apply_dead_zone(hx, hy, prev_hx, prev_hy, cfg.dead_zone)
            prev_hx, prev_hy = hx, hy

            screen_x, screen_y = map_to_screen(hx, hy, cfg, SCREEN_W, SCREEN_H)
            smooth_x, smooth_y = apply_smooth(screen_x, screen_y, smooth_x, smooth_y, cfg.smooth)

            move(mouse, smooth_x, smooth_y)

            px, py = int(landmarks[4].x * w), int(landmarks[4].y * h)
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(frame,
                f"head=({hx:.2f},{hy:.2f})  screen=({int(smooth_x)},{int(smooth_y)})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("face cursor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
