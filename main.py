import cv2

from config import Config
from face import load_landmarker, detect, face_relative_nose, detect_precision_mode
from cursor import get_screen_size, apply_dead_zone, map_to_screen, apply_smooth, move, make_mouse

cfg = Config()
SCREEN_W, SCREEN_H = get_screen_size()
mouse = make_mouse()

smooth_x = smooth_y = None
prev_hx = prev_hy  = None
mouth_counter = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

with load_landmarker(cfg.model_path, cfg.model_url) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        result = detect(landmarker, frame)

        mouth_counter, precise = detect_precision_mode(result.face_blendshapes, mouth_counter, cfg)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            hx, hy = face_relative_nose(landmarks)

            dz = cfg.dead_zone_precise if precise else cfg.dead_zone
            hx, hy = apply_dead_zone(hx, hy, prev_hx, prev_hy, dz)
            prev_hx, prev_hy = hx, hy

            screen_x, screen_y = map_to_screen(hx, hy, cfg, SCREEN_W, SCREEN_H)

            if precise and smooth_x is not None:
                screen_x = smooth_x + (screen_x - smooth_x) * cfg.precision_speed
                screen_y = smooth_y + (screen_y - smooth_y) * cfg.precision_speed

            alpha = cfg.smooth_precise if precise else cfg.smooth
            smooth_x, smooth_y = apply_smooth(screen_x, screen_y, smooth_x, smooth_y, alpha)

            move(mouse, smooth_x, smooth_y)

            # HUD
            px, py = int(landmarks[4].x * w), int(landmarks[4].y * h)
            colour = (0, 100, 255) if precise else (0, 255, 255)
            cv2.circle(frame, (px, py), 6, colour, -1)
            cv2.putText(frame,
                f"{'PRECISE' if precise else 'normal'}  head=({hx:.2f},{hy:.2f})  screen=({int(smooth_x)},{int(smooth_y)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        cv2.imshow("face cursor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
