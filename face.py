import mediapipe as mp
import urllib.request
import os


def load_landmarker(model_path, model_url):
    if not os.path.exists(model_path):
        print("Downloading face landmark model...")
        urllib.request.urlretrieve(model_url, model_path)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def detect(landmarker, bgr_frame):
    import cv2
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


def face_relative_nose(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose  = landmarks[4]

    cx = (left_eye.x + right_eye.x) / 2
    cy = (left_eye.y + right_eye.y) / 2

    face_w = abs(right_eye.x - left_eye.x) + 1e-6
    return (nose.x - cx) / face_w, (nose.y - cy) / face_w


def mouth_open_ratio(landmarks):
    upper = landmarks[13]
    lower = landmarks[14]
    face_w = abs(landmarks[263].x - landmarks[33].x) + 1e-6
    dist = abs(lower.y - upper.y)

    return dist / face_w


def detect_precision_mode(landmarks, counter, cfg):
    if not landmarks:
        return 0, False

    ratio = mouth_open_ratio(landmarks[0])

    is_open = ratio > cfg.mouth_open_threshold

    counter = counter + 1 if is_open else 0

    return counter, counter >= cfg.mouth_open_frames


if __name__ == "__main__":
    import cv2
    from config import Config

    cfg = Config()
    cap = cv2.VideoCapture(0)
    jaw_counter = 0

    with load_landmarker(cfg.model_path, cfg.model_url) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            result = detect(landmarker, frame)

            jaw_counter, precise = detect_precision_mode(
                result.face_landmarks, jaw_counter, cfg)

            jaw_score = mouth_open_ratio(result.face_landmarks[0]) if result.face_landmarks else 0.0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                box_colour = (0, 100, 255) if precise else (0, 255, 0)
                cv2.rectangle(frame,
                    (int(min(xs)), int(min(ys))),
                    (int(max(xs)), int(max(ys))),
                    box_colour, 2)

                hx, hy = face_relative_nose(landmarks)
                nx, ny = int(landmarks[4].x * w), int(landmarks[4].y * h)
                cv2.circle(frame, (nx, ny), 6, (0, 255, 255), -1)

                mode   = "PRECISE" if precise else "normal"
                colour = (0, 100, 255) if precise else (0, 255, 255)
                cv2.putText(frame, f"mode: {mode}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                cv2.putText(frame, f"mouth ratio: {jaw_score:.3f} / {cfg.mouth_open_threshold}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                cv2.putText(frame, f"head: ({hx:.2f}, {hy:.2f})",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            cv2.imshow("face.py debug", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
