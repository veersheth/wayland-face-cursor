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
    """Nose position in face-width units relative to the eye midpoint."""
    left_eye  = landmarks[33]
    right_eye = landmarks[263]
    nose      = landmarks[4]
    cx     = (left_eye.x + right_eye.x) / 2
    cy     = (left_eye.y + right_eye.y) / 2
    face_w = abs(right_eye.x - left_eye.x) + 1e-6
    return (nose.x - cx) / face_w, (nose.y - cy) / face_w


def detect_precision_mode(blendshapes, counter, cfg):
    """Return updated frame counter and whether precision mode is active."""
    if not blendshapes:
        return 0, False
    scores  = {b.category_name: b.score for b in blendshapes[0]}
    closed  = scores.get("mouthClose", 0) > cfg.mouth_close_threshold
    counter = counter + 1 if closed else 0
    return counter, counter >= cfg.mouth_close_frames
