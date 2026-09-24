class Config:
    # model
    model_path = "face_landmarker.task"
    model_url  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

    # smoothing, lower = more stable, higher = more responsive
    smooth         = 0.2
    smooth_precise = 0.07

    # dead zone in face width units
    dead_zone         = 0.02
    dead_zone_precise = 0.005

    # head range
    head_x_range = 0.3
    head_y_min   = 0.1
    head_y_max   = 0.6

    # precision mode
    precision_speed      = 0.25
    mouth_open_threshold = 0.08
    mouth_open_frames    = 3

    blink_threshold  = 0.5
    blink_min_frames = 2
    blink_max_frames = 15

    lip_roll_threshold = 0.4  
    lip_roll_frames    = 3   
