import cv2
import numpy as np


def compute_motion_between_frames(frame1, frame2):
    """
    Computes optical flow magnitude between two frames.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag), np.std(mag)


def scene_motion_score(video_path, scene, sample_step=5):
    """
    Computes average motion score for a scene by sampling frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(scene["start"] * fps)
    end_frame = int(scene["end"] * fps)

    motions = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0, 0.0

    for frame_id in range(start_frame + sample_step, end_frame, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        mean_mag, std_mag = compute_motion_between_frames(prev_frame, frame)
        motions.append(mean_mag)
        prev_frame = frame

    cap.release()

    if not motions:
        return 0.0, 0.0

    return float(np.mean(motions)), float(np.std(motions))
