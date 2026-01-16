import cv2
import numpy as np


def scene_motion_score(video_path, start_time, end_time):
    """
    Computes average motion score for a scene using optical flow.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_scores = []

    while cap.isOpened():
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time > end_time:
            break

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_scores.append(np.mean(mag))
        prev_gray = gray

    cap.release()
    return float(np.mean(motion_scores)) if motion_scores else 0.0
