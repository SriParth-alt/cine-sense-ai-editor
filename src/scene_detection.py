import cv2
import json
import os


def detect_scenes(video_path, output_json):
    """
    Detect scenes and store start frame index for each scene.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    scenes = []
    scene_id = 0
    start_frame = 0

    prev_gray = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = diff.mean()

            if score > 30:  # cut threshold
                scenes.append({
                    "scene_id": scene_id,
                    "start": round(start_frame / fps, 3),
                    "end": round(frame_idx / fps, 3),
                    "start_frame": start_frame
                })
                scene_id += 1
                start_frame = frame_idx

        prev_gray = gray
        frame_idx += 1

    # final scene
    scenes.append({
        "scene_id": scene_id,
        "start": round(start_frame / fps, 3),
        "end": round(frame_idx / fps, 3),
        "start_frame": start_frame
    })

    cap.release()

    with open(output_json, "w") as f:
        json.dump(scenes, f, indent=2)

    return scenes


def generate_scene_thumbnails(video_path, scenes, output_dir):
    """
    Generate thumbnails by sequentially reading frames.
    This works reliably for ALL video types.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    # Map scene start frames
    scene_frames = {
        scene["start_frame"]: scene["scene_id"]
        for scene in scenes
    }

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in scene_frames:
            scene_id = scene_frames[frame_idx]
            thumb_path = os.path.join(
                output_dir,
                f"scene_{scene_id:02d}.jpg"
            )
            cv2.imwrite(thumb_path, frame)

        frame_idx += 1

    cap.release()

