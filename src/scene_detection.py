import json
import os
import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip


def detect_scenes(video_path, output_json_path, threshold=15.0):
    """
    Detect scenes using content-based scene detection.
    Always returns at least one scene (fallback).
    """
    video = open_video(video_path)

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    scenes_out = []

    for i, (start, end) in enumerate(scene_list):
        scenes_out.append({
            "scene_id": i,
            "start": round(start.get_seconds(), 3),
            "end": round(end.get_seconds(), 3)
        })

    # -------- FALLBACK (IMPORTANT) --------
    # If no scenes detected, treat whole video as one scene
    if not scenes_out:
        clip = VideoFileClip(video_path)
        scenes_out = [{
            "scene_id": 0,
            "start": 0.0,
            "end": round(clip.duration, 3)
        }]
        clip.close()

    # Save scenes.json
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(scenes_out, f, indent=2)

    return scenes_out


def generate_scene_thumbnails(
    video_path,
    scenes,
    output_dir,
    resize=(320, 180)
):
    """
    Generate one thumbnail per scene using the middle frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for scene in scenes:
        scene_id = scene["scene_id"]
        start = scene["start"]
        end = scene["end"]

        # Middle timestamp of scene
        mid_time = (start + end) / 2.0
        frame_number = int(mid_time * fps)

        # Safety clamp
        frame_number = max(0, min(frame_number, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            print(f"[WARN] Could not read frame for scene {scene_id}")
            continue

        if resize:
            frame = cv2.resize(frame, resize)

        out_path = os.path.join(
            output_dir,
            f"scene_{scene_id:02d}.jpg"
        )
        cv2.imwrite(out_path, frame)

    cap.release()


# -------------------- MAIN RUNNER --------------------

if __name__ == "__main__":
    video_path = "sample_videos\\test.mp4"
    scenes_json_path = "outputs\\sample1\\scenes.json"
    thumbnails_dir = "outputs\\sample1\\scene_thumbnails"

    print("Detecting scenes...")
    scenes = detect_scenes(
        video_path,
        scenes_json_path,
        threshold=15.0
    )

    print(f"Detected {len(scenes)} scenes")

    print("Generating scene thumbnails...")
    generate_scene_thumbnails(
        video_path,
        scenes,
        thumbnails_dir
    )

    print("Scene detection + thumbnails complete")
