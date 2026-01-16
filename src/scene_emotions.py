import json
import os
from emotion_analysis import detect_face_emotion, color_mood_score


def compute_scene_emotions(
    scenes_json,
    thumbnails_dir,
    output_path
):
    with open(scenes_json, "r") as f:
        scenes = json.load(f)

    scene_emotions = {}

    for scene in scenes:
        scene_id = scene["scene_id"]
        thumbnail_path = os.path.join(
            thumbnails_dir,
            f"scene_{scene_id:02d}.jpg"
        )

        face_emotion, confidence = detect_face_emotion(thumbnail_path)
        color_mood = color_mood_score(thumbnail_path)

        scene_emotions[f"scene_{scene_id}"] = {
            "face_emotion": face_emotion,
            "confidence": round(confidence, 3),
            "color_mood": color_mood
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scene_emotions, f, indent=2)

    return scene_emotions


if __name__ == "__main__":
    compute_scene_emotions(
        scenes_json="outputs\\sample1\\scenes.json",
        thumbnails_dir="outputs\\sample1\\scene_thumbnails",
        output_path="outputs\\sample1\\scene_emotions.json"
    )

    print("Scene-level emotions generated successfully")
