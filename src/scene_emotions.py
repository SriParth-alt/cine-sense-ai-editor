import json
import os

# ✅ CORRECT ABSOLUTE IMPORT
from src.emotion_analysis import detect_face_emotion, color_mood_score


def compute_scene_emotions(scenes_json, thumbnails_dir, output_path):
    """
    Computes face-based emotion (if any) and color-based mood for each scene.
    Works safely even when no faces are present.
    """

    with open(scenes_json, "r") as f:
        scenes = json.load(f)

    results = {}

    for scene in scenes:
        scene_id = scene["scene_id"]
        thumb_path = os.path.join(
            thumbnails_dir,
            f"scene_{scene_id:02d}.jpg"
        )

        face_emotion, confidence = detect_face_emotion(thumb_path)
        color_mood = color_mood_score(thumb_path)

        results[f"scene_{scene_id}"] = {
            "face_emotion": face_emotion,
            "confidence": round(confidence, 3),
            "color_mood": color_mood
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
