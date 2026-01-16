import json

from src.motion_analysis import scene_motion_score
from src.audio_analysis import scene_audio_features


def compute_scene_features(video_path, scenes_json, output_path):
    """
    Computes motion and audio features for each detected scene.
    Audio extraction is fully optional and safe.
    """

    with open(scenes_json, "r") as f:
        scenes = json.load(f)

    features = {}

    for scene in scenes:
        scene_id = scene["scene_id"]
        start = scene["start"]
        end = scene["end"]

        motion = scene_motion_score(video_path, start, end)
        audio_energy, tempo = scene_audio_features(video_path, start, end)

        features[f"scene_{scene_id}"] = {
            "motion_mean": round(motion, 4),
            "audio_energy": round(audio_energy, 6),
            "tempo": round(tempo, 2)
        }

    with open(output_path, "w") as f:
        json.dump(features, f, indent=2)

    return features

