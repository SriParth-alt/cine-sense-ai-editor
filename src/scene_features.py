import json
from motion_analysis import scene_motion_score
from audio_analysis import load_audio, audio_features_for_scene


def compute_scene_features(video_path, audio_path, scenes_json, output_path):
    with open(scenes_json, "r") as f:
        scenes = json.load(f)

    y, sr = load_audio(audio_path)

    features = {}

    for scene in scenes:
        scene_id = f"scene_{scene['scene_id']}"

        motion_mean, motion_std = scene_motion_score(
            video_path, scene
        )

        audio_energy, tempo = audio_features_for_scene(
            y, sr, scene
        )

        features[scene_id] = {
            "motion_mean": round(motion_mean, 3),
            "motion_std": round(motion_std, 3),
            "audio_energy": round(audio_energy, 4),
            "tempo": round(tempo, 2)
        }

    with open(output_path, "w") as f:
        json.dump(features, f, indent=2)

    return features


if __name__ == "__main__":
    compute_scene_features(
        video_path="sample_videos\\test.mp4",
        audio_path="outputs\\sample1\\audio.wav",
        scenes_json="outputs\\sample1\\scenes.json",
        output_path="outputs\\sample1\\scene_features.json"
    )

    print("Scene-level motion & audio features generated")
