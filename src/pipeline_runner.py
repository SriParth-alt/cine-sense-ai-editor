import os
import json

from src.scene_detection import detect_scenes, generate_scene_thumbnails
from src.scene_features import compute_scene_features
from src.scene_emotions import compute_scene_emotions
from src.transition_engine import run_transition_engine
from src.ml_cut_strength_predictor import predict


def run_full_pipeline(video_path, output_dir, style="nolan"):
    """
    Runs the complete CineSense pipeline on any uploaded video.
    Audio is optional and never causes failure.
    """

    os.makedirs(output_dir, exist_ok=True)

    thumbnails_dir = os.path.join(output_dir, "scene_thumbnails")
    os.makedirs(thumbnails_dir, exist_ok=True)

    # 1. Scene Detection
    scenes_json = os.path.join(output_dir, "scenes.json")
    scenes = detect_scenes(video_path, scenes_json)

    # 2. Scene Thumbnails
    generate_scene_thumbnails(video_path, scenes, thumbnails_dir)

    # 3. Scene Features (Motion + Optional Audio)
    features_json = os.path.join(output_dir, "scene_features.json")
    compute_scene_features(
        video_path=video_path,
        scenes_json=scenes_json,
        output_path=features_json
    )

    # 4. Emotion & Mood
    emotions_json = os.path.join(output_dir, "scene_emotions.json")
    compute_scene_emotions(scenes_json, thumbnails_dir, emotions_json)

    # 5. Transition Recommendation
    transitions_json = os.path.join(output_dir, "scene_transitions.json")
    run_transition_engine(
        features_json,
        emotions_json,
        transitions_json,
        style
    )

    # 6. Cut Strength ML
    cut_strength = predict(
        features_json,
        emotions_json,
        transitions_json,
        style
    )

    with open(os.path.join(output_dir, "cut_strength.json"), "w") as f:
        json.dump(cut_strength, f, indent=2)

    return True

