import json
import joblib
import numpy as np

MODEL_PATH = "models/transition_model.pkl"

LABEL_MAP = {
    0: "cut",
    1: "dissolve",
    2: "fade",
    3: "whip_pan"
}

COLOR_MOOD_MAP = {
    "dark": 0,
    "cool": 1,
    "neutral": 2,
    "warm": 3
}

SCENE_TYPE_MAP = {
    "static_establishing": 0,
    "slow_scene": 1,
    "dynamic_scene": 2
}


def predict_transitions(features_path, emotions_path, transitions_path):
    model = joblib.load(MODEL_PATH)

    with open(features_path) as f:
        features = json.load(f)
    with open(emotions_path) as f:
        emotions = json.load(f)
    with open(transitions_path) as f:
        transitions = json.load(f)

    ml_results = {}

    for scene_id, feat in features.items():
        emo = emotions.get(scene_id, {})
        base = transitions.get(scene_id, {})

        X = np.array([[
            feat["motion_mean"],
            feat["audio_energy"],
            feat["tempo"],
            COLOR_MOOD_MAP.get(emo.get("color_mood", "neutral"), 2),
            SCENE_TYPE_MAP.get(base.get("scene_type", "slow_scene"), 1)
        ]])

        pred = model.predict(X)[0]

        ml_results[scene_id] = {
            "ml_recommended_transition": LABEL_MAP[pred]
        }

    return ml_results


if __name__ == "__main__":
    results = predict_transitions(
        "outputs\\sample1\\scene_features.json",
        "outputs\\sample1\\scene_emotions.json",
        "outputs\\sample1\\scene_transitions.json"
    )

    with open("outputs\\sample1\\ml_scene_transitions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("ML-based transition predictions generated")
