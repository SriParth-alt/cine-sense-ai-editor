import json
import joblib
import numpy as np

MODEL_PATH = "models/cut_strength_model.pkl"
COLOR_MOOD_MAP = {"dark": 0, "cool": 1, "neutral": 2, "warm": 3}
SCENE_TYPE_MAP = {"static_establishing": 0, "slow_scene": 1, "dynamic_scene": 2}
STYLE_MAP = {"nolan": 0, "tarantino": 1, "hirani": 2}


def predict(features_path, emotions_path, transitions_path, style="nolan"):
    model = joblib.load(MODEL_PATH)
    with open(features_path) as f:
        features = json.load(f)
    with open(emotions_path) as f:
        emotions = json.load(f)
    with open(transitions_path) as f:
        transitions = json.load(f)

    out = {}
    for scene_id, feat in features.items():
        emo = emotions.get(scene_id, {})
        tr = transitions.get(scene_id, {})
        X = np.array([[
            feat["motion_mean"],
            feat["audio_energy"],
            feat["tempo"],
            COLOR_MOOD_MAP.get(emo.get("color_mood", "neutral"), 2),
            SCENE_TYPE_MAP.get(tr.get("scene_type", "slow_scene"), 1),
            STYLE_MAP.get(style, 0)
        ]])
        out[scene_id] = {"cut_strength": round(float(model.predict(X)[0]), 3)}
    return out


if __name__ == "__main__":
    res = predict(
        "outputs\\sample1\\scene_features.json",
        "outputs\\sample1\\scene_emotions.json",
        "outputs\\sample1\\scene_transitions.json",
        style="nolan"
    )
    with open("outputs\\sample1\\cut_strength.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Cut-strength predictions generated")
