import json
import pandas as pd
import random

COLOR_MOOD_MAP = {"dark": 0, "cool": 1, "neutral": 2, "warm": 3}
SCENE_TYPE_MAP = {"static_establishing": 0, "slow_scene": 1, "dynamic_scene": 2}
STYLE_MAP = {"nolan": 0, "tarantino": 1, "hirani": 2}


def base_strength(scene_type, motion, audio):
    if scene_type == 0:
        return random.uniform(0.15, 0.35)
    if motion > 0.6 or audio > 0.04:
        return random.uniform(0.7, 0.9)
    return random.uniform(0.4, 0.6)


def build_dataset(features_path, emotions_path, transitions_path, output_csv, style="nolan"):
    with open(features_path) as f:
        features = json.load(f)
    with open(emotions_path) as f:
        emotions = json.load(f)
    with open(transitions_path) as f:
        transitions = json.load(f)

    rows = []
    for scene_id, feat in features.items():
        emo = emotions.get(scene_id, {})
        tr = transitions.get(scene_id, {})
        stype = SCENE_TYPE_MAP.get(tr.get("scene_type", "slow_scene"), 1)

        y = base_strength(stype, feat["motion_mean"], feat["audio_energy"])

        for _ in range(3):  # augment
            rows.append({
                "motion": feat["motion_mean"] * random.uniform(0.9, 1.1),
                "audio": feat["audio_energy"] * random.uniform(0.9, 1.1),
                "tempo": feat["tempo"] * random.uniform(0.95, 1.05),
                "color_mood": COLOR_MOOD_MAP.get(emo.get("color_mood", "neutral"), 2),
                "scene_type": stype,
                "style": STYLE_MAP.get(style, 0),
                "cut_strength": min(1.0, max(0.0, y + random.uniform(-0.05, 0.05)))
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print("Cut-strength training dataset generated")


if __name__ == "__main__":
    build_dataset(
        "outputs\\sample1\\scene_features.json",
        "outputs\\sample1\\scene_emotions.json",
        "outputs\\sample1\\scene_transitions.json",
        "outputs\\sample1\\cut_strength_training_data.csv",
        style="nolan"
    )
