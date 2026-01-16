import json
import pandas as pd
import random

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

TRANSITION_MAP = {
    "cut": 0,
    "dissolve": 1,
    "fade": 2,
    "whip_pan": 3
}

REVERSE_TRANSITION_MAP = {
    0: "cut",
    1: "dissolve",
    2: "fade",
    3: "whip_pan"
}


def augment_label(original_label, scene_type):
    """
    Introduce controlled variability for training.
    """
    if scene_type == 0:  # static_establishing
        return random.choice([0, 1, 2])  # cut, dissolve, fade
    elif scene_type == 1:  # slow_scene
        return random.choice([0, 1])
    else:  # dynamic_scene
        return random.choice([0, 3])


def build_dataset(features_path, emotions_path, transitions_path, output_csv):
    with open(features_path) as f:
        features = json.load(f)
    with open(emotions_path) as f:
        emotions = json.load(f)
    with open(transitions_path) as f:
        transitions = json.load(f)

    rows = []

    for scene_id, feat in features.items():
        emo = emotions.get(scene_id, {})
        trans = transitions.get(scene_id, {})

        scene_type = SCENE_TYPE_MAP.get(trans.get("scene_type", "slow_scene"), 1)
        base_label = TRANSITION_MAP.get(trans.get("recommended_transition", "cut"), 0)

        # Add original sample
        rows.append({
            "motion": feat["motion_mean"],
            "audio": feat["audio_energy"],
            "tempo": feat["tempo"],
            "color_mood": COLOR_MOOD_MAP.get(emo.get("color_mood", "neutral"), 2),
            "scene_type": scene_type,
            "label": base_label
        })

        # Add augmented samples
        for _ in range(2):  # 2 synthetic variations
            rows.append({
                "motion": feat["motion_mean"] * random.uniform(0.9, 1.1),
                "audio": feat["audio_energy"] * random.uniform(0.9, 1.1),
                "tempo": feat["tempo"] * random.uniform(0.95, 1.05),
                "color_mood": COLOR_MOOD_MAP.get(emo.get("color_mood", "neutral"), 2),
                "scene_type": scene_type,
                "label": augment_label(base_label, scene_type)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("Augmented ML training dataset generated")
    return df


if __name__ == "__main__":
    build_dataset(
        "outputs\\sample1\\scene_features.json",
        "outputs\\sample1\\scene_emotions.json",
        "outputs\\sample1\\scene_transitions.json",
        "outputs\\sample1\\transition_training_data.csv"
    )
