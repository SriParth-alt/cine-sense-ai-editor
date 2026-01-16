import json
from director_styles import DIRECTOR_STYLES


def classify_scene_context(motion, audio, tempo):
    """
    Classify scene type based on activity level.
    """
    if motion < 0.15 and audio < 0.02:
        return "static_establishing"
    elif motion < 0.35:
        return "slow_scene"
    else:
        return "dynamic_scene"


def recommend_transition(scene_feat, scene_emo, style_name):
    style = DIRECTOR_STYLES[style_name]

    motion = scene_feat["motion_mean"]
    audio = scene_feat["audio_energy"]
    tempo = scene_feat["tempo"]
    mood = scene_emo.get("color_mood", "neutral")
    face_emotion = scene_emo.get("face_emotion", "no_face")

    scene_type = classify_scene_context(motion, audio, tempo)

    # ---------- BASE SCORES ----------
    scores = {
        "cut": style["cut_bias"],
        "dissolve": style["dissolve_bias"],
        "fade": style["fade_bias"],
        "whip_pan": style["whip_bias"]
    }

    # ---------- STATIC / ESTABLISHING SCENE ----------
    if scene_type == "static_establishing":
        preferred = style["static_preference"]
        scores[preferred] += 0.35
        scores["whip_pan"] = 0.01  # almost never for static shots

    # ---------- MOTION-BASED LOGIC ----------
    elif motion >= style["fast_cut_motion"]:
        scores["cut"] += 0.25
        scores["whip_pan"] += 0.1
    else:
        scores["dissolve"] += 0.15
        scores["fade"] += 0.1

    # ---------- AUDIO-BASED LOGIC ----------
    if audio > 0.03 or tempo > 110:
        scores["cut"] += 0.1
    else:
        scores["fade"] += 0.1

    # ---------- EMOTION / MOOD LOGIC ----------
    # If face emotion missing, rely on color mood
    if face_emotion == "no_face":
        if mood == "dark":
            scores["fade"] += 0.1
        elif mood == "cool":
            scores["dissolve"] += 0.1
        elif mood == "warm":
            scores["cut"] += 0.05
    else:
        # face emotion present (future-proof)
        scores["cut"] += 0.05

    # ---------- NORMALIZE ----------
    total = sum(scores.values())
    for k in scores:
        scores[k] = scores[k] / total

    best = max(scores, key=scores.get)
    confidence = round(scores[best], 3)

    return best, confidence, scene_type


def run_transition_engine(features_path, emotions_path, output_path, style_name="nolan"):
    with open(features_path, "r") as f:
        features = json.load(f)

    with open(emotions_path, "r") as f:
        emotions = json.load(f)

    results = {}

    for scene_id, feat in features.items():
        emo = emotions.get(scene_id, {})
        transition, conf, scene_type = recommend_transition(feat, emo, style_name)

        results[scene_id] = {
            "recommended_transition": transition,
            "confidence": conf,
            "style": style_name,
            "scene_type": scene_type
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_transition_engine(
        features_path="outputs\\sample1\\scene_features.json",
        emotions_path="outputs\\sample1\\scene_emotions.json",
        output_path="outputs\\sample1\\scene_transitions.json",
        style_name="nolan"
    )

    print("Robust scene transition recommendations generated")
