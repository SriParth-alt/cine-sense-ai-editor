import cv2
import numpy as np
from fer import FER

# Initialize emotion detector once
emotion_detector = FER(mtcnn=True)


def detect_face_emotion(image_path):
    """
    Safely detects dominant facial emotion from an image.
    If no face or any error occurs, returns ('no_face', 0.0).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "unknown", 0.0

        results = emotion_detector.detect_emotions(img)

        if not results:
            return "no_face", 0.0

        emotions = results[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)

        return dominant_emotion, float(emotions[dominant_emotion])

    except Exception:
        # Any FER / TensorFlow / MTCNN error → graceful fallback
        return "no_face", 0.0


def color_mood_score(image_path):
    """
    Estimates mood from color and brightness.
    Works even when no faces are present.
    """
    img = cv2.imread(image_path)
    if img is None:
        return "unknown"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2]

    avg_hue = np.mean(hue)
    avg_value = np.mean(value)

    if avg_value < 60:
        return "dark"
    elif avg_hue < 30:
        return "warm"
    elif avg_hue > 90:
        return "cool"
    else:
        return "neutral"
