import os
import pandas as pd
import joblib
from sklearn.linear_model import Ridge

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cut_strength_model.pkl")


def train(csv_path):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)

    X = df[["motion", "audio", "tempo", "color_mood", "scene_type", "style"]]
    y = df["cut_strength"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("Cut-strength model trained and saved")


if __name__ == "__main__":
    train("outputs\\sample1\\cut_strength_training_data.csv")
