import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "transition_model.pkl")


def train_model(csv_path):
    # Ensure models directory exists (robust fix)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)

    X = df[["motion", "audio", "tempo", "color_mood", "scene_type"]]
    y = df["label"]

    model = LogisticRegression(max_iter=500)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("Transition ML model trained and saved successfully")


if __name__ == "__main__":
    train_model("outputs\\sample1\\transition_training_data.csv")

