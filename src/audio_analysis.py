import librosa
import numpy as np


def load_audio(audio_path, sr=22050):
    return librosa.load(audio_path, sr=sr)


def audio_features_for_scene(y, sr, scene):
    """
    Computes audio energy and tempo for a scene.
    """
    start_sample = int(scene["start"] * sr)
    end_sample = int(scene["end"] * sr)

    segment = y[start_sample:end_sample]
    if len(segment) == 0:
        return 0.0, 0.0

    rms = np.mean(librosa.feature.rms(y=segment))
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)

    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    tempo = float(tempo[0]) if len(tempo) > 0 else 0.0

    return float(rms), tempo
