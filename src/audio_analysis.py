import librosa
import numpy as np


def scene_audio_features(video_path, start_time, end_time):
    """
    Attempts to load audio directly from video.
    If audio is missing, returns safe default values.
    """

    try:
        y, sr = librosa.load(
            video_path,
            offset=start_time,
            duration=max(0.1, end_time - start_time)
        )

        if len(y) == 0:
            return 0.0, 0.0

        energy = float(np.mean(np.square(y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return energy, float(tempo)

    except Exception:
        # Graceful fallback if video has no audio stream
        return 0.0, 0.0
