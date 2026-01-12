import cv2
import json
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0

    cap.release()

    return {
        "video_file": os.path.basename(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 2)
    }


def extract_audio(video_path, output_audio_path, sample_rate=22050):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(
        output_audio_path,
        fps=sample_rate,
        verbose=False,
        logger=None
    )
    return output_audio_path


def extract_frames(video_path, output_dir, fps=1, resize=None, max_frames=None):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(video_fps // fps))

    frame_idx = 0
    saved = 0
    index_data = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Extracting frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            if resize:
                frame = cv2.resize(frame, resize)

            filename = f"frame_{saved:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)

            timestamp = frame_idx / video_fps
            index_data[filename] = {
                "frame_number": saved,
                "time_seconds": round(timestamp, 3)
            }

            saved += 1
            if max_frames and saved >= max_frames:
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    index_file = os.path.join(output_dir, "frames_index.json")
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)

    return index_file
