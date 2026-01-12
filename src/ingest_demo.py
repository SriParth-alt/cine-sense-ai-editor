import argparse
import os
import json
from video_processing import (
    get_video_metadata,
    extract_audio,
    extract_frames
)


def main():
    parser = argparse.ArgumentParser(description="CineSense Day 1 Ingestion")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--outdir", default="outputs/sample1")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Reading video metadata...")
    metadata = get_video_metadata(args.video)

    print("Extracting audio...")
    audio_path = os.path.join(args.outdir, "audio.wav")
    extract_audio(args.video, audio_path)

    print("Extracting frames...")
    frames_dir = os.path.join(args.outdir, "frames")
    extract_frames(
        args.video,
        frames_dir,
        fps=args.fps,
        resize=(640, 360)
    )

    metadata["audio_file"] = "audio.wav"
    metadata["frames_folder"] = "frames"

    metadata_path = os.path.join(args.outdir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nIngestion complete")
    print("Metadata saved to:", metadata_path)


if __name__ == "__main__":
    main()
