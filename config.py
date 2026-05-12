import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VIDEO_FOLDER = Path("videos")
FRAME_FOLDER = Path("artifacts/generated/frames")
OUTPUT_FOLDER = Path("artifacts/generated/analysis")
DB_FOLDER = Path("artifacts/generated/video-db")

for folder in (VIDEO_FOLDER, FRAME_FOLDER, OUTPUT_FOLDER, DB_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

VISION_MODEL = os.getenv("VISION_MODEL", "gemma4:latest")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:latest")
