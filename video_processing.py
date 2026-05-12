import json
import cv2
from pathlib import Path

from agent import collection, vision_llm
from config import FRAME_FOLDER, OUTPUT_FOLDER
from langchain_core.documents import Document
from utils import image_to_base64


def analyze_frame(frame_path: str) -> str:
    image_b64 = image_to_base64(frame_path)

    prompt = (
        "Analyze this frame in detail so that another person can answer questions about the scene.\n\n"
        "Use simple, clear language and return a structured summary with these sections:\n"
        "- objects\n"
        "- object positions\n"
        "- actions\n"
        "- lighting\n"
        "- weather\n"
        "- scene\n"
        "- emotions\n"
        "- camera angle\n"
        "- motion\n"
        "- important events\n\n"
        "Include key spatial relationships, visible details, and what is happening in the frame.\n"
        "Be concise, factual, and detailed enough for someone else to answer follow-up questions.\n"
    )

    response = vision_llm.invoke(
        [
            (
                "human",
                [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                ],
            )
        ]
    )

    return response.content


def _save_outputs(video_name: str, all_results: list[dict]) -> None:
    json_path = OUTPUT_FOLDER / f"{video_name}_analysis.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    txt_path = OUTPUT_FOLDER / f"{video_name}_summary.txt"
    with txt_path.open("w", encoding="utf-8") as file:
        for item in all_results:
            file.write("=" * 60 + "\n")
            file.write(f"Timestamp: {item['timestamp']} sec\n")
            file.write(item["analysis"] + "\n\n")


def process_video(video_path: str) -> str:
    video_file = Path(video_path)
    if not video_file.exists():
        return f"Video not found: {video_file}"

    video_name = video_file.stem

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return f"Could not open video: {video_file}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_number = 0
    saved_count = 0
    all_results: list[dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % 30 == 0:
            timestamp = round(frame_number / fps, 2)
            frame_file = FRAME_FOLDER / f"{video_name}_{saved_count}.jpg"
            cv2.imwrite(str(frame_file), frame)

            try:
                analysis = analyze_frame(str(frame_file))
            except Exception as error:
                print("Frame analysis error:", error)
                analysis = "Frame analysis failed"

            data = {
                "video": video_name,
                "frame": saved_count,
                "timestamp": timestamp,
                "analysis": analysis,
            }
            all_results.append(data)

            document = Document(
                page_content=analysis,
                metadata={
                    "video": video_name,
                    "frame": saved_count,
                    "timestamp": timestamp,
                    "path": str(frame_file),
                },
            )
            collection.add_documents(
                documents=[document],
                ids=[f"{video_name}_{saved_count}"],
            )
            saved_count += 1

        frame_number += 1

    cap.release()
    _save_outputs(video_name, all_results)

    return f"Finished processing {video_name}"
