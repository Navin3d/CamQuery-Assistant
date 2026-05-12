# ============================================================
# INSTALL
# ============================================================
# pip install -U \
# deepagents \
# langchain-ollama \
# langchain-chroma \
# langchain-community \
# pillow \
# opencv-python \
# chromadb \
# sentence-transformers \
# python-dotenv

# ============================================================
# IMPORTS
# ============================================================

import os
import cv2
import json
import base64
import copy
import gradio as gr

from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from deepagents import create_deep_agent
from langchain_pinecone import PineconeEmbeddings

# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

VIDEO_FOLDER = "./videos"
FRAME_FOLDER = "./artifacts/generated/frames"
OUTPUT_FOLDER = "./artifacts/generated/analysis"
DB_FOLDER = "./artifacts/generated/video-db"

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VISION_MODEL = os.getenv("VISION_MODEL", "gemma4:latest")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:latest")

# ============================================================
# OLLAMA MODELS
# ============================================================

vision_llm = ChatOllama(
    model=VISION_MODEL,
    temperature=0,
)

# llm = ChatOllama(
#     model=CHAT_MODEL,
#     temperature=0,
# )
# chat_llm = RunnableSequence(prompt, llm)

# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = PineconeEmbeddings()

# ============================================================
# VECTOR DATABASE
# ============================================================

collection = Chroma(
    collection_name="video-assistant",
    embedding_function=embeddings,
    persist_directory=DB_FOLDER
)

# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(image_path: str):

    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_frame(frame_path: str):

    image_b64 = image_to_base64(frame_path)

    prompt = """
Analyze this frame in detail so that another person can answer questions about the scene.

Use simple, clear language and return a structured summary with these sections:
- objects
- object positions
- actions
- lighting
- weather
- scene
- emotions
- camera angle
- motion
- important events

Include key spatial relationships, visible details, and what is happening in the frame.
Be concise, factual, and detailed enough for someone else to answer follow-up questions.
"""

    response = vision_llm.invoke(
        [
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}"
                    }
                ]
            )
        ]
    )

    return response.content


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path: str):
    """Process a video by extracting selected frames, analyzing them, and storing results.

    This function opens the provided video file, extracts every 30th frame, saves each frame
    to disk, analyzes the frame with the vision model, stores the analysis in a Chroma
    vector database, and writes summary output files in the analysis folder.

    Args:
        video_path (str): Path to the source video file.

    Returns:
        str: A completion message naming the processed video, or an error message if the
            video cannot be accessed or opened.
    """

    video_path = str(video_path)

    if not os.path.exists(video_path):
        return f"Video not found: {video_path}"

    video_name = Path(video_path).stem

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return f"Could not open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_number = 0
    saved_count = 0

    all_results = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # analyze every 30th frame
        if frame_number % 30 == 0:

            timestamp = round(frame_number / fps, 2)

            frame_file = os.path.join(
                FRAME_FOLDER,
                f"{video_name}_{saved_count}.jpg"
            )

            cv2.imwrite(frame_file, frame)

            print(f"Analyzing frame: {frame_file}")

            try:

                analysis = analyze_frame(frame_file)

            except Exception as e:

                print("Frame analysis error:", e)
                analysis = "Frame analysis failed"

            data = {
                "video": video_name,
                "frame": saved_count,
                "timestamp": timestamp,
                "analysis": analysis
            }

            all_results.append(data)

            # CREATE DOCUMENT
            doc = Document(
                page_content=analysis,
                metadata={
                    "video": video_name,
                    "frame": saved_count,
                    "timestamp": timestamp,
                    "path": frame_file
                }
            )

            # SAVE TO VECTOR DB
            collection.add_documents(
                documents=[doc],
                ids=[f"{video_name}_{saved_count}"]
            )

            saved_count += 1

        frame_number += 1

    cap.release()

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_file = os.path.join(
        OUTPUT_FOLDER,
        f"{video_name}_analysis.json"
    )

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # ========================================================
    # SAVE TXT
    # ========================================================

    txt_file = os.path.join(
        OUTPUT_FOLDER,
        f"{video_name}_summary.txt"
    )

    with open(txt_file, "w", encoding="utf-8") as f:

        for item in all_results:

            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {item['timestamp']} sec\n")
            f.write(item["analysis"] + "\n\n")

    print(f"Finished processing {video_name}")

    return f"Finished processing {video_name}"


# ============================================================
# SEARCH MEMORY
# ============================================================

def search_video_memory(question: str):
    """Search analyzed video frame memories.

    ALWAYS use this tool whenever the user asks about video:
    - what is happening in a video
    - how many people are present
    - objects in scenes
    - actions performed
    - weather
    - lighting
    - timestamps
    - scene summaries
    - descriptions of videos
    - events in videos

    Input:
        Natural language question about analyzed videos.

    Returns:
        Relevant frame analyses from stored vector memory.
    """

    results = collection.similarity_search(
        question,
        k=5
    )

    if not results:
        return "No matching frames found."

    final = []

    for doc in results:

        meta = doc.metadata

        final.append(
            f"""
Video: {meta.get('video')}
Frame: {meta.get('frame')}
Timestamp: {meta.get('timestamp')} sec

Analysis:
{doc.page_content}
"""
        )

    return "\n".join(final)


# ============================================================
# CREATE AGENT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
    You are a CamQuery Assistant.

    Use:
    1. Retrieved video memory
    2. Previous conversation context

    to answer naturally and consistently.

    ==============================
    PREVIOUS CONVERSATION
    ==============================
    {conversation_context}

    ==============================
    RETRIEVED VIDEO MEMORY
    ==============================
    {memory}


    ==============================
    CURRENT USER QUESTION
    ==============================
    {message}

    Instructions:
    - Use ONLY retrieved frame/video data
    - Reference timestamps when possible
    - Understand follow-up questions using chat history
    - If information is unavailable, say so clearly
    """
)

chat_agent = create_deep_agent(
    model=f"ollama:{CHAT_MODEL}",
    tools=[],
)

chat_llm = RunnableSequence(prompt, chat_agent)
# chat_llm = chat_agent

# ========================================================
# MAIN UI
# ========================================================

with gr.Blocks(title="CamQuery Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🎥 CamQuery Assistant")
    gr.Markdown(
        "Upload videos, process them, then ask questions about content, timestamps, objects, and scenes!"
    )

    # States
    current_chat = gr.State([])
    saved_chats = gr.State([])

    with gr.Row():

        # ========================================================
        # LEFT PANEL
        # ========================================================
        with gr.Column(scale=1):

            # Video upload section
            video_upload = gr.File(
                label="📹 Upload Video",
                file_types=[".mp4", ".avi", ".mov", ".mkv"],
                type="filepath"
            )

            process_btn = gr.Button(
                "🔄 Process Video",
                variant="primary"
            )

            status = gr.Textbox(
                label="Status",
                interactive=False
            )

        # ========================================================
        # RIGHT PANEL
        # ========================================================
        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                height=500,
                avatar_images=("./artifacts/pictures/GMC_BW.jpg", "./artifacts/pictures/bot.webp"),
                # type="messages"
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about video content...",
                    label="💬 Your Question",
                    scale=3
                )

                send_btn = gr.Button(
                    "Send",
                    scale=1
                )

            clear_btn = gr.Button("🗑️ Clear Chat")

    # ========================================================
    # PROCESS VIDEO
    # ========================================================
    def process_uploaded_video(video_path):

        if not video_path:
            return "❌ No video uploaded!"

        try:
            result = process_video(video_path)
            return f"✅ {result}"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    # ========================================================
    # CHAT RESPONSE FUNCTION
    # ========================================================
    def respond(message, history):

        if not message.strip():
            return "", history

        try:
            # ====================================================
            # SEARCH VIDEO MEMORY
            # ====================================================
            memory = search_video_memory(message)

            # ====================================================
            # BUILD CONVERSATION CONTEXT
            # ====================================================

            # Keep only last 3 conversation turns
            # (1 turn = user + assistant)
            recent_history = history[-6:]

            conversation_context = ""

            for msg in recent_history:

                role = msg["role"].capitalize()

                conversation_context += (
                    f"{role}: {msg['content']}\n"
                )

            # ====================================================
            # LLM RESPONSE
            # ====================================================

            raw_response = chat_llm.invoke({
                "message": message,
                "conversation_context": conversation_context,
                "memory": memory
            })

            # print("RAW RESPONSE:", raw_response)

            response = raw_response["messages"][-1]

            # response = chat_llm.invoke({
            #     "message": message,
            #     "conversation_context": conversation_context,
            #     "memory": memory
            # })

            # Add messages to history
            history.append({
                "role": "user",
                "content": message
            })

            history.append({
                "role": "assistant",
                "content": response.content
            })

            return "", history

        except Exception as e:

            history.append({
                "role": "user",
                "content": message
            })

            history.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}"
            })

            return "", history

    # ========================================================
    # CLEAR CHAT
    # ========================================================
    def clear_chat():
        return [], ""

    # ========================================================
    # EVENT HANDLERS
    # ========================================================

    # Process video
    process_btn.click(
        process_uploaded_video,
        inputs=[video_upload],
        outputs=[status]
    )

    # Send message
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    send_btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    # Clear current chat
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )

# ========================================================
# LAUNCH
# ========================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )